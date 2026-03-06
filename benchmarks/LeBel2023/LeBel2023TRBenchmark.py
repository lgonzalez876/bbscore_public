import os
import pickle
import datetime
import numpy as np
import torch
from typing import Union, List, Optional

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.datasets import get_data_home
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from data.LeBel2023 import (
    LeBel2023TRStimulusSet, LeBel2023TRAssembly,
    LeBel2023FreeSurferLabels, RegionMapper,
)
from metrics import METRICS
from metrics.utils import pearson_correlation_scorer
from models import get_model_class_and_id
from benchmarks import BENCHMARK_REGISTRY


class LeBel2023TRBenchmark:
    """
    TR-level temporal alignment benchmark for LeBel et al. (2023).

    Pipeline:
    1. Parse TextGrid files with word-level timestamps
    2. Bin words into TRs (default 2s)
    3. For each TR, run language model on cumulative context
    4. Load fMRI time series per story (no temporal averaging)
    5. Apply HRF delay (shift features forward by hrf_delay TRs)
    6. Concatenate all stories' TRs
    7. Run ridge regression with story-level GroupKFold CV
    8. Apply language mask (threshold on prediction correlation)
    """

    def __init__(
        self,
        model_identifier: str,
        layer_name: Union[str, List[str]],
        subject_id: str = 'UTS01',
        tr_duration: float = 2.0,
        hrf_delay: int = 2,
        n_cv_folds: int = 5,
        lang_mask_threshold: float = 0.05,
        batch_size: Union[int, List[int]] = None,
        debug: bool = False,
    ):
        if batch_size is None:
            batch_size = [4]
        self.debug = debug
        self.subject_id = subject_id
        self.tr_duration = tr_duration
        self.hrf_delay = hrf_delay
        self.n_cv_folds = n_cv_folds
        self.lang_mask_threshold = lang_mask_threshold
        self.model_identifier = model_identifier

        if isinstance(batch_size, list):
            self.batch_size = batch_size[0]
        else:
            self.batch_size = batch_size

        self.layer_name = layer_name
        self.layer_names = (
            layer_name if isinstance(layer_name, list) else [layer_name]
        )

        # Initialize model
        self.model_class, self.model_id_mapping = get_model_class_and_id(
            model_identifier)
        self.model_instance = self.model_class()
        self.model = self.model_instance.get_model(self.model_id_mapping)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        self.model.to(self.device)

        # Hook storage and registration
        self.features = {l: [] for l in self.layer_names}
        self._register_hooks()

        self.metrics = {}
        self.metric_params = {}
        self.use_ridge_smart_memory = False

        data_home = get_data_home()
        results_base = os.environ.get('RESULTS_PATH', data_home)
        self.results_dir = os.path.join(results_base, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        def hook_fn_factory(layer_id):
            def hook_fn(module, input, output):
                self.features[layer_id].append(output)
            return hook_fn

        for l_name in self.layer_names:
            found = False
            for name, module in self.model.named_modules():
                if name == l_name:
                    if isinstance(module, torch.nn.ModuleList):
                        module[-1].register_forward_hook(
                            hook_fn_factory(l_name))
                    else:
                        module.register_forward_hook(
                            hook_fn_factory(l_name))
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Layer {l_name} not found in model.")

    def _extract_tr_feature(self, text: str) -> Optional[np.ndarray]:
        """
        Run the language model on a text string and extract the feature
        vector via the model's preprocess -> forward -> postprocess flow.

        Returns:
            Feature vector of shape (D,), or None if text is empty.
        """
        if not text.strip():
            return None

        # Clear hook storage
        for l in self.layer_names:
            self.features[l] = []

        # Preprocess (pushes alignment metadata to deque)
        input_ids = self.model_instance.preprocess_fn(text)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        # Forward pass
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            _ = self.model(input_ids)

        # Collect features from all target layers
        layer_features = {}
        for l_name in self.layer_names:
            if not self.features[l_name]:
                raise ValueError(
                    f"No features captured for layer {l_name}")
            feat = self.features[l_name][0]
            if isinstance(feat, tuple):
                feat = feat[0]

            # feat shape: (1, seq_len, D) or (1, D)
            # Use postprocess_fn to get word-aligned, last-word embedding
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            processed = self.model_instance.postprocess_fn(feat)
            if isinstance(processed, torch.Tensor):
                processed = processed.cpu().numpy()
            layer_features[l_name] = processed.squeeze()

        if len(self.layer_names) == 1:
            return layer_features[self.layer_names[0]]
        return np.concatenate(
            [layer_features[l] for l in self.layer_names], axis=-1)

    def _extract_story_features(
        self, stimulus_set: LeBel2023TRStimulusSet, story_idx: int
    ) -> np.ndarray:
        """
        Extract features for all TRs of a single story.

        Returns:
            features: (n_TRs, D) array
        """
        cumulative_texts, n_trs = stimulus_set.get_tr_texts(story_idx)

        if n_trs == 0:
            return np.array([])

        features_list = []
        for tr_idx in range(n_trs):
            feat = self._extract_tr_feature(cumulative_texts[tr_idx])
            features_list.append(feat)

        # Determine feature dimensionality from first non-None
        feat_dim = None
        for f in features_list:
            if f is not None:
                feat_dim = f.shape[-1]
                break

        if feat_dim is None:
            return np.array([])

        result = np.zeros((n_trs, feat_dim), dtype=np.float32)
        for i, f in enumerate(features_list):
            if f is not None:
                result[i] = f

        return result

    def _compute_temporal_rsa(self, per_story_features, per_story_fmri,
                              lang_mask=None):
        """
        Within-story temporal RSA.

        For each story:
          1. Compute model RDM: correlation distance between TRs
          2. Compute neural RDM: correlation distance between TRs
          3. Compare upper triangles with Spearman correlation

        Args:
            per_story_features: list of (n_TRs, D) arrays
            per_story_fmri: list of (n_TRs, n_voxels) arrays
            lang_mask: optional boolean mask over voxels

        Returns:
            dict with per-story and average temporal RSA scores
        """
        story_scores = []
        story_scores_lang = []

        for story_feat, story_fmri in zip(
                per_story_features, per_story_fmri):
            n_trs = story_feat.shape[0]
            if n_trs < 3:
                continue

            # Model RDM (correlation distance between TRs)
            model_rdm = pdist(story_feat, metric='correlation')

            # Neural RDM on all voxels
            neural_rdm = pdist(story_fmri, metric='correlation')

            # Handle NaN from constant rows
            valid = ~(np.isnan(model_rdm) | np.isnan(neural_rdm))
            if valid.sum() < 3:
                continue

            rho, _ = spearmanr(model_rdm[valid], neural_rdm[valid])
            story_scores.append(rho)

            # Lang-masked neural RDM
            if lang_mask is not None and np.sum(lang_mask) > 0:
                neural_rdm_lang = pdist(
                    story_fmri[:, lang_mask], metric='correlation')
                valid_lang = ~(
                    np.isnan(model_rdm) | np.isnan(neural_rdm_lang))
                if valid_lang.sum() >= 3:
                    rho_lang, _ = spearmanr(
                        model_rdm[valid_lang],
                        neural_rdm_lang[valid_lang])
                    story_scores_lang.append(rho_lang)

        results = {
            'per_story_rsa': np.array(story_scores),
            'median_rsa_all': (
                float(np.median(story_scores))
                if story_scores else 0.0),
            'n_stories': len(story_scores),
        }

        if story_scores_lang:
            results['per_story_rsa_lang'] = np.array(story_scores_lang)
            results['median_rsa_lang'] = float(
                np.median(story_scores_lang))

        return results

    def initialize_rp(self, rp):
        """Compatibility with run.py."""
        if rp is not None:
            print("Warning: Random projection not supported "
                  "for TR-level benchmark. Ignoring.")

    def initialize_aggregation(self, mode):
        """Compatibility with run.py."""
        pass

    def add_metric(self, name, metric_params=None):
        self.metrics[name] = METRICS[name]
        if metric_params:
            self.metric_params[name] = metric_params

    def run(self):
        """
        Main TR-level encoding model pipeline.
        """
        # 1. Load stimuli with timestamps
        print("Loading TR-level stimuli...")
        stimulus_set = LeBel2023TRStimulusSet(
            tr_duration=self.tr_duration
        )

        # 2. Load fMRI time series
        print(f"Loading fMRI assembly for {self.subject_id}...")
        assembly = LeBel2023TRAssembly(
            subjects=[self.subject_id]
        )
        story_fmri, ceiling, ceiling_mask = assembly.get_assembly(
            story_names=stimulus_set.story_names
        )

        # 3. Extract features and align with fMRI per story
        all_features = []
        all_fmri = []
        all_story_labels = []

        for story_idx, story_name in enumerate(stimulus_set.story_names):
            if story_name not in story_fmri:
                print(f"Warning: No fMRI data for story "
                      f"'{story_name}', skipping.")
                continue

            print(f"Processing story {story_idx + 1}/"
                  f"{len(stimulus_set.story_names)}: {story_name}")

            story_features = self._extract_story_features(
                stimulus_set, story_idx)
            fmri_data = story_fmri[story_name]

            if story_features.size == 0:
                print(f"  Skipping {story_name}: no features extracted.")
                continue

            # Align TR counts
            n_trs_feat = story_features.shape[0]
            n_trs_fmri = fmri_data.shape[0]
            n_trs = min(n_trs_feat, n_trs_fmri)

            if n_trs < self.hrf_delay + 1:
                print(f"  Skipping {story_name}: "
                      f"too few TRs ({n_trs})")
                continue

            story_features = story_features[:n_trs]
            fmri_data = fmri_data[:n_trs]

            # 4. Apply HRF delay
            # Feature at TR_t predicts fMRI at TR_(t + hrf_delay)
            shifted_features = story_features[:n_trs - self.hrf_delay]
            shifted_fmri = fmri_data[self.hrf_delay:]

            all_features.append(shifted_features)
            all_fmri.append(shifted_fmri)
            all_story_labels.extend(
                [story_idx] * shifted_features.shape[0])

            if self.debug:
                print(f"  TRs: feat={n_trs_feat}, fmri={n_trs_fmri}, "
                      f"used={shifted_features.shape[0]}")

        if not all_features:
            raise ValueError("No data after alignment.")

        # Determine which analyses to run
        run_ridge = any(
            m in self.metrics for m in ['ridge', 'torch_ridge'])
        run_temporal_rsa = 'temporal_rsa' in self.metrics
        # Default to ridge if no recognized metric
        if not run_ridge and not run_temporal_rsa:
            run_ridge = True

        results = {}
        lang_mask = None
        region_mapper = None

        # 5. Ridge regression
        if run_ridge:
            X = np.concatenate(all_features, axis=0)
            y = np.concatenate(all_fmri, axis=0)
            groups = np.array(all_story_labels)

            print(f"Total TRs: {X.shape[0]}, "
                  f"Feature dim: {X.shape[1]}, "
                  f"Voxels: {y.shape[1]}")

            fold_scores = self._run_group_kfold(X, y, groups)

            median_pearson = np.median(
                fold_scores['pearson'], axis=0)
            median_r2 = np.median(fold_scores['r2'], axis=0)

            # Apply ceiling filter to all scoring.
            # ceiling_mask: voxels with reliable inter-subject
            # agreement (ISC > 0.01). Excludes noise-floor voxels
            # (~40% of whole-brain mask).
            cm = ceiling_mask
            n_cm = int(cm.sum())

            # Intersect language mask with ceiling filter
            lang_mask = (
                (median_pearson > self.lang_mask_threshold) & cm)
            n_lang_voxels = int(np.sum(lang_mask))
            print(f"Ceiling filter: {n_cm}/{y.shape[1]} voxels")
            print(f"Language mask: {n_lang_voxels} voxels "
                  f"(r>{self.lang_mask_threshold} & ceiling)")

            ridge_key = ('torch_ridge' if 'torch_ridge' in self.metrics
                         else 'ridge')

            ceil_vals = ceiling[cm]
            results[ridge_key] = {
                'raw_pearson': fold_scores['pearson'],
                'raw_r2': fold_scores['r2'],
                'median_pearson_all': median_pearson,
                'median_r2_all': median_r2,
                'ceiling_mask': cm,
                'n_ceiling_voxels': n_cm,
                'final_pearson_unceiled': float(
                    np.median(median_pearson[cm])),
                'final_pearson_ceiled': float(
                    np.median(median_pearson[cm] / ceil_vals)),
                'final_r2_unceiled': float(
                    np.median(median_r2[cm])),
                'final_r2_ceiled': float(
                    np.median(median_r2[cm] / ceil_vals)),
                'lang_mask': lang_mask,
                'n_lang_voxels': n_lang_voxels,
                'final_pearson_lang_unceiled': (
                    float(np.median(median_pearson[lang_mask]))
                    if n_lang_voxels > 0 else 0.0),
                'final_pearson_lang_ceiled': (
                    float(np.median(
                        median_pearson[lang_mask]
                        / ceiling[lang_mask]))
                    if n_lang_voxels > 0 else 0.0),
                'final_r2_lang_unceiled': (
                    float(np.median(median_r2[lang_mask]))
                    if n_lang_voxels > 0 else 0.0),
                'final_r2_lang_ceiled': (
                    float(np.median(
                        median_r2[lang_mask]
                        / ceiling[lang_mask]))
                    if n_lang_voxels > 0 else 0.0),
            }
            r = results[ridge_key]
            print(f"Ridge ({n_cm} voxels): "
                  f"unceiled Pearson={r['final_pearson_unceiled']:.4f}, "
                  f"ceiled={r['final_pearson_ceiled']:.4f}")
            print(f"Ridge ({n_cm} voxels): "
                  f"unceiled R2={r['final_r2_unceiled']:.4f}, "
                  f"ceiled={r['final_r2_ceiled']:.4f}")
            if n_lang_voxels > 0:
                print(
                    f"Ridge lang ({n_lang_voxels}): "
                    f"unceiled Pearson="
                    f"{r['final_pearson_lang_unceiled']:.4f}, "
                    f"ceiled="
                    f"{r['final_pearson_lang_ceiled']:.4f}")

            # --- Region-based scoring ---
            # Scores for excluded voxels set to NaN so RegionMapper
            # only aggregates ceiling-filtered voxels.
            try:
                fs_labels = LeBel2023FreeSurferLabels()
                label_dir = fs_labels.ensure_downloaded(
                    self.subject_id)
                region_mapper = RegionMapper(label_dir)

                mp_masked = median_pearson.copy()
                mr2_masked = median_r2.copy()
                mp_masked[~cm] = np.nan
                mr2_masked[~cm] = np.nan
                region_scores = region_mapper.aggregate_scores(
                    mp_masked, mr2_masked)
                results[ridge_key]['regions'] = region_scores

                # Anatomical language mask intersected with ceiling
                anat_lang_idx = region_mapper.get_language_indices()
                n_voxels = y.shape[1]
                anat_lang_mask = np.zeros(n_voxels, dtype=bool)
                valid_idx = anat_lang_idx[
                    anat_lang_idx < n_voxels]
                anat_lang_mask[valid_idx] = True
                anat_lang_mask &= cm

                results[ridge_key]['lang_mask_functional'] = lang_mask
                lang_mask = anat_lang_mask
                results[ridge_key]['lang_mask'] = lang_mask
                n_lang_voxels = int(np.sum(lang_mask))
                results[ridge_key]['n_lang_voxels'] = n_lang_voxels
                results[ridge_key]['final_pearson_lang_unceiled'] = (
                    float(np.median(median_pearson[lang_mask]))
                    if n_lang_voxels > 0 else 0.0)
                results[ridge_key]['final_pearson_lang_ceiled'] = (
                    float(np.median(
                        median_pearson[lang_mask]
                        / ceiling[lang_mask]))
                    if n_lang_voxels > 0 else 0.0)
                results[ridge_key]['final_r2_lang_unceiled'] = (
                    float(np.median(median_r2[lang_mask]))
                    if n_lang_voxels > 0 else 0.0)
                results[ridge_key]['final_r2_lang_ceiled'] = (
                    float(np.median(
                        median_r2[lang_mask]
                        / ceiling[lang_mask]))
                    if n_lang_voxels > 0 else 0.0)

                lang_grp = region_scores['per_group'].get(
                    'language', {})
                print(
                    f"Region scoring: language group "
                    f"Pearson="
                    f"{lang_grp.get('median_pearson', 0):.4f}, "
                    f"non-language="
                    f"{region_scores['per_group'].get('non_language', {}).get('median_pearson', 0):.4f}")
            except Exception as e:
                print(f"Warning: Region-based scoring failed: {e}")

        # 6. Within-story temporal RSA
        if run_temporal_rsa:
            print("Computing within-story temporal RSA...")
            rsa_results = self._compute_temporal_rsa(
                all_features, all_fmri, lang_mask=lang_mask)
            results['temporal_rsa'] = rsa_results
            print(f"Temporal RSA - "
                  f"Median (all voxels): "
                  f"{rsa_results['median_rsa_all']:.4f} "
                  f"({rsa_results['n_stories']} stories)")
            if 'median_rsa_lang' in rsa_results:
                print(f"Temporal RSA - "
                      f"Median (lang voxels): "
                      f"{rsa_results['median_rsa_lang']:.4f}")

            # --- Region-based temporal RSA ---
            try:
                if region_mapper is None:
                    fs_labels = LeBel2023FreeSurferLabels()
                    label_dir = fs_labels.ensure_downloaded(
                        self.subject_id)
                    region_mapper = RegionMapper(label_dir)
                rsa_region_scores = region_mapper.aggregate_rsa_scores(
                    all_features, all_fmri)
                results['temporal_rsa']['regions'] = rsa_region_scores
                lang_rsa = rsa_region_scores['per_group'].get(
                    'language', {})
                print(
                    f"Region RSA: language="
                    f"{lang_rsa.get('median_rsa', 0):.4f}, "
                    f"non-language="
                    f"{rsa_region_scores['per_group'].get('non_language', {}).get('median_rsa', 0):.4f}")
            except Exception as e:
                print(f"Warning: Region-based RSA failed: {e}")

        results['timestamp'] = datetime.datetime.utcnow().isoformat()
        results['hrf_delay'] = self.hrf_delay
        results['tr_duration'] = self.tr_duration
        results['lang_mask_threshold'] = self.lang_mask_threshold
        results['n_stories'] = len(
            [s for s in stimulus_set.story_names if s in story_fmri])
        n_total_trs = sum(f.shape[0] for f in all_features)
        results['total_trs'] = n_total_trs

        # Save results
        layer_str = (self.layer_name if isinstance(self.layer_name, str)
                     else "_".join(self.layer_names))
        benchmark_name = self.__class__.__name__
        results_file = os.path.join(
            self.results_dir,
            f"{self.model_identifier}_{layer_str}_"
            f"{benchmark_name}.pkl"
        )

        merged = {
            "metrics": results,
            "ceiling": ceiling,
            "ceiling_mask": ceiling_mask,
        }
        with open(results_file, 'wb') as f:
            pickle.dump(merged, f)
        print(f"Results saved to {results_file}")

        return merged

    def _run_group_kfold(self, X, y, groups):
        """
        Ridge regression with GroupKFold CV (stories as groups).

        Returns:
            dict with 'pearson' and 'r2' keys, each (n_folds, n_voxels)
        """
        n_unique_groups = len(np.unique(groups))
        n_splits = min(self.n_cv_folds, n_unique_groups)

        if n_splits < 2:
            raise ValueError(
                f"Need at least 2 story groups for CV, "
                f"got {n_unique_groups}")

        gkf = GroupKFold(n_splits=n_splits)

        alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0,
                  100.0, 1e4, 1e6]

        pearson_scores = []
        r2_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(
                gkf.split(X, y, groups)):
            print(f"  Fold {fold_idx + 1}/{n_splits}: "
                  f"train={len(train_idx)}, val={len(val_idx)}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = RidgeCV(alphas=alphas, store_cv_results=False)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            # Per-voxel Pearson correlation
            fold_pearson = np.array([
                pearson_correlation_scorer(y_val[:, i], preds[:, i])
                for i in range(y.shape[1])
            ])
            pearson_scores.append(fold_pearson)

            # Per-voxel R2
            fold_r2 = np.array([
                r2_score(y_val[:, i], preds[:, i])
                for i in range(y.shape[1])
            ])
            r2_scores.append(fold_r2)

        return {
            'pearson': np.array(pearson_scores),
            'r2': np.array(r2_scores),
        }
