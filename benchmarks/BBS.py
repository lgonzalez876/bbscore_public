import os
import pickle
import configparser
import datetime
import getpass
import numpy as np
import pickle
import subprocess
import torch
import torch.nn as nn
import psutil
from torch.utils.data import DataLoader
from typing import Union, List

from sklearn.datasets import get_data_home

from extractor_wrapper import FeatureExtractor
from metrics import METRICS
from models import get_model_class_and_id, MODEL_REGISTRY
from data.utils import custom_collate  # custom collate function


def _run_git(cmd):
    try:
        out = subprocess.check_output(["git"] + cmd,
                                      stderr=subprocess.DEVNULL)
        return out.strip().decode()
    except Exception:
        return None


def get_local_commit():
    return _run_git(["rev-parse", "HEAD"])


def is_worktree_clean():
    status = _run_git(["status", "--porcelain"])
    return status == ""


def estimate_gram_bytes(N):
    return N * N * np.dtype(np.float64).itemsize


def get_mem_info():
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 2**30, vm.available / 2**30, vm.total / 2**30


class BenchmarkScore:
    def __init__(
        self,
        stimulus_train_class,
        model_identifier,
        layer_name: Union[str, List[str]] = None,
        stimulus_test_class=None,
        assembly_class=None,
        assembly_train_kwargs=None,
        assembly_test_kwargs=None,
        batch_size=32,
        num_workers=4,
        task='neural',
        save_features=False,
        debug=False,
        safety_factor=0.8,
        random_projection=None,
        # aggregation_mode removed from init to support inheritance cleanly
    ):

        self.debug = debug
        self.safety_factor = safety_factor
        self.random_projection = random_projection
        self.use_ridge_smart_memory = False
        self.aggregation_mode = "none"  # Default

        # Instantiate model and preprocessing
        self.model_class, self.model_id_mapping = get_model_class_and_id(
            model_identifier)
        self.model_instance = self.model_class()
        self.model_identifier = model_identifier

        # layer_name can be list or str
        self.layer_names = layer_name if isinstance(
            layer_name, list) else [layer_name]
        # Keep original for filename if single
        self.layer_name = layer_name

        # Prepare stimuli
        self.stimulus_train = stimulus_train_class(
            preprocess=self.model_instance.preprocess_fn
        )
        self.stimulus_test = None
        if stimulus_test_class is not None:
            self.stimulus_test = stimulus_test_class(
                preprocess=self.model_instance.preprocess_fn
            )

        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) == 2:
                self.batch_size, self.test_batch_size = batch_size[0], batch_size[1]
            else:
                self.batch_size = batch_size[0]
                self.test_batch_size = None
        else:
            self.batch_size = int(batch_size)
            self.test_batch_size = None

        # Retrieve the model and create a feature extractor
        self.model = self.model_instance.get_model(self.model_id_mapping)
        self.extractor = FeatureExtractor(self.model, self.layer_names,
                                          postprocess_fn=self.model_instance.postprocess_fn,
                                          batch_size=self.batch_size,
                                          num_workers=num_workers,
                                          static=self.model_instance.static,
                                          aggregation_mode="none")  # Default init

        self.task = task
        self.metrics = {}
        self.metric_params = {}

        self.assembly_class = assembly_class
        self.assembly_train_kwargs = assembly_train_kwargs or {}
        self.assembly_test_kwargs = assembly_test_kwargs or {}

        data_home = get_data_home()
        self.features_path = os.path.join(data_home, 'features')
        os.makedirs(self.features_path, exist_ok=True)

        results_base = os.environ.get('RESULTS_PATH', data_home)
        self.results_dir = os.path.join(results_base, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        self.save_features = save_features

    def initialize_rp(self, rp):
        self.extractor.random_projection = rp

    def initialize_aggregation(self, mode):
        """
        Initialize aggregation mode after instantiation.
        This allows subclasses to inherit without modifying __init__ signatures.
        """
        self.aggregation_mode = mode
        self.extractor.aggregation_mode = mode

    def add_metric(self, name, metric_params=None):
        self.metrics[name] = METRICS[name]
        if metric_params:
            self.metric_params[name] = metric_params

    def _process_single_layer_result(self, features_train, features_test, labels_train, labels_test, current_layer_name):
        # Handle dict features (aggregation_mode="none") by extracting the current layer
        if isinstance(features_train, dict):
            if current_layer_name in features_train:
                features_train = features_train[current_layer_name]
            elif len(features_train) == 1:
                features_train = next(iter(features_train.values()))
            else:
                raise ValueError(
                    f"features_train is a dict with keys {list(features_train.keys())} "
                    f"but current_layer_name '{current_layer_name}' not found."
                )
        if isinstance(features_test, dict):
            if current_layer_name in features_test:
                features_test = features_test[current_layer_name]
            elif len(features_test) == 1:
                features_test = next(iter(features_test.values()))
            else:
                features_test = None
        stratify_labels_train = None
        if self.assembly_class:
            assembly = self.assembly_class()
            try:
                assembly_train_data = assembly.get_assembly(
                    **self.assembly_train_kwargs)
                if len(assembly_train_data) == 3:
                    target_train, ceiling, stratify_labels_train = assembly_train_data
                elif len(assembly_train_data) == 2:
                    target_train, ceiling = assembly_train_data
                else:
                    raise ValueError(
                        f"Assembly get_assembly returned {len(assembly_train_data)} values.")
            except Exception as e:
                print(f"Error calling get_assembly for training: {e}")
                raise

            if self.stimulus_test is not None:
                target_test, _ = assembly.get_assembly(
                    **self.assembly_test_kwargs)
            else:
                target_test = None
        else:
            target_train = labels_train
            target_test = labels_test
            ceiling = None
            stratify_labels_train = None

        results = {}
        n_metrics = len(self.metrics)
        for name, metric_class in self.metrics.items():
            try:
                extra = self.metric_params.get(name, {})
                if ceiling is not None:
                    metric_instance = metric_class(
                        ceiling=ceiling, **extra)
                else:
                    metric_instance = metric_class(**extra)
                results[name] = metric_instance.compute(
                    features_train,
                    target_train,
                    test_source=features_test,
                    test_target=target_test,
                    stratify_on=stratify_labels_train
                )
            except Exception as e:
                if n_metrics == 1:
                    raise RuntimeError(f"Metric '{name}' failed: {e}") from e
                print(f"⚠️ Metric '{name}' failed: {e}")
                continue

        results['timestamp'] = datetime.datetime.utcnow().isoformat()
        results['aggregation_mode'] = self.aggregation_mode

        benchmark_name = self.__class__.__name__
        results_file = os.path.join(
            self.results_dir,
            f"{self.model_identifier}_{current_layer_name}_{benchmark_name}.pkl"
        )

        if os.path.exists(results_file):
            try:
                with open(results_file, 'rb') as f:
                    prev = pickle.load(f)
                prev_metrics = prev.get("metrics", [])
                if isinstance(prev_metrics, dict):
                    prev_metrics = [prev_metrics]
                prev_metrics.append(results)
                merged = {"metrics": prev_metrics, "ceiling": ceiling}
                if self.aggregation_mode != "none":
                    merged["constituent_layers"] = self.layer_names
            except Exception:
                merged = {"metrics": results, "ceiling": ceiling}
        else:
            merged = {"metrics": results, "ceiling": ceiling}
            if self.aggregation_mode != "none":
                merged["constituent_layers"] = self.layer_names

        with open(results_file, 'wb') as f:
            pickle.dump(merged, f)

        if self.save_features:
            stim_name = self.stimulus_train.__class__.__name__
            feat_file = os.path.join(
                self.features_path,
                f"{self.model_identifier}_{current_layer_name}_{stim_name}_features.pkl"
            )
            try:
                os.makedirs(os.path.dirname(feat_file), exist_ok=True)
                with open(feat_file, 'wb') as f:
                    pickle.dump({'train': features_train, 'train_labels': labels_train,
                                 'test': features_test, 'test_labels': labels_test}, f)
            except Exception as e:
                print(f"Error saving features: {e}")

        return results, ceiling

    def run(self):

        # --- Memory Estimation (Warmup) ---
        ridge_metrics_present = any(
            'ridge' in name.lower() for name in self.metrics)
        downsample_factor = 1.0

        if ridge_metrics_present and self.use_ridge_smart_memory:
            # 1. Warmup Extract
            warmup_loader = DataLoader(self.stimulus_train, batch_size=1, shuffle=False,
                                       num_workers=self.extractor.num_workers, collate_fn=custom_collate)
            batch = next(iter(warmup_loader))
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = self.extractor.get_activations(inputs)

            # 2. Estimate "Elements per Sample" based on Aggregation Mode
            elements_per_sample = 0
            if self.aggregation_mode in ["concatenate", "stack"]:
                # Use logic similar to extract_features but just for shape
                # We can hack this by calling process_sequence_features on the captured warmup features
                # But careful: features is a dict
                shapes = []
                for l_name in self.layer_names:
                    raw = self.extractor.features[l_name]
                    processed = self.extractor._process_sequence_features(raw)
                    if isinstance(processed, torch.Tensor):
                        processed = processed.cpu().numpy()
                    shapes.append(processed.shape)  # (B, [T], D)

                # Assuming B=1 from warmup
                if self.aggregation_mode == "concatenate":
                    # Sum of last dims
                    total_dim = sum(s[-1] for s in shapes)
                    # Other dims (e.g. T) are shared
                    temporal_dim = np.prod(
                        shapes[0][1:-1]) if len(shapes[0]) > 2 else 1
                    elements_per_sample = temporal_dim * total_dim
                elif self.aggregation_mode == "stack":
                    # New dim L, must have shared D (via RP target_dim)
                    # If RP is not set, we assume they match or it will fail later
                    target_d = self.extractor.target_dim if self.extractor.target_dim else shapes[
                        0][-1]
                    temporal_dim = np.prod(
                        shapes[0][1:-1]) if len(shapes[0]) > 2 else 1
                    elements_per_sample = temporal_dim * \
                        len(self.layer_names) * target_d
            else:
                # Mode = "none". Bottleneck is the largest single layer.
                max_elements = 0
                for l_name in self.layer_names:
                    raw = self.extractor.features[l_name]
                    processed = self.extractor._process_sequence_features(raw)
                    size = np.prod(processed.shape[1:])  # Exclude batch
                    if size > max_elements:
                        max_elements = size
                elements_per_sample = max_elements

            # 3. Calculate Budget (Same logic as original)
            rss, avail_gb, total_gb = get_mem_info()
            budget_bytes = avail_gb * (2**30) * self.safety_factor
            byte_f64 = np.dtype(np.float64).itemsize
            N_train = len(self.stimulus_train)
            N_test = len(self.stimulus_test) if self.stimulus_test else 0

            # Rough overhead model
            overhead_factor = 2.0
            effective_budget = budget_bytes / overhead_factor

            # Simple assumption: targets are small compared to features
            available_for_features = effective_budget
            cost_per_sample = elements_per_sample * byte_f64

            total_needed = (N_train + N_test) * cost_per_sample

            if total_needed > available_for_features:
                downsample_factor = available_for_features / total_needed
                print(
                    f"--- Smart Memory: Downsampling factor set to {downsample_factor:.4f} ---")
            else:
                print("--- Smart Memory: Sufficient memory, no downsampling. ---")

        # --- Extraction ---
        print("Extracting features...")
        features_train_raw, labels_train = self.extractor.extract_features(
            self.stimulus_train, downsample_factor)

        features_test_raw, labels_test = None, None
        if self.stimulus_test is not None:
            features_test_raw, labels_test = self.extractor.extract_features(
                self.stimulus_test, downsample_factor, self.test_batch_size)

        # Free model memory before running metrics
        del self.extractor
        del self.model
        del self.model_instance
        if hasattr(self, 'stimulus_train') and hasattr(self.stimulus_train, 'test_image_data'):
            del self.stimulus_train.test_image_data
        if hasattr(self, 'stimulus_test') and self.stimulus_test is not None:
            if hasattr(self.stimulus_test, 'test_image_data'):
                del self.stimulus_test.test_image_data
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_results = {}

        # CASE 1: Aggregated (Concatenate or Stack)
        if self.aggregation_mode in ["concatenate", "stack"]:
            if len(self.layer_names) == 1:
                combined_name = self.layer_names[0]
            else:
                base = "_".join(self.layer_names)
                if len(base) > 80:
                    import hashlib
                    h = hashlib.md5(base.encode()).hexdigest()[:8]
                    combined_name = f"{self.aggregation_mode.capitalize()}_{len(self.layer_names)}Layers_{h}"
                else:
                    combined_name = f"{self.aggregation_mode.capitalize()}_{base}"

            print(
                f"Running metrics for Aggregated ({self.aggregation_mode}): {combined_name}")

            res, ceil = self._process_single_layer_result(
                features_train_raw, features_test_raw, labels_train, labels_test, combined_name
            )
            return {'metrics': res, 'ceiling': ceil}

        # CASE 2: Separate Layers (Dict)
        else:
            if not isinstance(features_train_raw, dict):
                # Fallback for single layer legacy
                features_train_raw = {self.layer_names[0]: features_train_raw}
                if features_test_raw is not None:
                    features_test_raw = {
                        self.layer_names[0]: features_test_raw}

            for layer_key, f_train in features_train_raw.items():
                print(f"Running metrics for Separate Layer: {layer_key}")
                f_test = features_test_raw[layer_key] if features_test_raw else None
                res, ceil = self._process_single_layer_result(
                    f_train, f_test, labels_train, labels_test, layer_key
                )
                all_results[layer_key] = {'metrics': res, 'ceiling': ceil}

            return all_results


class AssemblyBenchmarkScorer:
    def __init__(
        self,
        source_assembly_class,
        target_assembly_class,
        source_assembly_train_kwargs=None,
        source_assembly_test_kwargs=None,
        target_assembly_train_kwargs=None,
        target_assembly_test_kwargs=None,
        task='neural',
        debug=False,
    ):
        self.debug = debug
        self.task = task
        self.metrics = {}
        self.metric_params = {}

        self.source_assembly_class = source_assembly_class
        self.target_assembly_class = target_assembly_class
        self.source_assembly_train_kwargs = source_assembly_train_kwargs or {}
        self.source_assembly_test_kwargs = source_assembly_test_kwargs or {}
        self.target_assembly_train_kwargs = target_assembly_train_kwargs or {}
        self.target_assembly_test_kwargs = target_assembly_test_kwargs or {}

        self.source_name = self.source_assembly_class.__name__
        self.target_name = self.target_assembly_class.__name__

        data_home = get_data_home()
        results_base = os.environ.get('RESULTS_PATH', data_home)
        results_dir = os.path.join(results_base, 'results')
        os.makedirs(results_dir, exist_ok=True)

        benchmark_class_name = self.__class__.__name__
        self.results_file = os.path.join(
            results_dir,
            f"{benchmark_class_name}.pkl"
        )

    def initialize_rp(self, rp):
        if rp is not None:
            print(
                "Warning: Random Projection is not supported for AssemblyBenchmarkScorer. Ignoring.")

    def initialize_aggregation(self, mode):
        if mode != "none":
            print(
                "Warning: Layer Aggregation is not supported for AssemblyBenchmarkScorer. Ignoring.")

    def add_metric(self, name, metric_params=None):
        self.metrics[name] = METRICS[name]
        if metric_params:
            self.metric_params[name] = metric_params

    def run(self):

        # 1. Load source
        source_assembly = self.source_assembly_class()
        source_train, _ = source_assembly.get_assembly(
            **self.source_assembly_train_kwargs)
        source_test, _ = (source_assembly.get_assembly(**self.source_assembly_test_kwargs)
                          if self.source_assembly_test_kwargs else (None, None))
        print('Load source data. Train shape:',
              source_train.shape, 'Test shape:', source_test.shape)

        # 2. Load target
        target_assembly = self.target_assembly_class()

        try:
            target_train_data = target_assembly.get_assembly(
                **self.target_assembly_train_kwargs)
            if len(target_train_data) == 3:
                target_train, ceiling, stratify_labels_train = target_train_data
            elif len(target_train_data) == 2:
                target_train, ceiling = target_train_data
                stratify_labels_train = None
            else:
                raise ValueError(
                    f"Target assembly returned {len(target_train_data)} values, expected 2 or 3.")
        except Exception as e:
            print(f"Error calling get_assembly for training on target: {e}")
            raise

        target_test = None
        if self.target_assembly_test_kwargs:
            target_test, _ = target_assembly.get_assembly(
                **self.target_assembly_test_kwargs)
            print('Load target data. Train shape:',
                  target_train.shape, 'Test shape:', target_test.shape)

        # 3. Compute metrics
        results = {}
        for name, metric_class in self.metrics.items():
            try:
                extra = self.metric_params.get(name, {})
                if ceiling is not None:
                    metric_instance = metric_class(
                        ceiling=ceiling, **extra)
                else:
                    metric_instance = metric_class(**extra)
                results[name] = metric_instance.compute(
                    source_train,
                    target_train,
                    test_source=source_test,
                    test_target=target_test,
                    stratify_on=stratify_labels_train
                )
            except Exception as e:
                print(f"Metric '{name}' failed and will be skipped: {e}")
                continue

        results['timestamp'] = datetime.datetime.utcnow().isoformat()

        # 4. Save
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'rb') as f:
                    prev = pickle.load(f)
                metrics_list = prev.get("metrics", [])
                if isinstance(metrics_list, dict):
                    metrics_list = [metrics_list]
                metrics_list.append(results)
                merged = {"metrics": metrics_list, "ceiling": ceiling}
            except Exception as e:
                print(
                    f"Could not load existing results file, overwriting: {e}")
                merged = {"metrics": [results], "ceiling": ceiling}
        else:
            merged = {"metrics": [results], "ceiling": ceiling}

        with open(self.results_file, 'wb') as f:
            pickle.dump(merged, f)

        return {'metrics': results, 'ceiling': ceiling}
