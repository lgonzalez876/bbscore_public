import os
import glob
import logging
import numpy as np
import h5py
import re
import soundfile as sf
from collections import defaultdict
from typing import Optional, List, Union, Callable, Dict, Tuple
from data.base import BaseDataset

logger = logging.getLogger(__name__)


class LeBel2023StimulusSet(BaseDataset):
    """
    Stimulus set for the LeBel et al. (2023) dataset (OpenNeuro ds003020).
    Consists of natural language narratives.
    """

    def __init__(self, root_dir: Optional[str] = None, preprocess: Optional[Callable] = None):
        super().__init__(root_dir)
        self.preprocess = preprocess
        self.stimuli = []
        self.stimuli_ids = []

        self.dataset_dir = os.path.join(self.root_dir, "ds003020")
        self.textgrid_dir = os.path.join(
            self.dataset_dir, "derivative", "TextGrids")

        self._prepare_stimuli()

    def _prepare_stimuli(self):
        s3_source = "s3://openneuro.org/ds003020/derivative/TextGrids/"

        # Download TextGrids if not present
        if not os.path.exists(self.textgrid_dir) or not os.listdir(self.textgrid_dir):
            try:
                print(f"Downloading TextGrids from {s3_source}...")
                self.fetch(
                    source=s3_source,
                    target_dir=os.path.dirname(self.textgrid_dir),
                    filename="TextGrids",
                    method="s3",
                    anonymous=True
                )
            except Exception as e:
                print(f"Error downloading TextGrids: {e}")

        # Parse TextGrids
        tg_files = sorted(
            glob.glob(os.path.join(self.textgrid_dir, "*.TextGrid")))
        print(f"DEBUG: Found {len(tg_files)} TextGrid files.")
        if not tg_files:
            raise FileNotFoundError(
                f"No .TextGrid files found in {self.textgrid_dir}")

        for tg_file in tg_files:
            story_name = os.path.basename(tg_file).replace(".TextGrid", "")
            try:
                text = self._parse_textgrid(tg_file)
                self.stimuli.append(text)
                self.stimuli_ids.append(story_name)
            except Exception as e:
                print(f"Failed to parse {tg_file}: {e}")

    def _parse_textgrid(self, filepath):
        """
        Simple TextGrid parser to extract words from the 'words' tier.
        Assumes standard Praat short/long text format.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the 'words' tier
        # This is a heuristic parser. A robust one would use a library.
        # We look for: name = "words" ... intervals [x]: ... text = "the"

        words = []
        # Split by intervals usually works if we find the right tier
        # But file structure can vary.
        # Let's try a regex approach for "text =" fields inside the words tier.

        # Locate words tier
        tier_match = re.search(
            r'name = "word"(.*?)(name = |$)', content, re.DOTALL)
        if tier_match:
            tier_content = tier_match.group(1)
            # Extract text = "..."
            # Note: Praat saves text = "..." or text = ""
            matches = re.findall(r'text = "(.*?)"', tier_content)
            # Filter out empty strings (silences) if needed, or keep them?
            # Usually we want the transcript.
            filtered_words = [w for w in matches if w.strip()]
            return " ".join(filtered_words)

        return ""

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, idx):
        text = self.stimuli[idx]
        if self.preprocess:
            return self.preprocess(text)
        return text


class LeBel2023Assembly(BaseDataset):
    """
    fMRI Assembly for LeBel et al. (2023).
    """

    def __init__(self, root_dir: Optional[str] = None, subjects: Union[str, List[str]] = ['UTS01']):
        super().__init__(root_dir)
        if isinstance(subjects, str):
            subjects = [subjects]
        self.subjects = subjects
        self.dataset_dir = os.path.join(self.root_dir, "ds003020")
        self.data_dir = os.path.join(
            self.dataset_dir, "derivative", "preprocessed_data")

    def get_assembly(self):
        """
        Returns:
            fmri_data: (n_stories, n_voxels) - AVERAGED over time for now to match benchmark structure
                       OR concatenated time series if benchmark supports it.
            noise_ceiling: (n_voxels,)
        """
        s3_base = "s3://openneuro.org/ds003020/derivative/preprocessed_data/"

        all_subject_data = []

        # We assume the StimulusSet has loaded stories in sorted order of filenames.
        # We must load fMRI files in the SAME order.
        # StimulusSet loads *.TextGrid sorted.
        # We should load matching *.hf5 files.

        # Get list of stories from the TextGrid directory to ensure alignment
        tg_dir = os.path.join(self.dataset_dir, "derivative", "TextGrids")
        if not os.path.exists(tg_dir):
            # Try to instantiate stimulus set to trigger download?
            # Or just trust it exists if Benchmark calls stimulus first.
            # We'll just assume sorted glob of hf5 matches sorted glob of TextGrid
            pass

        for subj in self.subjects:
            subj_path = os.path.join(self.data_dir, subj)

            # Check for existing data
            hf5_files = sorted(glob.glob(os.path.join(subj_path, "*.hf5")))

            # Check recursive if needed
            if not hf5_files and os.path.exists(subj_path):
                hf5_files = sorted(glob.glob(os.path.join(
                    subj_path, "**", "*.hf5"), recursive=True))

            # Validate count (we expect 84 stories)
            if len(hf5_files) < 84:
                print(
                    f"Found {len(hf5_files)} files for {subj}, expected 84. Redownloading...")
                import shutil
                if os.path.exists(subj_path):
                    shutil.rmtree(subj_path)
                hf5_files = []  # Force download

            if not hf5_files:
                # If directory exists but empty (or contains wrong stuff), clean it up to ensure fetch runs
                if os.path.exists(subj_path):
                    try:
                        # Check if it has the weird hash dir and move files?
                        # Or just nuke it to be safe and redownload (safer for reproducibility)
                        # But user might have slow connection.
                        # Let's check for subdirectories and flatten if possible?
                        # Too complex. Let's just remove empty dir.
                        if not os.listdir(subj_path):
                            os.rmdir(subj_path)
                    except OSError:
                        pass

                # Download subject data
                try:
                    print(f"Downloading fMRI data for {subj}...")
                    self.fetch(
                        source=f"{s3_base}{subj}/",
                        target_dir=self.data_dir,
                        filename=subj,
                        method="s3",
                        anonymous=True
                    )
                except Exception as e:
                    print(f"Error downloading data for {subj}: {e}")

            # Reload file list
            hf5_files = sorted(glob.glob(os.path.join(subj_path, "*.hf5")))

            # If still no files, check if they are nested (e.g. from previous bad download)
            if not hf5_files and os.path.exists(subj_path):
                nested_hf5 = sorted(glob.glob(os.path.join(
                    subj_path, "**", "*.hf5"), recursive=True))
                if nested_hf5:
                    hf5_files = nested_hf5

            print(f"DEBUG: Found {len(hf5_files)} HF5 files for {subj}.")

            if not hf5_files:
                raise FileNotFoundError(f"No .hf5 files found for {subj}")

            subj_data_list = []
            for f in hf5_files:
                # Load data from HDF5
                # Key inside hf5 is typically 'data' or the story name
                try:
                    with h5py.File(f, 'r') as hf:
                        # Inspect keys
                        keys = list(hf.keys())
                        # Usually 'data' or 'dset'
                        # Based on inspection of similar datasets, it's often 'data'
                        # We'll try common keys
                        dset = None
                        for k in ['data', 'dset', 'roi', 'rep']:
                            if k in keys:
                                dset = hf[k][:]
                                break
                        if dset is None:
                            # Fallback: take the first key
                            dset = hf[keys[0]][:]

                        # dset shape: (time, voxels)
                        # To match the "Story" granularity of StimulusSet (1 string),
                        # we might need to average over time or similar.
                        # OR we assume the pipeline handles (1, T, V).

                        # For now, let's just take the MEAN over time to get (1, voxels)
                        # This creates a "story vector".

                        subj_data_list.append(np.nanmean(dset, axis=0))

                except Exception as e:
                    print(f"Error loading {f}: {e}")

            # Stack stories: (n_stories, n_voxels)
            if subj_data_list:
                print(
                    f"DEBUG: Stacking {len(subj_data_list)} stories for {subj}")
                all_subject_data.append(np.stack(subj_data_list, axis=0))

        if not all_subject_data:
            raise ValueError("No data loaded.")

        # Check if shapes match for averaging/stacking
        shapes = [d.shape for d in all_subject_data]

        if len(all_subject_data) > 1:
            # If shapes differ, we MUST concatenate (cannot average)
            if len(set(shapes)) > 1:
                print(
                    f"Warning: Subject shapes differ {shapes}. Concatenating features.")
                fmri_data = np.concatenate(all_subject_data, axis=1)
            else:
                # If shapes match, we usually average for shared benchmarks,
                # BUT since we know these subjects aren't aligned, averaging is geometrically wrong.
                # Concatenation is safer to preserve information.
                print(
                    f"Note: Multiple subjects loaded. Concatenating {len(all_subject_data)} subjects.")
                fmri_data = np.concatenate(all_subject_data, axis=1)
        else:
            fmri_data = all_subject_data[0]

        # Handle NaNs
        fmri_data = np.nan_to_num(fmri_data)

        print(f"DEBUG: Final fmri_data shape: {fmri_data.shape}")

        # Noise ceiling: per-voxel leave-one-out inter-subject correlation.
        # Precomputed by compute_ceiling.py using all 9 subjects (UTS01-UTS09).
        # For each subject S, each story, each voxel:
        #   ceiling = pearson_r(S_timeseries, mean(other_subjects)_timeseries)
        # Then median across stories -> (81126,) per subject.
        # Falls back to np.ones if ceiling file not found or subject missing.
        n_voxels = fmri_data.shape[1]
        ncsnr = self._load_ceiling(n_voxels)

        return fmri_data, ncsnr

    def _load_ceiling(self, n_voxels):
        """Load precomputed per-voxel noise ceiling for the requested subject."""
        ceiling_path = os.path.join(
            os.path.dirname(__file__), "lebel2023_ceiling.npz")
        if not os.path.exists(ceiling_path):
            print("Warning: Ceiling file not found, using placeholder ceiling=1.0")
            return np.ones(n_voxels, dtype=np.float32)

        data = np.load(ceiling_path, allow_pickle=True)
        subj = self.subjects[0]
        if subj in data:
            ceiling = data[subj].astype(np.float32)
            ceiling = np.clip(ceiling, 1e-3, None)
            print(f"Loaded noise ceiling for {subj}: "
                  f"median={np.median(ceiling):.4f}, "
                  f"mean={np.mean(ceiling):.4f}")
            return ceiling
        else:
            print(f"Warning: No ceiling for {subj}, using placeholder ceiling=1.0")
            return np.ones(n_voxels, dtype=np.float32)

    def __len__(self):
        # Placeholder or compute
        # Since get_assembly loads data, we can call it or use a default if known
        # In this dataset, there are 27 stories/files.
        return 27

    def __getitem__(self, idx):
        # This dataset usually accessed via get_assembly() for the whole matrix.
        # But for BaseDataset compatibility:
        # We can lazy load or just return a placeholder if not loaded yet.
        # But to be safe, we can try to load just that one file if possible.
        # However, BaseDataset structure here seems to focus on get_assembly for the benchmark.
        # We'll return a dummy value if get_assembly hasn't been called, or look it up.
        # Given the complexity, let's just return None or raise NotImplementedError
        # unless we cache the data in __init__ which is expensive.
        # BUT, the Scorer might not use __getitem__ for Assembly if it uses get_assembly.
        # So we just need to satisfy the abstract class.
        return None


class LeBel2023TRStimulusSet(BaseDataset):
    """
    TR-level stimulus set for LeBel et al. (2023).
    Parses TextGrid files preserving word-level timestamps,
    then bins words into TR windows with cumulative context.
    """

    def __init__(self, root_dir: Optional[str] = None,
                 tr_duration: float = 2.0):
        super().__init__(root_dir)
        self.tr_duration = tr_duration
        self.stories: List[List[Tuple[str, float, float]]] = []
        self.story_names: List[str] = []

        self.dataset_dir = os.path.join(self.root_dir, "ds003020")
        self.textgrid_dir = os.path.join(
            self.dataset_dir, "derivative", "TextGrids")

        self._prepare_stimuli()

    def _prepare_stimuli(self):
        s3_source = "s3://openneuro.org/ds003020/derivative/TextGrids/"

        if not os.path.exists(self.textgrid_dir) or \
                not os.listdir(self.textgrid_dir):
            try:
                print(f"Downloading TextGrids from {s3_source}...")
                self.fetch(
                    source=s3_source,
                    target_dir=os.path.dirname(self.textgrid_dir),
                    filename="TextGrids",
                    method="s3",
                    anonymous=True
                )
            except Exception as e:
                print(f"Error downloading TextGrids: {e}")

        tg_files = sorted(
            glob.glob(os.path.join(self.textgrid_dir, "*.TextGrid")))
        print(f"Found {len(tg_files)} TextGrid files.")
        if not tg_files:
            raise FileNotFoundError(
                f"No .TextGrid files found in {self.textgrid_dir}")

        for tg_file in tg_files:
            story_name = os.path.basename(tg_file).replace(".TextGrid", "")
            try:
                words_with_times = self._parse_textgrid_with_timestamps(
                    tg_file)
                self.stories.append(words_with_times)
                self.story_names.append(story_name)
            except Exception as e:
                print(f"Failed to parse {tg_file}: {e}")

    def _parse_textgrid_with_timestamps(
        self, filepath: str
    ) -> List[Tuple[str, float, float]]:
        """
        Parse a Praat TextGrid file, extracting (word, xmin, xmax) tuples
        from the 'words' tier.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tier_match = re.search(
            r'name = "word"(.*?)(name = |$)', content, re.DOTALL)
        if not tier_match:
            return []

        tier_content = tier_match.group(1)
        pattern = r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"'
        matches = re.findall(pattern, tier_content)

        words_with_times = []
        for xmin_str, xmax_str, word in matches:
            if word.strip():
                words_with_times.append((
                    word.strip(),
                    float(xmin_str),
                    float(xmax_str)
                ))
        return words_with_times

    def get_tr_texts(self, story_idx: int) -> Tuple[List[str], int]:
        """
        For a given story, return cumulative context strings per TR.

        Words are assigned to TRs by their midpoint time.
        Each TR's text is the concatenation of ALL words up to and
        including that TR (cumulative context).

        Returns:
            cumulative_texts: list of cumulative text strings, one per TR
            n_trs: number of TRs
        """
        words_with_times = self.stories[story_idx]
        if not words_with_times:
            return [], 0

        max_time = max(xmax for _, _, xmax in words_with_times)
        n_trs = int(np.ceil(max_time / self.tr_duration))

        cumulative_texts = []
        all_words_so_far = []
        word_idx = 0

        for tr_idx in range(n_trs):
            tr_end = (tr_idx + 1) * self.tr_duration
            # Assign words by midpoint
            while (word_idx < len(words_with_times) and
                   (words_with_times[word_idx][1] +
                    words_with_times[word_idx][2]) / 2.0 < tr_end):
                all_words_so_far.append(words_with_times[word_idx][0])
                word_idx += 1
            cumulative_texts.append(" ".join(all_words_so_far))

        return cumulative_texts, n_trs

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        return self.stories[idx]


class LeBel2023TRAssembly(BaseDataset):
    """
    TR-level fMRI assembly for LeBel et al. (2023).
    Returns per-story fMRI time series without temporal averaging.
    Multiple runs of the same story are averaged together.
    """

    def __init__(self, root_dir: Optional[str] = None,
                 subjects: Union[str, List[str]] = None):
        super().__init__(root_dir)
        if subjects is None:
            subjects = ['UTS01']
        if isinstance(subjects, str):
            subjects = [subjects]
        self.subjects = subjects
        self.dataset_dir = os.path.join(self.root_dir, "ds003020")
        self.data_dir = os.path.join(
            self.dataset_dir, "derivative", "preprocessed_data")

    @staticmethod
    def _extract_story_name(filepath: str) -> str:
        """Extract canonical story name from HDF5 filename."""
        return os.path.basename(filepath).replace(".hf5", "")

    def _load_hf5(self, filepath: str) -> np.ndarray:
        """Load a single HDF5 file, returning (n_TRs, n_voxels)."""
        with h5py.File(filepath, 'r') as hf:
            keys = list(hf.keys())
            dset = None
            for k in ['data', 'dset', 'roi', 'rep']:
                if k in keys:
                    dset = hf[k][:]
                    break
            if dset is None:
                dset = hf[keys[0]][:]
        return dset

    def _ensure_data_downloaded(self, subj: str) -> List[str]:
        """Download subject data if needed, return sorted list of hf5 paths."""
        s3_base = "s3://openneuro.org/ds003020/derivative/preprocessed_data/"
        subj_path = os.path.join(self.data_dir, subj)

        hf5_files = sorted(
            glob.glob(os.path.join(subj_path, "*.hf5")))

        if not hf5_files and os.path.exists(subj_path):
            hf5_files = sorted(glob.glob(os.path.join(
                subj_path, "**", "*.hf5"), recursive=True))

        if len(hf5_files) < 84:
            print(f"Found {len(hf5_files)} files for {subj}, "
                  f"expected 84. Downloading...")
            import shutil
            if os.path.exists(subj_path) and len(hf5_files) > 0:
                shutil.rmtree(subj_path)
            try:
                self.fetch(
                    source=f"{s3_base}{subj}/",
                    target_dir=self.data_dir,
                    filename=subj,
                    method="s3",
                    anonymous=True
                )
            except Exception as e:
                print(f"Error downloading data for {subj}: {e}")

            hf5_files = sorted(
                glob.glob(os.path.join(subj_path, "*.hf5")))
            if not hf5_files and os.path.exists(subj_path):
                hf5_files = sorted(glob.glob(os.path.join(
                    subj_path, "**", "*.hf5"), recursive=True))

        if not hf5_files:
            raise FileNotFoundError(
                f"No .hf5 files found for {subj}")

        print(f"Found {len(hf5_files)} HF5 files for {subj}.")
        return hf5_files

    def get_assembly(
        self, story_names: Optional[List[str]] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Load per-story fMRI time series.

        Args:
            story_names: If provided, only load stories matching these names.

        Returns:
            story_data: dict mapping story_name -> (n_TRs, n_voxels)
            ceiling: (n_voxels,) per-voxel noise ceiling
            ceiling_mask: (n_voxels,) bool — voxels with reliable ceiling
        """
        # Accumulate per-story data across subjects
        # For multiple subjects: concatenate along voxel axis
        merged_story_data: Dict[str, List[np.ndarray]] = defaultdict(list)

        for subj in self.subjects:
            hf5_files = self._ensure_data_downloaded(subj)

            # Group files by story name
            story_files: Dict[str, List[str]] = defaultdict(list)
            for f in hf5_files:
                sname = self._extract_story_name(f)
                if story_names is not None and sname not in story_names:
                    continue
                story_files[sname].append(f)

            for sname, files in sorted(story_files.items()):
                runs = []
                for f in files:
                    try:
                        dset = self._load_hf5(f)
                        runs.append(dset)
                    except Exception as e:
                        print(f"Error loading {f}: {e}")

                if not runs:
                    continue

                if len(runs) > 1:
                    # Average across repeated runs, align by min TR count
                    min_trs = min(r.shape[0] for r in runs)
                    aligned = [r[:min_trs] for r in runs]
                    subj_story = np.nanmean(
                        np.stack(aligned, axis=0), axis=0)
                else:
                    subj_story = runs[0]

                merged_story_data[sname].append(subj_story)

        if not merged_story_data:
            raise ValueError("No fMRI data loaded.")

        # Combine across subjects
        story_data: Dict[str, np.ndarray] = {}
        for sname, subj_arrays in merged_story_data.items():
            if len(subj_arrays) > 1:
                # Concatenate subjects along voxel axis, align TRs
                min_trs = min(a.shape[0] for a in subj_arrays)
                aligned = [a[:min_trs] for a in subj_arrays]
                story_data[sname] = np.concatenate(aligned, axis=1)
            else:
                story_data[sname] = subj_arrays[0]

        # Handle NaNs
        for sname in story_data:
            story_data[sname] = np.nan_to_num(story_data[sname])

        # Determine voxel count from first story
        n_voxels = next(iter(story_data.values())).shape[1]

        # Noise ceiling: leave-one-out inter-subject correlation (ISC).
        # Precomputed by compute_ceiling.py using all 9 subjects in volume
        # space. For subject S, each story, each voxel:
        #   ceiling = pearson_r(S, mean(others)) across TRs
        # Then median across stories. Voxels with ceiling <= 0.01 are
        # excluded (below noise floor).
        ceiling, ceiling_mask = self._load_ceiling(n_voxels)

        total_trs = sum(v.shape[0] for v in story_data.values())
        print(f"Loaded {len(story_data)} stories, "
              f"{total_trs} total TRs, {n_voxels} voxels.")

        return story_data, ceiling, ceiling_mask

    def _load_ceiling(self, n_voxels):
        """Load precomputed per-voxel noise ceiling and validity mask."""
        ceiling_path = os.path.join(
            os.path.dirname(__file__), "lebel2023_ceiling.npz")
        subj = self.subjects[0]
        CEILING_THRESHOLD = 0.01

        if not os.path.exists(ceiling_path):
            print("Warning: Ceiling file not found, "
                  "using placeholder ceiling=1.0")
            return (np.ones(n_voxels, dtype=np.float32),
                    np.ones(n_voxels, dtype=bool))

        data = np.load(ceiling_path, allow_pickle=True)
        if subj not in data:
            print(f"Warning: No ceiling for {subj}, "
                  "using placeholder ceiling=1.0")
            return (np.ones(n_voxels, dtype=np.float32),
                    np.ones(n_voxels, dtype=bool))

        ceiling = data[subj].astype(np.float32)
        valid_key = f'{subj}_valid'
        valid = (data[valid_key] if valid_key in data
                 else np.ones(len(ceiling), dtype=bool))

        # Combined mask: validity filter AND ceiling above noise floor
        ceiling_mask = valid & (ceiling > CEILING_THRESHOLD)
        n_kept = ceiling_mask.sum()
        print(f"Noise ceiling for {subj}: "
              f"median={np.median(ceiling[ceiling_mask]):.4f}, "
              f"{n_kept}/{len(ceiling)} voxels above threshold "
              f"({100 * n_kept / len(ceiling):.1f}%)")

        ceiling = np.clip(ceiling, 1e-3, None)
        return ceiling, ceiling_mask

    def __len__(self):
        return 27

    def __getitem__(self, idx):
        return None


class LeBel2023AudioStimulusSet(BaseDataset):
    """
    Story-level audio stimulus set for LeBel et al. (2023).
    Downloads WAV files and returns preprocessed waveforms per story.
    Truncates to first max_duration seconds (analogous to GPT-2 context length).
    """

    def __init__(self, root_dir: Optional[str] = None,
                 preprocess: Optional[Callable] = None,
                 sample_rate: int = 16000,
                 max_duration: float = 30.0):
        super().__init__(root_dir)
        self.preprocess = preprocess
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.stimuli = []
        self.stimuli_ids = []

        self.dataset_dir = os.path.join(self.root_dir, "ds003020")
        self.stimuli_dir = os.path.join(self.dataset_dir, "stimuli")

        self._prepare_stimuli()

    def _prepare_stimuli(self):
        s3_source = "s3://openneuro.org/ds003020/stimuli/"

        if not os.path.exists(self.stimuli_dir) or not os.listdir(self.stimuli_dir):
            try:
                print(f"Downloading audio stimuli from {s3_source}...")
                self.fetch(
                    source=s3_source,
                    target_dir=os.path.dirname(self.stimuli_dir),
                    filename="stimuli",
                    method="s3",
                    anonymous=True
                )
            except Exception as e:
                print(f"Error downloading audio stimuli: {e}")

        wav_files = sorted(
            glob.glob(os.path.join(self.stimuli_dir, "*.wav")))
        print(f"Found {len(wav_files)} audio stimulus files.")
        if not wav_files:
            raise FileNotFoundError(
                f"No .wav files found in {self.stimuli_dir}")

        for wav_file in wav_files:
            story_name = os.path.basename(wav_file).replace(".wav", "")
            self.stimuli.append(wav_file)
            self.stimuli_ids.append(story_name)

    def _load_audio(self, filepath):
        """Load and resample audio to target sample rate."""
        audio, sr = sf.read(filepath, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.sample_rate)
        # Truncate to max_duration
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        return audio

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, idx):
        audio = self._load_audio(self.stimuli[idx])
        if self.preprocess:
            return self.preprocess(audio)
        return audio


class LeBel2023AudioTRStimulusSet(BaseDataset):
    """
    TR-level audio stimulus set for LeBel et al. (2023).
    Downloads WAV files and provides per-TR audio segments.
    """

    def __init__(self, root_dir: Optional[str] = None,
                 tr_duration: float = 2.0,
                 sample_rate: int = 16000):
        super().__init__(root_dir)
        self.tr_duration = tr_duration
        self.sample_rate = sample_rate
        self.story_names = []
        self.audio_paths = []

        self.dataset_dir = os.path.join(self.root_dir, "ds003020")
        self.stimuli_dir = os.path.join(self.dataset_dir, "stimuli")

        self._prepare_stimuli()

    def _prepare_stimuli(self):
        s3_source = "s3://openneuro.org/ds003020/stimuli/"

        if not os.path.exists(self.stimuli_dir) or not os.listdir(self.stimuli_dir):
            try:
                print(f"Downloading audio stimuli from {s3_source}...")
                self.fetch(
                    source=s3_source,
                    target_dir=os.path.dirname(self.stimuli_dir),
                    filename="stimuli",
                    method="s3",
                    anonymous=True
                )
            except Exception as e:
                print(f"Error downloading audio stimuli: {e}")

        wav_files = sorted(
            glob.glob(os.path.join(self.stimuli_dir, "*.wav")))
        print(f"Found {len(wav_files)} audio stimulus files.")
        if not wav_files:
            raise FileNotFoundError(
                f"No .wav files found in {self.stimuli_dir}")

        for wav_file in wav_files:
            story_name = os.path.basename(wav_file).replace(".wav", "")
            self.story_names.append(story_name)
            self.audio_paths.append(wav_file)

    def _load_audio(self, filepath):
        """Load and resample audio to target sample rate."""
        audio, sr = sf.read(filepath, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio

    def get_tr_audio_segments(self, story_idx):
        """
        Slice a story's audio into non-overlapping TR-sized segments.

        Returns:
            segments: list of np.ndarray, each shape (samples_per_tr,)
            n_trs: int, number of TR segments
        """
        audio = self._load_audio(self.audio_paths[story_idx])
        samples_per_tr = int(self.tr_duration * self.sample_rate)
        n_trs = len(audio) // samples_per_tr

        if n_trs == 0:
            return [], 0

        segments = []
        for i in range(n_trs):
            start = i * samples_per_tr
            end = start + samples_per_tr
            segments.append(audio[start:end])

        # Include partial last segment if significant (>50% of TR)
        remainder = len(audio) - n_trs * samples_per_tr
        if remainder > samples_per_tr // 2:
            last_seg = np.zeros(samples_per_tr, dtype=np.float32)
            last_seg[:remainder] = audio[n_trs * samples_per_tr:]
            segments.append(last_seg)
            n_trs += 1

        return segments, n_trs

    def get_tr_audio_segments_with_context(self, story_idx, context_duration):
        """
        Slice a story's audio into TR-sized windows, each padded with
        preceding audio context up to `context_duration` seconds total.

        For each TR, the returned segment is:
            [preceding_context ... | current_2s_TR]
        Zero-padded on the left for early TRs that don't have enough history.

        Args:
            story_idx: int
            context_duration: float, total segment duration in seconds

        Returns:
            segments: list of np.ndarray, each shape (context_samples,)
            n_trs: int
        """
        audio = self._load_audio(self.audio_paths[story_idx])
        samples_per_tr = int(self.tr_duration * self.sample_rate)
        context_samples = int(context_duration * self.sample_rate)
        n_trs = len(audio) // samples_per_tr

        if n_trs == 0:
            return [], 0

        segments = []
        for i in range(n_trs):
            tr_end = (i + 1) * samples_per_tr
            tr_start = max(0, tr_end - context_samples)
            raw_segment = audio[tr_start:tr_end]

            # Zero-pad on the left if not enough preceding audio
            if len(raw_segment) < context_samples:
                padded = np.zeros(context_samples, dtype=np.float32)
                padded[context_samples - len(raw_segment):] = raw_segment
                segments.append(padded)
            else:
                segments.append(raw_segment)

        # Partial last segment (same logic as get_tr_audio_segments)
        remainder = len(audio) - n_trs * samples_per_tr
        if remainder > samples_per_tr // 2:
            tr_end = len(audio)
            tr_start = max(0, tr_end - context_samples)
            raw_segment = audio[tr_start:tr_end]
            if len(raw_segment) < context_samples:
                padded = np.zeros(context_samples, dtype=np.float32)
                padded[context_samples - len(raw_segment):] = raw_segment
                segments.append(padded)
            else:
                segments.append(raw_segment)
            n_trs += 1

        return segments, n_trs

    def __len__(self):
        return len(self.story_names)

    def __getitem__(self, idx):
        return self._load_audio(self.audio_paths[idx])


# ---------------------------------------------------------------------------
# Desikan-Killiany region definitions
# ---------------------------------------------------------------------------

LANGUAGE_REGIONS = {
    'superiortemporal',
    'middletemporal',
    'parsopercularis',
    'parstriangularis',
    'supramarginal',
    'inferiorparietal',
    'bankssts',
    'temporalpole',
    'fusiform',
    'transversetemporal',
}

REGION_GROUPS = {
    'temporal': [
        'superiortemporal', 'middletemporal', 'inferiortemporal',
        'bankssts', 'transversetemporal', 'temporalpole',
        'fusiform', 'entorhinal', 'parahippocampal',
    ],
    'frontal': [
        'parsopercularis', 'parstriangularis', 'parsorbitalis',
        'rostralmiddlefrontal', 'caudalmiddlefrontal', 'superiorfrontal',
        'lateralorbitofrontal', 'medialorbitofrontal', 'frontalpole',
        'precentral', 'paracentral',
    ],
    'parietal': [
        'supramarginal', 'inferiorparietal', 'superiorparietal',
        'postcentral', 'precuneus',
    ],
    'occipital': [
        'lateraloccipital', 'cuneus', 'pericalcarine', 'lingual',
    ],
    'cingulate': [
        'rostralanteriorcingulate', 'caudalanteriorcingulate',
        'posteriorcingulate', 'isthmuscingulate',
    ],
    'insular': ['insula'],
    'language': sorted(LANGUAGE_REGIONS),
    'auditory': ['transversetemporal', 'superiortemporal'],
    'brocas': ['parsopercularis', 'parstriangularis'],
    'wernickes': ['superiortemporal', 'middletemporal', 'supramarginal'],
}

EXPECTED_CORTEX_VERTICES = 81126


class LeBel2023FreeSurferLabels(BaseDataset):
    """
    Downloads FreeSurfer parcellation files (aparc.annot, cortex.label)
    for subjects in the LeBel et al. (2023) dataset (OpenNeuro ds003020).
    """

    S3_BASE = ("s3://openneuro.org/ds003020/"
               "derivative/freesurfer_subjdir/")
    NEEDED_FILES = [
        'lh.aparc.annot', 'rh.aparc.annot',
        'lh.cortex.label', 'rh.cortex.label',
    ]

    def __init__(self, root_dir: Optional[str] = None):
        super().__init__(root_dir)
        self.dataset_dir = os.path.join(self.root_dir, "ds003020")

    def _label_dir(self, subject: str) -> str:
        return os.path.join(
            self.dataset_dir, "derivative",
            "freesurfer_subjdir", subject, "label")

    def ensure_downloaded(self, subject: str) -> str:
        """Download FreeSurfer label files for *subject* if needed.

        Returns the local label directory path.
        """
        label_dir = self._label_dir(subject)
        os.makedirs(label_dir, exist_ok=True)

        for fname in self.NEEDED_FILES:
            local_path = os.path.join(label_dir, fname)
            if os.path.exists(local_path):
                continue
            s3_path = (f"{self.S3_BASE}{subject}/label/{fname}")
            print(f"Downloading {s3_path} ...")
            self.fetch(
                source=s3_path,
                target_dir=label_dir,
                filename=fname,
                method="s3",
                anonymous=True,
            )
        return label_dir

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None


class RegionMapper:
    """Maps the 81,126 cortex-surface voxel indices used in LeBel2023
    HDF5 files to Desikan-Killiany brain regions via FreeSurfer
    ``aparc.annot`` and ``cortex.label`` files.
    """

    def __init__(self, label_dir: str):
        import nibabel.freesurfer as fs

        # --- load cortex masks (vertex indices that are cortex) ---
        lh_cortex_verts = fs.read_label(
            os.path.join(label_dir, 'lh.cortex.label'))
        rh_cortex_verts = fs.read_label(
            os.path.join(label_dir, 'rh.cortex.label'))

        # --- load aparc annotations (full-hemisphere arrays) ---
        lh_labels, lh_ctab, lh_names = fs.read_annot(
            os.path.join(label_dir, 'lh.aparc.annot'))
        rh_labels, rh_ctab, rh_names = fs.read_annot(
            os.path.join(label_dir, 'rh.aparc.annot'))

        # Decode bytes → str for region names
        lh_names = [n.decode() if isinstance(n, bytes) else n
                    for n in lh_names]
        rh_names = [n.decode() if isinstance(n, bytes) else n
                    for n in rh_names]

        # --- build cortex-only region arrays ---
        lh_cortex_verts = np.sort(lh_cortex_verts)
        rh_cortex_verts = np.sort(rh_cortex_verts)

        n_cortex = len(lh_cortex_verts) + len(rh_cortex_verts)
        if n_cortex != EXPECTED_CORTEX_VERTICES:
            logger.warning(
                "Cortex vertex count %d differs from expected %d. "
                "Region mapping may be approximate.",
                n_cortex, EXPECTED_CORTEX_VERTICES)

        # Region label per cortex vertex (combined LH + RH)
        lh_region_ids = lh_labels[lh_cortex_verts]
        rh_region_ids = rh_labels[rh_cortex_verts]

        # Build name arrays — index into names list
        self.n_cortex = n_cortex
        self.region_names: List[str] = []  # one per cortex vertex
        self.hemi_labels: List[str] = []   # 'lh' or 'rh' per vertex

        for rid in lh_region_ids:
            name = lh_names[rid] if 0 <= rid < len(lh_names) else 'unknown'
            self.region_names.append(name)
            self.hemi_labels.append('lh')
        for rid in rh_region_ids:
            name = rh_names[rid] if 0 <= rid < len(rh_names) else 'unknown'
            self.region_names.append(name)
            self.hemi_labels.append('rh')

        self.region_names = np.array(self.region_names)
        self.hemi_labels = np.array(self.hemi_labels)

        # Pre-compute unique region set (excluding 'unknown' and
        # 'corpuscallosum' which shouldn't appear in cortex)
        self.unique_regions = sorted(set(
            r for r in self.region_names
            if r not in ('unknown', 'corpuscallosum')))

        # Build non_language group dynamically
        all_dk = set(self.unique_regions)
        REGION_GROUPS['non_language'] = sorted(
            all_dk - LANGUAGE_REGIONS)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_region_indices(self, region_name: str) -> np.ndarray:
        """Return cortex-voxel indices for a single DK region
        (both hemispheres combined)."""
        return np.where(self.region_names == region_name)[0]

    def get_hemi_region_indices(
        self, hemi: str, region_name: str
    ) -> np.ndarray:
        """Return cortex-voxel indices for one hemisphere + region."""
        return np.where(
            (self.region_names == region_name) &
            (self.hemi_labels == hemi)
        )[0]

    def get_group_indices(self, group_name: str) -> np.ndarray:
        """Return cortex-voxel indices for a region group."""
        regions = REGION_GROUPS.get(group_name, [])
        mask = np.isin(self.region_names, regions)
        return np.where(mask)[0]

    def get_language_indices(self) -> np.ndarray:
        return self.get_group_indices('language')

    def get_non_language_indices(self) -> np.ndarray:
        return self.get_group_indices('non_language')

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_scores(
        self,
        per_voxel_pearson: np.ndarray,
        per_voxel_r2: np.ndarray,
    ) -> dict:
        """Aggregate per-voxel scores into region / group summaries.

        Args:
            per_voxel_pearson: (n_voxels,) median Pearson r per voxel
            per_voxel_r2: (n_voxels,) median R² per voxel

        Returns:
            dict with 'per_region', 'per_group', 'per_hemi_region' keys
        """
        n_voxels = len(per_voxel_pearson)

        # Handle potential mismatch between mapping and actual voxel count
        if n_voxels != self.n_cortex:
            logger.warning(
                "Voxel count (%d) != cortex mapping size (%d). "
                "Truncating to min.",
                n_voxels, self.n_cortex)
            limit = min(n_voxels, self.n_cortex)
            per_voxel_pearson = per_voxel_pearson[:limit]
            per_voxel_r2 = per_voxel_r2[:limit]
            region_names = self.region_names[:limit]
            hemi_labels = self.hemi_labels[:limit]
        else:
            region_names = self.region_names
            hemi_labels = self.hemi_labels

        def _agg(indices):
            if len(indices) == 0:
                return {'median_pearson': 0.0, 'median_r2': 0.0,
                        'n_voxels': 0}
            p = per_voxel_pearson[indices]
            r = per_voxel_r2[indices]
            valid = ~(np.isnan(p) | np.isnan(r))
            n_valid = int(valid.sum())
            if n_valid == 0:
                return {'median_pearson': 0.0, 'median_r2': 0.0,
                        'n_voxels': 0}
            return {
                'median_pearson': float(np.median(p[valid])),
                'median_r2': float(np.median(r[valid])),
                'n_voxels': n_valid,
            }

        # Per individual region (both hemispheres)
        per_region = {}
        for rname in self.unique_regions:
            idx = np.where(region_names == rname)[0]
            per_region[rname] = _agg(idx)

        # Per region group
        per_group = {}
        for gname, gregions in REGION_GROUPS.items():
            idx = np.where(np.isin(region_names, gregions))[0]
            per_group[gname] = _agg(idx)

        # Per hemisphere × region
        per_hemi_region = {}
        for hemi in ('lh', 'rh'):
            hemi_mask = hemi_labels == hemi
            for rname in self.unique_regions:
                idx = np.where(
                    hemi_mask & (region_names == rname))[0]
                if len(idx) > 0:
                    per_hemi_region[f'{hemi}_{rname}'] = _agg(idx)

        return {
            'per_region': per_region,
            'per_group': per_group,
            'per_hemi_region': per_hemi_region,
        }

    def aggregate_rsa_scores(
        self,
        per_story_features: list,
        per_story_fmri: list,
    ) -> dict:
        """Compute within-story temporal RSA restricted to region groups.

        Args:
            per_story_features: list of (n_TRs, D) model feature arrays
            per_story_fmri: list of (n_TRs, n_voxels) fMRI arrays

        Returns:
            dict with per_group RSA scores
        """
        from scipy.spatial.distance import pdist
        from scipy.stats import spearmanr

        target_groups = [
            'language', 'non_language', 'temporal', 'frontal',
            'parietal', 'occipital', 'auditory', 'brocas', 'wernickes',
        ]
        group_stories: Dict[str, list] = {g: [] for g in target_groups}

        for story_feat, story_fmri in zip(
                per_story_features, per_story_fmri):
            n_trs = story_feat.shape[0]
            if n_trs < 3:
                continue
            model_rdm = pdist(story_feat, metric='correlation')

            for gname in target_groups:
                idx = self.get_group_indices(gname)
                n_voxels = story_fmri.shape[1]
                idx = idx[idx < n_voxels]
                if len(idx) == 0:
                    continue
                neural_rdm = pdist(
                    story_fmri[:, idx], metric='correlation')
                valid = ~(np.isnan(model_rdm) | np.isnan(neural_rdm))
                if valid.sum() < 3:
                    continue
                rho, _ = spearmanr(
                    model_rdm[valid], neural_rdm[valid])
                group_stories[gname].append(rho)

        per_group = {}
        for gname in target_groups:
            scores = group_stories[gname]
            per_group[gname] = {
                'median_rsa': (float(np.median(scores))
                               if scores else 0.0),
                'n_stories': len(scores),
            }
        return {'per_group': per_group}
