import gdown
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image
from sklearn.datasets import get_data_home
from torchvision import transforms
from typing import List, Dict, Union, Tuple, Optional, Callable

from data.base import BaseDataset

# Constants (can be made configurable)
NCSNR_THRESHOLD = 0.2


class NSDStimulusSet(BaseDataset):
    """Dataset for the NSD stimulus set (images)."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        """
        Initialize the NSDStimulusSet.

        Args:
            root_dir: Root directory.
            overwrite: Overwrite existing files.
            preprocess: Preprocessing transform.
        """
        super().__init__(root_dir)
        self.overwrite = overwrite
        if preprocess is None:
            self.preprocess = self._define_default_preprocess()
        else:
            self.preprocess = preprocess
        self.test_image_data = None

    def _define_default_preprocess(self):
        """Define the default torchvision preprocessing transform."""
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    _GCS_BASE_URL = 'https://storage.googleapis.com/bbscore_datasets/nsd'

    _GDRIVE_FILE_IDS = {
        'subj01': '13cRiwhjurCdr4G2omRZSOMO_tmatjdQr',
        'subj02': '1MO9reLoV4fqu6Weh4gmE78KJVtxg72ID',
        'subj05': '11dPt3Llj6eAEDJnaRy8Ch5CxfeKijX_t',
        'subj07': '1HX-6t4c6js6J_vP4Xo0h1fbK2WINpwem',
    }

    def _download_nsd_data(self, subj: str):
        """Download NSD data for a subject (for images)."""
        if subj not in self._GDRIVE_FILE_IDS:
            raise ValueError(
                "Invalid subject ID. Choose: 'subj01', 'subj02', 'subj05', 'subj07'."
            )

        filename = f'{subj}_nativesurface_nsdgeneral.pkl'
        output = os.path.join(self.root_dir, filename)

        if not os.path.exists(output) or self.overwrite:
            gcs_url = f'{self._GCS_BASE_URL}/{filename}'
            try:
                self.fetch(
                    source=gcs_url,
                    filename=filename,
                    method='http',
                    force_download=self.overwrite,
                )
            except Exception as e:
                print(
                    f"GCS download failed ({e}), falling back to Google Drive...")
                file_id = self._GDRIVE_FILE_IDS[subj]
                url = f'https://drive.google.com/uc?id={file_id}&export=download'
                self.fetch(
                    source=url,
                    filename=filename,
                    method='gdown',
                    force_download=self.overwrite,
                )
        return np.load(output, allow_pickle=True)

    def _prepare_images(self, subj: str = 'subj01'):
        """Load and preprocess the test images."""
        Y = self._download_nsd_data(subj)
        self.test_image_data = Y['image_data']['test']

    def __len__(self):
        """Return the number of test images."""
        if self.test_image_data is None:
            self._prepare_images()
        return len(self.test_image_data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a preprocessed image."""
        if self.test_image_data is None:
            self._prepare_images()
        image = self.test_image_data[idx]
        # Convert numpy array to PIL image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.preprocess(image)


class NSDAssembly(BaseDataset):
    """Dataset for the NSD fMRI data (assembly)."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        subjects: Union[str, List[str]] = [
            'subj01', 'subj02', 'subj05', 'subj07'],
        regions: Union[str, List[str]] = ['V1'],
        overwrite: bool = False,
        ncsnr_threshold: float = NCSNR_THRESHOLD,
    ):
        """
        Initialize the NSDAssembly.

        Args:
            root_dir: Root directory.
            subjects: Subject ID(s).
            overwrite: Overwrite existing files.
            ncsnr_threshold: NCSNR threshold.
        """
        super().__init__(root_dir)
        if isinstance(subjects, str):
            subjects = [subjects]
        self.subjects = subjects
        self.regions = regions
        self.overwrite = overwrite
        self.ncsnr_threshold = ncsnr_threshold
        self.data = {}  # Store data for each subject
        self.test_fmri_data = None
        self.ncsnr_data = None

    _GCS_BASE_URL = 'https://storage.googleapis.com/bbscore_datasets/nsd'

    _GDRIVE_FILE_IDS = {
        'subj01': '13cRiwhjurCdr4G2omRZSOMO_tmatjdQr',
        'subj02': '1MO9reLoV4fqu6Weh4gmE78KJVtxg72ID',
        'subj05': '11dPt3Llj6eAEDJnaRy8Ch5CxfeKijX_t',
        'subj07': '1HX-6t4c6js6J_vP4Xo0h1fbK2WINpwem',
    }

    def _download_nsd_data(self, subj: str):
        """Download NSD data for a subject."""
        if subj not in self._GDRIVE_FILE_IDS:
            raise ValueError(
                "Invalid subject ID. Choose: 'subj01', 'subj02', 'subj05', 'subj07'."
            )

        filename = f'{subj}_nativesurface_nsdgeneral.pkl'
        output = os.path.join(self.root_dir, filename)

        if not os.path.exists(output) or self.overwrite:
            gcs_url = f'{self._GCS_BASE_URL}/{filename}'
            try:
                self.fetch(
                    source=gcs_url,
                    filename=filename,
                    method='http',
                    force_download=self.overwrite,
                )
            except Exception as e:
                print(
                    f"GCS download failed ({e}), falling back to Google Drive...")
                file_id = self._GDRIVE_FILE_IDS[subj]
                url = f'https://drive.google.com/uc?id={file_id}&export=download'
                self.fetch(
                    source=url,
                    filename=filename,
                    method='gdown',
                    force_download=self.overwrite,
                )
        self.data[subj] = np.load(output, allow_pickle=True)

    def _get_metadata_concat_hemi(self, Y: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
        """Concatenate metadata and return ncsnr/metadata."""
        ncsnr_full = np.concatenate(
            (
                Y['voxel_metadata']['lh']['lh.ncsnr'],
                Y['voxel_metadata']['rh']['rh.ncsnr'],
            )
        )
        nsdgeneral_idx = np.concatenate(
            (
                Y['voxel_metadata']['lh']['lh.nsdgeneral.label'],
                Y['voxel_metadata']['rh']['rh.nsdgeneral.label'],
            )
        )
        nsdgeneral_mask = np.logical_and(
            nsdgeneral_idx == 'nsdgeneral', ncsnr_full > 0
        )
        ncsnr_nsdgeneral = ncsnr_full[nsdgeneral_mask]

        metadata_lh = pd.DataFrame(Y['voxel_metadata']['lh'])
        metadata_rh = pd.DataFrame(Y['voxel_metadata']['rh'])
        nsdgeneral_metadata_df = pd.concat([metadata_lh, metadata_rh])[
            nsdgeneral_mask
        ]

        return ncsnr_nsdgeneral, nsdgeneral_metadata_df

    def _get_data_dict(
        self,
        brain_data_rep_averaged: np.ndarray,
        ncsnr_nsdgeneral: np.ndarray,
        nsdgeneral_metadata_df: pd.DataFrame,
        regions: List[str],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get fMRI data for specified areas, return single array.

        Also returns area-specific NCSNR values.

        Args:
            brain_data_rep_averaged: Averaged brain data (numpy array).
            ncsnr_nsdgeneral: NCSNR values for nsdgeneral ROI.
            nsdgeneral_metadata_df: Metadata DataFrame for nsdgeneral ROI.
            regions: List of brain regions to extract.
            verbose: Whether to print the size of each area.

        Returns:
            Tuple: (Combined brain data, Dictionary of area-specific NCSNR values).
        """

        data_dict: Dict[str, np.ndarray] = {}  # Temporary dict for brain data
        ncsnr_dict: Dict[str, np.ndarray] = {}  # Dict for area-specific NCSNR

        for region in regions:
            if region in ['ventral', 'parietal', 'lateral',
                          'highventral', 'highparietal', 'highlateral',
                          'midparietal', 'midlateral', 'midventral']:
                if region.startswith("high"):
                    # Extract the base region name, e.g. 'highventral' becomes 'ventral'
                    base_region = region.replace("high", "")
                    lh_area_mask = nsdgeneral_metadata_df['lh.streams.label'].astype(
                        str) == base_region
                    rh_area_mask = nsdgeneral_metadata_df['rh.streams.label'].astype(
                        str) == base_region
                else:
                    # For other cases (including mid*), use substring matching
                    lh_area_mask = nsdgeneral_metadata_df['lh.streams.label'].astype(
                        str).str.contains(region, na=False)
                    rh_area_mask = nsdgeneral_metadata_df['rh.streams.label'].astype(
                        str).str.contains(region, na=False)

            elif region in ['V1', 'V2', 'V3', 'V4', 'V1d', 'V2d', 'V1v', 'V3d', 'V2v', 'V3v']:
                lh_area_mask = (
                    nsdgeneral_metadata_df['lh.prf-visualrois.label']
                    .astype(str)
                    .str.contains(region, na=False)
                )
                rh_area_mask = (
                    nsdgeneral_metadata_df['rh.prf-visualrois.label']
                    .astype(str)
                    .str.contains(region, na=False)
                )
            else:
                raise ValueError(f"Invalid region: {region}")

            area_mask = np.logical_or(lh_area_mask, rh_area_mask)
            area_mask_thresholded = np.logical_and(
                area_mask, ncsnr_nsdgeneral > self.ncsnr_threshold
            )

            if verbose:
                print(
                    f"Size of area {region}: {np.sum(area_mask_thresholded)}")

            area_data = brain_data_rep_averaged[:, area_mask_thresholded]
            # Crucial: get NCSNR *after* thresholding
            area_ncsnr = ncsnr_nsdgeneral[area_mask_thresholded]

            if region not in data_dict:
                data_dict[region] = area_data
                ncsnr_dict[region] = area_ncsnr
            else:
                data_dict[region] = np.concatenate(
                    (data_dict[region], area_data), axis=1)
                # Concatenate NCSNR too.
                ncsnr_dict[region] = np.concatenate(
                    (ncsnr_dict[region], area_ncsnr))

        combined_data = []
        for region_data in data_dict.values():
            combined_data.append(region_data)
        combined_data = np.concatenate(combined_data, axis=1)

        # Combine NCSNR values in the same order as combined_data
        combined_ncsnr = []
        for region in data_dict.keys():  # Iterate in the *same* order as data_dict
            combined_ncsnr.append(ncsnr_dict[region])
        combined_ncsnr = np.concatenate(
            combined_ncsnr)  # No axis needed, it's 1D

        return combined_data, combined_ncsnr

    def prepare_data(self, regions: Union[str, List[str]] = 'V1'):
        """Prepare fMRI data (helper function)."""
        if isinstance(regions, str):
            regions = [regions]

        all_responses = []
        all_ncsnr = []

        for subj in self.subjects:
            if subj not in self.data:
                self._download_nsd_data(subj)

            Y = self.data[subj]
            ncsnr_nsdgeneral, nsdgeneral_metadata_df = (
                self._get_metadata_concat_hemi(Y)
            )

            test_brain_data_cat = np.concatenate(
                (
                    Y['brain_data']['test']['lh'],
                    Y['brain_data']['test']['rh'],
                ),
                axis=2,
            )
            del self.data[subj]

            test_brain_data_cat = np.mean(test_brain_data_cat, axis=1)

            subj_test_fmri_data, subj_test_ncsnr = self._get_data_dict(
                test_brain_data_cat,
                ncsnr_nsdgeneral,
                nsdgeneral_metadata_df,
                regions,
            )
            del test_brain_data_cat

            all_responses.append(subj_test_fmri_data)
            all_ncsnr.append(subj_test_ncsnr)

        self.test_fmri_data = np.concatenate(all_responses, axis=1)
        self.ncsnr_data = np.concatenate(all_ncsnr, axis=0)
        # self.ncsnr_data = (np.power(self.ncsnr_data,2) / (np.power(self.ncsnr_data,2) + 1/3))

    def get_assembly(self):
        """
        Prepare and return the fMRI data assembly and ceiling.

        This function calls `prepare_data` to ensure the data is prepared,
        then returns the prepared `test_fmri_data` and `ncsnr_data`.

        Args:
            regions: The brain region(s) to prepare data for.

        Returns:
            A tuple containing the fMRI data assembly (np.ndarray) and the
            NCSNR ceiling array (np.ndarray).
        """
        self.prepare_data(self.regions)  # Ensure data is prepared
        return self.test_fmri_data, self.ncsnr_data

    def __len__(self):
        if self.test_fmri_data is None:
            self.prepare_data(regions)
        return self.test_fmri_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (fMRI data, ncsnr) for the index."""
        if self.test_fmri_data is None or self.ncsnr_data is None:
            self.prepare_data(regions)

        return self.test_fmri_data[idx], self.ncsnr_data[idx]


class NSDAssemblyV1(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V1', **kwargs)


class NSDAssemblyV1d(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V1d', **kwargs)


class NSDAssemblyV1v(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V1v', **kwargs)


class NSDAssemblyV2(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V2', **kwargs)


class NSDAssemblyV2d(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V2d', **kwargs)


class NSDAssemblyV2v(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V2v', **kwargs)


class NSDAssemblyV3(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V3', **kwargs)


class NSDAssemblyV3d(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V3d', **kwargs)


class NSDAssemblyV3v(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V3v', **kwargs)


class NSDAssemblyV4(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='V4', **kwargs)


class NSDAssemblyLateral(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='lateral', **kwargs)


class NSDAssemblyVentral(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='ventral', **kwargs)


class NSDAssemblyParietal(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='parietal', **kwargs)


class NSDAssemblyHighLateral(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='highlateral', **kwargs)


class NSDAssemblyHighVentral(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='highventral', **kwargs)


class NSDAssemblyHighParietal(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='highparietal', **kwargs)


class NSDAssemblyMidLateral(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='midlateral', **kwargs)


class NSDAssemblyMidVentral(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='midventral', **kwargs)


class NSDAssemblyMidParietal(NSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), NSDAssembly.__name__
        )
        super().__init__(root_dir=root, regions='midparietal', **kwargs)
