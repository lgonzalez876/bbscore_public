import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

from .sd_unet_wrapper import SDUNetWrapper


class StableDiffusion:
    """BBScore model wrapper for Stable Diffusion 2.1."""

    def __init__(self):
        self.model_mappings = {
            "SD21-T50": 50,
            "SD21-T200": 200,
            "SD21-T500": 500,
            "SD21-T999": 999,
        }
        self.static = True

    def _to_numpy_rgb_uint8(self, x):
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return arr
        elif isinstance(x, Image.Image):
            return np.asarray(x.convert("RGB"))
        else:
            raise ValueError("Frame must be np.ndarray or PIL.Image")

    def preprocess_fn(self, frames, device="cpu"):
        """
        Preprocess to (B, 3, 512, 512), normalized to [-1, 1].
        SD 2.1 was trained on 512x512 images.

        Args:
            frames: list of PIL.Image or HxWx3 uint8 numpy arrays
        Returns:
            torch.Tensor of shape (B, 3, 512, 512)
        """
        if not isinstance(frames, list):
            frames = [frames]
        np_frames = [self._to_numpy_rgb_uint8(f) for f in frames]
        x = torch.from_numpy(np.stack(np_frames, axis=0))  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        transform = T.Compose([
            T.ToDtype(torch.float32, scale=True),  # [0,255] -> [0,1]
            T.Resize(
                512,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(512),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),  # [0,1] -> [-1,1]
        ])

        x = transform(x)
        return x

    def get_model(self, identifier):
        """Load SDUNetWrapper with the appropriate timestep."""
        timestep = self.model_mappings[identifier]
        self.model = SDUNetWrapper(timestep=timestep)
        return self.model

    def postprocess_fn(self, features):
        """Pool spatial dims: (B, [1,] C, H, W) -> (B, C) via global average pooling."""
        # Squeeze all singleton dims between batch (dim 0) and the last 3 dims (C, H, W).
        # e.g. (1, 1, 1, 320, 64, 64) -> (1, 320, 64, 64)
        while features.ndim > 4 and features.shape[1] == 1:
            features = features.squeeze(1)

        if features.ndim == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(
                features, (1, 1)
            ).flatten(1)
        elif features.ndim == 3:
            features = features.reshape(features.shape[0], -1)

        # Safety net: if still > 2D after above checks, force pool
        if features.ndim > 2:
            batch = features.shape[0]
            features = features.reshape(batch, -1)

        return features
