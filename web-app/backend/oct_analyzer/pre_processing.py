from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch

def get_preprocessing_pipeline():
    """
    Builds a deterministic pipeline to standardize all incoming OCT volumes.
    """
    try:
        from monai.transforms import (
            Compose,
            EnsureChannelFirst,
            ScaleIntensityRangePercentiles,
            ToTensor,
        )

        return Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensityRangePercentiles(
                lower=1.0,
                upper=99.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            ToTensor(dtype=torch.float32),
        ])
    except ModuleNotFoundError:
        return _fallback_preprocessing


def _fallback_preprocessing(volume):
    """
    Minimal preprocessing used when MONAI is unavailable.

    Returns a channel-first float tensor with intensities clipped to the
    1st-99th percentile range and normalized to [0, 1].
    """
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D OCT volume, got shape {array.shape}")

    lower, upper = np.percentile(array, (1.0, 99.0))
    if upper <= lower:
        normalized = np.zeros_like(array, dtype=np.float32)
    else:
        normalized = np.clip(array, lower, upper)
        normalized = (normalized - lower) / (upper - lower)

    return torch.from_numpy(normalized).unsqueeze(0).float()
