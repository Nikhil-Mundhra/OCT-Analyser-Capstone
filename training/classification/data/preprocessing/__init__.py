"""
data/preprocessing package initialization.
Re-exports public symbols for clean modular access and backward compatibility.
"""

from .white_bars import detect_and_process_white_bars, detect_and_remove_compass_artifacts
from .outliers import has_intensity_support, reject_outliers_1d
from .masking import generate_tissue_mask
from .pipeline import process_image, VALID_EXT

# Backwards compatibility alias for _reject_outliers_1d
_reject_outliers_1d = reject_outliers_1d
_has_intensity_support = has_intensity_support

__all__ = [
    "detect_and_process_white_bars",
    "detect_and_remove_compass_artifacts",
    "has_intensity_support",
    "reject_outliers_1d",
    "_reject_outliers_1d",
    "_has_intensity_support",
    "generate_tissue_mask",
    "process_image",
    "VALID_EXT",
]
