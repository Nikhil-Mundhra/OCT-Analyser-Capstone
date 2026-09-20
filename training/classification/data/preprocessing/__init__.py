"""
data/preprocessing package initialization.
Re-exports public symbols for clean modular access and backward compatibility.
"""

from .white_bars import detect_and_process_white_bars, detect_and_remove_compass_artifacts
from .outliers import has_intensity_support, reject_outliers_1d
from .masking import generate_tissue_mask
from .pipeline import process_image, VALID_EXT

from .mask_diagnostics import evaluate_mask_health, render_diagnostic_panel, MaskHealthReport
from .geometry import letterbox_pad_and_resize, center_and_letterbox_tissue

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
    "evaluate_mask_health",
    "render_diagnostic_panel",
    "MaskHealthReport",
    "letterbox_pad_and_resize",
    "center_and_letterbox_tissue",
    "VALID_EXT",
]

