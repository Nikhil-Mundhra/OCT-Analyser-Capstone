"""
data/preprocessing.py

Modular OCT B-scan preprocessing entry point.
Re-exports modular components from data.preprocessing package for backward compatibility.
"""

import sys
from pathlib import Path

# Ensure package path is resolvable
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from data.preprocessing import (
    detect_and_process_white_bars,
    detect_and_remove_compass_artifacts,
    has_intensity_support,
    reject_outliers_1d,
    _reject_outliers_1d,
    _has_intensity_support,
    generate_tissue_mask,
    process_image,
    VALID_EXT,
)

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
