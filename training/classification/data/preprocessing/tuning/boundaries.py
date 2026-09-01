"""
data/preprocessing/tuning/boundaries.py

Modular Re-Export Façade for Retinal Boundary Detection & Multi-Surface Segmentation.
Preserves complete backward compatibility across scripts, tests, and API endpoints
while delegating core domain logic to dedicated single-responsibility modules.
"""

from data.preprocessing.masking import (
    _detect_ilm_dp,
    _estimate_adaptive_thresholds,
    _extract_raw_boundary_contours,
    _interpolate_and_filter_boundaries,
    detect_boundaries_intelligent_auto,
    generate_tissue_mask,
    generate_tissue_mask_custom,
    suppress_boundary_spikes,
)
from data.preprocessing.choroid import (
    chaikin_subdivision,
    compute_sfcm_choroid_boundary,
    detect_choroidal_caverns,
    detect_choroidal_holes,
    detect_rpe_band,
    get_sfcm_cache_key,
    refine_lumen_boundary,
)
from data.preprocessing.geometry import (
    letterbox_pad_and_resize,
    project_and_downsample_vectors,
    render_boundary_overlay,
)
from data.preprocessing.multisurface import (
    BSplineRegularizer,
    JointMultiSurfaceOptimizer,
    MultiSurfaceConfig,
)
from data.preprocessing.outliers import (
    has_intensity_support,
    reject_outliers_1d,
)

__all__ = [
    "_detect_ilm_dp",
    "_estimate_adaptive_thresholds",
    "_extract_raw_boundary_contours",
    "_interpolate_and_filter_boundaries",
    "detect_boundaries_intelligent_auto",
    "generate_tissue_mask",
    "generate_tissue_mask_custom",
    "suppress_boundary_spikes",
    "chaikin_subdivision",
    "compute_sfcm_choroid_boundary",
    "detect_choroidal_caverns",
    "detect_choroidal_holes",
    "detect_rpe_band",
    "get_sfcm_cache_key",
    "refine_lumen_boundary",
    "letterbox_pad_and_resize",
    "project_and_downsample_vectors",
    "render_boundary_overlay",
    "BSplineRegularizer",
    "JointMultiSurfaceOptimizer",
    "MultiSurfaceConfig",
    "has_intensity_support",
    "reject_outliers_1d",
]
