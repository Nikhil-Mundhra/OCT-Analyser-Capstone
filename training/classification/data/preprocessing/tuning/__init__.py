"""
Preprocessing Tuning Sub-System.

Provides an interactive local web server and calibration dashboard for
fine-tuning folder-specific OCT retinal tissue masking and binarization parameters.
"""

from data.preprocessing.tuning.boundaries import (
    _estimate_adaptive_thresholds,
    _extract_raw_boundary_contours,
    _interpolate_and_filter_boundaries,
    compute_sfcm_choroid_boundary,
    detect_rpe_band,
    detect_choroidal_caverns,
    detect_choroidal_holes,
    generate_tissue_mask_custom,
    get_sfcm_cache_key,
    letterbox_pad_and_resize,
    project_and_downsample_vectors,
    suppress_boundary_spikes,
)
from data.preprocessing.tuning.diagnostics import (
    InProcessSocket,
    check_filesystem_access,
    dispatch_in_process_request,
    perform_preflight_checks,
    run_standalone_self_tests,
    verify_server_endpoints,
)
from data.preprocessing.tuning.processor import (
    FOLDER_SAMPLES_CACHE,
    MASKED_DATASET_DIR,
    OUTPUT_DIR,
    SFCM_CACHE,
    SOURCE_DIR,
    curate_folder_batch,
    find_folder_path,
    find_image_path,
    get_available_subfolders,
    get_curated_manifest,
    get_masked_dataset_dir,
    get_output_dir,
    get_source_dir,
    process_and_save_image,
    remove_curated_mask_sample,
    reprocess_folder_sample,
    reprocess_single_image,
    save_curated_mask_sample,
)
from data.preprocessing.tuning.server import (
    FineTuningRequestHandler,
    ReusableHTTPServer,
    main,
    run_server,
)

__all__ = [
    "FineTuningRequestHandler",
    "ReusableHTTPServer",
    "run_server",
    "main",
    "suppress_boundary_spikes",
    "detect_rpe_band",
    "compute_sfcm_choroid_boundary",
    "generate_tissue_mask_custom",
    "letterbox_pad_and_resize",
    "project_and_downsample_vectors",
    "find_folder_path",
    "find_image_path",
    "get_available_subfolders",
    "process_and_save_image",
    "reprocess_folder_sample",
    "reprocess_single_image",
    "save_curated_mask_sample",
    "remove_curated_mask_sample",
    "get_curated_manifest",
    "curate_folder_batch",
    "get_source_dir",
    "get_output_dir",
    "get_masked_dataset_dir",
    "check_filesystem_access",
    "perform_preflight_checks",
    "verify_server_endpoints",
    "run_standalone_self_tests",
    "SFCM_CACHE",
    "FOLDER_SAMPLES_CACHE",
    "SOURCE_DIR",
    "OUTPUT_DIR",
    "MASKED_DATASET_DIR",
]
