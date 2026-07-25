from pathlib import Path
from typing import Any

import numpy as np
import torch

from .anatomical_flattener import flatten_volume_to_rpe
from .pre_processing import get_preprocessing_pipeline
from .scan_types import NormalizedScan
from .segmentation import placeholder_segment_layers as _placeholder_segment_layers
from .segmentation import segment_retinal_layers
import os
import cv2
import sys
from dataclasses import asdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_ML_SEG_DIR = PROJECT_ROOT / "backend" / "core_ml" / "segmentation"

try:
    from backend.core_ml.segmentation.inference.analyzer import SegmentationAnalyzer
except ImportError:
    SegmentationAnalyzer = None


LAYER_NAMES = [
    "ILM",
    "RNFL",
    "GCL",
    "IPL",
    "INL",
    "OPL",
    "ONL",
    "ELM",
    "IS/OS",
    "RPE",
    "Choroid",
    "Sclera",
]


def process_scan(scan: NormalizedScan, preview_dir: Path | None = None, progress_cb=None) -> dict[str, Any]:
    if progress_cb: progress_cb("Validating volume")
    validation = validate_volume(scan.volume)
    if scan.source_format == "single-image":
        if progress_cb: progress_cb("Skipping 3D depth cropping for 2D image")
        foveal_crop = scan.volume
        crop_info = {"crop_applied": False, "crop_bounds": [], "warnings": []}
        fovea_info = {"crop_bounds": [], "warnings": ["Skipped foveal center crop for 2D image"]}
    else:
        if progress_cb: progress_cb("Preprocessing volume (padding/cropping)")
        cropped, crop_info = crop_black_padding(scan.volume)
        foveal_crop, fovea_info = center_crop_volume(cropped, scan.spacing_mm)

    pipeline = get_preprocessing_pipeline()
    tensor = pipeline(foveal_crop)
    
    # Only run the anatomical flattener on actual 3D volumes.
    # Single 2D slices (e.g. PNG/JPEG uploads) should preserve their curvature.
    if tensor.shape[1] > 1:
        if progress_cb: progress_cb("Flattening anatomical structures to RPE")
        flattened = flatten_volume_to_rpe(tensor)
        flattened_volume = flattened.detach().cpu().numpy()[0]
    else:
        if progress_cb: progress_cb("Skipping anatomical flattening for 2D image")
        flattened_volume = tensor.detach().cpu().numpy()[0]

    model_type = os.environ.get("OCT_MODEL_TYPE", "legacy_convnext")
    if progress_cb:
        if model_type == "unified_unet":
            progress_cb("Running hierarchical UNet inference (segmentation & classification)")
        else:
            progress_cb("Running segmentation model")
    
    segmentation_result, pipeline_results = segment_retinal_layers(flattened_volume, scan.spacing_mm)
    segmentation = segmentation_result.labels
    if progress_cb: progress_cb("Extracting biomarker features")
    layers = extract_layer_features(flattened_volume, segmentation)
    
    # Run Segmentation Analyzer on the appropriate slice
    segmentation_analysis = None
    if SegmentationAnalyzer is not None:
        try:
            model_type = os.environ.get("OCT_MODEL_TYPE", "legacy_convnext")
            if model_type == "unified_unet":
                best_slice_idx = pipeline_results.get("best_slice_idx", 0)
                best_mask = segmentation[best_slice_idx, :, :]
            else:
                z_dim, y_dim, x_dim = segmentation.shape
                best_mask = segmentation[:, int(y_dim * 0.5), :]
                
            analyzer = SegmentationAnalyzer()
            seg_obj = analyzer.analyze(best_mask)
            segmentation_analysis = asdict(seg_obj)
        except Exception as e:
            print(f"Failed to run SegmentationAnalyzer: {e}")

    model_type = os.environ.get("OCT_MODEL_TYPE", "legacy_convnext")
    if model_type == "unified_unet":
        diagnosis = pipeline_results.get("Final_Diagnosis", "Unknown")
        confidence = pipeline_results.get("confidence", 0.0)
    else:
        # Run hierarchical classifier (L1/L2/L3) on the 2D flattened volume
        try:
            if progress_cb: progress_cb("Running Multi-Head classification model")
            from .classifier_integration import get_classifier
            import tempfile
            from uuid import uuid4
            classifier = get_classifier()
            # Save flattened volume to temporary file for classifier
            temp_img = Path(tempfile.gettempdir()) / f"flattened_{uuid4().hex}.png"
            cv2.imwrite(str(temp_img), flattened_volume[0] * 255)
            legacy_pipeline_results = classifier.predict(str(temp_img), gradcam=True)
            temp_img.unlink(missing_ok=True)
            
            # Merge results for frontend display
            pipeline_results["Level1"] = legacy_pipeline_results.get("Level1", {})
            pipeline_results["Level2"] = legacy_pipeline_results.get("Level2", {})
            pipeline_results["Level3"] = legacy_pipeline_results.get("Level3", {})
            pipeline_results["gradcams"] = legacy_pipeline_results.get("gradcams", {})
            
            # Set the top-level diagnosis to the deepest available prediction
            if "Level3" in legacy_pipeline_results and legacy_pipeline_results["Level3"].get("prediction"):
                diagnosis = legacy_pipeline_results["Level3"]["prediction"]
                confidence = legacy_pipeline_results["Level3"]["confidence"]
            elif "Level2" in legacy_pipeline_results and legacy_pipeline_results["Level2"].get("prediction"):
                diagnosis = legacy_pipeline_results["Level2"]["prediction"]
                confidence = legacy_pipeline_results["Level2"]["confidence"]
            else:
                diagnosis = legacy_pipeline_results.get("Level1", {}).get("prediction", "Unknown")
                confidence = legacy_pipeline_results.get("Level1", {}).get("confidence", 0.0)
                
        except Exception as e:
            print(f"Failed to run ClassifierWrapper: {e}")
            # Fallback
            diagnosis, confidence = classify_layers(layers)

    previews = {}
    if preview_dir is not None:
        from .preview import write_previews

        previews = write_previews(
            preview_dir,
            raw_volume=scan.volume,
            cropped_volume=foveal_crop,
            segmentation=segmentation,
            layers=layers,
        )

    warnings = [*scan.warnings, *validation["warnings"], *crop_info["warnings"], *fovea_info["warnings"]]
    if segmentation_result.warning:
        warnings.append(segmentation_result.warning)

    return {
        "status": "completed",
        "diagnosis": diagnosis,
        "confidence": confidence,
        "source_format": scan.source_format,
        "volume_shape": list(scan.volume_shape),
        "spacing_mm": list(scan.spacing_mm),
        "is_demo_model": segmentation_result.mode == "placeholder",
        "model_type": model_type,
        "qc": {
            "signal_range": validation["signal_range"],
            "crop_applied": crop_info["crop_applied"],
            "crop_bounds": crop_info["crop_bounds"],
            "fovea_crop_bounds": fovea_info["crop_bounds"],
            "warnings": warnings,
        },
        "layers": layers,
        "previews": previews,
        "metadata": scan.metadata,
        "level1": pipeline_results.get("Level1", {}),
        "level2": pipeline_results.get("Level2", {}),
        "level3": pipeline_results.get("Level3", {}),
        "gradcams": pipeline_results.get("gradcams", {}),
        "segmentation": segmentation_analysis,
    }


def validate_volume(volume: np.ndarray) -> dict[str, Any]:
    array = np.asarray(volume)
    warnings: list[str] = []

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D OCT volume, got shape {array.shape}")
    if min(array.shape) <= 0:
        raise ValueError(f"Volume dimensions must be non-empty, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Volume contains NaN or infinite values")

    signal_min = float(array.min())
    signal_max = float(array.max())
    if signal_max <= signal_min:
        warnings.append("Volume has no intensity variation")

    return {
        "signal_range": [signal_min, signal_max],
        "warnings": warnings,
    }


def crop_black_padding(volume: np.ndarray, threshold_ratio: float = 0.02) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(volume)
    signal_min = float(array.min())
    signal_max = float(array.max())
    threshold = signal_min + (signal_max - signal_min) * threshold_ratio
    foreground = array > threshold
    warnings: list[str] = []

    if not foreground.any():
        warnings.append("Could not detect non-background tissue for padding crop")
        return array, {
            "crop_applied": False,
            "crop_bounds": [0, array.shape[0], 0, array.shape[1], 0, array.shape[2]],
            "warnings": warnings,
        }

    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    cropped = array[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    bounds = [int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), int(mins[2]), int(maxs[2])]

    return cropped, {
        "crop_applied": cropped.shape != array.shape,
        "crop_bounds": bounds,
        "warnings": warnings,
    }


def center_crop_volume(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    target_depth_mm: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(volume)
    z_spacing = max(float(spacing_mm[0]), 1e-6)
    target_z = max(1, min(array.shape[0], int(round(target_depth_mm / z_spacing))))
    center_z = int(np.argmax(array.mean(axis=(1, 2))))
    start_z = max(0, min(array.shape[0] - target_z, center_z - target_z // 2))
    end_z = start_z + target_z

    bounds = [int(start_z), int(end_z), 0, int(array.shape[1]), 0, int(array.shape[2])]
    warnings = []
    if target_z == array.shape[0]:
        warnings.append("Fovea crop kept full depth because scan is smaller than target depth")

    return array[start_z:end_z], {
        "crop_bounds": bounds,
        "warnings": warnings,
    }


def placeholder_segment_layers(shape: tuple[int, int, int], num_layers: int = 12) -> np.ndarray:
    return _placeholder_segment_layers(shape, num_layers)


def extract_layer_features(volume: np.ndarray, segmentation: np.ndarray) -> list[dict[str, Any]]:
    energy = second_order_reflectivity_energy(volume)
    layers = []

    for index, name in enumerate(LAYER_NAMES, start=1):
        values = energy[segmentation == index]
        if values.size == 0:
            deciles = [0.0] * 9
            score = 0.0
        else:
            deciles = [float(value) for value in np.percentile(values, np.arange(10, 100, 10))]
            score = float(np.clip(np.mean(deciles) / (np.std(volume) + 1e-6), 0.0, 1.0))

        vote = "DR" if score >= 0.42 else "Healthy"
        layers.append({
            "name": name,
            "vote": vote,
            "score": round(score, 4),
            "cdf_deciles": [round(value, 6) for value in deciles],
        })

    return layers


def second_order_reflectivity_energy(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    energy = np.zeros_like(array, dtype=np.float32)
    for axis in range(3):
        forward = np.roll(array, -1, axis=axis)
        backward = np.roll(array, 1, axis=axis)
        energy += np.abs(array - forward) + np.abs(array - backward)
    return energy / 6.0


def classify_layers(layers: list[dict[str, Any]]) -> tuple[str, float]:
    dr_votes = sum(1 for layer in layers if layer["vote"] == "DR")
    diagnosis = "DR" if dr_votes > len(layers) / 2 else "Healthy"
    confidence = dr_votes / len(layers) if diagnosis == "DR" else 1.0 - (dr_votes / len(layers))
    return diagnosis, round(float(confidence), 4)
