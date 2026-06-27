from pathlib import Path
from typing import Any

import numpy as np
import torch

from .anatomical_flattener import flatten_volume_to_rpe
from .ipnv2_adapter import failed_ipnv2_metadata, ipnv2_metadata, run_ipnv2_smoke_inference
from .pre_processing import get_preprocessing_pipeline
from .scan_types import NormalizedScan
from .segmentation import placeholder_segment_layers as _placeholder_segment_layers
from .segmentation import segment_retinal_layers


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


def process_scan(scan: NormalizedScan, preview_dir: Path | None = None) -> dict[str, Any]:
    validation = validate_volume(scan.volume)
    cropped, crop_info = crop_black_padding(scan.volume)
    foveal_crop, fovea_info = center_crop_volume(cropped, scan.spacing_mm)

    pipeline = get_preprocessing_pipeline()
    tensor = pipeline(foveal_crop)
    flattened = flatten_volume_to_rpe(tensor)
    flattened_volume = flattened.detach().cpu().numpy()[0]

    segmentation_result = segment_retinal_layers(flattened_volume, scan.spacing_mm)
    segmentation = segmentation_result.labels
    layers = extract_layer_features(flattened_volume, segmentation)
    diagnosis, confidence = classify_layers(layers)
    ipnv2_result = None
    try:
        ipnv2_result = run_ipnv2_smoke_inference(flattened_volume)
        ipnv2 = ipnv2_metadata(ipnv2_result)
    except Exception as exc:
        ipnv2 = failed_ipnv2_metadata(exc)

    previews = {}
    if preview_dir is not None:
        from .preview import write_previews

        previews = write_previews(
            preview_dir,
            raw_volume=scan.volume,
            cropped_volume=foveal_crop,
            segmentation=segmentation,
            layers=layers,
            ipnv2_result=ipnv2_result,
        )
        ipnv2["previews"] = {
            key: previews[key]
            for key in ("ipnv2_probability", "ipnv2_overlay")
            if key in previews
        }

    warnings = [*scan.warnings, *validation["warnings"], *crop_info["warnings"], *fovea_info["warnings"]]
    if segmentation_result.warning:
        warnings.append(segmentation_result.warning)
    if ipnv2.get("warning"):
        warnings.append(ipnv2["warning"])

    return {
        "status": "completed",
        "diagnosis": diagnosis,
        "confidence": confidence,
        "source_format": scan.source_format,
        "volume_shape": list(scan.volume_shape),
        "spacing_mm": list(scan.spacing_mm),
        "is_demo_model": segmentation_result.mode == "placeholder",
        "qc": {
            "signal_range": validation["signal_range"],
            "crop_applied": crop_info["crop_applied"],
            "crop_bounds": crop_info["crop_bounds"],
            "fovea_crop_bounds": fovea_info["crop_bounds"],
            "warnings": warnings,
        },
        "layers": layers,
        "previews": previews,
        "ipnv2": ipnv2,
        "metadata": scan.metadata,
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
