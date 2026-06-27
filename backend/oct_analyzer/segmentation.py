from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


SEGMENTATION_ATLAS_ENV = "OCT_LAYER_ATLAS"
DEFAULT_LAYER_COUNT = 12


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    mode: str
    warning: str = ""


LayerSegmenter = Callable[[np.ndarray, tuple[float, float, float]], SegmentationResult]


def segment_retinal_layers(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    segmenter: LayerSegmenter | None = None,
) -> SegmentationResult:
    if segmenter is not None:
        result = segmenter(volume, spacing_mm)
        return _validated_result(result, volume.shape)

    atlas_path = atlas_path_from_env()
    warning = "Using deterministic placeholder layer segmentation"
    if atlas_path is not None:
        warning = (
            f"Atlas asset configured at {atlas_path}, but atlas registration is not connected yet; "
            "using deterministic placeholder layer segmentation"
        )

    return SegmentationResult(
        labels=placeholder_segment_layers(volume.shape),
        mode="placeholder",
        warning=warning,
    )


def placeholder_segment_layers(shape: tuple[int, int, int], num_layers: int = DEFAULT_LAYER_COUNT) -> np.ndarray:
    z_dim, y_dim, x_dim = shape
    if z_dim < num_layers:
        num_layers = z_dim
    labels = np.zeros(shape, dtype=np.uint8)
    edges = np.linspace(0, z_dim, num_layers + 1, dtype=int)
    for index in range(num_layers):
        labels[edges[index]:edges[index + 1], :, :] = index + 1
    if num_layers < DEFAULT_LAYER_COUNT:
        labels[labels == 0] = num_layers
    return labels


def atlas_path_from_env(env: dict[str, str] | None = None) -> Path | None:
    import os

    values = os.environ if env is None else env
    raw_path = values.get(SEGMENTATION_ATLAS_ENV, "").strip()
    if not raw_path:
        return None
    return Path(raw_path).expanduser()


def _validated_result(result: SegmentationResult, expected_shape: tuple[int, int, int]) -> SegmentationResult:
    labels = validate_segmentation_labels(result.labels, expected_shape)
    return SegmentationResult(labels=labels, mode=result.mode, warning=result.warning)


def validate_segmentation_labels(
    labels: np.ndarray,
    expected_shape: tuple[int, int, int],
    max_label: int = DEFAULT_LAYER_COUNT,
) -> np.ndarray:
    array = np.asarray(labels)
    if array.shape != tuple(expected_shape):
        raise ValueError(f"Segmentation labels must match volume shape {expected_shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Segmentation labels must use integer layer IDs")
    if array.size and (int(array.min()) < 0 or int(array.max()) > max_label):
        raise ValueError(f"Segmentation labels must be between 0 and {max_label}")
    return array.astype(np.uint8, copy=False)
