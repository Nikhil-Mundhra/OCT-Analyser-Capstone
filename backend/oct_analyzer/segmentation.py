from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import sys
import os
import cv2
import torch
import numpy as np


SEGMENTATION_ATLAS_ENV = "OCT_LAYER_ATLAS"
DEFAULT_LAYER_COUNT = 15 # Changed from 12 to 15

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_ML_SEG_DIR = PROJECT_ROOT / "backend" / "core_ml" / "segmentation"

try:
    from backend.core_ml.segmentation.models.unet import HierarchicalUNet
except ImportError:
    HierarchicalUNet = None

class UNetSegmenter:
    _instance = None

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        checkpoint_path = CORE_ML_SEG_DIR / "weights" / "unet_hierarchical_best.pth"
            
        if HierarchicalUNet is not None and checkpoint_path.exists():
            try:
                self.model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
                # Ensure we load on the correct device
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Failed to load UNet model: {e}")
                self.model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def segment(self, volume: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("UNet model not loaded")
        
        # volume shape is (Z, Y, X)
        z_dim, y_dim, x_dim = volume.shape
        labels_3d = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)
        
        batch_array = np.zeros((y_dim, 1, 512, 512), dtype=np.float32)
        
        for y in range(y_dim):
            slice_2d = volume[:, y, :] # (Z, X)
            
            slice_min = slice_2d.min()
            slice_max = slice_2d.max()
            if slice_max > slice_min:
                slice_norm = (slice_2d - slice_min) / (slice_max - slice_min)
            else:
                slice_norm = np.zeros_like(slice_2d, dtype=np.float32)
                
            # Resize using bilinear for the input image
            resized = cv2.resize(slice_norm, (512, 512), interpolation=cv2.INTER_LINEAR)
            batch_array[y, 0, :, :] = resized
            
        tensor_batch = torch.from_numpy(batch_array).to(self.device)
        
        with torch.no_grad():
            coarse_logits, granular_logits = self.model(tensor_batch)
            granular_preds = torch.argmax(granular_logits, dim=1).cpu().numpy().astype(np.uint8) # (y_dim, 512, 512)
            
        for y in range(y_dim):
            pred_2d = granular_preds[y]
            # Resize back to (x_dim, z_dim) using nearest neighbor for the mask
            resized_back = cv2.resize(pred_2d, (x_dim, z_dim), interpolation=cv2.INTER_NEAREST)
            labels_3d[:, y, :] = resized_back
            
        return labels_3d

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

    unet = UNetSegmenter.get_instance()
    if unet.model is not None:
        try:
            labels = unet.segment(volume)
            return _validated_result(SegmentationResult(labels=labels, mode="unet_15_layer"), volume.shape)
        except Exception as e:
            warning = f"UNet segmentation failed: {e}. Falling back to placeholder."
    else:
        warning = "UNet model not loaded. Falling back to deterministic placeholder layer segmentation"

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
