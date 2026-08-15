"""
tests/test_models_suite.py

Comprehensive unit test battery verifying all 5 production models in `models_suite/`:
  Model 1: RetinalLayersUNet (6-Class Retinal Layer Segmentation)
  Model 2: Choroidalyzer UNet (Choroid Thickness & Region Segmentation)
  Model 3: HRFAttentionUNet (High-Res Fluid & Lesion Attention U-Net)
  Model 4: OIMHSUNet (Macular Hole & Intraretinal Cysts)
  Model 5: OCTPathologyDetector (Faster R-CNN Biomarker Bounding Box Detector)
"""

import sys
import pytest
import torch
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from models_suite.model1_oct5k_layers.unet_layers import RetinalLayersUNet
from models_suite.model2_choroidalyzer.choroidalyze.model import UNet as ChoroidalyzerUNet
from models_suite.model3_hrf_dme.hrf_aunet import HRFAttentionUNet
from models_suite.model4_oimhs_hole_cysts.oimhs_unet import OIMHSUNet
from models_suite.model5_oct5k_detection.detector import OCTPathologyDetector, OCT5K_DETECTION_CLASSES


def test_model1_retinal_layers_unet_forward():
    """Verify Model 1 instantiation and forward pass tensor dimensions."""
    model = RetinalLayersUNet(in_channels=1, num_classes=6)
    model.eval()

    dummy_input = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 6, 128, 128), f"Expected shape (2, 6, 128, 128), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_model2_choroidalyzer_unet_forward():
    """Verify Model 2 (Choroidalyzer) forward pass."""
    model = ChoroidalyzerUNet(
        in_channels=1,
        out_channels=3,
        channels=[32, 64, 128],
        depth=2,
        dynamic_padding=True
    )
    model.eval()

    dummy_input = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 3, 128, 128), f"Expected shape (2, 3, 128, 128), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_model3_hrf_attention_unet_forward():
    """Verify Model 3 (HRF DME Attention U-Net) forward pass."""
    model = HRFAttentionUNet(n_channels=3, n_classes=1, base_filters=16)
    model.eval()

    dummy_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1, 128, 128), f"Expected shape (2, 1, 128, 128), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_model4_oimhs_unet_forward():
    """Verify Model 4 (OIMHS Macular Hole & Cysts) forward pass."""
    model = OIMHSUNet(in_channels=1, num_classes=5)
    model.eval()

    dummy_input = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 5, 128, 128), f"Expected shape (2, 5, 128, 128), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_model5_oct_pathology_detector_eval_forward():
    """Verify Model 5 (Faster R-CNN Detector) forward pass in eval mode."""
    model = OCTPathologyDetector(num_classes=len(OCT5K_DETECTION_CLASSES))
    model.eval()

    dummy_images = [torch.rand(3, 128, 128)]
    with torch.no_grad():
        detections = model(dummy_images)

    assert isinstance(detections, list), "Expected list of detections"
    assert len(detections) == 1, "Expected 1 detection dictionary"
    assert "boxes" in detections[0], "Missing 'boxes' key in detector output"
    assert "labels" in detections[0], "Missing 'labels' key in detector output"
    assert "scores" in detections[0], "Missing 'scores' key in detector output"


def test_model5_oct_pathology_detector_train_loss():
    """Verify Model 5 forward pass in training mode produces valid bounding box losses."""
    model = OCTPathologyDetector(num_classes=len(OCT5K_DETECTION_CLASSES))
    model.train()

    dummy_images = [torch.rand(3, 128, 128)]
    dummy_targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64)
    }]

    losses = model(dummy_images, dummy_targets)
    assert isinstance(losses, dict), "Expected dictionary of losses during training"
    assert "loss_classifier" in losses
    assert "loss_box_reg" in losses
    assert not torch.isnan(losses["loss_classifier"])
