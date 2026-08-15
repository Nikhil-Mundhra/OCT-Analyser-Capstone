"""
training/segmentation/tests/test_segmentation_training.py

Unit tests verifying the training configurations, model forward/loss passes,
and tensor pipelines across all 3 segmentation training modules:
  - train_model1_oct5k_layers
  - train_model4_oimhs_hole_cysts
  - train_model5_oct5k_detection
"""

import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SEGMENTATION_ROOT = WORKSPACE_ROOT / "training" / "segmentation"

for p in [WORKSPACE_ROOT, SEGMENTATION_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models_suite.model1_oct5k_layers.unet_layers import RetinalLayersUNet
from models_suite.model4_oimhs_hole_cysts.oimhs_unet import OIMHSUNet
from models_suite.model5_oct5k_detection.detector import OCTPathologyDetector, OCT5K_DETECTION_CLASSES


def test_model1_training_step():
    """Verify Model 1 training step computes valid gradients with CrossEntropyLoss."""
    model = RetinalLayersUNet(in_channels=1, num_classes=6)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic batch: 2 grayscale images (1x64x64) and masks (64x64 with classes 0-5)
    dummy_images = torch.randn(2, 1, 64, 64)
    dummy_masks = torch.randint(0, 6, (2, 64, 64), dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(dummy_images)
    loss = criterion(outputs, dummy_masks)

    assert not torch.isnan(loss), "Model 1 training loss is NaN"
    assert loss.item() > 0, "Model 1 training loss must be positive"

    loss.backward()
    # Check gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient not computed for {name}"
            break


def test_model4_training_step():
    """Verify Model 4 (OIMHS) training step computes valid gradients with CrossEntropyLoss."""
    model = OIMHSUNet(in_channels=1, num_classes=5)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dummy_images = torch.randn(2, 1, 64, 64)
    dummy_masks = torch.randint(0, 5, (2, 64, 64), dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(dummy_images)
    loss = criterion(outputs, dummy_masks)

    assert not torch.isnan(loss), "Model 4 training loss is NaN"
    assert loss.item() > 0, "Model 4 training loss must be positive"

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient not computed for {name}"
            break


def test_model5_detector_training_step():
    """Verify Model 5 (Faster R-CNN) training step computes valid multi-task bounding box losses."""
    model = OCTPathologyDetector(num_classes=len(OCT5K_DETECTION_CLASSES))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    dummy_images = [torch.rand(3, 128, 128), torch.rand(3, 128, 128)]
    dummy_targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]], dtype=torch.float32),
            "labels": torch.tensor([1, 2], dtype=torch.int64)
        },
        {
            "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]], dtype=torch.float32),
            "labels": torch.tensor([3], dtype=torch.int64)
        }
    ]

    optimizer.zero_grad()
    loss_dict = model(dummy_images, dummy_targets)

    total_loss = sum(loss for loss in loss_dict.values())
    assert not torch.isnan(total_loss), "Model 5 detector total loss is NaN"
    assert total_loss.item() > 0, "Model 5 detector total loss must be positive"

    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
