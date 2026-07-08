"""
models/level1_gatekeeper.py

Level 1 Binary Gatekeeper — ResNet-50 pretrained on ImageNet.

Task:    NORMAL (0) vs ABNORMAL (1)
Input:   224×224 RGB tensors (maximum throughput for 86K-image screening)
Backbone: torchvision ResNet-50 (IMAGENET1K_V2 weights — best accuracy preset)

Two-Phase Training Protocol
────────────────────────────
Phase 1 — Warm-up (backbone frozen):
  All ResNet-50 feature layers are frozen. Only the custom classifier head
  is trained with a relatively high LR (1e-3). This warms the randomly
  initialised head before gradient flow reaches the pretrained backbone,
  preventing the pretrained features from being corrupted early in training.

Phase 2 — Fine-tuning (full network):
  Backbone is unfrozen via unfreeze_backbone(). Differential LRs are applied
  via get_param_groups(): backbone at 1e-4 (preserves pretrained features),
  head at 1e-3 (continues fast adaptation of classification layers).
"""

import logging
from typing import Dict, List

import os
# Prevent Apple Silicon segmentation fault in torch.load when loading torchvision weights
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)


class GatekeeperModel(nn.Module):
    """
    ResNet-50 binary classifier for OCT gatekeeper screening.

    Architecture:
        ResNet-50 feature extractor (2048-d global avg pool output)
        → Dropout(0.3)
        → Linear(2048, 512) + ReLU
        → Dropout(0.15)
        → Linear(512, num_classes)

    Args:
        num_classes:     Output size. 2 for NORMAL/ABNORMAL binary task.
        dropout_rate:    Dropout probability before the first FC layer.
        pretrained:      Load IMAGENET1K_V2 weights if True.
        freeze_backbone: Start with backbone frozen (Phase 1 warm-up).
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Strip the original 1000-class FC head; keep everything up to avgpool
        # children: conv1 → bn1 → relu → maxpool → layer1-4 → avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Custom head
        in_features = backbone.fc.in_features  # 2048 for ResNet-50
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        logger.info(
            "GatekeeperModel ready | backbone=ResNet-50 | "
            "in_features=%d | num_classes=%d | frozen=%s",
            in_features, num_classes, freeze_backbone,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Freeze / Unfreeze API (called by HierarchyTrainer between phases)
    # ──────────────────────────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters — only head trains (Phase 1)."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("Backbone FROZEN — training classifier head only.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone — full network fine-tuning (Phase 2)."""
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info("Backbone UNFROZEN — full network fine-tuning active.")

    def get_param_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
    ) -> List[Dict]:
        """
        Differential LR parameter groups for Phase 2 optimiser.

        Pretrained backbone layers train at a lower rate to preserve
        ImageNet representations while the head adapts to OCT domain.

        Args:
            backbone_lr: LR for pretrained ResNet-50 feature layers.
            head_lr:     LR for the custom classifier head.

        Returns:
            List of dicts compatible with ``torch.optim.AdamW``.
        """
        return [
            {"params": self.features.parameters(),    "lr": backbone_lr},
            {"params": self.classifier.parameters(),  "lr": head_lr},
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor, shape ``(B, 3, 224, 224)``.

        Returns:
            Logits tensor, shape ``(B, num_classes)``.
        """
        x = self.features(x)       # (B, 2048, 1, 1)
        x = self.classifier(x)     # (B, num_classes)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_gatekeeper(
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> GatekeeperModel:
    """
    Factory function — preferred way to instantiate the Level 1 Gatekeeper.

    Args:
        num_classes:     2 for binary NORMAL/ABNORMAL.
        dropout_rate:    Dropout rate in classifier head.
        pretrained:      Use IMAGENET1K_V2 pretrained weights.
        freeze_backbone: Start with frozen backbone for warm-up phase.

    Returns:
        Configured :class:`GatekeeperModel`.
    """
    return GatekeeperModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
