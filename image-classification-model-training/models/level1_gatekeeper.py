"""
models/level1_gatekeeper.py

Level 1 Binary Gatekeeper — EfficientNet-B3 pretrained on ImageNet.

Task:    NORMAL (0) vs ABNORMAL (1)
Input:   224×224 RGB tensors (maximum throughput for 86K-image screening)
Backbone: torchvision EfficientNet-B3 (IMAGENET1K_V1 weights)
          12.2M parameters vs 25.6M for ResNet-50 (~3× fewer params)
          ImageNet Top-1: 82.2% vs 80.9% for ResNet-50
          Compound scaling (depth × width × resolution) captures both fine-grained
          local texture and broader structural context — well-suited for retinal
          pathology detection (drusen deposits, fluid accumulation, membrane changes).

Two-Phase Training Protocol
────────────────────────────
Phase 1 — Warm-up (backbone frozen):
  All EfficientNet-B3 feature layers are frozen. Only the custom classifier head
  is trained with a relatively high LR (1e-3). This warms the randomly
  initialised head before gradient flow reaches the pretrained backbone,
  preventing the pretrained features from being corrupted early in training.

Phase 2 — Fine-tuning (full network):
  Backbone is unfrozen via unfreeze_backbone(). Differential LRs are applied
  via get_param_groups(): backbone at 1e-4 (preserves pretrained features),
  head at 1e-3 (continues fast adaptation of classification layers).

Grad-CAM Target Layer
─────────────────────
  Use model.features[-1] (the last MBConv block) as the Grad-CAM target layer.
  This is the final spatial feature map before global average pooling.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights

logger = logging.getLogger(__name__)


class GatekeeperModel(nn.Module):
    """
    EfficientNet-B3 binary classifier for OCT gatekeeper screening.

    Architecture:
        EfficientNet-B3 feature extractor (1536-d spatial feature maps)
        → AdaptiveAvgPool2d(1)           (1536 × 1 × 1)
        → Flatten
        → Dropout(dropout_rate)
        → Linear(1536, 512) + ReLU
        → Dropout(dropout_rate / 2)
        → Linear(512, num_classes)

    Args:
        num_classes:     Output size. 2 for NORMAL/ABNORMAL binary task.
        dropout_rate:    Dropout probability before the first FC layer (0.3).
        pretrained:      Load IMAGENET1K_V1 weights if True.
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

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # EfficientNet has a clean three-part structure: features / avgpool / classifier.
        # We keep features (MBConv blocks) and avgpool; strip only the classifier head.
        self.features = backbone.features        # nn.Sequential of MBConv blocks
        self.avgpool  = backbone.avgpool         # AdaptiveAvgPool2d(output_size=1)

        # in_features = 1536 for EfficientNet-B3
        in_features = backbone.classifier[1].in_features

        # Custom classification head — same topology as the previous ResNet-50 head
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
            "GatekeeperModel ready | backbone=EfficientNet-B3 | "
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
            backbone_lr: LR for pretrained EfficientNet-B3 feature layers.
            head_lr:     LR for the custom classifier head.

        Returns:
            List of dicts compatible with ``torch.optim.AdamW``.
        """
        return [
            {"params": self.features.parameters(),   "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
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
        x = self.features(x)      # (B, 1536, 7, 7) for 224×224 input
        x = self.avgpool(x)       # (B, 1536, 1, 1)
        x = self.classifier(x)    # (B, num_classes)
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
        dropout_rate:    Dropout rate in classifier head (0.3 default).
        pretrained:      Use IMAGENET1K_V1 pretrained weights.
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
