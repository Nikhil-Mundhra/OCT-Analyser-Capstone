"""
models/level2_router.py

Level 2 Disease Router — EfficientNet-B2 pretrained on ImageNet.

Task:   Route ABNORMAL scans into 5 disease families
Input:  224×224 RGB tensors (same resolution as L1 for pipeline consistency)
Output: 5-class logits
  0 → Macular_Degeneration   (CNV + DRUSEN + AMD  → 47,107 images, 79%)
  1 → Diabetic_Complications  (DME + DR            → 11,602 images)
  2 → Vascular_Occlusions     (MH + RVO + RAO     →    225 images) ← aggregated
  3 → Fluid_Accumulation      (CSR                →    102 images)
  4 → Structural_Issues       (ERM + VID          →    231 images)

Design Notes:
  - EfficientNet-B2 is chosen over ResNet-50 for L2 because it achieves
    higher accuracy with fewer parameters (compound scaling), which matters
    when the training signal from minority families is weak.
  - Label smoothing (ε=0.1) is applied to mitigate annotation ambiguity
    from merging three heterogeneous source datasets.
  - FocalLoss (γ=2) is the primary imbalance mitigation at the loss level.
  - Backbone progressive unfreezing follows the same 2-phase protocol as L1.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B2_Weights

logger = logging.getLogger(__name__)


class DiseaseRouterModel(nn.Module):
    """
    EfficientNet-B2 multi-class disease family router.

    Args:
        num_classes:     Number of disease families (5).
        dropout_rate:    Dropout in classifier head (0.4 recommended for B2).
        pretrained:      Load IMAGENET1K_V1 weights if True.
        freeze_backbone: Start with backbone frozen.
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout_rate: float = 0.4,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)

        # EfficientNet anatomy: features → avgpool → classifier
        self.features = backbone.features   # MBConv blocks
        self.avgpool  = backbone.avgpool    # AdaptiveAvgPool2d(1, 1)

        # EfficientNet-B2 produces 1408 channels after avgpool
        in_features = backbone.classifier[-1].in_features  # 1408

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        logger.info(
            "DiseaseRouterModel ready | backbone=EfficientNet-B2 | "
            "in_features=%d | num_classes=%d | frozen=%s",
            in_features, num_classes, freeze_backbone,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Freeze / Unfreeze API
    # ──────────────────────────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze MBConv feature layers — head-only warm-up."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("Router backbone FROZEN.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info("Router backbone UNFROZEN.")

    def get_param_groups(
        self,
        backbone_lr: float = 5e-5,
        head_lr: float = 5e-4,
    ) -> List[Dict]:
        """
        Differential LR groups for Phase 2 AdamW.

        EfficientNet-B2 uses a lower backbone LR than ResNet-50 because
        compound scaling makes its feature extraction more specialised —
        larger perturbations risk destroying learned representations.
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
            Logits tensor, shape ``(B, num_classes=5)``.
        """
        x = self.features(x)          # (B, 1408, H', W')
        x = self.avgpool(x)           # (B, 1408, 1, 1)
        x = torch.flatten(x, 1)       # (B, 1408)
        x = self.classifier(x)        # (B, 5)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_router(
    num_classes: int = 5,
    dropout_rate: float = 0.4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> DiseaseRouterModel:
    """
    Factory function for the Level 2 Disease Router.

    Args:
        num_classes:     5 disease families.
        dropout_rate:    Head dropout (0.4 default for EfficientNet-B2).
        pretrained:      Use ImageNet pretrained weights.
        freeze_backbone: Start with frozen backbone.

    Returns:
        Configured :class:`DiseaseRouterModel`.
    """
    return DiseaseRouterModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
