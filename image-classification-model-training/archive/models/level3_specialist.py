"""
models/level3_specialist.py

Level 3 Specialist Models — EfficientNet-B0 fine-grained classifiers.

Input:  384×384 RGB tensors (2.9× more pixels than L1/L2 for structural detail)
Backbone: EfficientNet-B0 (smaller/faster than B2 — appropriate for specialists
          that operate on a much smaller subset of the total dataset)

Specialist Instances:
  ┌───────────────┬──────────────────────────────────────────────┬────────┐
  │ Key           │ Classes                                      │ Total  │
  ├───────────────┼──────────────────────────────────────────────┼────────┤
  │ Macular       │ CNV (Wet AMD) / DRUSEN (Dry AMD) / Generic_AMD│ 47,107│
  │ Diabetic      │ DME / DR                                     │ 11,602 │
  │ Vascular      │ MH / RVO / RAO                               │    225 │
  │ Fluid         │ CSR (single-class anomaly)                   │    102 │
  │ Structural    │ ERM / VID                                    │    231 │
  └───────────────┴──────────────────────────────────────────────┴────────┘

AMD Mapping (as per architectural directive):
  - CNV:         Wet AMD / Choroidal Neovascularization (class 0)
  - DRUSEN:      Dry AMD / Drusen deposits (class 1)
  - Generic_AMD: Unclassified AMD from OCTID source (class 2)
  These are STRICTLY SEPARATED — never merged.

Vascular Mapping (as per architectural directive):
  - L2 routes all Vascular into a single Vascular_Occlusions bucket.
  - L3_Vascular re-separates them: MH (0) / RVO (1) / RAO (2).

384px Justification:
  At 224px, the subtle textural difference between drusen deposits (dry AMD)
  and sub-retinal fluid (wet AMD/CNV) can be lost. 384px preserves the
  fine-grained structural detail needed for specialist discrimination.
  Batch size is reduced to 16 to fit within 32GB MPS unified memory.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Registry of all specialist configurations
# ──────────────────────────────────────────────────────────────────────────────
SPECIALIST_CONFIGS: Dict[str, Dict] = {
    "Macular": {
        "num_classes":     3,
        "specialist_name": "L3_Macular",
        "description":     "CNV (Wet AMD) vs DRUSEN (Dry AMD) vs Generic_AMD — strictly separated",
        "classes":         {0: "CNV", 1: "DRUSEN", 2: "Generic_AMD"},
    },
    "Diabetic": {
        "num_classes":     2,
        "specialist_name": "L3_Diabetic",
        "description":     "Diabetic Macular Edema (DME) vs Diabetic Retinopathy (DR)",
        "classes":         {0: "DME", 1: "DR"},
    },
    "Vascular": {
        "num_classes":     3,
        "specialist_name": "L3_Vascular",
        "description":     "Macular Hole (MH) vs RVO vs RAO (re-separated from L2 aggregate)",
        "classes":         {0: "MH", 1: "RVO", 2: "RAO"},
    },
    "Fluid": {
        "num_classes":     1,
        "specialist_name": "L3_Fluid",
        "description":     "Central Serous Retinopathy — single-class anomaly detection",
        "classes":         {0: "CSR"},
    },
    "Structural": {
        "num_classes":     2,
        "specialist_name": "L3_Structural",
        "description":     "Epiretinal Membrane (ERM) vs Vitreomacular Interface Disease (VID)",
        "classes":         {0: "ERM", 1: "VID"},
    },
}


class SpecialistModel(nn.Module):
    """
    EfficientNet-B0 fine-grained classifier for Level 3 specialist tasks.

    Operates on 384×384 input for maximum structural resolution.

    Architecture:
        EfficientNet-B0 features (1280-d after avgpool)
        → Dropout(dropout_rate)
        → Linear(1280, 512) + SiLU (Swish — native EfficientNet activation)
        → Dropout(dropout_rate / 2)
        → Linear(512, num_classes)

    Args:
        num_classes:     Number of fine-grained classes for this specialist.
        specialist_name: Human-readable name for logging.
        dropout_rate:    Dropout probability in classifier head.
        pretrained:      Load IMAGENET1K_V1 weights if True.
        freeze_backbone: Start with backbone frozen.
    """

    def __init__(
        self,
        num_classes: int,
        specialist_name: str = "Specialist",
        dropout_rate: float = 0.4,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.specialist_name = specialist_name

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features    # MBConv blocks
        self.avgpool  = backbone.avgpool     # AdaptiveAvgPool2d(1, 1)

        in_features = backbone.classifier[-1].in_features  # 1280 for B0

        # Richer head than the router — specialists need more discriminative power
        # for subtle inter-class differences (e.g., CNV vs DRUSEN fluid patterns)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),    # Swish — consistent with EfficientNet internals
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        logger.info(
            "%s ready | backbone=EfficientNet-B0 | in_features=%d | "
            "num_classes=%d | input=384×384 | frozen=%s",
            specialist_name, in_features, num_classes, freeze_backbone,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Freeze / Unfreeze API
    # ──────────────────────────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info("%s: backbone UNFROZEN.", self.specialist_name)

    def get_param_groups(
        self,
        backbone_lr: float = 5e-5,
        head_lr: float = 5e-4,
    ) -> List[Dict]:
        """Differential LR groups for Phase 2 fine-tuning."""
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
            x: Float tensor, shape ``(B, 3, 384, 384)``.

        Returns:
            Logits tensor, shape ``(B, num_classes)``.
        """
        x = self.features(x)       # (B, 1280, H', W')
        x = self.avgpool(x)        # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)    # (B, 1280)
        x = self.classifier(x)     # (B, num_classes)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_specialist(
    specialist_key: str,
    dropout_rate: float = 0.4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> SpecialistModel:
    """
    Factory function for Level 3 specialist models.

    Args:
        specialist_key:  One of 'Macular', 'Diabetic', 'Vascular',
                         'Fluid', 'Structural'.
        dropout_rate:    Head dropout probability.
        pretrained:      Use ImageNet pretrained weights.
        freeze_backbone: Start with frozen backbone (Phase 1 warm-up).

    Returns:
        Configured :class:`SpecialistModel` instance.

    Raises:
        ValueError: If specialist_key is not in SPECIALIST_CONFIGS.

    Example::

        model = build_specialist('Macular')
        # → L3_Macular: 3 classes (CNV / DRUSEN / Generic_AMD), 384×384 input
    """
    if specialist_key not in SPECIALIST_CONFIGS:
        raise ValueError(
            f"Unknown specialist: '{specialist_key}'. "
            f"Valid keys: {list(SPECIALIST_CONFIGS.keys())}"
        )

    cfg = SPECIALIST_CONFIGS[specialist_key]
    logger.info(
        "Building specialist [%s]: %s",
        specialist_key, cfg["description"],
    )

    return SpecialistModel(
        num_classes=cfg["num_classes"],
        specialist_name=cfg["specialist_name"],
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
