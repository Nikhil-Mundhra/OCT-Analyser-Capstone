"""
models/level1_gatekeeper.py

Level 1 Binary Gatekeeper — ResNet-50 pretrained on ImageNet.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class GatekeeperModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Load ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Extract features (everything except avgpool and fc)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = 2048

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

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def build_gatekeeper(
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> GatekeeperModel:
    return GatekeeperModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
