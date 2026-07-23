import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

OCT5K_DETECTION_CLASSES = [
    "Background",
    "Fluid",
    "Hyperfluorescent_spots",
    "Hard_drusen",
    "Soft_drusen",
    "Soft_drusen_PED",
    "Choroidal_folds",
    "Geographic_atrophy",
    "PR_layer_disruption",
    "Reticular_drusen"
]

class OCTPathologyDetector(nn.Module):
    """
    Model 5: Object Detector for 9 Pathological Biomarker Bounding Boxes.
    Built on Faster R-CNN with a ResNet50 backbone.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.backbone.out_channels = 2048

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pooler=roi_pooler
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)
