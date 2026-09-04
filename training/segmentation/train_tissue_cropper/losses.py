"""
training/segmentation/train_tissue_cropper/losses.py

Loss functions and evaluation metrics for Attention U-Net Binary Tissue Segmentation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class / Binary Soft Dice Loss with Laplace smoothing."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dice_per_class = []
        # Focus especially on foreground tissue (class 1)
        for c in range(num_classes):
            p_flat = probs[:, c].contiguous().view(-1)
            t_flat = targets_one_hot[:, c].contiguous().view(-1)

            intersection = (p_flat * t_flat).sum()
            union = p_flat.sum() + t_flat.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)

        dice_tensor = torch.stack(dice_per_class)
        return 1.0 - dice_tensor.mean()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for hard boundary transition mining."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CombinedTissueLoss(nn.Module):
    """Hybrid loss combining Cross Entropy, Soft Dice, and Focal Loss."""
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, focal_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets)

        total_loss = (self.ce_weight * loss_ce) + (self.dice_weight * loss_dice) + (self.focal_weight * loss_focal)
        return total_loss


def compute_dice_and_iou(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Computes Mean Dice Score and Intersection over Union (IoU) for foreground tissue."""
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1) # (B, H, W)
        pred_fg = (preds == 1).float().view(-1)
        target_fg = (targets == 1).float().view(-1)

        intersection = (pred_fg * target_fg).sum().item()
        union = pred_fg.sum().item() + target_fg.sum().item()

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)

        return dice, iou
