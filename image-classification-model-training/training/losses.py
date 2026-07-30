"""
training/losses.py

Loss functions and class-weight utilities for the hierarchical OCT pipeline.

Primary Loss: FocalLoss
  FocalLoss (Lin et al., 2017) addresses class imbalance by down-weighting
  easy-to-classify dominant-class examples (CNV, DRUSEN, DME) so that
  gradient updates are dominated by hard minority examples (RAO, CSR, VID).

  FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

  At L2: Vascular_Occlusions has 225 samples vs Macular_Degeneration's 47,107.
  Without focal loss, the router would learn to predict Macular_Degeneration
  for nearly everything and achieve ~79% accuracy without learning anything
  clinically useful.

Label Smoothing:
  Applied at L2 (ε=0.1) to account for annotation uncertainty when merging
  three datasets with different labelling protocols. Not applied at L1 (binary
  classification is unambiguous) or at L3 (specialist labels are more precise).

Usage::

    # L1 binary gatekeeper
    criterion = build_criterion('level1', class_weights=dataset.class_weights)

    # L2 router with label smoothing
    criterion = build_criterion('level2', class_weights=dataset.class_weights)

    # L3 specialist (e.g., Vascular)
    criterion = build_criterion('level3_vascular', class_weights=dataset.class_weights)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional alpha weighting and label smoothing.

    Combines three imbalance-mitigation strategies into one loss:
      1. Alpha weighting:     per-class weights dampen dominant class gradients.
      2. Focal modulation:    (1-p_t)^γ concentrates learning on hard examples.
      3. Label smoothing:     prevents over-confident predictions on noisy labels.

    Args:
        gamma:           Focusing parameter. γ=0 → standard CE. γ=2 recommended.
        alpha:           Per-class weight tensor, shape ``[num_classes]``.
                         If None, all classes are weighted equally.
        reduction:       ``'mean'`` (default) or ``'sum'``.
        label_smoothing: Smoothing ε. 0.0 disables it.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.reduction       = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index    = ignore_index

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs:  Raw logits, shape ``(B, C)``.
            targets: Integer class indices, shape ``(B,)``.

        Returns:
            Scalar loss.
        """
        if inputs.ndim > 2:
            inputs = inputs.view(-1, inputs.size(-1))
        if targets.ndim > 1:
            targets = targets.view(-1)

        # Force FP32 to prevent NaN from FP16 exp/log overflow on MPS/CUDA
        inputs = inputs.float()

        num_classes = inputs.size(-1)

        # Mask out targets that are ignored (-1) or out of bounds to prevent CUDA illegal memory access
        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < num_classes)

        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        targets_clamped = targets.clamp(0, num_classes - 1)

        # ── Compute per-sample cross-entropy (with optional label smoothing) ─
        if self.label_smoothing > 0.0 and num_classes > 1:
            eps = self.label_smoothing
            with torch.no_grad():
                # Build soft target distribution
                smooth_targets = torch.full_like(inputs, eps / (num_classes - 1))
                smooth_targets.scatter_(1, targets_clamped.unsqueeze(1), 1.0 - eps)
            log_prob = F.log_softmax(inputs, dim=1)
            # KL-divergence-style CE with soft labels
            ce_loss = -(smooth_targets * log_prob).sum(dim=1)
        else:
            log_prob = F.log_softmax(inputs, dim=1)
            ce_loss  = F.nll_loss(log_prob, targets_clamped, reduction="none")

        # ── Focal modulation: (1 - p_t)^gamma ────────────────────────────────
        prob  = F.softmax(inputs, dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
        p_t   = prob.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)
        focal = (1.0 - p_t) ** self.gamma

        # ── Alpha class weighting ─────────────────────────────────────────────
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets_clamped)
            loss    = alpha_t * focal * ce_loss
        else:
            loss    = focal * ce_loss

        # Mask out ignored target indices
        loss = loss * valid_mask.float()

        # ── Reduction ─────────────────────────────────────────────────────────
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1)
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Standard cross-entropy with label smoothing and optional class weights.

    A lighter alternative to FocalLoss when imbalance is moderate and you
    primarily want to guard against label noise. Not used by default in this
    pipeline but available as a swap-in.

    Args:
        smoothing: Label smoothing ε.
        weight:    Optional per-class weight tensor.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.smoothing    = smoothing
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(1)
        log_prob  = F.log_softmax(inputs, dim=1)

        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < n_classes)
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        targets_clamped = targets.clamp(0, n_classes - 1)

        with torch.no_grad():
            smooth = torch.full_like(log_prob, self.smoothing / (n_classes - 1))
            smooth.scatter_(1, targets_clamped.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth * log_prob).sum(dim=1)

        if self.weight is not None:
            w    = self.weight.to(inputs.device)
            loss = loss * w.gather(0, targets_clamped)

        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.sum().clamp(min=1)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_criterion(
    mode: str,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """
    Build the appropriate loss function for a given pipeline mode.

    Strategy:
      level1:  FocalLoss, no label smoothing (binary, clean labels).
      level2:  FocalLoss + label smoothing ε=0.1 (multi-source label noise).
      level3_*: FocalLoss + label smoothing ε=0.1 (fine-grained, noisy minority).

    Args:
        mode:            Dataset mode string.
        class_weights:   Per-class weight tensor from OCTHierarchicalDataset.
        focal_gamma:     Focal loss γ. 2.0 is the standard recommendation.
        label_smoothing: ε for label smoothing. Applied at L2 and L3 only.

    Returns:
        Configured :class:`FocalLoss` module.
    """
    # Apply smoothing only for multi-class levels (L2 and L3)
    smoothing = label_smoothing if mode != "level1" else 0.0

    logger.info(
        "Criterion: FocalLoss | mode=%s | gamma=%.1f | smoothing=%.2f | "
        "class_weights=%s",
        mode, focal_gamma, smoothing,
        None if class_weights is None else class_weights.tolist(),
    )

    return FocalLoss(
        gamma=focal_gamma,
        alpha=class_weights,
        reduction="mean",
        label_smoothing=smoothing,
    )
