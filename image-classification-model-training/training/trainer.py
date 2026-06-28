"""
training/trainer.py

HierarchyTrainer — Reusable training engine for all pipeline levels.

Design Principles:
  1. Device-agnostic: Auto-selects MPS > CUDA > CPU.
  2. Two-phase protocol: warm-up (backbone frozen) → fine-tuning (unfrozen).
  3. Differential LRs: backbone and head trained at different learning rates.
  4. Early stopping: patience-based monitoring of validation loss.
  5. Gradient clipping (max_norm=1.0) for training stability.
  6. CosineAnnealingWarmRestarts: prevents LR from decaying to near-zero,
     allowing periodic exploration of the loss landscape.
  7. TensorBoard logging: loss, accuracy, AUROC, LR, per-class F1 per epoch.
  8. Checkpoint saving: best_model.pth (highest val macro F1) + last_model.pth.

Two-Phase Protocol
──────────────────
Phase 1 — Warm-up  (backbone frozen, high LR on head only)
  Rationale: The classifier head is randomly initialised. If we immediately
  fine-tune the backbone, large gradients from the random head corrupt the
  pretrained ImageNet features. Phase 1 'warms' the head into the OCT domain
  before gradient flow reaches the backbone.

Phase 2 — Fine-tuning  (full network, differential LRs via AdamW)
  The backbone is unfrozen. Differential LRs (backbone_lr << head_lr) ensure
  the pretrained layers are nudged gently while the head adapts aggressively.
  CosineAnnealingWarmRestarts provides periodic LR spikes to escape local minima.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.metrics import MetricAccumulator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Auto-select the best available compute device.

    Priority: MPS (Apple Silicon 32 GB unified memory) > CUDA > CPU.

    Returns:
        :class:`torch.device` instance.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Device selected: MPS (Apple Silicon GPU — 32 GB unified memory)")
        return torch.device("mps")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("Device selected: CUDA (%s)", name)
        return torch.device("cuda")
    logger.warning("No GPU available — falling back to CPU.")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Halts training when the monitored metric stops improving.

    Args:
        patience:  Epochs without improvement before halting.
        min_delta: Minimum absolute change to qualify as an improvement.
        mode:      ``'min'`` (lower is better, e.g., val_loss) or
                   ``'max'`` (higher is better, e.g., val_macro_f1).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_value: Optional[float] = None
        self.counter:    int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """
        Update state with the latest metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter    = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset state — call between folds."""
        self.best_value  = None
        self.counter     = 0
        self.should_stop = False


# ──────────────────────────────────────────────────────────────────────────────
# HierarchyTrainer
# ──────────────────────────────────────────────────────────────────────────────

class HierarchyTrainer:
    """
    Unified training engine for all levels of the OCT classification hierarchy.

    Supports:
      - Two-phase training (warm-up → fine-tuning with backbone unfreeze).
      - Differential learning rates via ``model.get_param_groups()``.
      - Gradient clipping.
      - Early stopping on validation loss.
      - CosineAnnealingWarmRestarts LR scheduling.
      - TensorBoard scalar and confusion-matrix logging.
      - Best-model and last-model checkpoint saving.

    Args:
        model:          PyTorch model implementing ``freeze_backbone()``,
                        ``unfreeze_backbone()``, and ``get_param_groups()``.
        criterion:      Loss function (typically :class:`training.losses.FocalLoss`).
        mode:           Dataset mode string used for checkpoint / log namespacing.
        checkpoint_dir: Root directory for checkpoint subdirectories.
        log_dir:        Root directory for TensorBoard event files.
        device:         Compute device. Auto-selected if None.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        mode: str,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: Optional[torch.device] = None,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model     = model
        self.criterion = criterion
        self.mode      = mode
        self.device    = device or get_device()

        # Ensure output directories exist
        self.ckpt_dir = Path(checkpoint_dir) / mode
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        tb_dir = Path(log_dir) / mode
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        # Move model and loss alpha weights to device
        self.model.to(self.device)
        if hasattr(criterion, "alpha") and criterion.alpha is not None:
            criterion.alpha = criterion.alpha.to(self.device)

        # ── torch.compile() note ─────────────────────────────────────────────
        # torch.compile() is intentionally DISABLED for MPS.
        # The Inductor backend targets CUDA SM architecture ("SMs" = CUDA streaming
        # multiprocessors). On MPS it falls through to a generic path that adds
        # Python tracing overhead WITHOUT any kernel fusion, causing a measured
        # 12× slowdown on M2 Pro (0.267s/batch eager → 3.167s/batch "compiled").
        # When Apple ships a native Metal Inductor backend this can be re-enabled.
        # For CUDA users, add: self.model = torch.compile(self.model, mode='reduce-overhead')

        # ── Mixed precision (float16) ────────────────────────────────────────
        # Apple Silicon has native float16 GPU paths. float16 activations:
        #   - Halve memory bandwidth pressure (critical on 200 GB/s M2 Pro)
        #   - Fit larger batches in 24 GB unified memory
        #   - Speed up matrix multiplications in the GPU's shader cores
        # MPS does NOT support CUDA GradScaler, so gradients stay in float32.
        # Only the forward pass and loss computation run in float16.
        self._amp_enabled = (self.device.type in ("mps", "cuda"))
        self._amp_dtype   = amp_dtype
        if self._amp_enabled:
            logger.info(
                "AMP enabled — forward pass dtype=%s | gradients=float32 | device=%s",
                amp_dtype, self.device,
            )

        logger.info(
            "HierarchyTrainer ready | mode=%s | device=%s | amp=%s | checkpoints=%s",
            mode, self.device, self._amp_enabled, self.ckpt_dir,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Single-epoch steps
    # ──────────────────────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        accumulator: MetricAccumulator,
        max_norm: float = 1.0,
    ) -> float:
        """
        Run one training epoch.

        Args:
            loader:      Training DataLoader (with WeightedRandomSampler).
            optimizer:   Configured AdamW optimiser.
            accumulator: MetricAccumulator instance (reset before calling).
            max_norm:    Gradient clipping threshold.

        Returns:
            Mean batch loss for the epoch.
        """
        self.model.train()
        accumulator.reset()
        total_loss = 0.0
        n_batches  = len(loader)

        # Determine autocast context: mps/cuda use float16; cpu uses bfloat16
        _amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
            if self._amp_enabled
            else torch.autocast(device_type="cpu", enabled=False)
        )

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with _amp_ctx:
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            # NOTE: No GradScaler on MPS (CUDA-only). The loss.backward() call
            # uses float32 gradients even though the forward pass ran in float16.
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            optimizer.step()

            total_loss += loss.item()
            accumulator.update(logits.detach(), labels)

            if (batch_idx + 1) % max(1, n_batches // 5) == 0:
                logger.debug(
                    "  [%d/%d] loss=%.4f", batch_idx + 1, n_batches, loss.item()
                )

        return total_loss / n_batches

    @torch.no_grad()
    def _val_epoch(
        self,
        loader: DataLoader,
        accumulator: MetricAccumulator,
    ) -> float:
        """
        Run one validation epoch (no gradients, no augmentation).

        Returns:
            Mean batch loss for the epoch.
        """
        self.model.eval()
        accumulator.reset()
        total_loss = 0.0

        _amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
            if self._amp_enabled
            else torch.autocast(device_type="cpu", enabled=False)
        )

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with _amp_ctx:
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item()
            accumulator.update(logits.detach(), labels)

        return total_loss / len(loader)

    # ──────────────────────────────────────────────────────────────────────────
    # Main training method — two-phase protocol
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        # ── Phase 1 settings
        warmup_epochs: int = 5,
        warmup_lr: float = 1e-3,
        # ── Phase 2 settings
        finetune_epochs: int = 95,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        # ── Regularisation
        weight_decay: float = 1e-4,
        # ── Early stopping
        patience: int = 10,
        # ── Misc
        fold_id: int = 0,
        max_grad_norm: float = 1.0,
    ) -> Dict:
        """
        Run the full two-phase training for one cross-validation fold.

        Args:
            train_loader:    Training DataLoader (weighted sampler already applied).
            val_loader:      Validation DataLoader (deterministic).
            class_names:     Ordered class name list for metric keys.
            warmup_epochs:   Warm-up epochs with backbone frozen.
            warmup_lr:       Classifier head LR during warm-up.
            finetune_epochs: Max fine-tuning epochs (early stopping may cut short).
            backbone_lr:     Backbone LR for Phase 2 (lower than head).
            head_lr:         Classifier head LR for Phase 2.
            weight_decay:    AdamW L2 regularisation.
            patience:        Early stopping patience in epochs.
            fold_id:         Current fold index (0-indexed) for TensorBoard.
            max_grad_norm:   Gradient clipping threshold.

        Returns:
            Dict of best validation metrics from this fold (accuracy, auroc,
            macro_f1, weighted_f1, per-class F1s, epoch reached, phase).
        """
        # Use a unique global-step offset per fold so TensorBoard curves don't overlap
        global_step = fold_id * 100_000
        best_val_loss   = float("inf")
        best_val_macro_f1 = 0.0
        best_metrics: Dict = {}

        train_acc = MetricAccumulator()
        val_acc   = MetricAccumulator()

        # ── PHASE 1: WARM-UP ──────────────────────────────────────────────────
        self._log_section(f"PHASE 1 — Warm-up | {warmup_epochs} epochs | backbone FROZEN")

        optimizer_warmup = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=warmup_lr,
            weight_decay=weight_decay,
        )

        for epoch in range(warmup_epochs):
            t0         = time.perf_counter()
            train_loss = self._train_epoch(train_loader, optimizer_warmup, train_acc, max_grad_norm)
            val_loss   = self._val_epoch(val_loader, val_acc)
            elapsed    = time.perf_counter() - t0

            train_m = train_acc.compute(class_names, prefix="train_")
            val_m   = val_acc.compute(class_names, prefix="val_")

            global_step += 1
            self._log_epoch(
                epoch=epoch,
                phase="warmup",
                fold_id=fold_id,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_m,
                val_metrics=val_m,
                lr=warmup_lr,
                step=global_step,
                elapsed=elapsed,
            )

            # Track best by val loss during warm-up (macro F1 still unstable)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics  = {**val_m, "epoch": epoch, "phase": "warmup"}
                self._save_checkpoint(f"fold{fold_id}_best_model.pth")

        self._save_checkpoint(f"fold{fold_id}_last_model.pth")

        # ── PHASE 2: FINE-TUNING ──────────────────────────────────────────────
        self._log_section(
            f"PHASE 2 — Fine-tuning | max {finetune_epochs} epochs | "
            f"backbone UNFROZEN | patience={patience}"
        )

        self.model.unfreeze_backbone()

        optimizer_ft = torch.optim.AdamW(
            self.model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr),
            weight_decay=weight_decay,
        )
        # CosineAnnealingWarmRestarts: T_0=20 epochs, doubles after each restart
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_ft,
            T_0=20,
            T_mult=2,
            eta_min=1e-6,
        )
        early_stopper = EarlyStopping(patience=patience, mode="min")

        for epoch in range(finetune_epochs):
            t0         = time.perf_counter()
            train_loss = self._train_epoch(train_loader, optimizer_ft, train_acc, max_grad_norm)
            val_loss   = self._val_epoch(val_loader, val_acc)
            scheduler.step()
            elapsed    = time.perf_counter() - t0

            train_m = train_acc.compute(class_names, prefix="train_")
            val_m   = val_acc.compute(class_names, prefix="val_")

            global_step   += 1
            current_lr     = scheduler.get_last_lr()[0]
            abs_epoch      = warmup_epochs + epoch

            self._log_epoch(
                epoch=abs_epoch,
                phase="finetune",
                fold_id=fold_id,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_m,
                val_metrics=val_m,
                lr=current_lr,
                step=global_step,
                elapsed=elapsed,
            )

            # Model selection: best val loss (most stable signal across folds)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics  = {**val_m, "epoch": abs_epoch, "phase": "finetune"}
                self._save_checkpoint(f"fold{fold_id}_best_model.pth")
                logger.info(
                    "  ✓ New best — val_loss=%.4f | macro_f1=%.4f | auroc=%.4f",
                    val_loss,
                    val_m.get("val_macro_f1", float("nan")),
                    val_m.get("val_auroc",    float("nan")),
                )

            self._save_checkpoint(f"fold{fold_id}_last_model.pth")

            if early_stopper.step(val_loss):
                logger.info(
                    "Early stopping at epoch %d | best_val_loss=%.4f | "
                    "no improvement for %d epochs.",
                    abs_epoch, best_val_loss, patience,
                )
                break

        # Print best classification report for this fold
        if "val_report" in best_metrics:
            logger.info(
                "\nFold %d — Best Classification Report:\n%s",
                fold_id, best_metrics["val_report"],
            )

        return best_metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Logging helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _log_epoch(
        self,
        epoch: int,
        phase: str,
        fold_id: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        step: int,
        elapsed: float,
    ) -> None:
        """Write epoch metrics to TensorBoard and console."""
        tag = f"fold{fold_id}/{phase}"

        # Scalars
        self.writer.add_scalar(f"{tag}/loss/train", train_loss, step)
        self.writer.add_scalar(f"{tag}/loss/val",   val_loss,   step)
        self.writer.add_scalar(f"{tag}/lr",         lr,         step)

        for key, val in {**train_metrics, **val_metrics}.items():
            if isinstance(val, float) and not np.isnan(val):
                self.writer.add_scalar(f"{tag}/{key}", val, step)

        # Console — compact summary line
        logger.info(
            "Ep %3d [%s|fold%d] "
            "loss %.4f/%.4f | acc %.4f | macro_f1 %.4f | auroc %.4f | "
            "lr %.2e | %.1fs",
            epoch, phase, fold_id,
            train_loss, val_loss,
            val_metrics.get("val_accuracy",  float("nan")),
            val_metrics.get("val_macro_f1",  float("nan")),
            val_metrics.get("val_auroc",     float("nan")),
            lr, elapsed,
        )

    def _log_section(self, title: str) -> None:
        border = "=" * 60
        logger.info("\n%s\n  %s\n%s", border, title, border)

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str) -> None:
        path = self.ckpt_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "mode":             self.mode,
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        """Load a saved checkpoint back into the model."""
        path = self.ckpt_dir / filename
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint: %s", path)

    def close(self) -> None:
        """Flush and close TensorBoard writer — call after each fold."""
        self.writer.close()
