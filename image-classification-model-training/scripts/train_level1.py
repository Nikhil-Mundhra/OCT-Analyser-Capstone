"""
scripts/train_level1.py

Level 1 Gatekeeper Training — Stratified 5-Fold Cross-Validation.

Model:       ResNet-50 (IMAGENET1K_V2 weights)
Task:        NORMAL (0) vs ABNORMAL (1)
Input:       224×224 RGB (batch_size=64 for 32 GB MPS unified memory)
Loss:        FocalLoss (γ=2, α=class_weights) — no label smoothing (binary)
Optimiser:   AdamW with differential LRs (backbone vs head)
LR Schedule: CosineAnnealingWarmRestarts (T_0=20, T_mult=2)
Validation:  Stratified 5-fold CV — macro F1 is primary selection criterion

Two-Phase Training per Fold:
  Phase 1 — Warm-up      (5 epochs, backbone frozen, LR=1e-3)
  Phase 2 — Fine-tuning  (up to 50 epochs, full network, early stop patience=10)
             backbone_lr=1e-4  |  head_lr=1e-3

Usage:
    # Full 5-fold training
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/train_level1.py

    # Smoke test — 1 fold, 1 epoch each phase, batch_size=8
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/train_level1.py --smoke-test

    # Custom run
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/train_level1.py \\
        --config config/hierarchy.yaml \\
        --n-folds 5 \\
        --warmup-epochs 5 \\
        --epochs 50 \\
        --batch-size 64 \\
        --patience 10

Success Criteria:
    val_accuracy  ≥ 0.92
    val_auroc     ≥ 0.96
    val_macro_f1  ≥ 0.90
"""

import argparse
import json
import logging
import os

# -----------------------------------------------------------------------------
# APPLE SILICON / MAC OS FIXES
# Prevent fatal segmentation faults inside libomp.dylib (OpenMP) during dataloading
# -----------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path


# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from data.dataset import build_kfold_dataloaders
from data.transforms import get_transforms
from models.level1_gatekeeper import build_gatekeeper
from training.trainer import HierarchyTrainer, get_device
from training.losses import build_criterion


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Level 1 OCT Gatekeeper (NORMAL vs ABNORMAL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--config",         default="config/hierarchy.yaml",
                   help="Path to hierarchy.yaml")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Root directory for model checkpoints")
    p.add_argument("--log-dir",        default="logs",
                   help="Root directory for TensorBoard logs")

    # Cross-validation
    p.add_argument("--n-folds",        type=int,   default=5,
                   help="Number of stratified k-folds")
    p.add_argument("--seed",           type=int,   default=42,
                   help="Random seed for reproducibility")

    # Phase 1 — Warm-up
    p.add_argument("--warmup-epochs",  type=int,   default=5,
                   help="Warm-up epochs (backbone frozen)")
    p.add_argument("--warmup-lr",      type=float, default=1e-3,
                   help="LR for warm-up phase (head only)")

    # Phase 2 — Fine-tuning
    p.add_argument("--epochs",         type=int,   default=50,
                   help="Max fine-tuning epochs per fold")
    p.add_argument("--backbone-lr",    type=float, default=1e-4,
                   help="Backbone LR for fine-tuning phase")
    p.add_argument("--head-lr",        type=float, default=1e-3,
                   help="Classifier head LR for fine-tuning phase")
    p.add_argument("--patience",       type=int,   default=10,
                   help="Early stopping patience (epochs)")

    # Optimiser
    p.add_argument("--weight-decay",   type=float, default=1e-4,
                   help="AdamW L2 weight decay")
    p.add_argument("--resume-fold",    type=int, default=0,
                   help="Fold index to resume from (0-indexed). Skips prior folds.")

    # Data loading
    p.add_argument("--batch-size",     type=int,   default=48,
                   help="Training batch size (48 for M2 Pro 24GB | 64 for M2 Max/Ultra 32GB+)")
    p.add_argument("--num-workers",    type=int,   default=4,
                   help="DataLoader workers (4 recommended for M2 Pro 6P+4E cores)")

    # Loss
    p.add_argument("--focal-gamma",    type=float, default=2.0,
                   help="Focal loss gamma parameter")

    # Smoke test override
    p.add_argument("--smoke-test",     action="store_true",
                   help="1 fold, 1 epoch each phase, batch_size=8")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, smoke_test: bool) -> None:
    """Configure root logger to write to stdout and a log file."""
    log_path = Path(log_dir) / "level1"
    log_path.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    if not smoke_test:
        handlers.append(
            logging.FileHandler(log_path / "training.log", mode="a")
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cross-fold summary helpers
# ──────────────────────────────────────────────────────────────────────────────

_KEY_METRICS = [
    "val_accuracy",
    "val_auroc",
    "val_macro_f1",
    "val_weighted_f1",
]

_SUCCESS_CRITERIA = {
    "val_accuracy":  0.92,
    "val_auroc":     0.96,
    "val_macro_f1":  0.90,
}


def summarise_folds(
    all_fold_metrics: list,
    logger: logging.Logger,
) -> dict:
    """Compute mean ± std across folds and check success criteria."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  CROSS-FOLD SUMMARY  (Level 1 — NORMAL vs ABNORMAL)")
    logger.info("=" * 60)

    summary = {}

    for metric in _KEY_METRICS:
        values = [m.get(metric, float("nan")) for m in all_fold_metrics]
        values = [v for v in values if not np.isnan(v)]
        if not values:
            continue
        mean = float(np.mean(values))
        std  = float(np.std(values))
        summary[metric] = {"mean": mean, "std": std, "values": values}
        logger.info("  %-20s  %.4f ± %.4f", metric, mean, std)

    logger.info("")
    logger.info("─── Target Verification ───")
    all_pass = True
    for metric, target in _SUCCESS_CRITERIA.items():
        if metric not in summary:
            continue
        achieved = summary[metric]["mean"]
        passed   = achieved >= target
        all_pass = all_pass and passed
        status   = "✅ PASS" if passed else "❌ BELOW TARGET"
        logger.info(
            "  %-20s  %.4f  (target ≥ %.2f)  %s",
            metric, achieved, target, status,
        )

    if all_pass:
        logger.info("")
        logger.info("  🎉  All targets met — Level 1 Gatekeeper ready for deployment.")
    else:
        logger.warning(
            "  ⚠  Some targets not met — consider longer training or tuning "
            "hyperparameters before promoting to Level 2."
        )

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Smoke-test overrides ──────────────────────────────────────────────────
    if args.smoke_test:
        args.n_folds       = 2   # StratifiedKFold requires n_splits >= 2; only fold 0 will run
        args.warmup_epochs = 1
        args.epochs        = 1
        args.batch_size    = 8
        args.num_workers   = 0
        args.patience      = 9999

    setup_logging(args.log_dir, args.smoke_test)
    logger = logging.getLogger(__name__)

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Banner ────────────────────────────────────────────────────────────────
    device = get_device()
    logger.info("")
    logger.info("=" * 60)
    logger.info("  OCT Gatekeeper — Level 1 Training")
    if args.smoke_test:
        logger.info("  🔥 SMOKE TEST MODE")
    logger.info("  Config     : %s", args.config)
    logger.info("  Device     : %s", device)
    logger.info("  Folds      : %d", args.n_folds)
    logger.info("  Batch size : %d", args.batch_size)
    logger.info(
        "  Epochs     : %d warmup + %d finetune (patience=%d)",
        args.warmup_epochs, args.epochs, args.patience,
    )
    logger.info("  Seed       : %d", args.seed)
    logger.info("=" * 60)

    MODE = "level1"

    # ── Build k-fold DataLoaders ──────────────────────────────────────────────
    logger.info("\nBuilding stratified %d-fold DataLoaders...", args.n_folds)

    fold_loaders = build_kfold_dataloaders(
        config_path=args.config,
        mode=MODE,
        n_splits=args.n_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=get_transforms(MODE, "train"),
        val_transform=get_transforms(MODE, "val"),
        use_weighted_sampler=True,
        seed=args.seed,
    )

    # Retrieve dataset metadata from the first fold
    first_train_ds = fold_loaders[0][0].dataset
    class_names    = first_train_ds.class_names
    num_classes    = first_train_ds.num_classes
    class_weights  = first_train_ds.class_weights

    logger.info("Class names   : %s", class_names)
    logger.info(
        "Class weights : %s  (used as FocalLoss alpha)",
        [f"{w:.4f}" for w in class_weights.tolist()],
    )

    # ── K-Fold Cross-Validation Loop ──────────────────────────────────────────
    all_fold_metrics = []

    for fold_id, (train_loader, val_loader) in enumerate(fold_loaders):
        if fold_id < args.resume_fold:
            logger.info("Skipping FOLD %d (resuming from %d)", fold_id + 1, args.resume_fold + 1)
            continue
            
        logger.info("")
        logger.info("###  FOLD %d / %d  ###", fold_id + 1, args.n_folds)
        logger.info(
            "     train batches: %d | val batches: %d",
            len(train_loader), len(val_loader),
        )

        # Fresh model, criterion, and trainer for every fold
        model = build_gatekeeper(
            num_classes=num_classes,
            dropout_rate=0.3,
            pretrained=True,
            freeze_backbone=True,   # Phase 1 starts frozen
        )

        criterion = build_criterion(
            mode=MODE,
            class_weights=class_weights,
            focal_gamma=args.focal_gamma,
            label_smoothing=0.0,    # Binary — no smoothing needed
        )

        trainer = HierarchyTrainer(
            model=model,
            criterion=criterion,
            mode=MODE,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            device=device,
        )

        fold_metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            finetune_epochs=args.epochs,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            fold_id=fold_id,
        )

        all_fold_metrics.append(fold_metrics)
        trainer.close()   # Flush TensorBoard writer

        logger.info(
            "Fold %d complete — best epoch=%s, phase=%s",
            fold_id + 1,
            fold_metrics.get("epoch", "N/A"),
            fold_metrics.get("phase", "N/A"),
        )

        # In smoke-test mode, we only run fold 0 (n_folds was set to 2 to satisfy
        # StratifiedKFold's minimum requirement, but we don't need the second fold)
        if args.smoke_test:
            logger.info("Smoke test: exiting after fold 0.")
            break

    # ── Cross-fold summary & target check ─────────────────────────────────────
    summary = summarise_folds(all_fold_metrics, logger)

    # Save JSON summary for downstream inspection
    summary_path = Path(args.checkpoint_dir) / MODE / "cv_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serialisable types before dumping
    clean_summary = {
        k: {sk: sv for sk, sv in v.items() if sk != "values"}
        for k, v in summary.items()
    }
    with open(summary_path, "w") as f:
        json.dump(clean_summary, f, indent=2)

    logger.info("")
    logger.info("CV summary saved → %s", summary_path)
    logger.info(
        "TensorBoard logs  → tensorboard --logdir %s",
        Path(args.log_dir) / MODE,
    )
    logger.info(
        "Best checkpoints  → %s/fold*_best_model.pth",
        Path(args.checkpoint_dir) / MODE,
    )


if __name__ == "__main__":
    main()
