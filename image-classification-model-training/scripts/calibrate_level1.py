"""
scripts/calibrate_level1.py

Post-training calibration for Level 1 Gatekeeper.

Runs two calibration steps on a held-out validation fold:

  1. Threshold Calibration (ROC-derived)
     ─────────────────────────────────────
     Finds the optimal decision threshold from the validation fold ROC curve
     using one of two strategies:

       - youden: Maximises Sensitivity + Specificity - 1 (Youden's Index).
         Gives the geometrically optimal operating point.

       - sensitivity_constrained: Finds the highest-specificity threshold that
         keeps Sensitivity >= a target (default 0.95). Clinically appropriate
         for a binary screener where missing a diseased patient (False Negative)
         is the critical failure mode.

     Replaces the hardcoded 0.35 threshold in the test script.

  2. Temperature Scaling
     ─────────────────────
     Fits a single scalar temperature T on the validation set so that
     softmax(logits / T) produces well-calibrated probabilities. Raw softmax
     from deep networks is systematically overconfident; T > 1.0 softens
     predictions toward calibrated uncertainty.

     Saves a reliability diagram (calibration plot) before and after scaling.

Outputs:
    checkpoints/level1/calibration.json  — threshold, temperature, all metrics
    checkpoints/level1/roc_curve.png     — ROC curve with selected threshold
    checkpoints/level1/calibration_before.png — reliability diagram (raw)
    checkpoints/level1/calibration_after.png  — reliability diagram (T-scaled)

Usage:
    python3 scripts/calibrate_level1.py \\
        --checkpoint checkpoints/level1/fold0_best_model.pth \\
        --strategy sensitivity_constrained \\
        --sensitivity-target 0.95

    # Youden's Index instead:
    python3 scripts/calibrate_level1.py \\
        --checkpoint checkpoints/level1/fold0_best_model.pth \\
        --strategy youden
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from data.dataset import build_kfold_dataloaders
from data.transforms import get_transforms
from models.level1_gatekeeper import build_gatekeeper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Threshold selection strategies
# ──────────────────────────────────────────────────────────────────────────────

def find_youden_threshold(fpr, tpr, thresholds):
    """
    Youden's Index: choose the threshold that maximises TPR - FPR.

    Equivalent to maximising Sensitivity + Specificity - 1.
    Gives the geometrically optimal operating point on the ROC curve.

    Returns:
        (threshold, sensitivity, specificity)
    """
    youden_idx = np.argmax(tpr - fpr)
    return (
        float(thresholds[youden_idx]),
        float(tpr[youden_idx]),
        float(1.0 - fpr[youden_idx]),
    )


def find_sensitivity_constrained_threshold(fpr, tpr, thresholds, min_sensitivity=0.95):
    """
    Sensitivity-constrained strategy: find the highest-specificity threshold
    that keeps sensitivity >= min_sensitivity.

    Clinically appropriate for a screening task where False Negatives (missed
    disease) are far more costly than False Positives (unnecessary referrals).

    Falls back to Youden's Index if no threshold achieves the target.

    Returns:
        (threshold, sensitivity, specificity)
    """
    valid = np.where(tpr >= min_sensitivity)[0]
    if len(valid) == 0:
        logger.warning(
            "No threshold achieves sensitivity >= %.2f. "
            "Falling back to Youden's Index.",
            min_sensitivity,
        )
        return find_youden_threshold(fpr, tpr, thresholds)

    # Among valid thresholds, pick the one with lowest FPR (= highest specificity)
    best = valid[np.argmin(fpr[valid])]
    return (
        float(thresholds[best]),
        float(tpr[best]),
        float(1.0 - fpr[best]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Temperature Scaling
# ──────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for probability calibration.

    Wraps a trained model and divides all logits by a learnable scalar T
    before softmax. T is fitted by minimising NLL on a held-out calibration set.

    Interpretation:
        T > 1.0  — original model was overconfident (most common). Scaling
                   softens probabilities toward more calibrated uncertainty.
        T < 1.0  — original model was underconfident. Scaling sharpens predictions.
        T = 1.0  — no calibration needed.

    Reference:
        Guo et al. (2017), "On Calibration of Modern Neural Networks", ICML.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model       = model
        # Initialise at 1.5 — most trained CNNs are overconfident
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature

    def fit(self, cal_loader, device, max_iter: int = 100) -> float:
        """
        Fit temperature on a calibration DataLoader using L-BFGS.

        Collects all logits in a single pass (no temperature), then optimises
        T to minimise NLL on the collected logit/label pairs.

        Args:
            cal_loader: DataLoader of (images, labels) calibration data.
            device:     Compute device.
            max_iter:   L-BFGS maximum number of iterations.

        Returns:
            The fitted temperature value (float).
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()

        # Collect all logits without temperature scaling
        all_logits, all_labels = [], []
        with torch.no_grad():
            for images, labels in cal_loader:
                images = images.to(device)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # L-BFGS minimises NLL with respect to temperature only
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=max_iter,
        )

        def eval_closure():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature
            loss   = nll_criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        # Guard against degenerate values
        with torch.no_grad():
            self.temperature.clamp_(min=0.05)

        fitted_T = self.temperature.item()
        logger.info("Temperature fitted: T = %.4f", fitted_T)
        return fitted_T


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def collect_predictions(model, loader, device):
    """
    Run inference and return (y_true, y_prob_abnormal).

    Args:
        model:  Trained model in eval mode.
        loader: DataLoader of (images, labels).
        device: Compute device.

    Returns:
        Tuple of (all_labels np.ndarray, all_probs_abnormal np.ndarray).
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_with_threshold(
    fpr, tpr, auc,
    threshold, sensitivity, specificity,
    out_path: Path,
) -> None:
    """Save ROC curve PNG with the selected operating point marked."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#BDBDBD", linestyle="--", lw=1,
            label="Chance")

    # Mark the selected threshold
    ax.scatter(
        [1.0 - specificity], [sensitivity],
        color="#F44336", zorder=5, s=120,
        label=(
            f"Selected point\n"
            f"Threshold={threshold:.3f}\n"
            f"Sensitivity={sensitivity:.3f}   Specificity={specificity:.3f}"
        ),
    )

    ax.set_xlabel("False Positive Rate  (1 − Specificity)")
    ax.set_ylabel("True Positive Rate  (Sensitivity)")
    ax.set_title("Level 1 Gatekeeper — ROC Curve & Calibrated Threshold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("ROC plot saved → %s", out_path)


def plot_calibration(
    model, cal_loader, device,
    out_path: Path,
    label: str = "Model",
    n_bins: int = 10,
) -> float:
    """
    Generate a reliability diagram (calibration plot) and compute ECE.

    A perfectly calibrated model follows the diagonal y = x.
    Points below the diagonal → overconfident.
    Points above the diagonal → underconfident.

    Args:
        model:      Model (or TemperatureScaler wrapping a model) in eval mode.
        cal_loader: DataLoader of (images, labels).
        device:     Compute device.
        out_path:   Path to save the PNG.
        label:      Legend label for the model curve.
        n_bins:     Number of probability bins.

    Returns:
        Expected Calibration Error (ECE) as a float.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in cal_loader:
            logits = model(images.to(device))
            probs  = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (all_probs >= lo) & (all_probs < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(all_labels[mask].mean())
        bin_confs.append(all_probs[mask].mean())
        bin_counts.append(mask.sum())

    ece = (
        sum(cnt * abs(acc - conf) for acc, conf, cnt in zip(bin_accs, bin_confs, bin_counts))
        / len(all_labels)
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#BDBDBD",
            label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=(1.0 / n_bins) * 0.8,
           alpha=0.65, color="#2196F3", label=f"{label}  (ECE = {ece:.4f})")
    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Fraction of Positives (Actual Probability)")
    ax.set_title("Reliability Diagram — Level 1 Gatekeeper")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Calibration plot saved → %s  (ECE=%.4f)", out_path, ece)
    return ece


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Level 1 Gatekeeper: ROC-derived threshold + temperature scaling"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/level1/fold0_best_model.pth",
        help="Path to trained Level 1 checkpoint (.pth)",
    )
    parser.add_argument(
        "--config", default="config/hierarchy.yaml",
        help="Path to hierarchy.yaml",
    )
    parser.add_argument(
        "--strategy",
        choices=["youden", "sensitivity_constrained"],
        default="sensitivity_constrained",
        help=(
            "Threshold selection strategy. "
            "'youden' maximises Sensitivity + Specificity. "
            "'sensitivity_constrained' maximises Specificity subject to "
            "Sensitivity >= --sensitivity-target."
        ),
    )
    parser.add_argument(
        "--sensitivity-target", type=float, default=0.95,
        help="Minimum acceptable sensitivity (for sensitivity_constrained strategy)",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Which CV fold to use as the calibration set (0-indexed)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/level1",
        help="Directory to save calibration.json and plots",
    )
    args = parser.parse_args()

    # Calibration runs on CPU — the dataset is small and this avoids MPS quirks
    device  = torch.device("cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model = build_gatekeeper(num_classes=2, pretrained=False)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ── Build calibration DataLoader (validation fold from CV) ────────────────
    logger.info("Building calibration DataLoader (fold %d val split)...", args.fold)
    fold_loaders = build_kfold_dataloaders(
        config_path=args.config,
        mode="level1",
        n_splits=5,
        batch_size=args.batch_size,
        num_workers=0,         # CPU calibration — keep simple
        train_transform=get_transforms("level1", "train"),
        val_transform=get_transforms("level1", "val"),
        use_weighted_sampler=True,
        seed=42,
    )
    _, cal_loader = fold_loaders[args.fold]
    logger.info("Calibration set size: %d samples", len(cal_loader.dataset))

    # ── Step 1: ROC Threshold Calibration ────────────────────────────────────
    logger.info("Running inference for ROC calibration...")
    y_true, y_prob = collect_predictions(model, cal_loader, device)
    auc = roc_auc_score(y_true, y_prob)
    logger.info("Calibration AUROC: %.4f", auc)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if args.strategy == "youden":
        threshold, sensitivity, specificity = find_youden_threshold(
            fpr, tpr, thresholds
        )
    else:
        threshold, sensitivity, specificity = find_sensitivity_constrained_threshold(
            fpr, tpr, thresholds,
            min_sensitivity=args.sensitivity_target,
        )

    logger.info("=" * 55)
    logger.info("  THRESHOLD CALIBRATION RESULT")
    logger.info("=" * 55)
    logger.info("  Strategy      : %s", args.strategy)
    logger.info("  Threshold     : %.4f  (was hardcoded 0.35)", threshold)
    logger.info("  Sensitivity   : %.4f", sensitivity)
    logger.info("  Specificity   : %.4f", specificity)
    logger.info("  AUROC         : %.4f", auc)
    logger.info("=" * 55)

    plot_roc_with_threshold(
        fpr, tpr, auc, threshold, sensitivity, specificity,
        out_path=out_dir / "roc_curve.png",
    )

    # ── Step 2: Temperature Scaling ───────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("  TEMPERATURE SCALING CALIBRATION")
    logger.info("=" * 55)

    # Reliability diagram BEFORE temperature scaling
    ece_before = plot_calibration(
        model, cal_loader, device,
        out_path=out_dir / "calibration_before.png",
        label="Raw model",
    )

    # Fit temperature
    scaler = TemperatureScaler(model)
    temperature = scaler.fit(cal_loader, device)

    logger.info("  Temperature   : %.4f", temperature)
    if temperature > 1.0:
        logger.info("  Interpretation: Model was overconfident — scaling softens predictions.")
    elif temperature < 1.0:
        logger.info("  Interpretation: Model was underconfident — scaling sharpens predictions.")
    else:
        logger.info("  Interpretation: No calibration needed (T ≈ 1.0).")

    # Reliability diagram AFTER temperature scaling
    ece_after = plot_calibration(
        scaler, cal_loader, device,
        out_path=out_dir / "calibration_after.png",
        label="Temperature-scaled",
    )
    logger.info("  ECE before    : %.4f", ece_before)
    logger.info("  ECE after     : %.4f", ece_after)
    logger.info("=" * 55)

    # ── Save calibration.json ─────────────────────────────────────────────────
    cal_result = {
        "threshold":          threshold,
        "sensitivity":        sensitivity,
        "specificity":        specificity,
        "auroc":              auc,
        "strategy":           args.strategy,
        "sensitivity_target": args.sensitivity_target,
        "temperature":        temperature,
        "ece_before":         ece_before,
        "ece_after":          ece_after,
        "checkpoint":         str(args.checkpoint),
        "calibration_fold":   args.fold,
    }
    cal_path = out_dir / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(cal_result, f, indent=2)
    logger.info("")
    logger.info("Calibration saved → %s", cal_path)
    logger.info("ROC plot        → %s", out_dir / "roc_curve.png")
    logger.info("Cal plot before → %s", out_dir / "calibration_before.png")
    logger.info("Cal plot after  → %s", out_dir / "calibration_after.png")
    logger.info("")
    logger.info(
        "Next step: run test_level1_on_test_set.py — it will auto-load "
        "threshold=%.4f and temperature=%.4f from calibration.json.",
        threshold, temperature,
    )


if __name__ == "__main__":
    main()
