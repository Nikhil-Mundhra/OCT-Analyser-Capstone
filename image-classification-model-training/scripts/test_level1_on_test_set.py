"""
scripts/test_level1_on_test_set.py

External test set evaluation for the Level 1 Gatekeeper (Normal vs Abnormal).

Improvements over original:
  - Threshold loaded from calibration.json (ROC-derived) instead of hardcoded 0.35.
  - Temperature scaling applied from calibration.json before softmax.
  - Sensitivity and specificity reported explicitly as primary clinical metrics.
  - Test-Time Augmentation (TTA): averages predictions over 5 augmented views.
  - Grad-CAM visualisations for all four outcome categories (TP, FN, TN, FP).
  - Full confusion matrix breakdown in logs.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Apple Silicon environment guards — must be set before torch import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data.transforms import get_transforms
from models.level1_gatekeeper import build_gatekeeper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class OCTTestDataset(Dataset):
    """
    External test set reader.

    Reads a CSV with columns [Directory, Label, Class] and maps:
        normal → 0  (NORMAL)
        drusen → 1  (ABNORMAL)
        cnv    → 1  (ABNORMAL)

    Args:
        csv_path:  Path to data_information.csv.
        data_dir:  Root directory prepended to each row's Directory column.
        transform: torchvision transform (should be val_transforms — includes CLAHE).
        option:    1 = all images; 2 = only rows where Class == Label.
    """

    def __init__(self, csv_path, data_dir, transform=None, option=1):
        self.data_dir  = Path(data_dir)
        self.transform = transform

        df = pd.read_csv(csv_path)
        if option == 2:
            df = df[df["Class"] == df["Label"]]
        self.manifest  = df.reset_index(drop=True)
        self.label_map = {"normal": 0, "drusen": 1, "cnv": 1}

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row       = self.manifest.iloc[idx]
        img_path  = self.data_dir / row["Directory"]
        label_str = str(row["Label"]).lower().strip()
        label     = self.label_map.get(label_str, -1)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error("Failed to open %s: %s", img_path, exc)
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ──────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA)
# ──────────────────────────────────────────────────────────────────────────────

# Five deterministic views: original + 4 spatial transforms
_TTA_TRANSFORMS = [
    lambda x: x,                                   # (1) Original
    lambda x: torch.flip(x, dims=[-1]),            # (2) Horizontal flip
    lambda x: torch.flip(x, dims=[-2]),            # (3) Vertical flip
    lambda x: torch.rot90(x, k=1, dims=[-2, -1]), # (4) Rotate +90°
    lambda x: torch.rot90(x, k=-1, dims=[-2, -1]),# (5) Rotate −90°
]


def predict_batch_tta(model, images, device, n_views: int = 5, temperature: float = 1.0):
    """
    Run Test-Time Augmentation on a batch of images.

    Averages softmax probabilities over n_views augmented copies of each image.
    TTA reduces prediction variance and typically improves AUROC by 0.5–1.5%.

    Args:
        model:       Model in eval mode (no grad).
        images:      Batch tensor (B, C, H, W), already on device.
        device:      Compute device.
        n_views:     Number of TTA views (1 = no TTA, 5 = all).
        temperature: Temperature scalar from calibration.json (default 1.0 = off).

    Returns:
        Averaged probability tensor (B, num_classes).
    """
    prob_accumulator = None
    for aug_fn in _TTA_TRANSFORMS[:n_views]:
        aug_images = aug_fn(images)
        logits     = model(aug_images)
        probs      = torch.softmax(logits / temperature, dim=1)

        if prob_accumulator is None:
            prob_accumulator = probs
        else:
            prob_accumulator = prob_accumulator + probs

    return prob_accumulator / n_views


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM Explainability
# ──────────────────────────────────────────────────────────────────────────────

def generate_gradcam_grid(
    model,
    dataset,
    indices,
    label_map,
    out_dir: Path,
    category_name: str,
    max_samples: int = 5,
    device=None,
):
    """
    Generate Grad-CAM heatmap overlays for a set of dataset indices.

    Saves a horizontal grid image (max_samples panels wide) to out_dir.
    Uses torchcam for gradient extraction — install with: pip install torchcam

    Args:
        model:         Trained model.
        dataset:       OCTTestDataset instance (untransformed images preferred).
        indices:       List of integer dataset indices to visualise.
        label_map:     Dict {int → str} e.g. {0: "Normal", 1: "Abnormal"}.
        out_dir:       Output directory.
        category_name: One of 'true_pos', 'false_neg', 'true_neg', 'false_pos'.
        max_samples:   Maximum panels in the grid.
        device:        Compute device.
    """
    try:
        from torchcam.methods import GradCAM
        from torchcam.utils import overlay_mask
        from torchvision.transforms.functional import to_pil_image
        from PIL import ImageDraw
    except ImportError:
        logger.warning(
            "torchcam not installed — skipping Grad-CAM. "
            "Install with: pip install torchcam"
        )
        return

    if device is None:
        device = torch.device("cpu")

    out_dir.mkdir(parents=True, exist_ok=True)

    if len(indices) == 0:
        logger.info("No samples for Grad-CAM category: %s", category_name)
        return

    indices     = list(indices)[:max_samples]
    images_out  = []

    # EfficientNet-B3: last MBConv block = model.features[-1]
    # This is the final spatial feature map before global average pooling
    cam_extractor = GradCAM(model, target_layer=model.features[-1])

    for idx in indices:
        raw_img, true_label = dataset[idx]
        input_tensor = raw_img.unsqueeze(0).to(device)

        # GradCAM requires gradients — temporarily switch to train mode
        model.train()
        logits     = model(input_tensor)
        pred_class = logits.argmax(dim=1).item()
        prob_abnormal = torch.softmax(logits, dim=1)[0, 1].item()

        # Extract activation map for the predicted class
        activation_map = cam_extractor(pred_class, logits)
        cam_extractor.reset()
        model.eval()

        # Overlay heatmap on the original image
        raw_pil = to_pil_image(raw_img.clamp(0, 1))
        cam_pil = to_pil_image(activation_map[0].squeeze(0), mode="F")
        result  = overlay_mask(
            raw_pil, cam_pil, alpha=0.55, colormap="jet", normalize=True
        )

        # Annotate with ground truth and prediction
        draw = ImageDraw.Draw(result)
        text = (
            f"True: {label_map.get(true_label, '?')}\n"
            f"Pred: {label_map.get(pred_class, '?')}  "
            f"({prob_abnormal:.1%} ABN)"
        )
        draw.text((5, 5), text, fill=(255, 255, 0))
        images_out.append(result)

    # Stitch into a horizontal grid
    w, h      = images_out[0].size
    grid_img  = Image.new("RGB", (w * len(images_out), h))
    for i, img in enumerate(images_out):
        grid_img.paste(img, (i * w, 0))

    out_path = out_dir / f"gradcam_{category_name}.png"
    grid_img.save(out_path)
    logger.info("Grad-CAM grid (%s) saved → %s", category_name, out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Level 1 Gatekeeper on External Test Set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=(
            "/Users/nikhilmundhra/Downloads/Capstone/DataSets/test/"
            "Labeled Retinal Optical Coherence Tomography Dataset for "
            "Classification of Normal, Drusen, and CNV Cases 2021"
        ),
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/level1/fold0_best_model.pth",
        help="Path to trained Level 1 checkpoint (.pth)",
    )
    parser.add_argument(
        "--calibration-json", type=str,
        default="checkpoints/level1/calibration.json",
        help=(
            "Path to calibration.json produced by calibrate_level1.py. "
            "Provides the ROC-derived threshold and temperature scalar. "
            "Falls back to threshold=0.5, temperature=1.0 if not found."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override decision threshold. If not set, loads from calibration.json.",
    )
    parser.add_argument(
        "--option", type=int, choices=[1, 2], default=1,
        help="1 = all images; 2 = worst-case images only (Class == Label rows)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
    )
    parser.add_argument(
        "--tta-views", type=int, default=5,
        help="Number of TTA views (1 = disabled, 5 = all). Default: 5",
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Generate Grad-CAM visualisations for TP, FN, TN, FP categories.",
    )
    parser.add_argument(
        "--gradcam-dir", type=str,
        default="logs/level1/gradcam",
        help="Output directory for Grad-CAM images.",
    )
    args = parser.parse_args()

    csv_path = Path(args.data_dir) / "data_information.csv"
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return

    device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── Load calibration.json ─────────────────────────────────────────────────
    cal_path = Path(args.calibration_json)
    if args.threshold is not None:
        # Manual override — explicit flag takes priority
        threshold   = args.threshold
        temperature = 1.0
        logger.info("Using manually specified threshold: %.4f  (temperature: 1.0)", threshold)
    elif cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        threshold   = cal["threshold"]
        temperature = cal.get("temperature", 1.0)
        logger.info(
            "Loaded from calibration.json — "
            "threshold=%.4f  temperature=%.4f  "
            "(strategy=%s  val_sensitivity=%.4f  val_specificity=%.4f)",
            threshold, temperature,
            cal.get("strategy", "unknown"),
            cal.get("sensitivity", float("nan")),
            cal.get("specificity", float("nan")),
        )
    else:
        threshold   = 0.5
        temperature = 1.0
        logger.warning(
            "calibration.json not found at %s — using threshold=0.5, temperature=1.0. "
            "Run scripts/calibrate_level1.py first for principled calibration.",
            cal_path,
        )

    # ── Build model ───────────────────────────────────────────────────────────
    logger.info("Building model and loading checkpoint: %s", args.checkpoint)
    model = build_gatekeeper(num_classes=2, pretrained=False)
    ckpt  = Path(args.checkpoint)
    if not ckpt.exists():
        logger.error("Checkpoint not found: %s", ckpt)
        return
    checkpoint = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ── Build dataset & DataLoader ────────────────────────────────────────────
    logger.info("Loading dataset from %s (Option %d)", args.data_dir, args.option)
    transform = get_transforms("level1", "val")    # Includes CLAHE
    dataset   = OCTTestDataset(
        csv_path, args.data_dir,
        transform=transform,
        option=args.option,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    logger.info("Total test images: %d", len(dataset))
    if args.tta_views > 1:
        logger.info("TTA enabled: %d views per image", args.tta_views)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i % 10 == 0:
                logger.info("Batch %d / %d", i, len(dataloader))
            images = images.to(device)

            # TTA: average probabilities over n_views augmented copies.
            # temperature scaling is applied inside predict_batch_tta.
            probs = predict_batch_tta(
                model, images, device,
                n_views=args.tta_views,
                temperature=temperature,
            )
            preds = (probs[:, 1] >= threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    # ── Filter invalid labels (any -1 from unknown label strings) ─────────────
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    valid  = all_labels != -1
    y_true = all_labels[valid]
    y_pred = all_preds[valid]
    y_prob = all_probs[valid]

    # ── Compute metrics ───────────────────────────────────────────────────────
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")

    # Confusion matrix → clinical metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # TPR — recall for ABNORMAL
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TNR — recall for NORMAL
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # Precision for ABNORMAL
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0   # Precision for NORMAL

    # ── Print results ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("  TEST RESULTS — Level 1 Gatekeeper")
    logger.info("=" * 55)
    logger.info("  Threshold            : %.4f", threshold)
    logger.info("  Temperature          : %.4f", temperature)
    logger.info("  TTA views            : %d", args.tta_views)
    logger.info("  Total samples        : %d", len(y_true))
    logger.info("")
    logger.info("  ── Clinical Metrics ──────────────────────────────")
    logger.info("  Sensitivity (ABN recall) : %.4f  ← primary", sensitivity)
    logger.info("  Specificity (NRM recall) : %.4f", specificity)
    logger.info("  PPV (ABN precision)      : %.4f", ppv)
    logger.info("  NPV (NRM precision)      : %.4f", npv)
    logger.info("")
    logger.info("  ── Aggregate Metrics ────────────────────────────")
    logger.info("  Accuracy             : %.4f", acc)
    logger.info("  AUROC                : %.4f", auroc)
    logger.info("  Macro F1             : %.4f", macro_f1)
    logger.info("  Weighted F1          : %.4f", weighted_f1)
    logger.info("")
    logger.info("  ── Confusion Matrix ─────────────────────────────")
    logger.info("             Pred NRM   Pred ABN")
    logger.info("  True NRM :   %6d     %6d  (TN, FP)", tn, fp)
    logger.info("  True ABN :   %6d     %6d  (FN, TP)", fn, tp)
    logger.info("")
    logger.info("Classification Report:")
    logger.info(
        "\n"
        + classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"])
    )
    logger.info("=" * 55)

    # ── Grad-CAM Visualisations ───────────────────────────────────────────────
    if args.gradcam:
        logger.info("")
        logger.info("Generating Grad-CAM visualisations...")
        label_map = {0: "Normal", 1: "Abnormal"}
        gradcam_dir = Path(args.gradcam_dir)

        true_pos  = np.where((y_true == 1) & (y_pred == 1))[0]
        false_neg = np.where((y_true == 1) & (y_pred == 0))[0]  # Critical — missed disease
        true_neg  = np.where((y_true == 0) & (y_pred == 0))[0]
        false_pos = np.where((y_true == 0) & (y_pred == 1))[0]

        logger.info(
            "Outcome counts — TP:%d  FN:%d  TN:%d  FP:%d",
            len(true_pos), len(false_neg), len(true_neg), len(false_pos),
        )

        # Re-enable gradients for Grad-CAM
        for category_name, indices in [
            ("true_pos",  true_pos),
            ("false_neg", false_neg),   # Most important to understand — missed ABNORMAL
            ("true_neg",  true_neg),
            ("false_pos", false_pos),
        ]:
            generate_gradcam_grid(
                model=model,
                dataset=dataset,
                indices=indices,
                label_map=label_map,
                out_dir=gradcam_dir,
                category_name=category_name,
                max_samples=5,
                device=device,
            )

        logger.info("Grad-CAM grids saved to: %s", gradcam_dir)


if __name__ == "__main__":
    main()
