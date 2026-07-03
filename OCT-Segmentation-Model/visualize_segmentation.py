"""
visualize_segmentation.py
─────────────────────────
Runs the trained HierarchicalUNet on a single OCT image and produces a
side-by-side PNG:
  Left  – original grayscale scan
  Right – segmentation overlay with colour-coded retinal layers + lesions
  Bottom panel – legend + clinical metrics table

Usage:
    python visualize_segmentation.py --image <path> [--output <out.png>]
"""

import sys, argparse, os
from pathlib import Path

# Force non-interactive backend BEFORE matplotlib is imported
os.environ["MPLBACKEND"] = "Agg"

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

# ── Path gymnastics so we can import from the project root ──────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.unet import HierarchicalUNet


# ── Class definitions ────────────────────────────────────────────────────────
CLASS_MAP = {
    0:  ("Background",          "#1a1a2e"),
    1:  ("ILM",                 "#e94560"),
    2:  ("NFL-IPL",             "#f5a623"),
    3:  ("INL",                 "#f8e71c"),
    4:  ("OPL",                 "#7ed321"),
    5:  ("ONL-ISM",             "#4a90e2"),
    6:  ("ISE",                 "#9013fe"),
    7:  ("OS-RPE",              "#50e3c2"),
    8:  ("RPE",                 "#b8e986"),
    9:  ("Fluid",               "#ff3860"),   # lesion – highlighted
    10: ("Hard Drusen",         "#ff6f00"),
    11: ("Soft Drusen",         "#ff9a3c"),
    12: ("PED",                 "#c62828"),
    13: ("Geographic Atrophy",  "#6a1b9a"),
    14: ("Hyper-reflective Foci", "#00838f"),
}

LAYER_CLASSES  = list(range(1, 9))
LESION_CLASSES = list(range(9, 15))

# ── Helpers ──────────────────────────────────────────────────────────────────

def hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def build_colour_mask(pred: np.ndarray, alpha_layers=0.55, alpha_lesions=0.80):
    """
    Returns an BGRA overlay where each class has its own colour.
    Layers get a semi-transparent fill; lesions get a stronger alpha.
    Background stays fully transparent.
    """
    H, W = pred.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    for class_id, (_, hex_col) in CLASS_MAP.items():
        if class_id == 0:
            continue
        mask = pred == class_id
        if not mask.any():
            continue
        h = hex_col.lstrip("#")
        r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
        alpha = alpha_lesions if class_id in LESION_CLASSES else alpha_layers
        overlay[mask, 0] = r
        overlay[mask, 1] = g
        overlay[mask, 2] = b
        overlay[mask, 3] = alpha

    return overlay


def draw_layer_contours(ax, pred: np.ndarray, img_h: int, img_w: int):
    """Draws a 1-pixel boundary line for each detected retinal layer."""
    for class_id in LAYER_CLASSES:
        binary = (pred == class_id).astype(np.uint8)
        if not binary.any():
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, hex_col = CLASS_MAP[class_id]
        for cnt in contours:
            pts = cnt.squeeze()
            if pts.ndim < 2:
                continue
            ax.plot(pts[:, 0], pts[:, 1], linewidth=0.8, color=hex_col, alpha=0.9)


def compute_metrics(pred: np.ndarray, orig_h: int, orig_w: int):
    """Simple pixel-level metrics."""
    H, W = pred.shape
    scale_y = orig_h / H
    scale_x = orig_w / W

    # Retinal thickness: ILM (1) top to RPE (8) bottom
    ilm_rows = np.argwhere(pred == 1)
    rpe_rows = np.argwhere(pred == 8)
    if ilm_rows.size and rpe_rows.size:
        thickness_px = float(rpe_rows[:, 0].mean() - ilm_rows[:, 0].mean())
        thickness_px = max(thickness_px, 0)
    else:
        thickness_px = 0.0

    # Fluid
    fluid_px  = int((pred == 9).sum())
    fluid_pct = 100.0 * fluid_px / (H * W)

    # Lesion presence
    lesion_counts = {}
    for cid in LESION_CLASSES:
        cnt = int((pred == cid).sum())
        if cnt > 10:
            lesion_counts[CLASS_MAP[cid][0]] = cnt

    classes_present = [CLASS_MAP[cid][0] for cid in range(1, 15) if (pred == cid).any()]

    return {
        "est_retinal_thickness_px": round(thickness_px, 1),
        "fluid_area_px": fluid_px,
        "fluid_coverage_pct": round(fluid_pct, 3),
        "lesions_detected": lesion_counts,
        "classes_present": classes_present,
    }


# ── Main visualisation ───────────────────────────────────────────────────────

def visualize(image_path: Path, output_path: Path, checkpoint_path: Path):
    # 1. Load image (grayscale)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = img_gray.shape

    # 2. Prepare tensor  (512×512 as per main.py)
    MODEL_SIZE = 512
    img_resized = cv2.resize(img_gray, (MODEL_SIZE, MODEL_SIZE))
    img_norm    = img_resized.astype(np.float32) / 255.0
    tensor      = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)  # (1,1,512,512)

    # 3. Load model
    device = torch.device("cpu")   # safe on all Macs (no MPS threading issues)
    model  = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # 4. Inference
    with torch.no_grad():
        _, granular_logits = model(tensor.to(device))
    pred = torch.argmax(granular_logits, dim=1).squeeze(0).cpu().numpy()  # (512,512)

    # 5. Metrics
    metrics = compute_metrics(pred, orig_h, orig_w)
    print("\n=== Segmentation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── Build figure ──────────────────────────────────────────────────────────
    # Layout: 2 columns top (original | overlay) + bottom legend/metrics row
    fig = plt.figure(figsize=(16, 11), facecolor="#0d1117")

    gs  = fig.add_gridspec(2, 2,
                           height_ratios=[3.5, 1.2],
                           hspace=0.08, wspace=0.04,
                           left=0.03, right=0.97, top=0.93, bottom=0.02)

    ax_orig    = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_legend  = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])

    for ax in [ax_orig, ax_overlay, ax_legend, ax_metrics]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── Panel 1: original ────────────────────────────────────────────────────
    ax_orig.imshow(img_gray, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax_orig.set_title("Original OCT Scan", color="white", fontsize=13,
                      fontweight="bold", pad=8)

    # ── Panel 2: segmentation overlay ────────────────────────────────────────
    # First show the resized original scan as greyscale background
    ax_overlay.imshow(img_resized, cmap="gray", vmin=0, vmax=255, aspect="auto",
                      extent=[0, MODEL_SIZE, MODEL_SIZE, 0])

    # Build RGBA overlay and display it
    colour_mask = build_colour_mask(pred)
    ax_overlay.imshow(colour_mask, aspect="auto",
                      extent=[0, MODEL_SIZE, MODEL_SIZE, 0],
                      interpolation="nearest")

    # Thin contour lines for layer boundaries
    draw_layer_contours(ax_overlay, pred, orig_h, orig_w)

    ax_overlay.set_title("Hierarchical UNet – Segmentation Overlay",
                          color="white", fontsize=13, fontweight="bold", pad=8)
    ax_overlay.set_xlim(0, MODEL_SIZE)
    ax_overlay.set_ylim(MODEL_SIZE, 0)

    # ── Panel 3: legend ──────────────────────────────────────────────────────
    ax_legend.set_title("Class Legend", color="white", fontsize=11,
                         fontweight="bold", loc="left", pad=4)

    patches = []
    for cid, (name, col) in CLASS_MAP.items():
        if cid == 0:
            continue
        is_lesion = cid in LESION_CLASSES
        edge      = "white" if is_lesion else "none"
        lw        = 1.0     if is_lesion else 0
        p = mpatches.Patch(facecolor=col, edgecolor=edge, linewidth=lw,
                           label=("⚠ " if is_lesion else "• ") + name)
        patches.append(p)

    legend = ax_legend.legend(
        handles=patches,
        loc="center",
        ncol=3,
        fontsize=8.5,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
        framealpha=0.95,
        handlelength=1.4,
        handleheight=1.2,
        columnspacing=1.0,
        borderpad=0.8,
    )

    # ── Panel 4: metrics table ────────────────────────────────────────────────
    ax_metrics.set_title("Clinical Metrics", color="white", fontsize=11,
                          fontweight="bold", loc="left", pad=4)

    rows = [
        ["Est. Retinal Thickness", f"{metrics['est_retinal_thickness_px']} px"],
        ["Fluid Area",             f"{metrics['fluid_area_px']:,} px²"],
        ["Fluid Coverage",         f"{metrics['fluid_coverage_pct']:.3f}%"],
    ]
    for name, cnt in metrics["lesions_detected"].items():
        rows.append([f"  ↳ {name}", f"{cnt:,} px"])

    if not metrics["classes_present"]:
        rows.append(["Classes Detected", "None"])
    else:
        rows.append(["Classes Detected", str(len(metrics["classes_present"]))])

    tbl = ax_metrics.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Style table cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if row % 2 == 0 else "#0d1117")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#30363d")
        if row == 0:   # header
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", fontweight="bold")

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"OCT Retinal Segmentation  ·  {Path(image_path).name}",
        color="white", fontsize=15, fontweight="bold", y=0.97,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    print(f"\n✅  Saved → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input OCT image")
    parser.add_argument("--output", default="segmentation_output.png",
                        help="Output PNG path")
    parser.add_argument("--checkpoint", default=str(ROOT / "unet_hierarchical_best.pth"),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    visualize(
        image_path=Path(args.image),
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
    )
