"""
scripts/preview_preprocessed.py

Outputs 1 preprocessed image per disease class through the full training
transform pipeline (Rotate90CW → TissueMaskCrop → CLAHE → ScaleIntensity).
Saves a summary grid PNG for visual inspection.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import cv2

# Must set before any torch/cv2 import
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
cv2.setNumThreads(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified"
CONFIG    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "config", "hierarchy.yaml")
OUT_PATH  = "telemetry_outputs/preprocessed_per_class.png"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def preprocess_image(path: str) -> np.ndarray:
    """
    Applies the exact same steps as get_train_transforms() but
    without MONAI Compose, to avoid OMP/fork deadlocks in preview scripts.
    Returns HWC float32 in [0,1] (not normalized) for display.
    """
    # 1. Load as grayscale (MONAI LoadImage default for single-channel OCT)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cv2 could not open: {path}")

    # If colour, keep as-is; if grayscale, keep 2D
    if img.ndim == 2:
        pass  # (H, W)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # NOTE: cv2.imread reads natively — NO rotation needed here.
    # (Rotation is only needed in the MONAI pipeline to fix LoadImage's axis transpose.)

    # 3. TissueMaskCrop
    if img.ndim == 2:
        gray = img
    else:
        gray = img[:, :, 0]

    gray_f = gray.astype(np.float32)
    if gray_f.max() <= 1.0:
        gray_u8 = np.clip(gray_f * 255, 0, 255).astype(np.uint8)
    else:
        gray_u8 = np.clip(gray_f, 0, 255).astype(np.uint8)

    H, W = gray_u8.shape[:2]
    _, thresh = cv2.threshold(gray_u8, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray_u8)
        cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
        mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, mk, iterations=1)

        # Zero bottom 2 corners only (compass sits at bottom corners)
        ch, cw = int(H * 0.25), int(W * 0.20)
        mask[H - ch:, :cw]      = 0  # Bottom-left
        mask[H - ch:, W - cw:]  = 0  # Bottom-right

        if img.ndim == 2:
            img = np.where(mask > 0, img, 0)
        else:
            for c in range(img.shape[2]):
                img[:, :, c] = np.where(mask > 0, img[:, :, c], 0)


    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.ndim == 2:
        img = clahe.apply(np.clip(img, 0, 255).astype(np.uint8))
        img = np.stack([img, img, img], axis=-1)
    else:
        ch0 = np.clip(img[:, :, 0], 0, 255).astype(np.uint8)
        eq  = clahe.apply(ch0)
        img = np.stack([eq, eq, eq], axis=-1)

    # 5. Scale to [0,1]
    img = img.astype(np.float32) / 255.0

    # 6. Resize to 384×384
    img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)

    return img   # HWC [0,1]


# ── Load class map ──
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

seen_classes = {}
# Abnormal classes first
for entry in cfg["class_map"]:
    l3 = entry.get("l3_class") or entry.get("fine_class")
    if not l3 or l3 in seen_classes:
        continue
    dir_path = os.path.join(DATA_ROOT, entry["path"])
    if not os.path.exists(dir_path):
        continue
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            seen_classes[l3] = os.path.join(dir_path, fname)
            break

# Normal
for entry in cfg["class_map"]:
    if entry["l1"] == "NORMAL" and "Normal" not in seen_classes:
        dir_path = os.path.join(DATA_ROOT, entry["path"])
        if os.path.exists(dir_path):
            for fname in sorted(os.listdir(dir_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    seen_classes["Normal"] = os.path.join(dir_path, fname)
                    break

print(f"Processing {len(seen_classes)} classes...\n")

results = {}
for cls_name, img_path in sorted(seen_classes.items()):
    try:
        disp = preprocess_image(img_path)
        results[cls_name] = disp
        print(f"  ✓ {cls_name:<15}  {disp.shape}  {os.path.basename(img_path)}", flush=True)
    except Exception as e:
        import traceback
        print(f"  ✗ {cls_name}: {e}", flush=True)
        traceback.print_exc()

# ── Plot grid ──
if not results:
    print("\n✗ No images were successfully processed. Check errors above.")
    sys.exit(1)

n    = len(results)
cols = 4
rows = max(1, (n + cols - 1) // cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.8))
fig.patch.set_facecolor('#0d0d0d')
axes = axes.flatten()

for ax, (cls_name, img) in zip(axes, results.items()):
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
    ax.set_title(cls_name, fontsize=13, fontweight='bold', color='white', pad=6)
    ax.axis('off')

for ax in axes[len(results):]:
    ax.set_visible(False)

fig.suptitle(
    "Preprocessed OCT — 1 Sample Per Disease Class\n"
    "Pipeline: Rotate90CW → TissueMaskCrop → CLAHE → Scale [0,1] → Resize 384×384",
    fontsize=13, color='#cccccc', y=1.01
)

os.makedirs("telemetry_outputs", exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n✅ Saved → {OUT_PATH}")
