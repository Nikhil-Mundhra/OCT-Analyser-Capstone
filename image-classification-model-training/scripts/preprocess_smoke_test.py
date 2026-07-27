"""
scripts/preprocess_smoke_test.py

Runs seg-driven preprocessing on exactly 1 image per leaf folder
under --src, saves results to --dst, then outputs a side-by-side
grid: original | seg mask | masked result.
"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import cv2
cv2.setNumThreads(0)

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEG_ROOT = Path(__file__).resolve().parents[2] / "image-segmentation-model-training"
sys.path.insert(0, str(SEG_ROOT))
from models_suite.model1_oct5k_layers.unet_layers import RetinalLayersUNet

CHECKPOINT = SEG_ROOT / "models_suite/model1_oct5k_layers/checkpoints/best_model.pth"
VALID_EXT  = {'.jpg', '.jpeg', '.png', '.bmp'}
SEG_SIZE   = 512
DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

SRC = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified")
OUT = Path("telemetry_outputs/seg_smoke_test.png")

# ── Load model ──────────────────────────────────────────────────────────────
model = RetinalLayersUNet(in_channels=1, num_classes=6).to(DEVICE)
ckpt  = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✓ RetinalLayersUNet loaded  (epoch {ckpt['epoch']})  device={DEVICE}\n")

# ── Collect 1 image per leaf folder ────────────────────────────────────────
samples = {}   # folder_label → image_path
for folder in sorted(SRC.rglob('*')):
    if not folder.is_dir():
        continue
    imgs = [f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in VALID_EXT]
    if imgs:
        label = str(folder.relative_to(SRC))
        samples[label] = imgs[0]

print(f"Found {len(samples)} leaf folders. Processing 1 image each...\n")

# ── Run inference ───────────────────────────────────────────────────────────
rows = []
for label, img_path in samples.items():
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        print(f"  [SKIP] {img_path.name}")
        continue
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    orig_h, orig_w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor  = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
    pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    tissue = (pred > 0).astype(np.uint8) * 255
    mask   = cv2.resize(tissue, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask   = cv2.dilate(mask, k, iterations=2)

    masked = np.where(cv2.merge([mask, mask, mask]) > 0, img_bgr, 0).astype(np.uint8)

    orig_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked,  cv2.COLOR_BGR2RGB)

    rows.append((label, orig_rgb, mask, masked_rgb))
    print(f"  ✓ {label[:60]:<60}  {img_path.name}", flush=True)

# ── Plot grid ───────────────────────────────────────────────────────────────
n    = len(rows)
cols = 3   # orig | mask | result
fig, axes = plt.subplots(n, cols, figsize=(cols * 5, n * 3))
fig.patch.set_facecolor('#0d0d0d')

col_titles = ["Original", "Seg Mask (tissue=white)", "Masked Result"]
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, fontsize=11, color='cyan', fontweight='bold', pad=6)

for ri, (label, orig, mask, masked) in enumerate(rows):
    short = label.split('/')[-1]   # leaf folder name only
    for ci, img in enumerate([orig, mask, masked]):
        ax = axes[ri, ci]
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.axis('off')
        if ci == 0:
            ax.set_ylabel(short, fontsize=8, color='white', rotation=0,
                          labelpad=80, va='center')

fig.suptitle(
    "Seg-Driven Masking — RetinalLayersUNet (OCT5K, 6-class)\n1 sample per leaf folder",
    fontsize=13, color='white', y=1.002
)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(str(OUT), dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n✅ Grid saved → {OUT}")
