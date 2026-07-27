"""
Diagnostic script: verify Rotate90Clockwise is actually applied to images.
Saves side-by-side: raw PIL load vs post-transform, printed to PNG files for inspection.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import get_train_transforms, Rotate90Clockwise
from monai.transforms import Compose, LoadImage, EnsureChannelFirst

# ── Find 3 sample images from the dataset ──
DATA_ROOT = "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified"
sample_paths = []
for root, dirs, files in os.walk(DATA_ROOT):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_paths.append(os.path.join(root, f))
    if len(sample_paths) >= 3:
        break
sample_paths = sample_paths[:3]
print(f"Testing on {len(sample_paths)} sample images:")
for p in sample_paths:
    print(f"  {p}")

# ── Test 1: Does MONAI LoadImage + EnsureChannelFirst produce H×W or W×H? ──
print("\n--- Raw MONAI load dimensions (before rotation) ---")
raw_pipe = Compose([LoadImage(image_only=True), EnsureChannelFirst()])
for p in sample_paths:
    img = raw_pipe(p)
    img_np = img.numpy() if hasattr(img, 'numpy') else img
    print(f"  {os.path.basename(p)}: shape={img_np.shape}  (C, H, W)")

# ── Test 2: Does Rotate90Clockwise change the shape? ──
print("\n--- After Rotate90Clockwise ---")
rot_pipe = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Rotate90Clockwise()])
for p in sample_paths:
    img = rot_pipe(p)
    img_np = img.numpy() if hasattr(img, 'numpy') else img
    print(f"  {os.path.basename(p)}: shape={img_np.shape}  (C, H, W)")

# ── Test 3: Save visual comparison strips ──
print("\n--- Saving visual comparison images ---")
os.makedirs("telemetry_outputs/rotation_debug", exist_ok=True)

for i, p in enumerate(sample_paths):
    # Raw PIL load (ground truth orientation)
    pil_img = np.array(Image.open(p).convert("RGB"))

    # MONAI raw (no rotation)
    raw = raw_pipe(p)
    raw_np = raw.numpy() if hasattr(raw, 'numpy') else raw
    raw_disp = np.clip(raw_np, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    if raw_disp.shape[2] == 1:
        raw_disp = np.repeat(raw_disp, 3, axis=2)

    # MONAI with rotation
    rot = rot_pipe(p)
    rot_np = rot.numpy() if hasattr(rot, 'numpy') else rot
    rot_disp = np.clip(rot_np, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    if rot_disp.shape[2] == 1:
        rot_disp = np.repeat(rot_disp, 3, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(pil_img)
    axes[0].set_title(f"PIL Raw Load\n{pil_img.shape[1]}×{pil_img.shape[0]}")
    axes[0].axis('off')

    axes[1].imshow(raw_disp)
    axes[1].set_title(f"MONAI LoadImage (no rotation)\n{raw_disp.shape[1]}×{raw_disp.shape[0]}")
    axes[1].axis('off')

    axes[2].imshow(rot_disp)
    axes[2].set_title(f"After Rotate90Clockwise\n{rot_disp.shape[1]}×{rot_disp.shape[0]}")
    axes[2].axis('off')

    fname = f"telemetry_outputs/rotation_debug/sample_{i:02d}_{os.path.basename(p)}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")

print("\nDone. Check telemetry_outputs/rotation_debug/ for visuals.")
