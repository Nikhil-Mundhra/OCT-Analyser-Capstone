"""
scripts/preprocess_dataset.py

One-time offline preprocessing pipeline — morphological tissue masking.
Fast, CPU-parallel, no model required.

Pipeline per image:
  1. cv2.imread  (native orientation — no MONAI transpose issue)
  2. Otsu threshold → largest contour → dilate → tissue mask
  3. Zero bottom 2 corners (20% margin) — removes compass/UI artifacts
  4. Apply mask → background zeroed
  5. Save to dst/ preserving directory structure

Rotation NOTE: NO rotation applied here.
  - cv2.imread reads files natively (correctly oriented).
  - Rotate90Clockwise remains in the MONAI training pipeline to correct
    LoadImage's internal axis-transpose.

Usage:
  python3 scripts/preprocess_dataset.py \
      --src  /path/to/Classified \
      --dst  /path/to/Classified-tissue-cropped \
      --workers 8
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
import traceback

import cv2
import numpy as np

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


# ─────────────────────────────────────────────────────────────────────────────
# Core per-image transform
# ─────────────────────────────────────────────────────────────────────────────
def process_image(src_path: str, dst_path: str, quality: int = 95, frame: bool = False, frame_size: int = 224) -> bool:
    try:
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [SKIP] Cannot open: {src_path}", flush=True)
            return False

        # Normalise to 3-channel BGR uint8
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ── Morphological tissue mask ────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if gray.dtype != np.uint8:
            gray_u8 = np.clip(
                gray * (255.0 if gray.max() <= 1.0 else 1.0), 0, 255
            ).astype(np.uint8)
        else:
            gray_u8 = gray

        H, W = gray_u8.shape

        # 1. Threshold + morphological close
        _, thresh = cv2.threshold(gray_u8, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 2. Largest contour → tissue mask
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray_u8)
            cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, dilate_k, iterations=1)
        else:
            mask = np.ones_like(gray_u8, dtype=np.uint8) * 255

        # 3. Zero bottom 2 corners only (compass/UI box sits at bottom corners)
        ch, cw = int(H * 0.25), int(W * 0.20)
        mask[H - ch:, :cw]      = 0  # Bottom-left
        mask[H - ch:, W - cw:]  = 0  # Bottom-right

        # 4. Apply mask — background → pure black
        mask_3c = cv2.merge([mask, mask, mask])
        img = np.where(mask_3c > 0, img, 0).astype(np.uint8)

        # 5. Optional framing (letterbox pad to square + resize)
        if frame:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            pad_top = (max_dim - h) // 2
            pad_bottom = max_dim - h - pad_top
            pad_left = (max_dim - w) // 2
            pad_right = max_dim - w - pad_left
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
            if frame_size is not None:
                img = cv2.resize(img, (frame_size, frame_size), interpolation=cv2.INTER_AREA)

        # ────────────────────────────────────────────────────────────────────
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        ext = dst.suffix.lower()
        if ext in {'.jpg', '.jpeg'}:
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            cv2.imwrite(str(dst), img)

        return True

    except Exception as e:
        print(f"  [ERROR] {src_path}: {e}", flush=True)
        traceback.print_exc()
        return False


def _worker(args):
    src, dst, quality, frame, frame_size = args
    return process_image(src, dst, quality, frame=frame, frame_size=frame_size)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_job_list(src_root: Path, dst_root: Path):
    jobs = []
    for p in sorted(src_root.rglob('*')):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            rel = p.relative_to(src_root)
            dst = dst_root / rel
            jobs.append((str(p), str(dst)))
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Offline OCT morphological tissue-mask preprocessing")
    parser.add_argument('--src',     type=str, required=True,  help="Source: path to Classified/")
    parser.add_argument('--dst',     type=str, required=True,  help="Dest:   path to Classified-tissue-cropped/")
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2),
                        help="Parallel worker processes (default: cpu_count - 2)")
    parser.add_argument('--quality', type=int, default=95,     help="JPEG save quality (default 95)")
    parser.add_argument('--limit',   type=int, default=None,   help="Cap total images (for smoke tests)")
    parser.add_argument('--frame', dest='frame', action='store_true', help="Apply letterbox framing: pad to square with black borders and resize to --frame-size (enabled by default)")
    parser.add_argument('--no-frame', dest='frame', action='store_false', help="Disable framing (framing is enabled by default)")
    parser.set_defaults(frame=True)
    parser.add_argument('--frame-size', type=int, default=224, help="Framing size (default: 224)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"ERROR: Source path does not exist: {src_root}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  OCT Tissue-Mask Preprocessing (Morphological)")
    print(f"{'='*60}")
    print(f"  Source  : {src_root}")
    print(f"  Dest    : {dst_root}")
    print(f"  Workers : {args.workers}")
    print(f"  Corners : bottom-left + bottom-right (20% margin)")
    print(f"{'='*60}\n")

    jobs = build_job_list(src_root, dst_root)
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"  Limit   : {args.limit} images\n")

    total = len(jobs)
    print(f"Found {total:,} images to process.\n")

    if total == 0:
        print("No images found. Check --src path.")
        sys.exit(1)

    jobs_with_quality = [(s, d, args.quality, args.frame, args.frame_size) for s, d in jobs]

    ok = fail = 0
    report_every = max(1, total // 20)

    with mp.Pool(processes=args.workers) as pool:
        for i, success in enumerate(pool.imap_unordered(_worker, jobs_with_quality), start=1):
            if success:
                ok += 1
            else:
                fail += 1
            if i % report_every == 0 or i == total:
                pct = 100 * i / total
                print(f"  Progress: {i:>6,}/{total:,}  ({pct:5.1f}%)  "
                      f"✓ {ok:,}  ✗ {fail:,}", flush=True)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Processed : {ok:,}")
    print(f"  Failed    : {fail:,}")
    print(f"  Output    : {dst_root}")
    print(f"{'='*60}")
    print(f"\nTo train on pre-cropped data, run:")
    print(f"  OCT_DATA_ROOT=\"{dst_root}\" python3 scripts/train_convnext.py ...")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    main()
