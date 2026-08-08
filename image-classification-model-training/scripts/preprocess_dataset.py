"""
scripts/preprocess_dataset.py

One-time offline preprocessing pipeline — Morphological tissue masking + White bar artifact removal.
Fast, CPU-parallel, model-free.

Pipeline per image:
  1. cv2.imread (native orientation)
  2. Detect & remove irregular / slanted / diagonal top & bottom white bars via Column-Wise Vitreous-Moat Raycasting.
  3. Morphological tissue masking (Otsu threshold -> largest contour -> dilate -> background zeroed).
  4. Zero out 4 corner regions (top-left, top-right, bottom-left, bottom-right UI/logo boxes).
  5. Letterbox pad to square + resize to 384x384 (enabled by default).
  6. Save to dst/ preserving directory structure.

Evaluation Mode (--highlight-red):
  Paints detected top/bottom white bars in BRIGHT RED (BGR: [0, 0, 255]) for visual inspection dataset generation.

Usage:
  # Production mode (384x384 framed, white bars & background zeroed):
  python3 scripts/preprocess_dataset.py \
      --src /Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified \
      --dst /Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed \
      --workers 8

  # Red visualization evaluation mode:
  python3 scripts/preprocess_dataset.py \
      --src /Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified \
      --dst /Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed-R3 \
      --highlight-red \
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
# Column-Wise Vitreous-Moat Raycasting White Bar Detection & Removal
# ─────────────────────────────────────────────────────────────────────────────
def detect_and_process_white_bars(
    img: np.ndarray,
    white_thresh: int = 190,
    dark_bg_thresh: int = 70,
    gap_pixels: int = 3,
    highlight_red: bool = False
) -> np.ndarray:
    """
    Performs Column-Wise Raycasting from top (y=0) and bottom (y=H-1) to detect
    slanted, diagonal, wedge-shaped, or thick white bars of ANY angle or thickness.
    Halts each column ray at the dark vitreous/choroid background moat before reaching retinal tissue.
    """
    if img.ndim == 2:
        gray = img
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgr = img.copy()
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        return img

    H, W = gray.shape
    bar_mask = np.zeros((H, W), dtype=np.uint8)

    # 1. Top-Down Column Raycasting
    top_row_white_pct = np.mean(gray[0, :] > white_thresh)
    if top_row_white_pct > 0.15:
        for x in range(W):
            dark_count = 0
            for y in range(H):
                val = gray[y, x]
                if val >= white_thresh:
                    bar_mask[y, x] = 255
                    dark_count = 0
                elif val < dark_bg_thresh:
                    dark_count += 1
                    if dark_count >= gap_pixels:
                        break
                else:
                    dark_count += 1
                    if dark_count >= gap_pixels + 3:
                        break

    # 2. Bottom-Up Column Raycasting
    bottom_row_white_pct = np.mean(gray[H - 1, :] > white_thresh)
    if bottom_row_white_pct > 0.15:
        for x in range(W):
            dark_count = 0
            for y in range(H - 1, -1, -1):
                val = gray[y, x]
                if val >= white_thresh:
                    bar_mask[y, x] = 255
                    dark_count = 0
                elif val < dark_bg_thresh:
                    dark_count += 1
                    if dark_count >= gap_pixels:
                        break
                else:
                    dark_count += 1
                    if dark_count >= gap_pixels + 3:
                        break

    # Smooth raycast mask slightly with morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bar_mask = cv2.morphologyEx(bar_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Apply processing (Black out for production, Bright Red for evaluation)
    processed_img = bgr.copy()
    if highlight_red:
        processed_img[bar_mask == 255] = [0, 0, 255]  # Bright Red in BGR
    else:
        processed_img[bar_mask == 255] = [0, 0, 0]    # Black out in BGR

    return processed_img


# ─────────────────────────────────────────────────────────────────────────────
# Core per-image transform
# ─────────────────────────────────────────────────────────────────────────────
def process_image(
    src_path: str,
    dst_path: str,
    quality: int = 95,
    frame: bool = True,
    frame_size: int = 384,
    highlight_red: bool = False
) -> bool:
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

        # ── 1. Column-Wise Vitreous-Moat White Bar Removal ─────────────────────
        img = detect_and_process_white_bars(
            img, white_thresh=190, dark_bg_thresh=70, gap_pixels=3, highlight_red=highlight_red
        )

        # ── 2. Morphological tissue mask ──────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray_u8 = np.clip(gray * (255.0 if gray.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            gray_u8 = gray

        H, W = gray_u8.shape

        _, thresh = cv2.threshold(gray_u8, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray_u8)
            cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, dilate_k, iterations=1)
        else:
            mask = np.ones_like(gray_u8, dtype=np.uint8) * 255

        # ── 3. Zero bottom 2 corners (compass/UI box sits at bottom corners) ──
        ch, cw = int(H * 0.25), int(W * 0.20)
        mask[H - ch:, :cw]      = 0  # Bottom-left corner
        mask[H - ch:, W - cw:]  = 0  # Bottom-right corner

        # ── 4. Apply mask — background -> pure black (unless red highlighted) ─────
        if not highlight_red:
            mask_3c = cv2.merge([mask, mask, mask])
            img = np.where(mask_3c > 0, img, 0).astype(np.uint8)

        # ── 5. Framing: Letterbox pad to square + resize to 384x384 ───────────
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

        # ── 6. Save output ───────────────────────────────────────────────────
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
    src, dst, quality, frame, frame_size, highlight_red = args
    return process_image(src, dst, quality, frame=frame, frame_size=frame_size, highlight_red=highlight_red)


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
    parser = argparse.ArgumentParser(description="Offline OCT morphological tissue-mask, framing, & white bar preprocessing")
    parser.add_argument('--src',           type=str, required=True,  help="Source: path to Classified/")
    parser.add_argument('--dst',           type=str, required=True,  help="Dest:   path to preprocessed dataset")
    parser.add_argument('--workers',       type=int, default=max(1, mp.cpu_count() - 2),
                        help="Parallel worker processes (default: cpu_count - 2)")
    parser.add_argument('--quality',       type=int, default=95,     help="JPEG save quality (default 95)")
    parser.add_argument('--limit',         type=int, default=None,   help="Cap total images (for smoke tests)")
    parser.add_argument('--highlight-red', action='store_true',     help="Paint detected white bars in BRIGHT RED (evaluation mode)")
    parser.add_argument('--frame',         dest='frame', action='store_true', help="Apply letterbox framing & resize (enabled by default)")
    parser.add_argument('--no-frame',      dest='frame', action='store_false', help="Disable framing")
    parser.set_defaults(frame=True)  # Enabled by default: guarantees square 384x384 resolution
    parser.add_argument('--frame-size',    type=int, default=384, help="Framing target resolution (default: 384)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"ERROR: Source path does not exist: {src_root}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  OCT Preprocessing (Tissue Mask, White Bar Removal, 384x384 Framing)")
    print(f"{'='*60}")
    print(f"  Source        : {src_root}")
    print(f"  Dest          : {dst_root}")
    print(f"  Workers       : {args.workers}")
    print(f"  Highlight Red : {args.highlight_red}")
    print(f"  Framing       : {args.frame} (Target Resolution: {args.frame_size}x{args.frame_size})")
    print(f"{'='*60}\n")

    jobs = build_job_list(src_root, dst_root)
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"  Limit         : {args.limit} images\n")

    total = len(jobs)
    print(f"Found {total:,} images to process.\n")

    if total == 0:
        print("No images found. Check --src path.")
        sys.exit(1)

    jobs_with_args = [(s, d, args.quality, args.frame, args.frame_size, args.highlight_red) for s, d in jobs]

    ok = fail = 0
    report_every = max(1, total // 20)

    with mp.Pool(processes=args.workers) as pool:
        for i, success in enumerate(pool.imap_unordered(_worker, jobs_with_args), start=1):
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
    print(f"{'='*60}\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    main()
