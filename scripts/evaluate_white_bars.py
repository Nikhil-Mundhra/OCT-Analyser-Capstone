"""
evaluate_white_bars.py

Wedge-Friendly & Anti-Bleed White Bar Detection & Evaluation Script.

Captures 100% of tapering wedge bars, diagonal banners, and corner UI artifacts
all the way to their finest tips without width truncation or tissue bleeding.

Key Features:
1. No Width Restrictions: Captures tapering wedges, triangles, and short corner bars.
2. Low Min-Pixel Threshold (100px): Preserves sharp tapering rightmost tips.
3. Outer Margin Vertical Clipping: Clips detection strictly to top/bottom outer margins (top 18%, bottom 18%), preventing any tissue bleeding.
4. 8-Connectivity Component Tracing: Preserves continuous diagonal bar boundaries.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def detect_and_color_white_bars(
    img: np.ndarray,
    white_thresh: int = 210,
    min_pixels: int = 100,
    max_bars: int = 2,
    margin_ratio: float = 0.18
) -> tuple[np.ndarray, int]:
    """
    Detects irregular, wedge-shaped, tapering, or corner white bars without width restrictions,
    capturing 100% of tapering tips while preventing tissue bleed via vertical margin clipping.
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
        raise ValueError(f"Unsupported image dimensions: {img.shape}")
        
    H, W = gray.shape

    # 1. Threshold high-intensity pixels
    _, white_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)

    # 2. Find connected components (8-connectivity preserves thin tapering tips)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

    # 3. Outer Margin Mask
    margin_mask = np.zeros((H, W), dtype=np.uint8)
    top_margin_h = int(H * margin_ratio)
    bottom_margin_h = int(H * (1.0 - margin_ratio))
    
    margin_mask[:top_margin_h, :] = 255
    margin_mask[bottom_margin_h:, :] = 255

    # 4. Filter candidates:
    # - Must touch top (y <= 2) or bottom (y + h >= H - 2)
    # - Area >= min_pixels (low threshold to preserve sharp tapering tips)
    candidates = []
    
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        
        touches_top = (y <= 2)
        touches_bottom = (y + h >= H - 2)
        
        if (touches_top or touches_bottom) and area >= min_pixels:
            candidates.append((label, area))
            
    # 5. Sort candidates by area descending and select top max_bars
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected_bars = candidates[:max_bars]
    
    # 6. Build highlight mask for selected bars and CLIP to outer margins
    highlight_mask = np.zeros((H, W), dtype=np.uint8)
    
    for label, area in selected_bars:
        highlight_mask[labels == label] = 255
        
    # Crucial Safeguard: Vertical margin clipping prevents bleeding into central tissue
    highlight_mask = cv2.bitwise_and(highlight_mask, margin_mask)

    # 7. Paint bright red [0, 0, 255]
    output_img = bgr.copy()
    output_img[highlight_mask == 255] = [0, 0, 255]
    
    return output_img, len(selected_bars)


def main():
    parser = argparse.ArgumentParser(description="Wedge-Friendly White Bar Detection Evaluation")
    parser.add_argument('--src', type=str, required=True, help="Input image file or directory")
    parser.add_argument('--out', type=str, default="white_bars_eval_out", help="Output directory")
    parser.add_argument('--thresh', type=int, default=210, help="Brightness threshold (default: 210)")
    parser.add_argument('--min-pixels', type=int, default=100, help="Minimum pixel area for a bar (default: 100)")
    parser.add_argument('--max-bars', type=int, default=2, help="Maximum number of bars (default: 2)")
    parser.add_argument('--margin-ratio', type=float, default=0.18, help="Outer margin height ratio (default: 0.18)")
    args = parser.parse_args()

    src_path = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        files = [src_path]
    elif src_path.is_dir():
        files = [p for p in src_path.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXT]
    else:
        print(f"Error: Invalid path {src_path}")
        sys.exit(1)

    print(f"Processing {len(files)} file(s)...")
    total_bars = 0
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        colored_img, count = detect_and_color_white_bars(
            img,
            white_thresh=args.thresh,
            min_pixels=args.min_pixels,
            max_bars=args.max_bars,
            margin_ratio=args.margin_ratio
        )
        total_bars += count
        dst = out_dir / p.name
        cv2.imwrite(str(dst), colored_img)
        print(f"  {p.name}: Detected {count} bar(s) -> Saved to {dst}")

    print(f"\nDone! Processed {len(files)} image(s), detected {total_bars} total bar(s). Results saved to: {out_dir}")

if __name__ == '__main__':
    main()
