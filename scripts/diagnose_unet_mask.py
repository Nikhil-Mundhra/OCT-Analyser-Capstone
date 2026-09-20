"""
scripts/diagnose_unet_mask.py

Standalone CLI utility to inspect, validate, and diagnose U-Net segmentation masks.
Can be invoked on an individual scan or an entire directory to inspect anomalies,
generate visual cards, and print structured triage metrics.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "training" / "classification") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "classification"))

import cv2
import numpy as np
import torch

from data.preprocessing.mask_diagnostics import (
    evaluate_mask_health,
    render_diagnostic_panel,
    MaskHealthReport,
)
from data.preprocessing.geometry import center_and_letterbox_tissue
from data.preprocessing.unet_cropper import UNetTissueCropper
from data.preprocessing.white_bars import detect_and_process_white_bars


def diagnose_scan(
    img_path: Path,
    cropper: UNetTissueCropper,
    out_vis: Optional[Path] = None,
) -> MaskHealthReport:
    img = cv2.imread(str(img_path))
    if img is None:
        report = MaskHealthReport(is_suspicious=True, severity="CRITICAL", anomaly_flags=["FILE_NOT_FOUND"])
        return report

    # Pre-process white bars
    cleaned = detect_and_process_white_bars(img)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    probs, y_t, y_b = cropper.predict_mask_and_vectors(gray)

    # Build binary mask
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        yt = max(0, int(round(y_t[x] - 15)))
        yb = min(h - 1, int(round(y_b[x] + 15)))
        if yb > yt:
            mask[yt : yb + 1, x] = 255

    # Clamp lateral dead margin columns (preserving internal fluid cysts and shadows)
    col_max = np.max(gray, axis=0)
    l_dead = 0
    while l_dead < w and col_max[l_dead] <= 8:
        l_dead += 1
    if l_dead > 0:
        mask[:, :l_dead] = 0
    r_dead = w - 1
    while r_dead >= 0 and col_max[r_dead] <= 8:
        r_dead -= 1
    if r_dead < w - 1:
        mask[:, r_dead + 1:] = 0

    report = evaluate_mask_health(mask=mask, probs=probs, y_top=y_t, y_bot=y_b, raw_img=gray)

    print(f"\n--- Diagnostic Report: {img_path.name} ---")
    print(f"Severity       : [{report.severity}]")
    print(f"Is Suspicious  : {report.is_suspicious}")
    print(f"Anomaly Flags  : {report.anomaly_flags if report.anomaly_flags else 'None'}")
    print(f"Key Metrics    : {json.dumps(report.metrics, indent=2)}")
    if report.recommendations:
        print(f"Recommendations:")
        for r in report.recommendations:
            print(f"  - {r}")

    if out_vis:
        centered_img, centered_mask, _, _, _ = center_and_letterbox_tissue(
            img=np.where(cv2.merge([mask, mask, mask]) > 0, cleaned, 0),
            y_top=y_t,
            y_bot=y_b,
            mask=mask,
            target_dim=384,
        )
        card = render_diagnostic_panel(
            raw_img=img,
            clean_img=centered_img,
            mask=centered_mask,
            y_top=y_t,
            y_bot=y_b,
            report=report,
            target_dim=384,
        )
        out_vis.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_vis), card)
        print(f"Saved visual diagnostic panel to: {out_vis}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Diagnose U-Net segmentation mask on a scan.")
    parser.add_argument("--image", type=str, required=True, help="Path to input B-scan image.")
    parser.add_argument("--save-vis", type=str, default=None, help="Path to save visual diagnostic panel image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu').")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    cropper = UNetTissueCropper(checkpoint_path=ckpt, device=args.device)

    out_p = Path(args.save_vis) if args.save_vis else None
    diagnose_scan(Path(args.image), cropper, out_vis=out_p)


if __name__ == "__main__":
    main()
