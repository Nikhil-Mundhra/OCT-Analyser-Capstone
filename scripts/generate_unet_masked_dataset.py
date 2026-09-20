"""
scripts/generate_unet_masked_dataset.py

Production Dataset Generation & Autonomous Suspicious Mask Diagnostic Suite.
Processes the full Classified/ dataset into Classified-unet-masked/ with:
  1. Compass UI Removal (Spectralis bottom 30% wireframe boxes)
  2. Horizontal White Scanner Banner & Stamp Removal (Smooth envelope cuts)
  3. Pitch-Black Scanner Letterbox Sanitization (Hard invariant clamping for I <= 5)
  4. U-Net Neural Boundary Prediction (Dense 2D probs + 1D ILM/Choroid vectors)
  5. Real-Time Suspicious Mask Diagnostics (7-invariant medical health check)
  6. Anatomically Centered 384x384 Rescaling (Pixel-to-length accurate isotropic letterbox)
  7. Production Triage & Review (Saves quarantine visual panels for WARNING / CRITICAL)
  8. Resumable State Manifest (Crash-proof, can be stopped and resumed at any time)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "training" / "classification") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "classification"))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data.preprocessing.white_bars import (
    detect_and_process_white_bars,
    detect_and_remove_compass_artifacts,
)
from data.preprocessing.mask_diagnostics import (
    evaluate_mask_health,
    render_diagnostic_panel,
    MaskHealthReport,
)
from data.preprocessing.geometry import center_and_letterbox_tissue
from data.preprocessing.unet_cropper import UNetTissueCropper
from data.preprocessing.tuning.processor import detect_tissue_lateral_bounds

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def process_single_scan_unet(
    src_path: Path,
    cropper: UNetTissueCropper,
    margin_top: int = 15,
    margin_bot: int = 15,
    target_dim: int = 384,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], MaskHealthReport, Dict[str, Any]]:
    """
    Processes a single scan through the entire U-Net pipeline:
      - Artifact removal (compass + white bars + dead borders)
      - U-Net segmentation + 1D vector extraction
      - Diagnostic evaluation
      - Anatomical centering & 384x384 letterboxing
    """
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        report = MaskHealthReport(is_suspicious=True, severity="CRITICAL", anomaly_flags=["UNREADABLE_FILE"])
        return None, None, report, {}

    orig_h, orig_w = img_bgr.shape[:2]

    # 1. Compass UI Removal
    img_bgr, compass_bbox = detect_and_remove_compass_artifacts(
        img_bgr, src_path=str(src_path), enabled=None, location="auto", return_bbox=True
    )

    # 2. Horizontal White Bar / Scanner Banner Removal
    img_bgr = detect_and_process_white_bars(
        img_bgr, white_thresh=175, dark_bg_thresh=70, gap_pixels=4, pad_pixels=4
    )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. U-Net Neural Prediction
    probs_orig, y_top, y_bot = cropper.predict_mask_and_vectors(gray, threshold=0.50)

    # 4. Slanted Lateral Bounds
    c_lt, c_lb, c_rt, c_rb = detect_tissue_lateral_bounds(probs_orig, y_top, y_bot)

    # 5. Build High-Resolution Continuous Mask
    mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
    poly_pts = []
    for x in range(orig_w):
        yt = max(0, int(round(y_top[x] - margin_top)))
        poly_pts.append([x, yt])
    for x in range(orig_w - 1, -1, -1):
        yb = min(orig_h - 1, int(round(y_bot[x] + margin_bot)))
        poly_pts.append([x, yb])

    if poly_pts:
        cv2.fillPoly(mask_orig, [np.array(poly_pts, dtype=np.int32)], 255)

    # Apply Lateral Curtains
    if c_lt > 0 or c_lb > 0:
        left_poly = np.array([[0, 0], [c_lt, 0], [c_lb, orig_h], [0, orig_h]], dtype=np.int32)
        cv2.fillPoly(mask_orig, [left_poly], 0)
    if c_rt < orig_w or c_rb < orig_w:
        right_poly = np.array([[c_rt, 0], [orig_w, 0], [orig_w, orig_h], [c_rb, orig_h]], dtype=np.int32)
        cv2.fillPoly(mask_orig, [right_poly], 0)

    # 6. Dead Border Sanitization (Lateral Scanner Non-Acquisition Columns Only)
    # Note: Never clamp mask[gray <= 5] across 2D pixels, as hyporeflective fluid cysts
    # (DME) and vessel shadows inside the retina have I <= 5 and must remain inside the mask!
    col_max = np.max(gray, axis=0)
    l_dead = 0
    while l_dead < orig_w and col_max[l_dead] <= 8:
        l_dead += 1
    if l_dead > 0:
        mask_orig[:, :l_dead] = 0

    r_dead = orig_w - 1
    while r_dead >= 0 and col_max[r_dead] <= 8:
        r_dead -= 1
    if r_dead < orig_w - 1:
        mask_orig[:, r_dead + 1 :] = 0

    # 7. Real-Time Suspicious Mask Diagnostics Check
    report = evaluate_mask_health(
        mask=mask_orig,
        probs=probs_orig,
        y_top=y_top,
        y_bot=y_bot,
        raw_img=gray,
    )

    # 8. Background Suppression
    mask_3c = cv2.merge([mask_orig, mask_orig, mask_orig])
    suppressed_bgr = np.where(mask_3c > 0, img_bgr, 0).astype(np.uint8)

    # 9. Anatomical Tissue Centering & Isotropic 384x384 Rescaling
    centered_img, centered_mask, scale, pad_t, pad_l = center_and_letterbox_tissue(
        img=suppressed_bgr,
        y_top=y_top,
        y_bot=y_bot,
        mask=mask_orig,
        target_dim=target_dim,
    )

    meta = {
        "orig_shape": [orig_h, orig_w],
        "scale": round(float(scale), 4),
        "pad_top": pad_t,
        "pad_left": pad_l,
        "c_lt": c_lt,
        "c_lb": c_lb,
        "c_rt": c_rt,
        "c_rb": c_rb,
        "compass_bbox": compass_bbox,
    }

    return centered_img, centered_mask, report, meta


def run_batch_dataset_processing(
    src_root: Path,
    dst_root: Path,
    limit: Optional[int] = None,
    quarantine_all_suspicious: bool = True,
    device: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
):
    """
    Executes full Classified/ dataset processing into Classified-unet-masked/.
    """
    dst_images = dst_root / "Images"
    dst_masks = dst_root / "Masks"
    quarantine_dir = dst_root / "quarantine_review"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dst_root / "manifest.json"
    diagnostics_path = dst_root / "diagnostics_summary.json"

    # Load existing manifest for resume capability
    processed_keys = set()
    manifest_data = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                processed_keys = set(manifest_data.get("samples", {}).keys())
            print(f"[Resume] Loaded {len(processed_keys)} existing processed samples from manifest.")
        except Exception as e:
            print(f"[Warning] Failed to read existing manifest: {e}")

    # Discover images
    print(f"[Discovery] Scanning {src_root} for B-scans...")
    all_scans = []
    for p in sorted(src_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            if p.name.endswith("_mask.png") or p.name.startswith("."):
                continue
            all_scans.append(p)

    total_discovered = len(all_scans)
    print(f"[Discovery] Total B-scans found: {total_discovered}")

    if limit is not None:
        all_scans = all_scans[:limit]
        print(f"[Limit] Processing limited to first {limit} scans.")

    # Initialize U-Net
    print(f"[Model] Initializing Attention U-Net Cropper...")
    cropper = UNetTissueCropper(checkpoint_path=checkpoint_path, device=device, target_size=(384, 384))
    if not cropper.has_weights:
        raise RuntimeError("UNetTissueCropper failed to load trained weights. Aborting generation.")
    print(f"[Model] Cropper ready on device: {cropper.device}")

    # Diagnostics Tracking
    clean_init = 0
    warn_init = 0
    crit_init = 0
    flags_init = defaultdict(int)
    if diagnostics_path.exists():
        try:
            with open(diagnostics_path, "r", encoding="utf-8") as f:
                d_old = json.load(f)
                clean_init = d_old.get("clean_count", 0)
                warn_init = d_old.get("warning_count", 0)
                crit_init = d_old.get("critical_count", 0)
                for k, v in d_old.get("flags_tally", {}).items():
                    flags_init[k] = v
        except Exception:
            pass

    stats = {
        "total_processed": len(processed_keys),
        "clean_count": clean_init,
        "warning_count": warn_init,
        "critical_count": crit_init,
        "flags_tally": flags_init,
        "start_time": time.time(),
    }

    pbar = tqdm(all_scans, desc="Generating Classified-unet-masked", unit="scan")
    samples_dict = manifest_data.get("samples", {})

    save_every = 25
    counter = 0

    for src_p in pbar:
        rel_p = src_p.relative_to(src_root)
        rel_key = str(rel_p)

        if rel_key in processed_keys:
            continue

        try:
            centered_img, centered_mask, report, meta = process_single_scan_unet(
                src_path=src_p,
                cropper=cropper,
            )

            if centered_img is None or centered_mask is None:
                stats["critical_count"] += 1
                continue

            # Update stats
            if report.severity == "CLEAN":
                stats["clean_count"] += 1
            elif report.severity == "WARNING":
                stats["warning_count"] += 1
            else:
                stats["critical_count"] += 1

            for flag in report.anomaly_flags:
                stats["flags_tally"][flag] += 1

            # Output paths
            out_img_p = dst_images / rel_p.with_suffix(".png")
            out_mask_p = dst_masks / rel_p.with_suffix(".png")
            out_img_p.parent.mkdir(parents=True, exist_ok=True)
            out_mask_p.parent.mkdir(parents=True, exist_ok=True)

            # Save 384x384 image and mask
            cv2.imwrite(str(out_img_p), centered_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(out_mask_p), centered_mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            # If suspicious, render visual quarantine panel
            if quarantine_all_suspicious and report.is_suspicious:
                raw_orig = cv2.imread(str(src_p))
                if raw_orig is not None:
                    card = render_diagnostic_panel(
                        raw_img=raw_orig,
                        clean_img=centered_img,
                        mask=centered_mask,
                        y_top=None,
                        y_bot=None,
                        report=report,
                        target_dim=384,
                    )
                    safe_stem = f"{src_p.parent.name}_{src_p.stem}_diagnostic.jpg"
                    cv2.imwrite(str(quarantine_dir / safe_stem), card, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Record in manifest
            samples_dict[rel_key] = {
                "image_path": str(out_img_p.relative_to(dst_root)),
                "mask_path": str(out_mask_p.relative_to(dst_root)),
                "severity": report.severity,
                "is_suspicious": report.is_suspicious,
                "anomaly_flags": report.anomaly_flags,
                "metrics": report.metrics,
                "meta": meta,
                "timestamp": time.time(),
            }
            processed_keys.add(rel_key)
            counter += 1
            stats["total_processed"] += 1

            # Periodic manifest checkpoint
            if counter % save_every == 0:
                pbar.set_postfix({
                    "Clean": stats["clean_count"],
                    "Warn": stats["warning_count"],
                    "Crit": stats["critical_count"],
                })
                _save_manifest_and_stats(manifest_path, diagnostics_path, samples_dict, stats)

        except Exception as e:
            tqdm.write(f"[ERROR] Failed processing {src_p}: {e}")
            stats["critical_count"] += 1

    # Final save
    _save_manifest_and_stats(manifest_path, diagnostics_path, samples_dict, stats)

    elapsed = time.time() - stats["start_time"]
    print(f"\n{'='*70}")
    print(f"  PROCESSING COMPLETE | Elapsed: {elapsed/60.0:.2f} mins")
    print(f"  Total Processed : {stats['total_processed']}")
    print(f"  Clean Scans     : {stats['clean_count']} ({stats['clean_count']/max(1, counter)*100:.1f}%)")
    print(f"  Warning Scans   : {stats['warning_count']}")
    print(f"  Critical Scans  : {stats['critical_count']}")
    print(f"  Destination     : {dst_root}")
    print(f"{'='*70}\n")


def _save_manifest_and_stats(m_path: Path, d_path: Path, samples: dict, stats: dict):
    """Saves atomic json checkpoints of the manifest and diagnostics using temp files."""
    m_data = {
        "version": "1.0",
        "dataset": "Classified-unet-masked",
        "total_samples": len(samples),
        "samples": samples,
    }
    m_tmp = m_path.with_suffix(".tmp")
    with open(m_tmp, "w", encoding="utf-8") as f:
        json.dump(m_data, f, indent=2)
    os.replace(m_tmp, m_path)

    d_data = {
        "total_processed": stats["total_processed"],
        "clean_count": stats["clean_count"],
        "warning_count": stats["warning_count"],
        "critical_count": stats["critical_count"],
        "flags_tally": dict(stats["flags_tally"]),
        "updated_at": time.time(),
    }
    d_tmp = d_path.with_suffix(".tmp")
    with open(d_tmp, "w", encoding="utf-8") as f:
        json.dump(d_data, f, indent=2)
    os.replace(d_tmp, d_path)


def main():
    parser = argparse.ArgumentParser(description="Generate production Classified-unet-masked/ dataset with diagnostics.")
    parser.add_argument("--src-dir", type=str, default="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified")
    parser.add_argument("--dst-dir", type=str, default="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-unet-masked")
    parser.add_argument("--limit", type=int, default=None, help="Optional max scans to process for smoke testing.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu').")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth.")
    parser.add_argument("--no-quarantine", action="store_true", help="Disable saving visual quarantine cards.")
    args = parser.parse_args()

    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)
    ckpt = Path(args.checkpoint) if args.checkpoint else None

    run_batch_dataset_processing(
        src_root=src_root,
        dst_root=dst_root,
        limit=args.limit,
        quarantine_all_suspicious=not args.no_quarantine,
        device=args.device,
        checkpoint_path=ckpt,
    )


if __name__ == "__main__":
    main()
