"""
scripts/test_sam_preprocessing_suite.py

Comprehensive Test & Validation Suite for Retinal OCT SAM 2 / MedSAM Preprocessing.
Evaluates CLAHE, Sobel-Y, Synthetic 3-Channel Composites, Viridis false-color mapping,
and automatic prompt point / bounding box extraction across multiple retinal pathologies.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure repository root and training/classification are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training" / "classification"))

from data.preprocessing.sam_transforms import (
    build_sam_multichannel_inputs,
    generate_retinal_tissue_prompts,
    mask_to_smooth_envelope,
    draw_prompt_visualization,
)


def run_benchmark():
    print("=" * 80)
    print("RUNNING SAM 2 / MedSAM PREPROCESSING & PROMPT GENERATION BENCHMARK")
    print("=" * 80)

    # 1. Discover 5 representative scans across diverse disease categories
    sample_paths = []
    
    # Check micro_dataset categories
    micro_dir = REPO_ROOT / "training" / "classification" / "data" / "micro_dataset"
    if micro_dir.exists():
        for category in ["NORMAL", "DME", "CNV", "DRUSEN"]:
            cat_dir = micro_dir / category
            if cat_dir.exists():
                imgs = list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.jpg"))
                if imgs:
                    sample_paths.append((category, imgs[0]))

    # Add a difficult DME dome scan from scratch if available
    scratch_samples = list((REPO_ROOT / "scratch").glob("**/*Subject_01_slice_010*.png"))
    if scratch_samples:
        sample_paths.append(("DME_Dome_Macular_Edema", scratch_samples[0]))

    if not sample_paths:
        print("[WARN] No sample scans found in micro_dataset or scratch. Creating synthetic OCT scan.")
        synth_img = np.zeros((496, 512), dtype=np.uint8)
        # Simulate retina
        synth_img[180:320, :] = 90
        # Simulate RPE band
        synth_img[270:285, :] = 220
        synth_path = REPO_ROOT / "scratch" / "synthetic_oct_sample.png"
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(synth_path), synth_img)
        sample_paths.append(("Synthetic_OCT", synth_path))

    output_dir = REPO_ROOT / "scratch" / "sam_preprocessing_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(sample_paths)} test scans for benchmarking:")
    for tag, p in sample_paths:
        print(f"  - [{tag}] {p.name}")

    # 2. Process each sample and evaluate transformations & prompts
    summary_results = []

    for tag, path in sample_paths:
        raw_bgr = cv2.imread(str(path))
        if raw_bgr is None:
            print(f"[SKIP] Failed to load {path}")
            continue

        gray_u8 = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) if raw_bgr.ndim == 3 else raw_bgr
        h, w = gray_u8.shape

        # A. Build multi-channel and false-color variants
        variants = build_sam_multichannel_inputs(gray_u8)

        # Assert correct shapes and types
        assert variants["raw_gray"].shape == (h, w), "raw_gray shape mismatch"
        assert variants["clahe"].shape == (h, w), "clahe shape mismatch"
        assert variants["sobel_y"].shape == (h, w), "sobel_y shape mismatch"
        assert variants["composite_3c"].shape == (h, w, 3), "composite_3c shape mismatch"
        assert variants["viridis_3c"].shape == (h, w, 3), "viridis_3c shape mismatch"
        assert variants["jet_3c"].shape == (h, w, 3), "jet_3c shape mismatch"

        # B. Generate automatic high-confidence prompt points & bounding box
        prompts = generate_retinal_tissue_prompts(gray_u8, num_pos_points=7)
        pos_coords = prompts["pos_coords"]
        neg_coords = prompts["neg_coords"]
        box = prompts["box"]
        y_center = prompts["y_center"]

        # Validate prompt bounds
        assert 0 <= box[0] < box[2] <= w, f"Invalid box width bounds: {box}"
        assert 0 <= box[1] < box[3] <= h, f"Invalid box height bounds: {box}"
        assert len(pos_coords) == 7, "Positive points count mismatch"
        assert len(neg_coords) > 0, "Negative points empty"

        # C. Render Prompt Visualizations on Raw, Composite, and Viridis
        vis_raw = draw_prompt_visualization(cv2.cvtColor(variants["raw_gray"], cv2.COLOR_GRAY2BGR), prompts)
        vis_comp = draw_prompt_visualization(variants["composite_3c"], prompts)
        vis_vir = draw_prompt_visualization(variants["viridis_3c"], prompts)

        # D. Test Mask-to-Envelope Conversion
        # Simulate a candidate binary mask from the box
        synth_mask = np.zeros((h, w), dtype=np.uint8)
        synth_mask[int(box[1] + 20):int(box[3] - 20), :] = 255
        env_mask, y_top_out, y_bot_out = mask_to_smooth_envelope(synth_mask, margin_top=15, margin_bottom=20)

        assert env_mask.shape == (h, w), "Envelope mask shape mismatch"
        assert len(y_top_out) == w and len(y_bot_out) == w, "Boundary vectors length mismatch"
        assert np.all(y_top_out < y_bot_out), "Top boundary must be above bottom boundary"

        # E. Save Individual Visualizations & Composite Comparison Matrix
        sample_out_dir = output_dir / tag
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sample_out_dir / "01_raw_gray.png"), variants["raw_gray"])
        cv2.imwrite(str(sample_out_dir / "02_clahe.png"), variants["clahe"])
        cv2.imwrite(str(sample_out_dir / "03_sobel_y.png"), variants["sobel_y"])
        cv2.imwrite(str(sample_out_dir / "04_composite_3c.png"), variants["composite_3c"])
        cv2.imwrite(str(sample_out_dir / "05_viridis_3c.png"), variants["viridis_3c"])
        cv2.imwrite(str(sample_out_dir / "06_prompts_overlay.png"), vis_comp)

        # Build a 2x3 high-level diagnostic grid for quick visual inspection
        row1 = np.hstack([
            cv2.cvtColor(variants["raw_gray"], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(variants["clahe"], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(variants["sobel_y"], cv2.COLOR_GRAY2BGR)
        ])
        row2 = np.hstack([
            variants["composite_3c"],
            variants["viridis_3c"],
            vis_comp
        ])
        grid_vis = np.vstack([row1, row2])
        grid_path = output_dir / f"{tag}_diagnostic_panel.png"
        cv2.imwrite(str(grid_path), grid_vis)

        summary_results.append({
            "tag": tag,
            "dimensions": f"{w}x{h}",
            "y_center": y_center,
            "box": f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]",
            "pos_points": len(pos_coords),
            "neg_points": len(neg_coords),
            "panel_path": str(grid_path)
        })

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY & VALIDATION RESULTS")
    print("=" * 80)
    for res in summary_results:
        print(f"Pathology: {res['tag']:<25} | Size: {res['dimensions']:<10} | Retinal Center: y={res['y_center']}px | Box: {res['box']}")
        print(f"  -> Generated Diagnostic Grid: {res['panel_path']}")
    print("=" * 80)
    print("All transformations, shape assertions, and prompt verifications PASSED successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
