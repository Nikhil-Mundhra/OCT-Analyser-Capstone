"""
scratch/evaluate_medsam2_vs_regular_sam.py

Comprehensive Comparative Benchmark: MedSAM 2 vs Regular SAM on Retinal OCT Scans.
Evaluates Medical SAM 2 (MedSAM2_latest.pt) against Meta SAM ViT-B across 8 key pathologies.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training" / "classification"))

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data.preprocessing.sam_transforms import (
    build_sam_multichannel_inputs,
    generate_retinal_tissue_prompts,
    mask_to_smooth_envelope,
    draw_prompt_visualization
)
from data.preprocessing.tuning.boundaries import letterbox_pad_and_resize

print("=" * 80, flush=True)
print("INITIALIZING MedSAM 2 (Medical Segment Anything 2) BENCHMARK ON OCT DATASET", flush=True)
print("=" * 80, flush=True)

# 1. Load MedSAM 2 Model
medsam2_ckpt = "checkpoints/medsam2/MedSAM2_latest.pt"
medsam2_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

print(f"Loading MedSAM 2 ({medsam2_cfg}) with weights from {medsam2_ckpt}...", flush=True)
t0 = time.time()
medsam2_model = build_sam2(medsam2_cfg, medsam2_ckpt, device="cpu")
medsam2_predictor = SAM2ImagePredictor(medsam2_model)
print(f"MedSAM 2 initialized successfully in {time.time() - t0:.2f}s!\n", flush=True)

# 2. Select 8 representative scans from dataset
targets = [
    ("NORMAL", Path("training/classification/data/micro_dataset/NORMAL/NORMAL-2371458-11.jpeg")),
    ("DME_Cysts", Path("training/classification/data/micro_dataset/DME/DME-4441781-1.jpeg")),
    ("CNV_Neovascularization", Path("training/classification/data/micro_dataset/CNV/CNV-5557306-155.jpeg")),
    ("DRUSEN_Deposits", Path("training/classification/data/micro_dataset/DRUSEN/DRUSEN-9642260-40.jpeg")),
    ("Chiu_DME_Dome", Path("scratch/plan1_test/Subject_01_slice_030_raw.jpg")),
    ("CHU_MH_MacularHole", Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed-Otsu/CHU_MH/MH_surgery_others_267_V_raw.jpg")),
    ("ERM_Membrane", Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed-Otsu/ERM/erm_1043186_1_proc.jpg")),
    ("CSR_Fluid", Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed-Otsu/108503_OCTID_CSR/CSR47_proc.jpg")),
]

out_dir = Path("scratch/medsam2_actual_results")
out_dir.mkdir(parents=True, exist_ok=True)

summary = []

for idx, (tag, path) in enumerate(targets, 1):
    if not path.exists():
        print(f"[SKIP] Missing: {path}", flush=True)
        continue

    print(f"[{idx}/{len(targets)}] Evaluating MedSAM 2 on {tag:<22} ({path.name})...", flush=True)
    img_bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    h, w = gray.shape

    # Preprocessing & Automated Prompts
    variants = build_sam_multichannel_inputs(gray)
    prompts = generate_retinal_tissue_prompts(gray, num_pos_points=7)

    point_coords = prompts["point_coords"]
    point_labels = prompts["point_labels"]
    box = prompts["box"]

    sample_out = out_dir / tag
    sample_out.mkdir(parents=True, exist_ok=True)

    # We evaluate MedSAM 2 in two modes:
    # Mode A: Direct Raw Grayscale (MedSAM2 is trained natively on medical grayscale)
    # Mode B: Synthetic 3-Channel Composite [Raw, CLAHE, Sobel-Y]
    test_inputs = [
        ("MedSAM2_Raw_Gray", cv2.cvtColor(variants["raw_gray"], cv2.COLOR_GRAY2RGB)),
        ("MedSAM2_Synthetic_3C", variants["composite_3c"])
    ]

    mode_res = {}

    for mode_name, img_tensor in test_inputs:
        t_start = time.time()
        medsam2_predictor.set_image(img_tensor)

        # MedSAM 2 inference with Point Prompts + Bounding Box Prior
        masks, scores, _ = medsam2_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box[None, :],
            multimask_output=True
        )
        lat_ms = (time.time() - t_start) * 1000.0

        best_idx = int(np.argmax([m.sum() for m in masks]))
        pred_mask = (masks[best_idx] * 255).astype(np.uint8)
        conf_score = float(scores[best_idx])

        # Smooth boundary envelope extraction
        env_mask, y_top_out, y_bot_out = mask_to_smooth_envelope(
            pred_mask, margin_top=15, margin_bottom=20, gaussian_sigma=6.0
        )

        # Visual overlay
        overlay = img_bgr.copy()
        tissue_idx = (env_mask > 0)
        overlay[tissue_idx] = (overlay[tissue_idx] * 0.70 + np.array([0, 220, 0]) * 0.30).astype(np.uint8)
        for x in range(w - 1):
            cv2.line(overlay, (x, int(y_top_out[x])), (x + 1, int(y_top_out[x + 1])), (255, 255, 0), 2)
            cv2.line(overlay, (x, int(y_bot_out[x])), (x + 1, int(y_bot_out[x + 1])), (255, 0, 255), 2)

        # Standardized Letterboxed 384x384
        masked_img = np.where(env_mask[:, :, None] > 0, cv2.cvtColor(variants["raw_gray"], cv2.COLOR_GRAY2BGR), 0)
        letterboxed, _, _, _, _, _ = letterbox_pad_and_resize(masked_img, target_dim=384)

        cv2.imwrite(str(sample_out / f"{mode_name}_mask.png"), pred_mask)
        cv2.imwrite(str(sample_out / f"{mode_name}_overlay.png"), overlay)
        cv2.imwrite(str(sample_out / f"{mode_name}_letterboxed.png"), letterboxed)

        mode_res[mode_name] = {
            "latency_ms": lat_ms,
            "score": conf_score,
            "overlay": overlay,
            "letterboxed": letterboxed,
            "mask_area": int(np.sum(pred_mask > 0))
        }
        print(f"    - {mode_name:<20} | Latency: {lat_ms:5.1f}ms | Confidence: {conf_score:.3f} | Area: {np.sum(pred_mask > 0)}px", flush=True)

    # Create Side-by-Side Diagnostic Comparison Panel
    # Panel: [Prompts on Raw, MedSAM2 Raw Gray Overlay, MedSAM2 Synthetic 3C Overlay, Standardized 384x384]
    prompt_vis = draw_prompt_visualization(img_bgr, prompts)
    panel = np.hstack([
        prompt_vis,
        mode_res["MedSAM2_Raw_Gray"]["overlay"],
        mode_res["MedSAM2_Synthetic_3C"]["overlay"],
        cv2.resize(mode_res["MedSAM2_Synthetic_3C"]["letterboxed"], (w, h))
    ])

    panel_path = out_dir / f"{tag}_medsam2_diagnostic_panel.png"
    cv2.imwrite(str(panel_path), panel)
    print(f"  -> Saved MedSAM 2 Diagnostic Panel: {panel_path.name}\n", flush=True)

    summary.append({
        "category": tag,
        "filename": path.name,
        "size": f"{w}x{h}",
        "raw_score": mode_res["MedSAM2_Raw_Gray"]["score"],
        "comp_score": mode_res["MedSAM2_Synthetic_3C"]["score"],
        "latency_ms": mode_res["MedSAM2_Synthetic_3C"]["latency_ms"],
        "panel_path": str(panel_path)
    })

print("=" * 80, flush=True)
print("MedSAM 2 EVALUATION SUMMARY ACROSS 8 OCT PATHOLOGIES", flush=True)
print("=" * 80, flush=True)
for s in summary:
    print(f"Category: {s['category']:<26} | MedSAM 2 Raw Score: {s['raw_score']:.3f} | MedSAM 2 Comp Score: {s['comp_score']:.3f} | Latency: {s['latency_ms']:5.1f}ms")
print("=" * 80, flush=True)
print(f"All MedSAM 2 outputs and comparison panels saved in: {out_dir}", flush=True)
print("=" * 80, flush=True)
