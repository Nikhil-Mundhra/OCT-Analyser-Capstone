import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
from pathlib import Path
from functools import partial

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training" / "classification"))

import cv2
import numpy as np
import torch
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from segment_anything import SamPredictor
from data.preprocessing.sam_transforms import (
    build_sam_multichannel_inputs,
    generate_retinal_tissue_prompts,
    mask_to_smooth_envelope,
    draw_prompt_visualization
)
from data.preprocessing.tuning.boundaries import letterbox_pad_and_resize

print("1. Initializing SAM ViT-B...", flush=True)
prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size

image_encoder = ImageEncoderViT(
    depth=12,
    embed_dim=768,
    img_size=image_size,
    mlp_ratio=4,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    num_heads=12,
    patch_size=vit_patch_size,
    qkv_bias=True,
    use_rel_pos=True,
    global_attn_indexes=[2, 5, 8, 11],
    window_size=14,
    out_chans=prompt_embed_dim,
)

prompt_encoder = PromptEncoder(
    embed_dim=prompt_embed_dim,
    image_embedding_size=(image_embedding_size, image_embedding_size),
    input_image_size=(image_size, image_size),
    mask_in_chans=16,
)

mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
    transformer=TwoWayTransformer(
        depth=2,
        embedding_dim=prompt_embed_dim,
        mlp_dim=2048,
        num_heads=8,
    ),
    transformer_dim=prompt_embed_dim,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
)

sam = Sam(
    image_encoder=image_encoder,
    prompt_encoder=prompt_encoder,
    mask_decoder=mask_decoder,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
)
sam.eval()

ckpt_path = "checkpoints/sam/sam_vit_b_01ec64.pth"
with open(ckpt_path, "rb") as f:
    state_dict = torch.load(f, map_location="cpu", weights_only=False)
sam.load_state_dict(state_dict)
predictor = SamPredictor(sam)
print("2. SAM ViT-B ready for batch inference!", flush=True)

# Define 8 diverse category targets matching user's dropdown
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

out_dir = Path("scratch/sam_live_results")
out_dir.mkdir(parents=True, exist_ok=True)

results_summary = []

for tag, path in targets:
    if not path.exists():
        print(f"[SKIP] Path does not exist: {path}", flush=True)
        continue

    print(f"\nProcessing {tag:<22} ({path.name})...", flush=True)
    img_bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    h, w = gray.shape

    variants = build_sam_multichannel_inputs(gray)
    prompts = generate_retinal_tissue_prompts(gray, num_pos_points=7)

    # Predict on Synthetic 3C composite
    t0 = time.time()
    predictor.set_image(variants["composite_3c"])
    masks, scores, _ = predictor.predict(
        point_coords=prompts["point_coords"],
        point_labels=prompts["point_labels"],
        box=prompts["box"][None, :],
        multimask_output=True
    )
    lat_s = time.time() - t0

    best_idx = int(np.argmax([m.sum() for m in masks]))
    pred_mask = (masks[best_idx] * 255).astype(np.uint8)
    env_mask, y_top_out, y_bot_out = mask_to_smooth_envelope(pred_mask, margin_top=15, margin_bottom=20)

    # 1. Overlay
    overlay = img_bgr.copy()
    overlay[env_mask > 0] = (overlay[env_mask > 0] * 0.70 + np.array([0, 200, 0]) * 0.30).astype(np.uint8)
    for x in range(w - 1):
        cv2.line(overlay, (x, int(y_top_out[x])), (x + 1, int(y_top_out[x + 1])), (255, 255, 0), 2)
        cv2.line(overlay, (x, int(y_bot_out[x])), (x + 1, int(y_bot_out[x + 1])), (255, 0, 255), 2)

    # 2. Letterboxed Standardized Output
    masked_img = np.where(env_mask[:, :, None] > 0, cv2.cvtColor(variants["raw_gray"], cv2.COLOR_GRAY2BGR), 0)
    letterboxed, _, _, _, _, _ = letterbox_pad_and_resize(masked_img, target_dim=384)

    # 3. Diagnostic Panel: Left = Prompts on Raw, Mid = SAM Segmentation Overlay, Right = Letterboxed 384x384
    prompt_vis = draw_prompt_visualization(img_bgr, prompts)
    panel = np.hstack([prompt_vis, overlay, cv2.resize(letterboxed, (w, h))])

    out_file = out_dir / f"{tag}_sam_diagnostic_panel.png"
    cv2.imwrite(str(out_file), panel)
    print(f"  Done in {lat_s:.2f}s | SAM Confidence: {scores.max():.3f} | Output: {out_file.name}", flush=True)

    results_summary.append({
        "tag": tag,
        "filename": path.name,
        "size": f"{w}x{h}",
        "y_center": prompts["y_center"],
        "confidence": float(scores.max()),
        "latency": lat_s,
        "panel_path": str(out_file)
    })

print("\n" + "=" * 80, flush=True)
print("SAM EVALUATION SUMMARY ACROSS 8 OCT PATHOLOGIES", flush=True)
print("=" * 80, flush=True)
for r in results_summary:
    print(f"Category: {r['tag']:<24} | Center: y={r['y_center']:<3}px | SAM Score: {r['confidence']:.3f} | Latency: {r['latency']:.2f}s")
print("=" * 80, flush=True)
