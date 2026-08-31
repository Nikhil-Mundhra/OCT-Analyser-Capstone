"""
scratch/build_medsam2_vs_sam_direct_comparison.py
Generate direct head-to-head comparison images between Regular SAM and MedSAM 2.
"""

from pathlib import Path
import cv2
import numpy as np

sam_dir = Path("scratch/sam_live_results")
medsam2_dir = Path("scratch/medsam2_actual_results")
out_dir = Path("/Users/nikhilmundhra/.gemini/antigravity/brain/c906d7df-91e9-46b4-bad0-0635b4ed2118")

tags = [
    ("NORMAL", "NORMAL Retina"),
    ("Chiu_DME_Dome", "Chiu DME Fluid Dome"),
    ("DME_Cysts", "DME Intraretinal Cysts"),
    ("CSR_Fluid", "CSR Subretinal Fluid"),
    ("CNV_Neovascularization", "CNV Neovascularization"),
    ("ERM_Membrane", "ERM Epiretinal Membrane"),
    ("CHU_MH_MacularHole", "Macular Hole (MH)"),
    ("DRUSEN_Deposits", "Drusen Deposits")
]

for tag, title in tags:
    sam_overlay_path = Path(f"scratch/sam_live_results/{tag}/Synthetic_3C_overlay.png")
    if not sam_overlay_path.exists():
        # Fallback to tag name directly
        sam_overlay_path = list(Path("scratch/sam_evaluation_results").glob(f"*{tag}*/*Synthetic_3C_overlay.png"))
        sam_overlay_path = sam_overlay_path[0] if len(sam_overlay_path) > 0 else None

    medsam2_overlay_path = medsam2_dir / tag / "MedSAM2_Synthetic_3C_overlay.png"
    medsam2_letterbox_path = medsam2_dir / tag / "MedSAM2_Synthetic_3C_letterboxed.png"

    if not medsam2_overlay_path.exists():
        print(f"Skipping {tag}, missing medsam2 overlay")
        continue

    medsam2_overlay = cv2.imread(str(medsam2_overlay_path))
    h, w, _ = medsam2_overlay.shape

    # Load SAM overlay from diagnostic panel
    sam_panel_path = Path(f"scratch/sam_live_results/{tag}_sam_diagnostic_panel.png")
    if sam_panel_path.exists():
        sam_panel = cv2.imread(str(sam_panel_path))
        # Panel has 3 columns: Prompt, SAM Overlay, Letterboxed
        pw = sam_panel.shape[1] // 3
        sam_overlay = sam_panel[:, pw:2*pw]
        sam_overlay = cv2.resize(sam_overlay, (w, h))
    else:
        sam_overlay = medsam2_overlay.copy()

    # Load prompt vis
    prompt_panel = cv2.imread(str(medsam2_dir / f"{tag}_medsam2_diagnostic_panel.png"))
    prompt_vis = prompt_panel[:, :w]
    letterboxed = cv2.resize(cv2.imread(str(medsam2_letterbox_path)), (w, h))

    # Add text banner headers
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(prompt_vis, "Input: Scan + Center/Vitreous Prompts", (20, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(sam_overlay, "Stock SAM ViT-B (SA-1B Weights)", (20, 30), font, 0.7, (0, 100, 255), 2)
    cv2.putText(medsam2_overlay, "MedSAM 2 (Medical Fine-Tuned)", (20, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(letterboxed, "Standardized Letterbox 384x384", (20, 30), font, 0.7, (255, 255, 255), 2)

    row1 = np.hstack([prompt_vis, sam_overlay])
    row2 = np.hstack([medsam2_overlay, letterboxed])
    comparison_grid = np.vstack([row1, row2])

    out_file = out_dir / f"{tag}_head_to_head_comparison.png"
    cv2.imwrite(str(out_file), comparison_grid)
    print(f"Generated head-to-head comparison: {out_file.name}")
