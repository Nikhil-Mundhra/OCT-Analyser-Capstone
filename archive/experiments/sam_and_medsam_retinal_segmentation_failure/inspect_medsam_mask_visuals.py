"""
scratch/inspect_medsam_mask_visuals.py
Examine the raw masks vs scan structure.
"""

from pathlib import Path
import cv2
import numpy as np

out_dir = Path("scratch/medsam2_debug_visuals")
out_dir.mkdir(parents=True, exist_ok=True)

samples = ["NORMAL", "Chiu_DME_Dome", "DME_Cysts", "CSR_Fluid", "ERM_Membrane"]

for s in samples:
    p_img = Path(f"scratch/medsam2_actual_results/{s}/MedSAM2_Synthetic_3C_overlay.png")
    p_mask = Path(f"scratch/medsam2_actual_results/{s}/MedSAM2_Synthetic_3C_mask.png")
    
    if p_img.exists() and p_mask.exists():
        img = cv2.imread(str(p_img))
        mask = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
        
        # Overlay mask directly in bright red
        debug_vis = img.copy()
        debug_vis[mask > 0] = [0, 0, 255] # Pure red mask
        
        out_path = out_dir / f"{s}_raw_mask_overlay.png"
        cv2.imwrite(str(out_path), debug_vis)
        print(f"Saved: {out_path}")
