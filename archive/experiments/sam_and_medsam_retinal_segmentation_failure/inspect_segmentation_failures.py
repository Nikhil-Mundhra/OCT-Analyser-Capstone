"""
scratch/inspect_segmentation_failures.py
Inspect the actual mask shapes, coverage, and visual quality.
"""

from pathlib import Path
import cv2
import numpy as np

sam_dir = Path("scratch/sam_live_results")
medsam2_dir = Path("scratch/medsam2_actual_results")

samples = [
    "NORMAL",
    "Chiu_DME_Dome",
    "DME_Cysts",
    "CSR_Fluid",
    "CNV_Neovascularization",
    "ERM_Membrane",
    "CHU_MH_MacularHole",
    "DRUSEN_Deposits"
]

print("=" * 80)
print("AUDITING ACTUAL SEGMENTATION MASKS")
print("=" * 80)

for s in samples:
    p_medsam2_mask = medsam2_dir / s / "MedSAM2_Synthetic_3C_mask.png"
    p_medsam2_raw_mask = medsam2_dir / s / "MedSAM2_Raw_Gray_mask.png"
    
    if p_medsam2_mask.exists():
        m_comp = cv2.imread(str(p_medsam2_mask), cv2.IMREAD_GRAYSCALE)
        m_raw = cv2.imread(str(p_medsam2_raw_mask), cv2.IMREAD_GRAYSCALE)
        h, w = m_comp.shape
        total_px = h * w
        comp_area = np.sum(m_comp > 0)
        raw_area = np.sum(m_raw > 0)
        
        # Check top and bottom rows where mask exists
        y_indices_comp = np.where(m_comp > 0)[0]
        y_min_comp = y_indices_comp.min() if len(y_indices_comp) > 0 else -1
        y_max_comp = y_indices_comp.max() if len(y_indices_comp) > 0 else -1
        
        # Check column coverage
        x_indices_comp = np.where(m_comp > 0)[1]
        x_min_comp = x_indices_comp.min() if len(x_indices_comp) > 0 else -1
        x_max_comp = x_indices_comp.max() if len(x_indices_comp) > 0 else -1

        print(f"Sample: {s:<24} ({w}x{h})")
        print(f"  - MedSAM2 Comp Area: {comp_area:6d}px ({comp_area/total_px*100:4.1f}%) | y: [{y_min_comp}, {y_max_comp}] | x: [{x_min_comp}, {x_max_comp}]")
        print(f"  - MedSAM2 Raw Area:  {raw_area:6d}px ({raw_area/total_px*100:4.1f}%)")
    else:
        print(f"Sample: {s:<24} MISSING MASK FILE")

print("=" * 80)
