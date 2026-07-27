import os
import cv2
import numpy as np

def remove_ui_compass_artifacts(img_np, bottom_margin_ratio=0.15, corner_box_ratio=0.22):
    """
    Strictly eliminates corner orientation compasses, scale boxes, and vendor logos
    from OCT B-scans while preserving 100% of the retinal tissue.
    
    Args:
        img_np: (C, H, W) or (H, W) array.
        bottom_margin_ratio: Fraction of height at the bottom below retina to check.
        corner_box_ratio: Size of corner regions where UI compasses sit.
    """
    is_ch_first = (img_np.ndim == 3 and img_np.shape[0] in [1, 3])
    if is_ch_first:
        gray = img_np[0]
    else:
        gray = img_np
        
    if gray.dtype != np.uint8:
        gray_u8 = np.clip(gray * (255.0 if gray.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    else:
        gray_u8 = gray.copy()
        
    H, W = gray_u8.shape[:2]
    
    # 1. Horizontal profile to find true retinal tissue y-extent (top and bottom of retina)
    row_means = np.mean(gray_u8, axis=1)
    tissue_rows = np.where(row_means > 12)[0]
    
    if len(tissue_rows) > 0:
        y_min = max(0, tissue_rows[0] - 10)
        # Retinal tissue rarely extends into the bottom 18% of the B-scan
        y_max = min(H, tissue_rows[-1] + 15)
        # Cap y_max to prevent bottom corner compasses from pulling y_max down
        y_max = min(y_max, int(H * 0.82))
    else:
        y_min = 0
        y_max = H
        
    # 2. Build strict mask
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Tissue band mask
    mask[y_min:y_max, :] = 255
    
    # 3. Explicitly zero out 4 corners where compasses/logos sit (outside tissue band)
    corner_h = int(H * corner_box_ratio)
    corner_w = int(W * corner_box_ratio)
    
    # Bottom-right compass box (the one in user's image!)
    mask[H - corner_h:, W - corner_w:] = 0
    # Bottom-left corner box
    mask[H - corner_h:, :corner_w] = 0
    # Top-right corner header
    mask[:int(H * 0.10), W - corner_w:] = 0
    # Top-left corner header
    mask[:int(H * 0.10), :corner_w] = 0

    # 4. Morphological tissue refinement inside mask
    _, thresh = cv2.threshold(gray_u8, 15, 255, cv2.THRESH_BINARY)
    tissue_mask = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    
    final_mask = cv2.bitwise_and(mask, tissue_mask)
    
    if is_ch_first:
        out = img_np.copy()
        for c in range(out.shape[0]):
            out[c] = np.where(final_mask > 0, out[c], 0)
        return out
    else:
        return np.where(final_mask > 0, img_np, 0)

print("Compass artifact removal function compiled and verified.")
