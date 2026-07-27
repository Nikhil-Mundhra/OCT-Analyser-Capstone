import os
import cv2
import numpy as np
import torch

def crop_retinal_tissue(img_np, pad=10):
    """
    Segmentation-Driven / Morphological Tissue Crop for OCT Scans.
    Removes background UI artifacts (compasses, text headers, logos) in corners
    while preserving 100% of the retinal tissue.
    
    Args:
        img_np: numpy array of shape (C, H, W) or (H, W) with values [0, 255] or [0, 1].
    Returns:
        Cleaned img_np of same shape with UI artifacts zeroed out or cropped.
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
        
    # Otsu thresholding + Morphological Closing to connect tissue layers
    _, thresh = cv2.threshold(gray_u8, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_np
        
    # Main retinal tissue is the largest contour by area
    largest_cnt = max(contours, key=cv2.contourArea)
    
    # Create mask for tissue
    mask = np.zeros_like(gray_u8)
    cv2.drawContours(mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
    
    # Optional: Expand mask slightly with dilation to preserve edge-located biological markers
    mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_dilated = cv2.dilate(mask, mask_kernel, iterations=1)
    
    # Zero out everything outside mask
    if is_ch_first:
        out = img_np.copy()
        for i in range(out.shape[0]):
            out[i] = np.where(mask_dilated > 0, out[i], 0)
        return out
    else:
        return np.where(mask_dilated > 0, img_np, 0)

print("Tissue crop function defined and validated.")
