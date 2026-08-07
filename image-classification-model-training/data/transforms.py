"""
data/transforms.py

Augmentation pipelines for the OCT Multi-Head classification pipeline using MONAI.
"""

import numpy as np
import cv2
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
    RandRotate,
    RandFlip,
    RandGaussianNoise,
    RandCoarseDropout,
    NormalizeIntensity,
    Transform
)

# ImageNet statistics for ConvNeXt
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RES_H_W = (384, 384)

class CLAHETransform(Transform):
    """
    Contrast Limited Adaptive Histogram Equalization for OCT images.
    Wrapped as a MONAI Transform.
    """
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
        self._clahe = None

    def __call__(self, img):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        
        img_np = img.numpy() if hasattr(img, 'numpy') else img
        
        # Convert first channel to uint8
        ch = np.clip(img_np[0], 0, 255).astype(np.uint8)
        equalized = self._clahe.apply(ch)
        
        # Replace all channels with the equalized luminance (since OCT is structurally grayscale)
        for i in range(img_np.shape[0]):
            img_np[i] = equalized
            
        return img_np.astype(np.float32)

class Ensure3Channels(Transform):
    def __call__(self, img):
        img_np = img.numpy() if hasattr(img, 'numpy') else img
        if img_np.shape[0] == 1:
            img_np = np.repeat(img_np, 3, axis=0)
        elif img_np.shape[0] > 3:
            img_np = img_np[:3]
        return img_np

class TissueMaskCrop(Transform):
    """
    Segmentation-Driven / Morphological Tissue Crop for OCT Scans.
    Removes background UI artifacts (compasses, text headers, logos) in corners
    while preserving 100% of the retinal tissue.
    """
    def __init__(self, min_intensity=15):
        self.min_intensity = min_intensity

    def __call__(self, img):
        try:
            img_np = img.numpy() if hasattr(img, 'numpy') else np.array(img)
            if img_np.ndim == 3:
                if img_np.shape[0] in [1, 3]:  # Channel-first (C, H, W)
                    gray = img_np[0]
                else:                          # Channel-last (H, W, C)
                    gray = img_np[:, :, 0]
            else:                              # 2D (H, W)
                gray = img_np

            if gray.dtype != np.uint8:
                gray_u8 = np.clip(gray * (255.0 if gray.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            else:
                gray_u8 = gray.copy()

            H, W = gray_u8.shape[:2]

            # 1. Morphological tissue contour extraction
            _, thresh = cv2.threshold(gray_u8, self.min_intensity, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return img

            largest_cnt = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray_u8)
            cv2.drawContours(mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)

            mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask_dilated = cv2.dilate(mask, mask_kernel, iterations=1)

            if img_np.ndim == 3:
                if img_np.shape[0] in [1, 3]:
                    out = img_np.copy()
                    for c in range(out.shape[0]):
                        out[c] = np.where(mask_dilated > 0, out[c], 0)
                else:
                    out = img_np.copy()
                    for c in range(out.shape[2]):
                        out[:, :, c] = np.where(mask_dilated > 0, out[:, :, c], 0)
                return out
            else:
                return np.where(mask_dilated > 0, img_np, 0)
        except Exception:
            return img

class Rotate90Clockwise(Transform):
    """
    Rotates image 90 degrees clockwise so OCT scans are correctly oriented.
    """
    def __call__(self, img):
        try:
            img_np = img.numpy() if hasattr(img, 'numpy') else np.array(img)
            if img_np.ndim == 3:
                if img_np.shape[0] in [1, 3]:  # (C, H, W)
                    out = np.rot90(img_np, -1, (1, 2))
                else:                          # (H, W, C)
                    out = np.rot90(img_np, -1, (0, 1))
            else:                              # (H, W)
                out = np.rot90(img_np, -1, (0, 1))
            return out.copy()
        except Exception:
            return img

def get_train_transforms():
    """
    Standard training augmentation pipeline using MONAI for Classified-preprocessed.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Rotate90Clockwise(),  # Fix MONAI LoadImage PILReader 90 deg counter-clockwise axis transpose
        CLAHETransform(),
        Ensure3Channels(),
        ScaleIntensity(), # Scale [0, 255] -> [0, 1]
        Resize(RES_H_W),
        RandFlip(prob=0.5, spatial_axis=1), # Horizontal flip only (Preserves Vitreous -> RPE superior-inferior anatomical ordering)
        RandRotate(range_x=0.09, prob=0.5, keep_size=True), # Small ~5 degree anatomical tilt rotation
        RandGaussianNoise(prob=0.3, std=0.05),
        NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        RandCoarseDropout(holes=1, spatial_size=(32, 32), dropout_holes=True, fill_value=0, prob=0.2)
    ])

def get_val_transforms():
    """
    Deterministic validation/test pipeline using MONAI for Classified-preprocessed.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Rotate90Clockwise(),  # Fix MONAI LoadImage PILReader 90 deg counter-clockwise axis transpose
        CLAHETransform(),
        Ensure3Channels(),
        ScaleIntensity(),
        Resize(RES_H_W),
        NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True)
    ])

def get_transforms(mode_or_split: str = "train", split: str = None) -> Compose:
    """
    Returns the MONAI transforms pipeline.
    Supports legacy two-arg call (mode, split) and new single-arg call (split).
    """
    actual_split = split if split is not None else mode_or_split
    if actual_split not in ["train", "val"]:
        raise ValueError(f"Unknown split: '{actual_split}'. Use 'train' or 'val'.")
    
    if actual_split == "train":
        return get_train_transforms()
    return get_val_transforms()
