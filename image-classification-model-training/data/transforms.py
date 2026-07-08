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

def get_train_transforms():
    """
    Standard training augmentation pipeline using MONAI.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        CLAHETransform(),
        Ensure3Channels(),
        ScaleIntensity(), # Scale [0, 255] -> [0, 1]
        Resize(RES_H_W),
        RandFlip(prob=0.5, spatial_axis=0), # Vertical
        RandFlip(prob=0.5, spatial_axis=1), # Horizontal
        RandRotate(range_x=0.26, prob=0.5, keep_size=True), # ~15 degrees
        RandGaussianNoise(prob=0.3, std=0.05),
        NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        RandCoarseDropout(holes=1, spatial_size=(32, 32), dropout_holes=True, fill_value=0, prob=0.2)
    ])

def get_val_transforms():
    """
    Deterministic validation/test pipeline using MONAI.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
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
