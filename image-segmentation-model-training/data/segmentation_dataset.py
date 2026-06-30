import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

import albumentations as A


class OCT5kSegmentationDataset(Dataset):
    """
    Dataset for Hierarchical Multi-Head Segmentation of OCT5k.

    Returns:
        image        : Normalised float tensor  (1, H, W), values in [0, 1].
        coarse_mask  : Long tensor (H, W) with 3 classes:
                         0 = Background, 1 = Structural Retina, 2 = Lesions/Fluid.
        granular_mask: Long tensor (H, W) with 15 classes (0–14).

    Note on transforms:
        The `transform` parameter accepts an albumentations Compose pipeline and
        is applied *before* tensor conversion. However, for training/validation
        split-specific transforms, use `TransformSubset` (defined below) rather
        than setting `transform` here — this ensures only the training split
        receives augmentation, not the validation split.
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir  : Path to the OCT5k_Segmentation_Subset directory,
                        which must contain Images/ and Masks/ subdirectories.
            transform : albumentations.Compose pipeline, applied to both image
                        and masks before tensor conversion. Pass None (default)
                        and use TransformSubset instead for split-specific transforms.
        """
        self.root_dir   = Path(root_dir)
        self.images_dir = self.root_dir / "Images"
        self.masks_dir  = self.root_dir / "Masks"
        self.transform  = transform

        # Gather all image files, sorted for reproducibility
        self.image_paths = sorted(list(self.images_dir.glob("*.png")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path  = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name

        # Load image as uint8 grayscale (H, W)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        # Load mask — pixel values ARE the class IDs (0–14)
        mask_granular = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_granular is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        # Dynamically generate the coarse (3-class) mask from the granular mask
        #   Class 0: Background
        #   Class 1: Retinal tissue layers (granular classes 1–8)
        #   Class 2: Lesions / Fluid       (granular classes 9–14)
        mask_coarse = np.zeros_like(mask_granular, dtype=np.uint8)
        mask_coarse[mask_granular > 0] = 1   # any retinal structure
        mask_coarse[mask_granular > 8] = 2   # pathology / fluid

        # Apply albumentations transforms (if provided)
        if self.transform is not None:
            augmented     = self.transform(image=image, masks=[mask_granular, mask_coarse])
            image         = augmented['image']
            mask_granular, mask_coarse = augmented['masks']

        # Convert to PyTorch tensors
        image         = torch.from_numpy(image).float().unsqueeze(0) / 255.0   # (1, H, W)
        mask_granular = torch.from_numpy(np.array(mask_granular)).long()        # (H, W)
        mask_coarse   = torch.from_numpy(np.array(mask_coarse)).long()          # (H, W)

        return image, mask_coarse, mask_granular


# ---------------------------------------------------------------------------
# Split-specific Augmentation Transforms
# ---------------------------------------------------------------------------

def get_training_transforms() -> A.Compose:
    """
    Albumentations augmentation pipeline for the **training split only**.

    Design rationale for each transform:
      - HorizontalFlip      : OCT B-scans are laterally symmetric (left/right eye).
                              Flipping doubles effective dataset size with no clinical cost.
      - NO VerticalFlip     : The retina has a strict top/bottom orientation (vitreous above,
                              choroid below). Vertical flipping would create anatomically
                              impossible images and actively mislead the model.
      - ShiftScaleRotate    : Simulates minor positioning variance between OCT acquisitions
                              (patient head tilt, scanner alignment drift). Small limits
                              keep augmentations plausible.
      - RandomBrightnessContrast: Simulates device-to-device intensity calibration
                              differences — important for cross-scanner generalisation.
      - CLAHE               : Contrast-Limited Adaptive Histogram Equalisation. Simulates
                              OCT post-processing pipelines that enhance layer visibility.
      - GaussNoise          : Directly simulates OCT speckle noise, a physical artefact
                              from coherent illumination that appears in every real scan.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            fill=0,       # constant (zero) border fill for image
            fill_mask=0,  # constant (zero) border fill for masks
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        # std_range is albumentations 2.0 API (proportional to pixel value range)
        # (3/255, 7/255) ≈ equivalent to std 3–7 on uint8 images
        A.GaussNoise(std_range=(3/255, 7/255), p=0.3),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
    ])


def get_val_transforms():
    """
    Validation/test transforms: **no augmentation**, but deterministic normalisation.

    Defined explicitly so future preprocessing (e.g., CLAHE or normalisation) can be added
    without touching the training pipeline.
    """
    return A.Compose([
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
    ])


# ---------------------------------------------------------------------------
# TransformSubset — apply split-specific transforms after random_split
# ---------------------------------------------------------------------------

class TransformSubset(torch.utils.data.Dataset):
    """
    Wraps a Subset returned by `torch.utils.data.random_split` and applies
    an albumentations pipeline to *only* that split.

    Why this is needed:
        `random_split` shares the **same** underlying Dataset instance between
        train and val splits. Setting `transform` on the parent dataset would
        therefore augment both splits, which must never happen — the validation
        set must always see unmodified images so metrics are comparable across
        epochs.

    Usage::

        from data.segmentation_dataset import (
            OCT5kSegmentationDataset,
            TransformSubset,
            get_training_transforms,
            get_val_transforms,
        )

        full_dataset = OCT5kSegmentationDataset(root_dir=dataset_path)
        train_sub, val_sub = random_split(full_dataset, [train_n, val_n])

        train_dataset = TransformSubset(train_sub, transform=get_training_transforms())
        val_dataset   = TransformSubset(val_sub,   transform=get_val_transforms())
    """

    def __init__(self, subset: torch.utils.data.Subset, transform=None):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, mask_coarse, mask_granular = self.subset[idx]

        if self.transform is not None:
            # The parent Dataset returns tensors; albumentations needs numpy uint8.
            # We convert back, augment, then re-convert — the overhead is negligible
            # compared to the I/O cost of loading the image from disk.
            image_np  = (image.squeeze(0).numpy() * 255.0).astype(np.uint8)
            mask_g_np = mask_granular.numpy().astype(np.uint8)
            mask_c_np = mask_coarse.numpy().astype(np.uint8)

            aug = self.transform(image=image_np, masks=[mask_g_np, mask_c_np])

            image_np  = aug['image']
            mask_g_np, mask_c_np = aug['masks']

            image         = torch.from_numpy(image_np).float().unsqueeze(0) / 255.0
            mask_granular = torch.from_numpy(mask_g_np.astype(np.int64))
            mask_coarse   = torch.from_numpy(mask_c_np.astype(np.int64))

        return image, mask_coarse, mask_granular
