"""
training/segmentation/dataset_classified_masked.py

PyTorch Dataset Loader for curated Classified-masked/ Retinal Tissue Segmentation.
Consumes verified image-mask pairs curated via the OCT Preprocessing Tuning Dashboard.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_MASKED_ROOT = Path(
    os.environ.get("MASKED_DATASET_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-masked")
)


class ClassifiedMaskedDataset(Dataset):
    """
    High-Quality PyTorch Dataset loader for curated Classified-masked OCT scans.
    Pairs raw uncropped scans with binary tissue segmentation masks (0: Background, 1: Retinal Tissue).
    """

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        target_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
        in_channels: int = 1,
        binary_threshold: int = 128,
    ):
        self.root_dir = Path(root_dir) if root_dir else DEFAULT_MASKED_ROOT
        self.images_dir = self.root_dir / "Images"
        self.masks_dir = self.root_dir / "Masks"
        self.target_size = target_size
        self.transform = transform
        self.in_channels = in_channels
        self.binary_threshold = binary_threshold

        self.samples: list[Tuple[Path, Path, str]] = []
        self._discover_pairs()

    def _discover_pairs(self):
        """Scans Images/ and matches corresponding Masks/ by relative folder hierarchy."""
        if not self.images_dir.exists() or not self.masks_dir.exists():
            return

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for img_path in sorted(self.images_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                rel = img_path.relative_to(self.images_dir)
                mask_path = self.masks_dir / rel
                if not mask_path.exists():
                    # Try with .png extension
                    mask_path = (self.masks_dir / rel).with_suffix(".png")

                if mask_path.exists() and mask_path.is_file():
                    folder_name = rel.parent.name if rel.parent.name else "root"
                    self.samples.append((img_path, mask_path, folder_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path, _ = self.samples[idx]

        # Read image
        if self.in_channels == 1:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

        # Read binary mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")

        # Standardize target dimensions
        if self.target_size is not None:
            th, tw = self.target_size
            if image.shape[:2] != (th, tw):
                image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Binarize mask to {0, 1}
        mask_binary = (mask >= self.binary_threshold).astype(np.uint8)

        # Augmentation pipeline (Albumentations compatible)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask_binary)
            image = augmented["image"]
            mask_binary = augmented["mask"]

        # Convert to Tensors
        if self.in_channels == 1:
            image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        mask_tensor = torch.from_numpy(mask_binary).long()

        return image_tensor, mask_tensor
