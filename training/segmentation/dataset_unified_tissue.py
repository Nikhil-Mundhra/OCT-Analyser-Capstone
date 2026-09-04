"""
training/segmentation/dataset_unified_tissue.py

Unified Multi-Source PyTorch Dataset for Retinal Tissue Cropping & Background Suppression.
Combines OCT5K Semantic Segmentation, OIMHS Cyst/Hole Lesions, OCTID, and Curated Classified-masked scans.
"""

import hashlib
import os
from pathlib import Path
import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from data.preprocessing.white_bars import detect_and_process_white_bars
except ImportError:
    try:
        from training.classification.data.preprocessing.white_bars import detect_and_process_white_bars
    except ImportError:
        from white_bars import detect_and_process_white_bars

DEFAULT_SEGMENTED_ROOT = Path(
    os.environ.get("SEGMENTED_DATASET_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Segmented")
)
DEFAULT_MASKED_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "data" / "Classified-masked"
DEFAULT_MASKED_EXTERNAL_ROOT = Path(
    os.environ.get("MASKED_DATASET_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-masked")
)


class UnifiedTissueDataset(Dataset):
    """
    Unified Dataset Loader for Binary Retinal Tissue Segmentation (0: Background, 1: Retinal Tissue).
    Aggregates multi-source scans into a standardized (512x512) tensor pipeline.
    """

    def __init__(
        self,
        segmented_root: Optional[Path] = None,
        masked_root: Optional[Path] = None,
        target_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
        in_channels: int = 1,
        include_oct5k: bool = True,
        include_oimhs: bool = True,
        include_octid: bool = True,
        include_curated: bool = True,
        samples_override: Optional[List[Tuple[Path, Path, str]]] = None,
    ):
        self.segmented_root = Path(segmented_root) if segmented_root else DEFAULT_SEGMENTED_ROOT
        self.masked_root = Path(masked_root) if masked_root else (
            DEFAULT_MASKED_PROJECT_ROOT if DEFAULT_MASKED_PROJECT_ROOT.exists() else DEFAULT_MASKED_EXTERNAL_ROOT
        )
        self.target_size = target_size
        self.transform = transform
        self.in_channels = in_channels

        if samples_override is not None:
            self.samples = samples_override
        else:
            self.samples: List[Tuple[Path, Path, str]] = []
            if include_oct5k:
                self._discover_oct5k()
            if include_oimhs:
                self._discover_oimhs()
            if include_octid:
                self._discover_octid()
            if include_curated:
                self._discover_curated()

    def _discover_oct5k(self):
        """Discovers OCT5K matched image-mask pairs and sets label conversion."""
        oct5k_dir = self.segmented_root / "OCT5K_Semantic_Segmentation"
        if not oct5k_dir.exists():
            return

        images_dir = oct5k_dir / "Images"
        if images_dir.exists():
            masks_base = oct5k_dir / "Masks"
            masks_g1 = masks_base / "Grading_1"
            for img_p in sorted(images_dir.rglob("*.png")):
                rel_p = img_p.relative_to(images_dir)
                mask_p = masks_g1 / rel_p
                if not mask_p.exists():
                    mask_p = masks_base / rel_p
                if mask_p.exists():
                    self.samples.append((img_p, mask_p, "oct5k"))

    def _discover_oimhs(self):
        """Discovers OIMHS lesion and retinal tissue scans."""
        oimhs_dir = self.segmented_root / "OIMHS_Formatted"
        if not oimhs_dir.exists():
            return

        images_dir = oimhs_dir / "Images"
        masks_dir = oimhs_dir / "Masks"
        if images_dir.exists() and masks_dir.exists():
            for img_p in sorted(images_dir.rglob("*.png")):
                rel_p = img_p.relative_to(images_dir)
                mask_p = masks_dir / rel_p
                if mask_p.exists():
                    self.samples.append((img_p, mask_p, "oimhs"))

    def _discover_octid(self):
        """Discovers 108503-V1-OCTID layer boundary scans."""
        octid_dir = self.segmented_root / "108503-V1-OCTID"
        if not octid_dir.exists():
            return

        images_dir = octid_dir / "images"
        masks_dir = octid_dir / "masks"
        if images_dir.exists() and masks_dir.exists():
            for img_p in sorted(images_dir.rglob("*")):
                if img_p.is_file() and img_p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    rel_p = img_p.relative_to(images_dir)
                    mask_p = (masks_dir / rel_p).with_suffix(".png")
                    if mask_p.exists():
                        self.samples.append((img_p, mask_p, "octid"))

    def _discover_curated(self):
        """Discovers curated scans and binary masks from Classified-masked/."""
        if not self.masked_root.exists():
            return

        images_dir = self.masked_root / "Images"
        masks_dir = self.masked_root / "Masks"
        if images_dir.exists() and masks_dir.exists():
            for img_p in sorted(images_dir.rglob("*")):
                if img_p.is_file() and img_p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    rel_p = img_p.relative_to(images_dir)
                    mask_p = (masks_dir / rel_p).with_suffix(".png")
                    if mask_p.exists():
                        self.samples.append((img_p, mask_p, "curated"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p, mask_p, source_type = self.samples[idx]

        # 1. Read scan image
        if self.in_channels == 1:
            image = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_p}")

        # 1b. Clean white scanner annotation bars / metadata borders to pure black
        image = detect_and_process_white_bars(image, white_thresh=190, dark_bg_thresh=70, gap_pixels=3)
        if self.in_channels == 1 and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Read mask
        raw_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"Failed to read mask at: {mask_p}")

        # 3. Standardize dimensions
        th, tw = self.target_size
        if image.shape[:2] != (th, tw):
            image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        if raw_mask.shape[:2] != (th, tw):
            raw_mask = cv2.resize(raw_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # 4. Source-specific binarization
        if source_type == "oct5k":
            # OCT5K has 6 classes: 0 is background, 1..5 are retinal layers (ILM to Choroid)
            mask = (raw_mask >= 1).astype(np.uint8)
        elif source_type == "curated":
            # Curated masks are 8-bit binary (0 or 255)
            mask = (raw_mask >= 128).astype(np.uint8)
        else:
            # OIMHS and OCTID: any non-zero pixel is tissue/lesion
            mask = (raw_mask > 0).astype(np.uint8)

        # 4b. Pitch-Black Border & Non-Acquisition Sanitization
        # Enforce physical invariant: Dead scanner border pixels / pure black margins (<= 5)
        # can never be retinal tissue, even if manual or extrapolated vectors spanned across them.
        if self.in_channels == 1:
            dead_pixels = (image <= 5)
        else:
            dead_pixels = (np.max(image, axis=2) <= 5)
        mask[dead_pixels] = 0

        # 5. Apply augmentations (Albumentations compatible)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 6. Convert to PyTorch Tensors
        if self.in_channels == 1:
            img_f32 = (image.astype(np.float32) / 255.0)[np.newaxis, ...]
            image_tensor = torch.from_numpy(img_f32)
        else:
            img_f32 = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(img_f32)

        mask_int64 = mask.astype(np.int64)
        mask_tensor = torch.from_numpy(mask_int64)

        return image_tensor, mask_tensor


class MedicalTissueAugmenter:
    """Dependency-free non-destructive medical data augmenter for paired scans and masks."""
    def __init__(
        self,
        p_flip: float = 0.5,
        p_brightness: float = 0.4,
        p_noise: float = 0.2,
        p_gamma: float = 0.3,
    ):
        self.p_flip = p_flip
        self.p_brightness = p_brightness
        self.p_noise = p_noise
        self.p_gamma = p_gamma

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict:
        aug_img = image.copy()
        aug_mask = mask.copy()

        # 1. Horizontal Flip
        if random.random() < self.p_flip:
            aug_img = np.fliplr(aug_img)
            aug_mask = np.fliplr(aug_mask)

        # 2. Brightness & Contrast (affects scan only)
        if random.random() < self.p_brightness:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            aug_img = np.clip(alpha * aug_img.astype(float) + beta, 0, 255).astype(np.uint8)

        # 3. Gamma Correction (affects scan only)
        if random.random() < self.p_gamma:
            gamma = random.uniform(0.80, 1.25)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            aug_img = cv2.LUT(aug_img, table)

        # 4. Additive Gaussian Noise (affects scan only)
        if random.random() < self.p_noise:
            noise = np.random.normal(0, random.uniform(3, 10), aug_img.shape)
            aug_img = np.clip(aug_img.astype(float) + noise, 0, 255).astype(np.uint8)

        return {"image": np.ascontiguousarray(aug_img), "mask": np.ascontiguousarray(aug_mask)}


def get_tissue_augmentations(is_train: bool = True) -> Optional[Callable]:
    """Builds non-destructive medical augmentation pipeline."""
    if not is_train:
        return None
    return MedicalTissueAugmenter()


def get_train_val_tissue_datasets(
    segmented_root: Optional[Path] = None,
    masked_root: Optional[Path] = None,
    target_size: Tuple[int, int] = (512, 512),
    val_split: float = 0.15,
    seed: int = 42,
    in_channels: int = 1
) -> Tuple[UnifiedTissueDataset, UnifiedTissueDataset]:
    """Creates stratified train and validation dataset instances."""
    full_ds = UnifiedTissueDataset(
        segmented_root=segmented_root,
        masked_root=masked_root,
        target_size=target_size,
        in_channels=in_channels,
    )

    # Deterministic Path Hash-Based Partitioning
    # Guarantees zero leakage across iterative fine-tuning rounds as dataset grows
    train_samples = []
    val_samples = []
    threshold = int(val_split * 10000)

    for img_p, mask_p, src in full_ds.samples:
        # Use consistent identifier: source + relative filename
        sample_id = f"{src}:{img_p.name}"
        hash_val = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 10000
        if hash_val < threshold:
            val_samples.append((img_p, mask_p, src))
        else:
            train_samples.append((img_p, mask_p, src))

    # Safeguard: ensure at least 1 sample in validation if dataset is tiny
    if len(val_samples) == 0 and len(train_samples) > 1:
        val_samples.append(train_samples.pop())

    train_aug = get_tissue_augmentations(is_train=True)
    train_dataset = UnifiedTissueDataset(
        target_size=target_size,
        transform=train_aug,
        in_channels=in_channels,
        samples_override=train_samples
    )
    val_dataset = UnifiedTissueDataset(
        target_size=target_size,
        transform=None,
        in_channels=in_channels,
        samples_override=val_samples
    )

    return train_dataset, val_dataset
