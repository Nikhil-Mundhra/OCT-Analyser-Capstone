"""
data/transforms.py

Augmentation pipelines for the OCT hierarchical classification pipeline.

Resolution Strategy (from architectural directives):
  Level 1 (Gatekeeper):   224×224 — maximum throughput for binary screening.
  Level 2 (Router):       224×224 — consistent feature space with L1.
  Level 3 (Specialists):  384×384 — fine-grained structural detail for
                           CNV vs DRUSEN, RAO vs RVO, etc.

Pipeline Variants:
  - Standard Train:  Random crop/flip/rotation + ColorJitter + GaussianBlur +
                     RandomErasing. Used for L1, L2, L3_Macular, L3_Diabetic.
  - Heavy Train:     Adds RandomAffine + stronger erasing. Used for
                     extreme minority L3 specialists (Vascular, Fluid, Structural)
                     where RAO has only 22 samples and CSR has 102.
  - Val/Test:        Deterministic resize + CenterCrop + normalize only.

All pipelines use ImageNet mean/std for pretrained backbone compatibility.
"""

from torchvision import transforms

# ── ImageNet statistics ───────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Resolution constants ──────────────────────────────────────────────────────
RES_L1_L2: int = 224   # Level 1 & 2 input resolution
RES_L3:    int = 384   # Level 3 specialist input resolution

# Intermediate crop sizes (resize target before random/center crop)
_CROP_L1_L2: int = 256
_CROP_L3:    int = 416


# ──────────────────────────────────────────────────────────────────────────────
# Transform factory functions
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(resolution: int = RES_L1_L2) -> transforms.Compose:
    """
    Standard training augmentation pipeline.

    Designed to:
      - Increase geometric diversity (flip, rotate, crop).
      - Simulate OCT scan artefacts (GaussianBlur, ColorJitter).
      - Force the network to ignore local texture via RandomErasing.

    Args:
        resolution: Target output resolution (224 or 384).

    Returns:
        Composed torchvision transform.
    """
    crop_size = _CROP_L3 if resolution == RES_L3 else _CROP_L1_L2
    return transforms.Compose([
        transforms.Resize(
            crop_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=0.3,
        ),
        transforms.ToTensor(),
        # RandomErasing after ToTensor (operates on tensor, not PIL image)
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.10),
            ratio=(0.3, 3.3),
            value="random",
        ),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_heavy_train_transforms(resolution: int = RES_L3) -> transforms.Compose:
    """
    Heavy augmentation pipeline for extreme minority classes.

    Applied to L3_Vascular (RAO=22, RVO=101, MH=102), L3_Fluid (CSR=102),
    and L3_Structural (ERM=155, VID=76) to maximise synthetic variation.

    Adds on top of the standard pipeline:
      - RandomAffine (translate, scale, shear)
      - Stronger rotation (±30°)
      - Stronger RandomErasing scale

    Args:
        resolution: Target output resolution (typically 384 for L3).
    """
    crop_size = _CROP_L3 if resolution == RES_L3 else _CROP_L1_L2
    return transforms.Compose([
        transforms.Resize(
            crop_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.10, 0.10),
            scale=(0.85, 1.15),
            shear=10,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.10,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))],
            p=0.4,
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.35,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value="random",
        ),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(resolution: int = RES_L1_L2) -> transforms.Compose:
    """
    Deterministic validation/test pipeline (no augmentation).

    Args:
        resolution: Target output resolution (224 or 384).

    Returns:
        Composed torchvision transform.
    """
    crop_size = _CROP_L3 if resolution == RES_L3 else _CROP_L1_L2
    return transforms.Compose([
        transforms.Resize(
            crop_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Registry — keyed by (mode, split)
# ──────────────────────────────────────────────────────────────────────────────

#: Complete transform registry. Access via :func:`get_transforms`.
TRANSFORM_REGISTRY: dict = {
    # Level 1 — 224px, standard augmentation
    "level1": {
        "train": get_train_transforms(RES_L1_L2),
        "val":   get_val_transforms(RES_L1_L2),
    },
    # Level 2 — 224px, HEAVY augmentation (minority class collapse prevention)
    "level2": {
        "train": get_heavy_train_transforms(RES_L1_L2),
        "val":   get_val_transforms(RES_L1_L2),
    },
    # Level 3 Macular — 384px, standard (large enough dataset)
    "level3_macular": {
        "train": get_train_transforms(RES_L3),
        "val":   get_val_transforms(RES_L3),
    },
    # Level 3 Diabetic — 384px, standard (DME=11,495 samples)
    "level3_diabetic": {
        "train": get_train_transforms(RES_L3),
        "val":   get_val_transforms(RES_L3),
    },
    # Level 3 Vascular — 384px, HEAVY (MH=102, RVO=101, RAO=22)
    "level3_vascular": {
        "train": get_heavy_train_transforms(RES_L3),
        "val":   get_val_transforms(RES_L3),
    },
    # Level 3 Fluid — 384px, HEAVY (CSR=102 only)
    "level3_fluid": {
        "train": get_heavy_train_transforms(RES_L3),
        "val":   get_val_transforms(RES_L3),
    },
    # Level 3 Structural — 384px, HEAVY (ERM=155, VID=76)
    "level3_structural": {
        "train": get_heavy_train_transforms(RES_L3),
        "val":   get_val_transforms(RES_L3),
    },
}


def get_transforms(mode: str, split: str = "train") -> transforms.Compose:
    """
    Convenience accessor for the transform registry.

    Args:
        mode:  Dataset mode (e.g., ``'level1'``, ``'level3_vascular'``).
        split: ``'train'`` or ``'val'``.

    Returns:
        A ``torchvision.transforms.Compose`` instance.

    Raises:
        ValueError: If mode or split is invalid.
    """
    if mode not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown mode: '{mode}'. "
            f"Choose from: {sorted(TRANSFORM_REGISTRY.keys())}"
        )
    if split not in ("train", "val"):
        raise ValueError(f"Unknown split: '{split}'. Use 'train' or 'val'.")
    return TRANSFORM_REGISTRY[mode][split]
