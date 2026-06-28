"""
data/dataset.py

OCTHierarchicalDataset — PyTorch Dataset for the hierarchical OCT pipeline.

Key design choices:
  - All class-to-path mappings are driven by config/hierarchy.yaml.
    Adding a new data source requires only a YAML entry — no code changes.
  - Supports 7 modes: level1, level2, level3_{macular,diabetic,vascular,fluid,structural}
    Each mode filters the manifest and exposes the appropriate label column.
  - Stratified 5-fold cross-validation via build_kfold_dataloaders().
    Labels are stratified on the finest grain available per mode, ensuring
    even the rarest classes (RAO=22, CSR=102) appear in every fold's val set.
  - WeightedRandomSampler provides per-sample oversampling for minority classes
    to complement FocalLoss — a two-pronged imbalance mitigation strategy.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Valid dataset modes
DataMode = Literal[
    "level1",
    "level2",
    "level3_macular",
    "level3_diabetic",
    "level3_vascular",
    "level3_fluid",
    "level3_structural",
]

# Maps mode → l3_specialist key in the YAML
_L3_MODE_TO_SPECIALIST: Dict[str, str] = {
    "level3_macular":    "Macular",
    "level3_diabetic":   "Diabetic",
    "level3_vascular":   "Vascular",
    "level3_fluid":      "Fluid",
    "level3_structural": "Structural",
}

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class OCTHierarchicalDataset(Dataset):
    """
    Dataset for one mode and one fold of the OCT hierarchy.

    Args:
        config_path:   Absolute or relative path to config/hierarchy.yaml.
        mode:          One of the DataMode literals above.
        fold_indices:  Numpy integer array of row indices into the mode-filtered
                       manifest. Pass ``None`` to include all samples.
        transform:     torchvision transform applied to each loaded PIL image.

    Attributes:
        class_names (List[str]): Ordered class name list for this mode.
        num_classes (int):       Number of classes for this mode.
        class_weights (Tensor):  Inverse-frequency weights, shape [num_classes].
    """

    def __init__(
        self,
        config_path: str,
        mode: DataMode,
        fold_indices: Optional[np.ndarray] = None,
        transform=None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.transform = transform

        # ── Load YAML config ─────────────────────────────────────────────────
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        self._data_root    = Path(self._cfg["data_root"])
        self._l1_labels    = self._cfg["l1_labels"]     # {NORMAL: 0, ABNORMAL: 1}
        self._l2_labels    = self._cfg["l2_labels"]     # {Macular_Degeneration: 0, …}
        self._l3_specs     = self._cfg["l3_specialists"] # nested dict

        # ── Build full manifest (cached as class attr per config path) ────────
        self._full_manifest: pd.DataFrame = self._build_manifest()

        # ── Filter to this mode ───────────────────────────────────────────────
        self._manifest: pd.DataFrame = self._filter_manifest(mode)

        # ── Apply fold subset ─────────────────────────────────────────────────
        if fold_indices is not None:
            self._manifest = (
                self._manifest.iloc[fold_indices].reset_index(drop=True)
            )

        # ── Determine label column ────────────────────────────────────────────
        self._label_col: str = _label_column_for_mode(mode)

        # ── Pre-compute class weights ─────────────────────────────────────────
        self.class_weights: torch.Tensor = self._compute_class_weights()

        logger.info(
            "[%s] %d samples | %d classes | label_col=%s",
            mode,
            len(self._manifest),
            self.num_classes,
            self._label_col,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Manifest construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_manifest(self) -> pd.DataFrame:
        """
        Crawl data_root using class_map entries and build a flat DataFrame.

        Columns:
            image_path, fine_class,
            l1 (str), l1_idx (int),
            l2 (str|None), l2_idx (int, -1 if N/A),
            l3_specialist (str|None), l3_class (str|None), l3_idx (int, -1 if N/A)
        """
        records: List[Dict] = []

        for entry in self._cfg["class_map"]:
            dir_path = self._data_root / entry["path"]
            if not dir_path.exists():
                logger.warning("Directory not found, skipping: %s", dir_path)
                continue

            # Resolve L3 index once per directory entry
            spec_key  = entry.get("l3_specialist")
            l3_class  = entry.get("l3_class")
            l3_idx    = -1
            if spec_key and l3_class:
                l3_idx = self._l3_specs.get(spec_key, {}).get(
                    "classes", {}
                ).get(l3_class, -1)

            for img_path in sorted(dir_path.iterdir()):
                if img_path.suffix.lower() not in _VALID_EXTENSIONS:
                    continue

                records.append({
                    "image_path":    str(img_path),
                    "fine_class":    entry["fine_class"],
                    "l1":            entry["l1"],
                    "l1_idx":        self._l1_labels.get(entry["l1"], -1),
                    "l2":            entry.get("l2"),
                    "l2_idx":        self._l2_labels.get(entry.get("l2"), -1)
                                     if entry.get("l2") else -1,
                    "l3_specialist": spec_key,
                    "l3_class":      l3_class,
                    "l3_idx":        l3_idx,
                })

        df = pd.DataFrame(records)
        logger.info(
            "Manifest built: %d images | %d fine classes",
            len(df),
            df["fine_class"].nunique(),
        )
        return df

    def _filter_manifest(self, mode: str) -> pd.DataFrame:
        """Return only the rows relevant to this training mode."""
        df = self._full_manifest

        if mode == "level1":
            return df.reset_index(drop=True)

        if mode == "level2":
            # Only ABNORMAL samples flow into the router
            return df[df["l1_idx"] == 1].reset_index(drop=True)

        if mode in _L3_MODE_TO_SPECIALIST:
            specialist = _L3_MODE_TO_SPECIALIST[mode]
            return df[df["l3_specialist"] == specialist].reset_index(drop=True)

        raise ValueError(
            f"Unknown mode: '{mode}'. Valid modes: "
            f"level1, level2, level3_{{macular,diabetic,vascular,fluid,structural}}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Class weight helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_class_weights(self) -> torch.Tensor:
        """
        Compute balanced inverse-frequency weights.

        weight_c = n_total / (n_classes * n_c)

        Returns shape [num_classes] float32 tensor.
        """
        labels      = self._manifest[self._label_col].values
        classes, counts = np.unique(labels, return_counts=True)
        n_total     = len(labels)
        n_classes   = len(classes)
        weights     = np.zeros(n_classes, dtype=np.float32)

        for cls, cnt in zip(classes, counts):
            if 0 <= cls < n_classes:
                weights[cls] = n_total / (n_classes * cnt)

        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weights for ``WeightedRandomSampler``.

        Returns float64 PyTorch tensor of shape [len(self)].
        """
        labels = self._manifest[self._label_col].values
        weight_map = {
            int(c): float(self.class_weights[c])
            for c in range(len(self.class_weights))
        }
        # Build as a native Python list first, then directly to torch.Tensor
        # This completely avoids the PyTorch/NumPy C++ bridge that segfaults on MPS
        weight_list = [weight_map.get(int(lbl), 1.0) for lbl in labels]
        return torch.tensor(weight_list, dtype=torch.float64)

    def get_labels(self) -> np.ndarray:
        """Return integer label array — needed for StratifiedKFold splitting."""
        return self._manifest[self._label_col].values.astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def class_names(self) -> List[str]:
        """Ordered list of class name strings for this mode."""
        if self.mode == "level1":
            return _invert_label_dict(self._l1_labels)
        if self.mode == "level2":
            return _invert_label_dict(self._l2_labels)
        specialist = _L3_MODE_TO_SPECIALIST[self.mode]
        return _invert_label_dict(self._l3_specs[specialist]["classes"])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row   = self._manifest.iloc[idx]
        label = int(row[self._label_col])

        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception as exc:
            logger.error("Failed to open %s: %s", row["image_path"], exc)
            # Return a black placeholder to avoid crashing the DataLoader
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _label_column_for_mode(mode: str) -> str:
    if mode == "level1":
        return "l1_idx"
    if mode == "level2":
        return "l2_idx"
    return "l3_idx"


def _invert_label_dict(d: Dict[str, int]) -> List[str]:
    """Convert {name: idx} → ordered [name0, name1, …] list."""
    result = [""] * len(d)
    for name, idx in d.items():
        result[idx] = name
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Stratified k-Fold DataLoader Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_kfold_dataloaders(
    config_path: str,
    mode: DataMode,
    n_splits: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Build k (train_loader, val_loader) pairs via Stratified K-Fold CV.

    Stratification is performed on the label column for the given mode,
    guaranteeing proportional minority representation in every fold.

    Args:
        config_path:          Path to hierarchy.yaml.
        mode:                 Dataset mode string.
        n_splits:             Number of folds k (default 5).
        batch_size:           Training batch size.
        num_workers:          DataLoader worker processes.
        train_transform:      Augmentation pipeline for training folds.
        val_transform:        Deterministic transform for validation folds.
        use_weighted_sampler: Oversample minority classes in training fold.
        seed:                 Reproducibility seed.

    Returns:
        List of ``(train_loader, val_loader)`` tuples, length = n_splits.
    """
    # Build the full mode-filtered dataset (no transform) to get all labels
    probe_ds = OCTHierarchicalDataset(
        config_path=config_path,
        mode=mode,
        fold_indices=None,
        transform=None,
    )
    all_labels  = probe_ds.get_labels()
    all_indices = np.arange(len(probe_ds))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_loaders: List[Tuple[DataLoader, DataLoader]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_indices, all_labels)
    ):
        logger.info(
            "Fold %d/%d — train: %d | val: %d",
            fold_idx + 1, n_splits, len(train_idx), len(val_idx),
        )

        # ── Training fold ─────────────────────────────────────────────────
        train_ds = OCTHierarchicalDataset(
            config_path=config_path,
            mode=mode,
            fold_indices=train_idx,
            transform=train_transform,
        )

        if use_weighted_sampler:
            # get_sample_weights now returns a safe, native torch.Tensor
            sample_weights_t = train_ds.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights_t,
                num_samples=len(train_ds),
                replacement=True,
            )
            shuffle_train = False  # Mutually exclusive with sampler
        else:
            sampler      = None
            shuffle_train = True

        # pin_memory is a CUDA-only optimisation; it triggers a UserWarning
        # on MPS and has no effect, so we gate it on CUDA availability.
        _pin = torch.cuda.is_available()

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=_pin,
            persistent_workers=(num_workers > 0),
            drop_last=True,   # Avoid single-sample batches → BatchNorm instability
        )

        # ── Validation fold ───────────────────────────────────────────────
        val_ds = OCTHierarchicalDataset(
            config_path=config_path,
            mode=mode,
            fold_indices=val_idx,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size * 2,  # No gradients → can double batch size
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_pin,
            persistent_workers=(num_workers > 0),
        )

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders
