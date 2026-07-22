"""
data/dataset.py

MultiHeadOCTDataset — PyTorch Dataset for the Multi-Head ConvNeXt architecture.
Parses `config/hierarchy.yaml` and maps files to structured multi-head tensors.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)
_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

class MultiHeadOCTDataset(Dataset):
    """
    Dataset for the Multi-Head OCT pipeline.
    
    Args:
        config_path: Path to hierarchy.yaml.
        data_root: Optional override for the root data directory.
        fold_indices: Numpy integer array of row indices for this fold.
        transform: MONAI transform pipeline (must accept file paths, as LoadImage is used).
    """
    def __init__(
        self,
        config_path: str,
        data_root: Optional[str] = None,
        fold_indices: Optional[np.ndarray] = None,
        transform = None,
    ) -> None:
        super().__init__()
        self.transform = transform

        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        default_root = os.environ.get("OCT_DATA_ROOT", self._cfg.get("data_root", ""))
        self._data_root = Path(data_root) if data_root else Path(default_root)
        self._l1_labels = self._cfg["l1_labels"]
        self._l2_labels = self._cfg["l2_labels"]
        self._l3_specs  = self._cfg.get("l3_specialists", {})
        self._granular_classes = self._cfg.get("granular_classes", {})

        self._manifest: pd.DataFrame = self._build_manifest()

        if fold_indices is not None:
            self._manifest = self._manifest.iloc[fold_indices].reset_index(drop=True)

        logger.info(
            "MultiHeadOCTDataset initialized: %d samples.",
            len(self._manifest),
        )

    def _build_manifest(self) -> pd.DataFrame:
        records = []
        for entry in self._cfg["class_map"]:
            dir_path = self._data_root / entry["path"]
            if not dir_path.exists():
                logger.debug(f"Skipping {dir_path} as it does not exist.")
                continue

            spec_key = entry.get("l3_specialist")
            l3_class = entry.get("l3_class")

            for img_path in sorted(dir_path.rglob('*')):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in _VALID_EXTENSIONS:
                    continue

                records.append({
                    "image_path": str(img_path),
                    "l1_idx": self._l1_labels.get(entry["l1"], 0),
                    "l2_idx": self._l2_labels.get(entry.get("l2"), -1) if entry.get("l2") else -1,
                    "granular_idx": self._granular_classes.get(l3_class, -1) if l3_class else -1,
                    "spec_key": spec_key,
                    "l3_class": l3_class
                })
        return pd.DataFrame(records)

    def compute_class_weights(self, target="l2") -> torch.Tensor:
        """
        Computes inverse-frequency class weights for a given target level.
        target: 'l1', 'l2', or a specialist key (e.g., 'Macular')
        """
        if target == "l1":
            counts = self._manifest['l1_idx'].value_counts().sort_index()
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(counts)
            return torch.tensor(weights.values, dtype=torch.float32)
        elif target == "l2":
            # Exclude Normal (-1) for L2 weights
            df_abnormal = self._manifest[self._manifest['l2_idx'] != -1]
            if df_abnormal.empty:
                return torch.ones(5)
            counts = df_abnormal['l2_idx'].value_counts().sort_index()
            # Ensure all 5 classes are represented in counts
            for i in range(5):
                if i not in counts:
                    counts[i] = 1 # avoid inf
            counts = counts.sort_index()
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(counts)
            return torch.tensor(weights.values, dtype=torch.float32)
        elif target == "h2":
            df_abnormal = self._manifest[self._manifest['granular_idx'] != -1]
            if df_abnormal.empty:
                return torch.ones(len(self._granular_classes))
            counts = df_abnormal['granular_idx'].value_counts().sort_index()
            num_classes = len(self._granular_classes)
            for i in range(num_classes):
                if i not in counts:
                    counts[i] = 1
            counts = counts.sort_index()
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(counts)
            return torch.tensor(weights.values, dtype=torch.float32)
        else:
            df_spec = self._manifest[self._manifest['spec_key'] == target]
            if df_spec.empty:
                return torch.ones(1)
            # Map l3_class string to index
            class_map = self._l3_specs[target]["classes"]
            class_indices = df_spec['l3_class'].map(class_map)
            counts = class_indices.value_counts().sort_index()
            num_classes = len(class_map)
            for i in range(num_classes):
                if i not in counts:
                    counts[i] = 1
            counts = counts.sort_index()
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(counts)
            return torch.tensor(weights.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self._manifest.iloc[idx]
        image_path = row["image_path"]

        # MONAI pipelines usually start with LoadImage which takes the file path
        if self.transform is not None:
            try:
                # Apply pipeline
                image = self.transform(image_path)
                # If MONAI returns a MetaTensor, convert to standard torch.Tensor
                if hasattr(image, "as_tensor"):
                    image = image.as_tensor()
                elif isinstance(image, np.ndarray):
                    image = torch.from_numpy(image)
            except Exception as exc:
                logger.error("Failed to process %s: %s", image_path, exc)
                # Return black placeholder for stability
                image = torch.zeros(3, 384, 384, dtype=torch.float32)
        else:
            image = image_path

        # ── Target H1: Binary (Normal=0, Abnormal=1) ──
        # Cast to float32 for BCEWithLogitsLoss
        h1 = torch.tensor([row["l1_idx"]], dtype=torch.float32)

        # ── Target H2: Granular Pathology (Multi-class 0-11, or -1 for Normal) ──
        h2 = torch.tensor(row["granular_idx"], dtype=torch.long)

        targets = {
            "normal_abnormal": h1,
            "pathology": h2
        }

        return image, targets

def build_dataloader(
    config_path: str,
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    transform = None,
    shuffle: bool = True
) -> DataLoader:
    """
    Utility to quickly build a DataLoader (e.g. for the micro_dataset sanity test).
    """
    dataset = MultiHeadOCTDataset(
        config_path=config_path,
        data_root=data_root,
        transform=transform
    )
    
    _pin = torch.cuda.is_available()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=_pin,
        persistent_workers=(num_workers > 0)
    )
    return loader

def build_kfold_dataloaders(
    config_path: str,
    mode: str,
    n_splits: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform = None,
    val_transform = None,
    use_weighted_sampler: bool = False,
    seed: int = 42
) -> List[Tuple[DataLoader, DataLoader]]:
    dataset = MultiHeadOCTDataset(config_path=config_path)
    
    # Stratify by l1 for level1, l2 for level2, etc. (simplifying here to l1)
    labels = [row["l1_idx"] for _, row in dataset._manifest.iterrows()]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_loaders = []
    
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_ds = MultiHeadOCTDataset(config_path=config_path, fold_indices=train_idx, transform=train_transform)
        val_ds = MultiHeadOCTDataset(config_path=config_path, fold_indices=val_idx, transform=val_transform)
        
        _pin = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=_pin,
            persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_pin,
            persistent_workers=(num_workers > 0)
        )
        
        fold_loaders.append((train_loader, val_loader))
        
    return fold_loaders

