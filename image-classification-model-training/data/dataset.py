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
from PIL import Image
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
        manifest: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.transform = transform

        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        default_root = os.environ.get("OCT_DATA_ROOT", self._cfg.get("data_root", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed"))
        self._data_root = Path(data_root) if data_root else Path(default_root)
        self._l1_labels = self._cfg["l1_labels"]
        self._l2_labels = self._cfg["l2_labels"]
        self._l3_specs  = self._cfg.get("l3_specialists", {})
        self._granular_classes = self._cfg.get("granular_classes", {})

        if manifest is not None:
            self._manifest = manifest.copy().reset_index(drop=True)
        else:
            self._manifest = self._build_manifest()

        if fold_indices is not None:
            self._manifest = self._manifest.iloc[fold_indices].reset_index(drop=True)

        rank = int(os.environ.get("RANK", 0))
        if verbose and rank == 0:
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
                    "dataset_key": entry.get("dataset_key", "UNKNOWN"),
                    "l1_idx": self._l1_labels.get(entry["l1"], 0),
                    "l2_idx": self._l2_labels.get(entry.get("l2"), -1) if entry.get("l2") else -1,
                    "granular_idx": self._granular_classes.get(l3_class, -1) if l3_class else -1,
                    "spec_key": spec_key,
                    "l3_class": l3_class
                })
        if not records:
            raise ValueError(
                f"Dataset manifest is empty (0 images found)! "
                f"Please check OCT_DATA_ROOT='{self._data_root}'. "
                f"Ensure it points to the folder containing subfolders like 'Normal (Healthy)', 'AMD', etc."
            )
        return pd.DataFrame(records)

    def get_class_names(self, target="h2") -> List[str]:
        """Returns ordered list of class name strings for a target level."""
        if target in ("h2", "l2", "pathology", "granular"):
            if self._granular_classes:
                sorted_classes = sorted(self._granular_classes.items(), key=lambda x: x[1])
                return [k for k, _ in sorted_classes]
            elif self._l2_labels:
                sorted_l2 = sorted(self._l2_labels.items(), key=lambda x: x[1])
                return [k for k, _ in sorted_l2]
        elif target in ("h1", "l1", "normal_abnormal"):
            sorted_l1 = sorted(self._l1_labels.items(), key=lambda x: x[1])
            return [k for k, _ in sorted_l1]
        return []

    def compute_class_weights(self, target="l2") -> torch.Tensor:
        """
        Computes inverse-frequency class weights for a given target level.
        For 'h2', calculates inverse-square-root weights from UNIQUE TRAINING PATIENT COUNTS.
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
        elif target in ("h2", "granular", "pathology"):
            df_abnormal = self._manifest[self._manifest['granular_idx'] != -1].copy()
            if df_abnormal.empty:
                return torch.ones(len(self._granular_classes))
            
            # Ensure namespaced patient IDs are assigned
            if "patient_id" not in df_abnormal.columns:
                class_names = self.get_class_names("h2")
                def _row_pid(row):
                    g_idx = row["granular_idx"]
                    c_name = class_names[g_idx] if (0 <= g_idx < len(class_names)) else "NORMAL"
                    d_key = row.get("dataset_key", "UNKNOWN")
                    return extract_patient_id(row["image_path"], d_key, c_name)
                df_abnormal["patient_id"] = df_abnormal.apply(_row_pid, axis=1)

            # Compute weights based on UNIQUE TRAINING PATIENT COUNTS per class
            patient_counts = df_abnormal.groupby("granular_idx")["patient_id"].nunique().sort_index()
            num_classes = len(self._granular_classes)
            for i in range(num_classes):
                if i not in patient_counts:
                    patient_counts[i] = 1
            counts = patient_counts.sort_index().values.astype(np.float32)
            counts = np.maximum(counts, 1.0)
            weights = 1.0 / np.sqrt(counts)
            mean_w = float(np.mean(weights))
            weights = weights / max(1e-8, mean_w)
            weights = np.clip(weights, 0.4, 4.0)
            return torch.tensor(weights, dtype=torch.float32)
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

        if self.transform is not None:
            try:
                image = self.transform(image_path)
                if hasattr(image, "as_tensor"):
                    image = image.as_tensor()
                elif isinstance(image, np.ndarray):
                    image = torch.from_numpy(image)
            except Exception as exc:
                logger.warning("MONAI LoadImage failed for %s: %s. Retrying with PIL.", image_path, exc)
                try:
                    pil_img = Image.open(image_path).convert("RGB")
                    image = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
                except Exception as exc2:
                    logger.error("PIL fallback also failed for %s: %s. Using placeholder.", image_path, exc2)
                    image = torch.zeros(3, 384, 384, dtype=torch.float32)
        else:
            pil_img = Image.open(image_path).convert("RGB")
            image = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        targets = {
            "normal_abnormal": torch.tensor([float(row["l1_idx"])], dtype=torch.float32),
            "pathology": torch.tensor(row["granular_idx"], dtype=torch.int64)
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

import re

def extract_patient_id(image_path_str: str, dataset_key: str = "UNKNOWN", pathology_class: str = "UNKNOWN") -> str:
    """
    Extract exact local patient ID from image path string using dataset_key and pathology_class namespacing.
    Format: dataset_key::pathology_class::local_patient_id
    Prevents false cross-dataset or cross-class patient ID collisions in StratifiedGroupKFold.
    """
    path_obj = Path(image_path_str)
    stem = path_obj.stem
    
    # 1. OCT2017: CNV-5557306-155.jpeg -> OCT2017::CNV::5557306
    m = re.search(r'(?:CNV|DRUSEN|DME|NORMAL)-(\d+)-\d+', stem, re.IGNORECASE)
    if m: return f"{dataset_key}::{pathology_class}::{m.group(1)}"
        
    # 2. OCTDL / OCT-datasets with <class>_<patientID>_<sliceNum>.jpg
    m = re.search(r'(?:rvo|rao|erm|vid|dme|no|amd|cnv|drusen)_(\d+)_\d+', stem, re.IGNORECASE)
    if m: return f"{dataset_key}::{pathology_class}::{m.group(1)}"

    # 3. OCTID (108503_OCTID): AMRD37, DR105, MH93, CSR7, NORMAL67
    m = re.search(r'(?:AMRD|AMD|DR|MH|CSR|NORMAL)(\d+)', stem, re.IGNORECASE)
    if m and not stem.startswith("Subject"): return f"{dataset_key}::{pathology_class}::{m.group(0)}"

    # 4. Chiu BOE 2014: Subject_05_slice_028.png -> CHIU_BOE::DME::Subject_05
    m = re.search(r'(Subject_\d+)', stem, re.IGNORECASE)
    if m: return f"{dataset_key}::{pathology_class}::{m.group(1)}"

    # 5. CHU_MH: MH_surgery_others_219_V -> CHU_MH::MH::219
    m = re.search(r'MH.*_(\d+)_[A-Z]', stem, re.IGNORECASE)
    if m: return f"{dataset_key}::{pathology_class}::{m.group(1)}"

    # 6. OCT5K: e.g. Normal_Part1_Normal_26.E2E...
    if "OCT5K" in str(image_path_str):
        m = re.search(r'(\d+)\.E2E', stem)
        if m: return f"{dataset_key}::{pathology_class}::{m.group(1)}"
        return f"{dataset_key}::{pathology_class}::{stem[:15]}"

    return f"{dataset_key}::{pathology_class}::{stem}"

def build_kfold_dataloaders(
    config_path: str,
    mode: str,
    n_splits: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform = None,
    val_transform = None,
    use_weighted_sampler: bool = False,
    seed: int = 42,
    is_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> List[Tuple[DataLoader, DataLoader]]:
    dataset = MultiHeadOCTDataset(config_path=config_path, verbose=(rank == 0))
    full_manifest = dataset._manifest.copy()
    
    # Extract namespaced patient IDs and use StratifiedGroupKFold for patient-grouped splitting
    class_names = dataset.get_class_names("h2")
    def _row_patient_id(row):
        g_idx = row["granular_idx"]
        c_name = class_names[g_idx] if (0 <= g_idx < len(class_names)) else "NORMAL"
        d_key = row.get("dataset_key", "UNKNOWN")
        return extract_patient_id(row["image_path"], d_key, c_name)

    full_manifest["patient_id"] = full_manifest.apply(_row_patient_id, axis=1)
    
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    fold_loaders = []
    
    for train_idx, val_idx in sgkf.split(full_manifest, full_manifest["granular_idx"], full_manifest["patient_id"]):
        train_ds = MultiHeadOCTDataset(
            config_path=config_path,
            manifest=full_manifest.iloc[train_idx],
            transform=train_transform,
            verbose=False
        )
        val_ds = MultiHeadOCTDataset(
            config_path=config_path,
            manifest=full_manifest.iloc[val_idx],
            transform=val_transform,
            verbose=False
        )
        
        _pin = torch.cuda.is_available() or torch.backends.mps.is_available()
        if is_ddp and torch.cuda.is_available():
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=_pin,
                persistent_workers=(num_workers > 0)
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=_pin,
                persistent_workers=(num_workers > 0)
            )
        else:
            if use_weighted_sampler:
                from torch.utils.data import WeightedRandomSampler
                granular_indices = train_ds._manifest['granular_idx'].values
                # Map -1 (NORMAL) to index 12 so bincount handles 12 pathology classes + 1 NORMAL class
                valid_indices = np.where(granular_indices >= 0, granular_indices, 12)
                class_counts = np.bincount(valid_indices, minlength=13)
                class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
                # Apply 1.5x Priority Multiplier for ultra-minority pathology classes (< 30 samples)
                for c_idx in range(12):
                    if class_counts[c_idx] < 30:
                        class_weights[c_idx] *= 1.5
                sample_weights = torch.FloatTensor([class_weights[t] for t in valid_indices])
                train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    num_workers=num_workers,
                    pin_memory=_pin,
                    persistent_workers=(num_workers > 0)
                )
            else:
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

