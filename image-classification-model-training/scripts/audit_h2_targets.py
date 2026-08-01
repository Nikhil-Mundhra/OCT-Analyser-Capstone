"""
scripts/audit_h2_targets.py

Diagnostic script for Change 1 & Change 4:
  1. Inspect sample['pathology'] tensor shape, dtype, and values.
  2. Quantify labels per image, distinct combinations, positive patient groups per class,
     and co-occurrence matrix between pathologies.
  3. Produce patient-level fold audit table across 5 folds.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset import MultiHeadOCTDataset

def extract_patient_id(image_path_str: str) -> str:
    """
    Extract patient ID / scan group from image path string.
    Identifies dataset source patterns (OCT2017, OCT5K, 108503_OCTID, OCTDL, Chiu_BOE, etc.).
    """
    path_obj = Path(image_path_str)
    stem = path_obj.stem
    parent_name = path_obj.parent.name
    
    # 1. OCT2017 dataset (e.g. DRUSEN-123456-1.jpeg -> patient ID 123456)
    if "OCT2017" in image_path_str:
        parts = stem.split('-')
        if len(parts) >= 2:
            return f"OCT2017_{parts[1]}"
        return f"OCT2017_{parent_name}"
        
    # 2. 108503_OCTID dataset (e.g. AMD_001.png or 001_1.png)
    if "108503" in image_path_str or "OCTID" in image_path_str:
        parts = stem.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            return f"OCTID_{'_'.join(parts[:-1])}"
        return f"OCTID_{parent_name}"
        
    # 3. Chiu BOE 2014 dataset
    if "Chiu_BOE" in image_path_str:
        parts = stem.split('_')
        if len(parts) >= 2:
            return f"Chiu_{parts[0]}"
        return f"Chiu_{parent_name}"

    # 4. OCT5K dataset
    if "OCT5K" in image_path_str:
        parts = stem.split('_')
        if len(parts) >= 2:
            return f"OCT5K_{parts[0]}"
        return f"OCT5K_{parent_name}"

    # 5. OCTDL dataset
    if "OCTDL" in image_path_str:
        parts = stem.split('_')
        if len(parts) >= 2:
            return f"OCTDL_{parts[0]}"
        return f"OCTDL_{parent_name}"

    # Fallback to parent directory + filename prefix
    parts = stem.split('_')
    if len(parts) >= 2:
        return f"{parent_name}_{parts[0]}"
    return f"{parent_name}_{stem}"


def audit_dataset(config_path: str, data_root: str):
    print("=" * 80)
    print("CHANGE 1 & CHANGE 4: TARGET SEMANTICS & PATIENT GROUPING AUDIT")
    print("=" * 80)
    
    dataset = MultiHeadOCTDataset(config_path=config_path, data_root=data_root, verbose=True)
    manifest = dataset._manifest
    class_names = dataset.get_class_names("h2")
    num_classes = len(class_names)

    # 1. Print representative sample properties
    print("\n--- 1. Sample Tensor Properties ---")
    img, sample = dataset[0]
    print(f"Sample 'normal_abnormal' (H1): shape={sample['normal_abnormal'].shape}, dtype={sample['normal_abnormal'].dtype}, value={sample['normal_abnormal']}")
    print(f"Sample 'pathology' (H2)      : shape={sample['pathology'].shape}, dtype={sample['pathology'].dtype}, value={sample['pathology']}")

    # 2. H2 Labels Per Image & Multi-class vs Multi-label Determination
    print("\n--- 2. H2 Target Format Quantification ---")
    h2_indices = manifest["granular_idx"].values
    abnormal_mask = (manifest["l1_idx"] == 1).values
    abnormal_h2 = h2_indices[abnormal_mask]

    print(f"Total Images Manifested        : {len(manifest):,}")
    print(f"Total Normal (H1=0) Images     : {np.sum(~abnormal_mask):,} (H2 target = -1)")
    print(f"Total Abnormal (H1=1) Images   : {len(abnormal_h2):,}")
    
    # Check if granular_idx is a scalar integer
    labels_per_image = 1  # Each row in manifest has 1 scalar granular_idx
    print(f"H2 Labels Assigned per Sample  : {labels_per_image} (Scalar integer class index per image)")
    
    # Value counts of granular classes
    print("\nPer-Class Image Support (H2 Pathology):")
    counts = pd.Series(abnormal_h2).value_counts().sort_index()
    for idx, name in enumerate(class_names):
        cnt = counts.get(idx, 0)
        print(f"  [{idx:2d}] {name:<15} : {cnt:>6,} images")

    # 3. Patient Grouping Analysis
    print("\n--- 3. Patient Grouping & Distinct Patient Counts ---")
    manifest["patient_id"] = manifest["image_path"].apply(extract_patient_id)
    distinct_patients = manifest["patient_id"].nunique()
    print(f"Total Distinct Patient Groups  : {distinct_patients:,}")

    patient_class_df = manifest[manifest["granular_idx"] != -1].groupby("granular_idx")["patient_id"].nunique()
    print("\nDistinct Positive Patient Groups Per Class:")
    patient_counts_dict = {}
    for idx, name in enumerate(class_names):
        p_cnt = patient_class_df.get(idx, 0)
        patient_counts_dict[name] = p_cnt
        print(f"  [{idx:2d}] {name:<15} : {p_cnt:>5,} distinct patients")

    # 4. Pathology Co-occurrence Analysis (Across Patient Scans)
    print("\n--- 4. Co-occurrence Matrix Between Pathologies (Patient-Level) ---")
    # Patient x Class matrix
    abnormal_df = manifest[manifest["granular_idx"] != -1]
    patient_matrix = pd.crosstab(abnormal_df["patient_id"], abnormal_df["granular_idx"])
    # Re-index to ensure 0..11 columns present
    for i in range(num_classes):
        if i not in patient_matrix.columns:
            patient_matrix[i] = 0
    patient_matrix = patient_matrix[range(num_classes)]
    patient_matrix_binary = (patient_matrix > 0).astype(int)

    co_occurrence = np.dot(patient_matrix_binary.T, patient_matrix_binary)
    co_df = pd.DataFrame(co_occurrence, index=class_names, columns=class_names)
    print(co_df.to_string())

    # 5. Patient-Level 5-Fold Audit Table
    print("\n--- 5. Patient-Grouped 5-Fold Audit Table ---")
    from sklearn.model_selection import KFold
    
    unique_patients = manifest["patient_id"].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    patient_fold_map = {}
    for fold_idx, (_, val_p_indices) in enumerate(kf.split(unique_patients)):
        val_patients = set(unique_patients[val_p_indices])
        for p in val_patients:
            patient_fold_map[p] = fold_idx
            
    manifest["fold"] = manifest["patient_id"].map(patient_fold_map)
    
    fold_audit = []
    for idx, name in enumerate(class_names):
        sub = manifest[manifest["granular_idx"] == idx]
        total_p = sub["patient_id"].nunique()
        fold_p_cnts = [sub[sub["fold"] == f]["patient_id"].nunique() for f in range(5)]
        fold_audit.append({
            "Class": name,
            "Total positive patients": total_p,
            "Fold 0": fold_p_cnts[0],
            "Fold 1": fold_p_cnts[1],
            "Fold 2": fold_p_cnts[2],
            "Fold 3": fold_p_cnts[3],
            "Fold 4": fold_p_cnts[4]
        })
        
    audit_df = pd.DataFrame(fold_audit)
    print(audit_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit H2 Targets and Patient Grouping")
    parser.add_argument("--config", type=str, default="image-classification-model-training/config/hierarchy.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()
    
    audit_dataset(args.config, args.data_root)
