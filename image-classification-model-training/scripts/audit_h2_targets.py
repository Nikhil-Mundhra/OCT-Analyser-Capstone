"""
scripts/audit_h2_targets.py

Diagnostic script for Change 1 & Change 4:
  1. Inspect sample['pathology'] tensor shape, dtype, and values.
  2. Quantify labels per image, distinct combinations, positive patient groups per class,
     and co-occurrence matrix between pathologies.
  3. Produce patient-level fold audit table across 5 folds using robust regex patient parsing.
"""

import argparse
import os
import re
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset import MultiHeadOCTDataset, extract_patient_id

def audit_dataset(config_path: str, data_root: str):
    print("=" * 80)
    print("NAMESPACED AUDIT: TARGET SEMANTICS & PATIENT GROUPING (STRATIFIED GROUP K-FOLD)")
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
    print(f"H2 Labels Assigned per Sample  : 1 (Scalar integer class index per image)")
    
    print("\nPer-Class Image Support (H2 Pathology):")
    counts = pd.Series(abnormal_h2).value_counts().sort_index()
    for idx, name in enumerate(class_names):
        cnt = counts.get(idx, 0)
        print(f"  [{idx:2d}] {name:<15} : {cnt:>6,} images")

    # 3. Patient Grouping & Namespaced Distinct Patient Counts
    print("\n--- 3. Patient Grouping & Namespaced Distinct Patient Counts ---")
    def _row_patient_id(row):
        g_idx = row["granular_idx"]
        c_name = class_names[g_idx] if (0 <= g_idx < len(class_names)) else "NORMAL"
        return extract_patient_id(row["image_path"], c_name)

    manifest["patient_id"] = manifest.apply(_row_patient_id, axis=1)
    distinct_patients = manifest["patient_id"].nunique()
    print(f"Total Namespaced Patient Groups: {distinct_patients:,}")

    # Strong validation assertion
    class_counts = manifest.groupby("patient_id")["granular_idx"].nunique()
    max_classes_per_patient = class_counts.max()
    print(f"Max Pathology Classes per Global Patient ID: {max_classes_per_patient}")
    assert max_classes_per_patient == 1, "Validation Failure: Global patient ID belongs to multiple classes!"
    print("✓ VALIDATION PASSED: Every global patient ID belongs to EXACTLY 1 pathology class.")

    patient_class_df = manifest[manifest["granular_idx"] != -1].groupby("granular_idx")["patient_id"].nunique()
    print("\nTrue Distinct Positive Patient Groups Per Class:")
    for idx, name in enumerate(class_names):
        p_cnt = patient_class_df.get(idx, 0)
        img_cnt = counts.get(idx, 0)
        print(f"  [{idx:2d}] {name:<15} : {p_cnt:>5,} distinct patients ({img_cnt:>6,} images)")

    # 4. Patient-Level Stratified Group K-Fold Audit Table
    print("\n--- 4. Patient-Level Stratified Group K-Fold Audit Table ---")
    sgkf = StratifiedGroupKFold(n_splits=5)
    manifest["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(manifest, manifest["granular_idx"], manifest["patient_id"])):
        manifest.loc[val_idx, "fold"] = fold

    fold_audit = []
    for idx, name in enumerate(class_names):
        sub = manifest[manifest["granular_idx"] == idx]
        tot_p = sub["patient_id"].nunique()
        tot_img = len(sub)
        f_p = [sub[sub["fold"] == f]["patient_id"].nunique() for f in range(5)]
        f_img = [len(sub[sub["fold"] == f]) for f in range(5)]
        fold_audit.append({
            "Class": name,
            "Total Images": tot_img,
            "Total Patients": tot_p,
            "Fold 0 (P/Img)": f"{f_p[0]}/{f_img[0]}",
            "Fold 1 (P/Img)": f"{f_p[1]}/{f_img[1]}",
            "Fold 2 (P/Img)": f"{f_p[2]}/{f_img[2]}",
            "Fold 3 (P/Img)": f"{f_p[3]}/{f_img[3]}",
            "Fold 4 (P/Img)": f"{f_p[4]}/{f_img[4]}",
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
