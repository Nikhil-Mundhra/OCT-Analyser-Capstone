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

from data.dataset import MultiHeadOCTDataset

def extract_patient_id(image_path_str: str) -> str:
    """
    Extract exact patient ID / scan group from image path string across all OCT dataset sources.
    Handles OCT2017, OCTDL, Chiu BOE 2014, 108503_OCTID, CHU_MH, and OCT5K.
    """
    stem = Path(image_path_str).stem
    p_str = str(image_path_str)
    
    # 1. OCT2017: CNV-5557306-155.jpeg -> OCT2017_5557306
    m = re.search(r'(?:CNV|DRUSEN|DME|NORMAL)-(\d+)-\d+', stem, re.IGNORECASE)
    if m: return f"OCT2017_{m.group(1)}"
        
    # 2. OCTDL / OCT-datasets with <class>_<patientID>_<sliceNum>.jpg
    m = re.search(r'(?:rvo|rao|erm|vid|dme|no|amd|cnv|drusen)_(\d+)_\d+', stem, re.IGNORECASE)
    if m: return f"OCTDL_{m.group(1)}"

    # 3. OCTID (108503_OCTID): AMRD37, DR105, MH93, CSR7, NORMAL67
    m = re.search(r'(?:AMRD|AMD|DR|MH|CSR|NORMAL)(\d+)', stem, re.IGNORECASE)
    if m and not stem.startswith("Subject"): return f"OCTID_{m.group(0)}"

    # 4. Chiu BOE 2014: Subject_05_slice_028.png -> Chiu_Subject_05
    m = re.search(r'(Subject_\d+)', stem, re.IGNORECASE)
    if m: return f"Chiu_{m.group(1)}"

    # 5. CHU_MH: MH_surgery_others_219_V -> CHU_MH_219
    m = re.search(r'MH.*_(\d+)_[A-Z]', stem, re.IGNORECASE)
    if m: return f"CHU_MH_{m.group(1)}"

    # 6. OCT5K: e.g. Normal_Part1_Normal_26.E2E...
    if "OCT5K" in p_str:
        m = re.search(r'(\d+)\.E2E', stem)
        if m: return f"OCT5K_{m.group(1)}"
        return f"OCT5K_{stem[:15]}"

    return f"RAW_{Path(image_path_str).parent.name}_{stem}"


def audit_dataset(config_path: str, data_root: str):
    print("=" * 80)
    print("CORRECTED AUDIT: TARGET SEMANTICS & PATIENT GROUPING (STRATIFIED GROUP K-FOLD)")
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

    # 3. Patient Grouping Analysis
    print("\n--- 3. Patient Grouping & Distinct Patient Counts ---")
    manifest["patient_id"] = manifest["image_path"].apply(extract_patient_id)
    distinct_patients = manifest["patient_id"].nunique()
    print(f"Total Distinct Patient Groups  : {distinct_patients:,}")

    patient_class_df = manifest[manifest["granular_idx"] != -1].groupby("granular_idx")["patient_id"].nunique()
    print("\nTrue Distinct Positive Patient Groups Per Class:")
    for idx, name in enumerate(class_names):
        p_cnt = patient_class_df.get(idx, 0)
        img_cnt = counts.get(idx, 0)
        print(f"  [{idx:2d}] {name:<15} : {p_cnt:>5,} distinct patients ({img_cnt:>6,} images)")

    # 4. Pathology Co-occurrence Analysis (Across Patient Scans)
    print("\n--- 4. Co-occurrence Matrix Between Pathologies (Patient-Level) ---")
    abnormal_df = manifest[manifest["granular_idx"] != -1]
    patient_matrix = pd.crosstab(abnormal_df["patient_id"], abnormal_df["granular_idx"])
    for i in range(num_classes):
        if i not in patient_matrix.columns:
            patient_matrix[i] = 0
    patient_matrix = patient_matrix[range(num_classes)]
    patient_matrix_binary = (patient_matrix > 0).astype(int)

    co_occurrence = np.dot(patient_matrix_binary.T, patient_matrix_binary)
    co_df = pd.DataFrame(co_occurrence, index=class_names, columns=class_names)
    print(co_df.to_string())

    # 5. Patient-Level Stratified Group K-Fold Audit Table
    print("\n--- 5. Corrected Patient-Level Stratified Group K-Fold Audit Table ---")
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
