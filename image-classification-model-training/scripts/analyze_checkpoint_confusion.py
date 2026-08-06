"""
scripts/analyze_checkpoint_confusion.py

Diagnostic analysis script to inspect model checkpoint predictions, confusion matrix,
predicted-class distributions, patient-level aggregated metrics, and DRUSEN logit behavior.
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.multi_head_convnext import MultiHeadConvNeXt, build_multi_head_model
from data.dataset import build_kfold_dataloaders, MultiHeadOCTDataset, DEFAULT_PATHOLOGY_CLASSES
from transforms import get_transforms
from utils.device import get_raw_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def analyze_checkpoint(checkpoint_path: str, config_path: str, fold_id: int = 0):
    logger.info(f"Loading checkpoint for diagnostic analysis: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model architecture & weights
    model = build_multi_head_model(pretrained=False, warmup=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Build validation dataloader for specified fold
    val_transforms = get_transforms("val")
    fold_loaders = build_kfold_dataloaders(
        config_path=config_path,
        mode="multi_head",
        n_splits=5,
        batch_size=32,
        num_workers=0,
        val_transform=val_transforms,
        is_ddp=False,
    )
    
    train_loader, val_loader = fold_loaders[fold_id]
    dataset = val_loader.dataset
    class_names = dataset.get_class_names("h2")
    num_classes = len(class_names)

    all_logits = []
    all_probs = []
    all_targets = []
    all_patient_ids = []
    all_source_keys = []

    logger.info(f"Evaluating validation dataset for Fold {fold_id} ({len(dataset)} samples)...")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            targets = batch["h2"].to(device) # One-hot target [B, 12]
            
            outputs = model(images)
            out_h2 = outputs["pathology"] # Logits [B, 12]
            probs_h2 = torch.softmax(out_h2, dim=1)

            all_logits.append(out_h2.cpu().numpy())
            all_probs.append(probs_h2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            if "patient_id" in batch:
                all_patient_ids.extend(batch["patient_id"])
            if "dataset_key" in batch:
                all_source_keys.extend(batch["dataset_key"])

    logits = np.concatenate(all_logits, axis=0) # [N, 12]
    probs = np.concatenate(all_probs, axis=0)   # [N, 12]
    targets = np.concatenate(all_targets, axis=0) # [N, 12]
    
    true_labels = np.argmax(targets, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    print("\n==========================================================================")
    print(" 1. SLICE-LEVEL CLASSIFICATION REPORT")
    print("==========================================================================")
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4, zero_division=0)
    print(report)

    print("\n==========================================================================")
    print(" 2. PREDICTED VS TRUE CLASS DISTRIBUTION (MEASURING CLASS COLLAPSE)")
    print("==========================================================================")
    true_counts = np.bincount(true_labels, minlength=num_classes)
    pred_counts = np.bincount(pred_labels, minlength=num_classes)
    
    print(f"{'Class Name':<15} | {'True Count':<12} | {'Pred Count':<12} | {'Ratio (Pred/True)':<18}")
    print("-" * 65)
    for idx, c_name in enumerate(class_names):
        ratio = pred_counts[idx] / max(1, true_counts[idx])
        print(f"{c_name:<15} | {true_counts[idx]:<12} | {pred_counts[idx]:<12} | {ratio:.2f}")

    print("\n==========================================================================")
    print(" 3. 12x12 CONFUSION MATRIX (ROWS=TRUE, COLS=PREDICTED)")
    print("==========================================================================")
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())

    print("\n==========================================================================")
    print(" 4. DRUSEN DIAGNOSTIC DEEP-DIVE ANALYSIS")
    print("==========================================================================")
    drusen_idx = class_names.index("DRUSEN") if "DRUSEN" in class_names else 1
    drusen_mask = (true_labels == drusen_idx)
    num_drusen = np.sum(drusen_mask)
    
    if num_drusen > 0:
        drusen_logits = logits[drusen_mask] # [N_drusen, 12]
        drusen_probs = probs[drusen_mask]
        drusen_preds = pred_labels[drusen_mask]

        # Calculate rank of DRUSEN logit for each DRUSEN sample
        # (Rank 1 means highest logit, Rank 2 means 2nd highest, etc.)
        sorted_indices = np.argsort(-drusen_logits, axis=1) # Sort descending
        ranks = np.where(sorted_indices == drusen_idx)[1] + 1

        print(f"Total True DRUSEN Validation Samples: {num_drusen}")
        print(f"Correctly Predicted DRUSEN Count:     {np.sum(drusen_preds == drusen_idx)} ({np.mean(drusen_preds == drusen_idx)*100:.2f}%)")
        print(f"Mean DRUSEN Logit (True DRUSEN):      {np.mean(drusen_logits[:, drusen_idx]):.4f}")
        print(f"Mean DRUSEN Prob (True DRUSEN):       {np.mean(drusen_probs[:, drusen_idx]):.4f}")
        print(f"Average Rank of DRUSEN Logit:         {np.mean(ranks):.2f} (Rank 1 = Best)")

        print("\nRank Distribution for True DRUSEN Samples:")
        rank_counts = np.bincount(ranks, minlength=num_classes + 1)[1:]
        for r_val, count in enumerate(rank_counts, start=1):
            if count > 0:
                print(f"  Rank {r_val:<2}: {count:<5} samples ({count/num_drusen*100:.1f}%)")

        print("\nTop Misclassification Destinations for True DRUSEN:")
        mis_preds = drusen_preds[drusen_preds != drusen_idx]
        if len(mis_preds) > 0:
            mis_counts = np.bincount(mis_preds, minlength=num_classes)
            for idx in np.argsort(-mis_counts):
                if mis_counts[idx] > 0:
                    print(f"  Misclassified as {class_names[idx]:<12}: {mis_counts[idx]:<5} samples ({mis_counts[idx]/len(mis_preds)*100:.1f}%)")
    else:
        print("No True DRUSEN validation samples found in this fold.")

    print("\n==========================================================================")
    print(" 5. PATIENT-LEVEL LOGIT AGGREGATED EVALUATION")
    print("==========================================================================")
    if len(all_patient_ids) == len(true_labels):
        df_pat = pd.DataFrame({
            "patient_id": all_patient_ids,
            "true_label": true_labels,
        })
        for c_idx in range(num_classes):
            df_pat[f"logit_{c_idx}"] = logits[:, c_idx]

        # Average logits per patient volume
        logit_cols = [f"logit_{c_idx}" for c_idx in range(num_classes)]
        grouped = df_pat.groupby("patient_id")
        
        pat_true = []
        pat_pred = []
        for pat_id, group in grouped:
            # Mode of true labels (should be single-class per patient)
            pat_true.append(group["true_label"].iloc[0])
            avg_logits = group[logit_cols].mean(axis=0).values
            pat_pred.append(np.argmax(avg_logits))

        pat_true = np.array(pat_true)
        pat_pred = np.array(pat_pred)
        
        pat_macro_f1 = f1_score(pat_true, pat_pred, average="macro", zero_division=0)
        pat_acc = np.mean(pat_true == pat_pred)
        print(f"Total Unique Validation Patients: {len(np.unique(all_patient_ids))}")
        print(f"Patient-Level Macro-F1 Score:     {pat_macro_f1:.4f}")
        print(f"Patient-Level Accuracy:         {pat_acc*100:.2f}%")
        
        pat_report = classification_report(pat_true, pat_pred, target_names=class_names, digits=4, zero_division=0)
        print("\nPatient-Level Classification Report:")
        print(pat_report)
    else:
        print("Patient IDs not provided in dataset batch; skipping volume aggregation.")

    print("==========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model checkpoint predictions & DRUSEN logit distribution")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--config", type=str, default="image-classification-model-training/config/hierarchy.yaml", help="Path to config file")
    parser.add_argument("--fold", type=int, default=0, help="Fold ID to evaluate")
    args = parser.parse_args()

    analyze_checkpoint(args.checkpoint, args.config, args.fold)
