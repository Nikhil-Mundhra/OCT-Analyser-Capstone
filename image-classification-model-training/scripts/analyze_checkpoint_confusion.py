"""
scripts/analyze_checkpoint_confusion.py

Diagnostic script for Change 5:
  Loads a saved checkpoint (e.g. fold0_best_model.pth), evaluates it against
  the validation split, and generates a detailed 12x12 Confusion Matrix & Class Error Breakdown.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset import build_kfold_dataloaders, MultiHeadOCTDataset
from data.transforms import get_transforms
from models.multi_head_convnext import build_multi_head_model

def analyze_confusion(config_path: str, checkpoint_path: str, data_root: str = None):
    print("=" * 80)
    print("CHANGE 5: CHECKPOINT CONFUSION MATRIX & ERROR BREAKDOWN")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file does not exist: {checkpoint_path}")
        return

    full_ds = MultiHeadOCTDataset(config_path=config_path, data_root=data_root, transform=None)
    class_names = full_ds.get_class_names("h2")
    num_classes = len(class_names)

    # 1. Build validation dataloader for Fold 0
    val_transforms = get_transforms("val")
    fold_loaders = build_kfold_dataloaders(
        config_path=config_path,
        mode="multi_head",
        n_splits=5,
        batch_size=32,
        num_workers=2,
        val_transform=val_transforms
    )
    _, val_loader = fold_loaders[0]

    # 2. Load model and checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Using compute device: {device}")

    model = build_multi_head_model(pretrained=False, warmup=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # Strip module. prefix if DDP/DataParallel
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 3. Collect predictions & targets
    all_preds = []
    all_targets = []

    print("\nRunning inference over Fold 0 validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            targets = labels["pathology"].to(device)

            logits_dict = model(images)
            logits = logits_dict["pathology"]

            num_h2_classes = logits.size(-1)
            valid_mask = ((labels["normal_abnormal"] == 1).view(-1)) & (targets >= 0) & (targets < num_h2_classes)

            if valid_mask.sum() > 0:
                valid_targets = targets[valid_mask]
                valid_logits = logits[valid_mask]
                preds = torch.argmax(valid_logits, dim=1)

                all_targets.extend(valid_targets.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

    print(f"Evaluated {len(all_targets):,} abnormal validation samples.")

    # 4. Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])

    print("\n--- 12x12 Confusion Matrix (Rows = True Class, Columns = Predicted Class) ---")
    print(cm_df.to_string())

    # 5. Answer User Diagnostic Questions
    print("\n" + "=" * 80)
    print("DIAGNOSTIC QUESTION ANSWERS:")
    print("=" * 80)

    # Q1: What is DRUSEN being predicted as?
    drusen_idx = class_names.index("DRUSEN")
    drusen_row = cm[drusen_idx]
    drusen_total = np.sum(drusen_row)
    print(f"\n1. DRUSEN (True Total: {drusen_total:,}):")
    if drusen_total > 0:
        for idx, count in enumerate(drusen_row):
            pct = 100.0 * count / drusen_total
            print(f"   -> Predicted as {class_names[idx]:<15}: {count:>5d} ({pct:5.1f}%)")

    # Q2: Which true classes are being predicted as DME?
    dme_idx = class_names.index("DME")
    dme_col = cm[:, dme_idx]
    dme_total_pred = np.sum(dme_col)
    print(f"\n2. DME False Positives (Total Predicted as DME: {dme_total_pred:,}):")
    for idx, count in enumerate(dme_col):
        if count > 0:
            print(f"   <- True {class_names[idx]:<15}: {count:>5d} images")

    # Q3: What is CNV being predicted as?
    cnv_idx = class_names.index("CNV")
    cnv_row = cm[cnv_idx]
    cnv_total = np.sum(cnv_row)
    print(f"\n3. CNV Predictions Breakdown (True Total: {cnv_total:,}):")
    if cnv_total > 0:
        for idx, count in enumerate(cnv_row):
            pct = 100.0 * count / cnv_total
            print(f"   -> Predicted as {class_names[idx]:<15}: {count:>5d} ({pct:5.1f}%)")

    print("\n--- Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Checkpoint Confusion Matrix")
    parser.add_argument("--config", type=str, default="image-classification-model-training/config/hierarchy.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pth)")
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    analyze_confusion(args.config, args.checkpoint, args.data_root)
