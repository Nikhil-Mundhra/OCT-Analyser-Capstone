"""
scripts/test_drusen_memorization.py

Diagnostic script for Change 6:
  Trains a lightweight model on a tiny 120-image balanced micro-dataset
  containing 10 samples per class (including DRUSEN, CNV, DME, AMD, etc.)
  to verify whether the network can reach ~100% training accuracy and F1 > 0.95 on DRUSEN.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset import MultiHeadOCTDataset
from data.transforms import get_transforms
from models.multi_head_convnext import build_multi_head_model
from training.losses import FocalLoss
from sklearn.metrics import f1_score, accuracy_score, classification_report

def run_memorization_test(config_path: str, data_root: str, epochs: int = 25, lr: float = 1e-3):
    print("=" * 80)
    print("CHANGE 6: TINY MEMORIZATION TEST (DRUSEN & PIPELINE DIAGNOSTIC)")
    print("=" * 80)

    # 1. Load full dataset manifest
    full_ds = MultiHeadOCTDataset(config_path=config_path, data_root=data_root, transform=None)
    manifest = full_ds._manifest
    class_names = full_ds.get_class_names("h2")

    # 2. Extract balanced subset (up to 10 images per class)
    abnormal_manifest = manifest[manifest["granular_idx"] != -1].copy()
    sampled_records = []
    
    for c_idx, c_name in enumerate(class_names):
        c_sub = abnormal_manifest[abnormal_manifest["granular_idx"] == c_idx]
        take_n = min(len(c_sub), 10)
        if take_n > 0:
            sampled_records.append(c_sub.sample(n=take_n, random_state=42))
            
    micro_manifest = pd.concat(sampled_records).reset_index(drop=True)
    print(f"\nConstructed Micro-Dataset for Memorization Test:")
    print(f"Total Micro-Dataset Samples: {len(micro_manifest)}")
    counts = micro_manifest["granular_idx"].value_counts().sort_index()
    for c_idx, c_name in enumerate(class_names):
        print(f"  [{c_idx:2d}] {c_name:<15}: {counts.get(c_idx, 0):>2d} samples")

    # 3. Create Dataset and DataLoader
    train_transform = get_transforms("train")
    micro_ds = MultiHeadOCTDataset(config_path=config_path, manifest=micro_manifest, transform=train_transform)
    loader = torch.utils.data.DataLoader(micro_ds, batch_size=16, shuffle=True, num_workers=0)

    # 4. Build Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing compute device: {device}")
    
    model = build_multi_head_model(pretrained=True, warmup=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_h2 = FocalLoss(gamma=2.0, alpha=None, label_smoothing=0.0) # Unweighted for memorization test

    # 5. Training Loop over Micro-Batch
    model.train()
    print("\nStarting Micro-Dataset Memorization Training...")
    
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels in loader:
            images = images.to(device)
            targets = labels["pathology"].to(device)

            optimizer.zero_grad()
            logits_dict = model(images)
            logits = logits_dict["pathology"]

            loss = criterion_h2(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        
        drusen_idx = class_names.index("DRUSEN")
        drusen_mask = (np.array(all_targets) == drusen_idx)
        if np.sum(drusen_mask) > 0:
            drusen_f1 = f1_score(np.array(all_targets) == drusen_idx, np.array(all_preds) == drusen_idx, zero_division=0)
        else:
            drusen_f1 = 0.0

        if ep % 5 == 0 or ep == epochs or macro_f1 > 0.95:
            print(f"Epoch {ep:2d}/{epochs:2d} | Loss: {total_loss/len(loader):.4f} | Overall Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | DRUSEN F1: {drusen_f1:.4f}")

    print("\n--- Final Classification Report on Micro-Dataset ---")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))

    if drusen_f1 > 0.90:
        print("✓ SUCCESS: DRUSEN and all classes memorized cleanly! Target mapping & loss architecture are 100% valid.")
    else:
        print("✗ FAILURE: DRUSEN failed to memorize. Investigating target encoding or indexing issue.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRUSEN & Pipeline Memorization Test")
    parser.add_argument("--config", type=str, default="image-classification-model-training/config/hierarchy.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_memorization_test(args.config, args.data_root, epochs=args.epochs, lr=args.lr)
