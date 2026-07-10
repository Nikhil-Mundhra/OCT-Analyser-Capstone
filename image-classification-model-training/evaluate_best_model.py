import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report

from models.multi_head_convnext import build_multi_head_model
from train_convnext_mps import MultiHeadOCTDataset, H2_CLASS_TO_IDX, H3_MAPPINGS

def main():
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    manifest_path = "dataset_manifest.csv"
    print(f"Loading dataset from {manifest_path}...")
    full_dataset = MultiHeadOCTDataset(manifest_path, transform=val_transform)
    
    train_size = int(0.8 * len(full_dataset))
    
    np.random.seed(42)
    indices = np.random.permutation(len(full_dataset)).tolist()
    val_dataset = torch.utils.data.Subset(full_dataset, indices[train_size:])
    
    print(f"Validation size: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print("Building model...")
    model = build_multi_head_model(pretrained=False, warmup=False)
    model.to(device)
    
    checkpoint_path = "hf_space/weights/multi_head_mps/fold0_best_model.pth"
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    h1_preds, h1_targets = [], []
    h2_preds, h2_targets = [], []
    
    h3_preds = {k: [] for k in H3_MAPPINGS.keys()}
    h3_targets = {k: [] for k in H3_MAPPINGS.keys()}
    
    h3_key_map = {
        'Macular_Degeneration': 'macular',
        'Diabetic_Complications': 'diabetic',
        'Vascular_Occlusions': 'vascular',
        'Fluid_Accumulation': 'fluid',
        'Structural_Issues': 'structural'
    }
    
    print("Evaluating...")
    _amp_dtype = torch.float16 if device.type in ('mps', 'cuda') else torch.bfloat16
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            
            with torch.autocast(device_type=device.type, dtype=_amp_dtype, enabled=device.type in ('mps', 'cuda')):
                logits = model(images)
                
            h1_prob = torch.sigmoid(logits['normal_abnormal']).cpu().numpy()
            h1_pred = (h1_prob > 0.5).astype(int)
            h1_preds.extend(h1_pred)
            h1_targets.extend(labels['normal_abnormal'].numpy())
            
            valid_h2 = labels['pathology'] != -1
            if valid_h2.sum() > 0:
                h2_targets.extend(labels['pathology'][valid_h2].numpy())
                h2_preds.extend(logits['pathology'][valid_h2].argmax(dim=1).cpu().numpy())
                
            for family, branch_key in h3_key_map.items():
                idx_for_family = (labels['pathology'] == H2_CLASS_TO_IDX[family])
                if idx_for_family.sum() > 0:
                    probs = torch.sigmoid(logits['severity'][branch_key][idx_for_family]).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    targets = labels['severity'][branch_key][idx_for_family].numpy()
                    h3_preds[family].extend(preds)
                    h3_targets[family].extend(targets)
                    
            if i % 100 == 0:
                print(f"Batch {i}/{len(val_loader)}")
                
    print("\n" + "="*50)
    print("H1 (Normal vs Abnormal) Metrics")
    print("="*50)
    h1_f1 = f1_score(h1_targets, h1_preds, average='macro')
    h1_acc = accuracy_score(h1_targets, h1_preds)
    print(f"Accuracy: {h1_acc:.4f} | Macro F1: {h1_f1:.4f}")
    print(classification_report(h1_targets, h1_preds, target_names=["Normal", "Abnormal"]))
    
    print("\n" + "="*50)
    print("H2 (Pathology Routing) Metrics")
    print("="*50)
    h2_f1 = f1_score(h2_targets, h2_preds, average='macro')
    h2_acc = accuracy_score(h2_targets, h2_preds)
    print(f"Accuracy: {h2_acc:.4f} | Macro F1: {h2_f1:.4f}")
    h2_target_names = [k for k, v in sorted(H2_CLASS_TO_IDX.items(), key=lambda item: item[1])]
    print(classification_report(h2_targets, h2_preds, target_names=h2_target_names))
    
    print("\n" + "="*50)
    print("H3 (Severity & Subtype) Metrics")
    print("="*50)
    
    for family, mapping in H3_MAPPINGS.items():
        print(f"--- {family} ---")
        targets = np.array(h3_targets[family])
        preds = np.array(h3_preds[family])
        if len(targets) == 0:
            print("No samples in validation set for this family.")
            continue
            
        branch_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        print(f"Overall Macro F1 for {family}: {branch_f1:.4f}")
        class_names = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
        print(classification_report(targets, preds, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main()
