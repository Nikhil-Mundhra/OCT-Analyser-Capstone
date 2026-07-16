import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- MAC OS MPS FIX FOR TIMM SEGFAULT ---
# Must execute BEFORE torch is imported
import torch
import torch.nn.init
import timm.layers.weight_init
timm.layers.weight_init.trunc_normal_ = lambda tensor, mean=0., std=1., a=-2., b=2.: torch.nn.init.normal_(tensor, mean=mean, std=std)

import argparse
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from PIL import Image
import torchvision.transforms as transforms
import logging

from models.multi_head_convnext import build_multi_head_model
from training.multi_head_trainer import MultiHeadTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mappings
H2_CLASS_TO_IDX = {
    'Macular_Degeneration': 0,
    'Diabetic_Complications': 1,
    'Vascular_Occlusions': 2,
    'Fluid_Accumulation': 3,
    'Structural_Issues': 4
}

H3_MAPPINGS = {
    'Macular_Degeneration': {'CNV': 0, 'DRUSEN': 1, 'Generic_AMD': 2},
    'Diabetic_Complications': {'DME': 0, 'DR': 1},
    'Vascular_Occlusions': {'MH': 0, 'RVO': 1, 'RAO': 2},
    'Fluid_Accumulation': {'CSR': 0},
    'Structural_Issues': {'ERM': 0, 'VID': 1}
}

class MultiHeadOCTDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Parse labels
        h1 = int(row['head1_label'])
        
        h2 = -1
        if pd.notna(row['head2_label']) and row['head2_label'] != "":
            h2 = H2_CLASS_TO_IDX.get(row['head2_label'], -1)
            
        # Initialize h3 vectors (multi-label binary vectors)
        h3_m = torch.zeros(3)
        h3_d = torch.zeros(2)
        h3_v = torch.zeros(3)
        h3_f = torch.zeros(1)
        h3_s = torch.zeros(2)
        
        if pd.notna(row['head3_labels']) and row['head3_labels'] != "":
            labels = [l.strip() for l in str(row['head3_labels']).split(',')]
            for label in labels:
                if row['head2_label'] == 'Macular_Degeneration' and label in H3_MAPPINGS['Macular_Degeneration']:
                    h3_m[H3_MAPPINGS['Macular_Degeneration'][label]] = 1
                elif row['head2_label'] == 'Diabetic_Complications' and label in H3_MAPPINGS['Diabetic_Complications']:
                    h3_d[H3_MAPPINGS['Diabetic_Complications'][label]] = 1
                elif row['head2_label'] == 'Vascular_Occlusions' and label in H3_MAPPINGS['Vascular_Occlusions']:
                    h3_v[H3_MAPPINGS['Vascular_Occlusions'][label]] = 1
                elif row['head2_label'] == 'Fluid_Accumulation' and label in H3_MAPPINGS['Fluid_Accumulation']:
                    h3_f[H3_MAPPINGS['Fluid_Accumulation'][label]] = 1
                elif row['head2_label'] == 'Structural_Issues' and label in H3_MAPPINGS['Structural_Issues']:
                    h3_s[H3_MAPPINGS['Structural_Issues'][label]] = 1

        labels_dict = {
            'normal_abnormal': torch.tensor([h1], dtype=torch.float32),
            'pathology': torch.tensor(h2, dtype=torch.long),
            'severity': {
                'macular': h3_m,
                'diabetic': h3_d,
                'vascular': h3_v,
                'fluid': h3_f,
                'structural': h3_s
            }
        }
        return image, labels_dict

def compute_loss_weights(df, device):
    logger.info("Calculating dynamic loss weights to handle class imbalance...")
    
    # Mathematical Multiplier to force the network toward 0 False Negatives
    FALSE_NEGATIVE_PENALTY_MULTIPLIER = 1.5
    logger.info(f"Applying False Negative Penalty Multiplier: {FALSE_NEGATIVE_PENALTY_MULTIPLIER}x")

    # Head 1
    h1_pos = df['head1_label'].sum()
    h1_neg = len(df) - h1_pos
    h1_pos_weight = torch.tensor([float(h1_neg) / max(1, h1_pos) * FALSE_NEGATIVE_PENALTY_MULTIPLIER], dtype=torch.float32).to(device)

    # Head 2
    h2_counts = df['head2_label'].value_counts()
    h2_weights = torch.ones(5, dtype=torch.float32).to(device)
    total_h2 = h2_counts.sum()
    for class_name, count in h2_counts.items():
        if pd.notna(class_name) and class_name in H2_CLASS_TO_IDX:
            idx = H2_CLASS_TO_IDX[class_name]
            h2_weights[idx] = total_h2 / (5.0 * max(1, count))

    # Head 3
    h3_pos_weights = {}
    for family, mapping in H3_MAPPINGS.items():
        family_df = df[df['head2_label'] == family]
        family_total = len(family_df)
        
        weights = torch.ones(len(mapping), dtype=torch.float32).to(device)
        for label, idx in mapping.items():
            count = family_df['head3_labels'].apply(lambda x: label in str(x).split(',')).sum()
            neg_count = family_total - count
            weights[idx] = (neg_count / max(1, count))
        
        key_map = {
            'Macular_Degeneration': 'macular',
            'Diabetic_Complications': 'diabetic',
            'Vascular_Occlusions': 'vascular',
            'Fluid_Accumulation': 'fluid',
            'Structural_Issues': 'structural'
        }
        h3_pos_weights[key_map[family]] = weights

    return h1_pos_weight, h2_weights, h3_pos_weights

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Head ConvNeXt on MPS")
    parser.add_argument('--manifest', type=str, default='dataset_manifest.csv')
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test on 2 batches')
    parser.add_argument('--resume', action='store_true', help='Resume from latest fold0_last_model.pth checkpoint')
    args = parser.parse_args()

    # 1. Device Setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        amp_dtype = torch.float16 # MPS autocast uses float16
        logger.info("MPS (Metal Performance Shaders) backend enabled!")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        amp_dtype = torch.float16
        logger.info("CUDA backend enabled!")
    else:
        device = torch.device('cpu')
        amp_dtype = torch.bfloat16
        logger.info("CPU backend enabled (WARNING: Training will be extremely slow).")

    # 2. Data Preparation
    train_transform = transforms.Compose([
        # Center-Biased Safe Cropping (The UI / Border Defense)
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        # Constrained Affine Transformations (The Structural Defense)
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        # Mild Photometric Jitter (The Machine Laser Defense)
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0, hue=0),
        # Gaussian Blur / Speckle (The SNR Defense)
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    logger.info(f"Loading dataset from {args.manifest}...")
    full_dataset = MultiHeadOCTDataset(args.manifest, transform=train_transform)
    logger.info(f"Dataset loaded with {len(full_dataset)} items.")
    
    h1_w, h2_w, h3_w = compute_loss_weights(full_dataset.df, device)
    
    logger.info("Splitting dataset...")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    logger.info(f"Train size: {train_size}, Val size: {val_size}")
    import numpy as np
    np.random.seed(42)
    indices = np.random.permutation(len(full_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(full_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(full_dataset, indices[train_size:])
    
    logger.info("Setting validation transform (NOTE: doing this on Subset modifies original, we will fix this later)...")
    val_dataset.dataset.transform = val_transform

    batch_size = 16
    if args.smoke_test:
        batch_size = 8
        logger.info("SMOKE TEST MODE ENABLED - Small batches")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model & Loss Setup
    logger.info("Building model...")
    model = build_multi_head_model(pretrained=not args.smoke_test, warmup=True)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    logger.info("Model built.")
    
    criterions = {
        'h1': nn.BCEWithLogitsLoss(pos_weight=h1_w),
        'h2': nn.CrossEntropyLoss(weight=h2_w, ignore_index=-1),
        'h3': {
            'macular': nn.BCEWithLogitsLoss(pos_weight=h3_w['macular']),
            'diabetic': nn.BCEWithLogitsLoss(pos_weight=h3_w['diabetic']),
            'vascular': nn.BCEWithLogitsLoss(pos_weight=h3_w['vascular']),
            'fluid': nn.BCEWithLogitsLoss(pos_weight=h3_w['fluid']),
            'structural': nn.BCEWithLogitsLoss(pos_weight=h3_w['structural'])
        }
    }
    logger.info("Criterions initialized.")
    
    loss_weights = {
        'h1': 1.0,
        'h2': 2.0,
        'h3': 0.2  # Scaled down because H3 sums 5 individual BCE losses
    }

    # 4. Trainer Initialization
    logger.info("Initializing Trainer...")
    trainer = MultiHeadTrainer(
        model=model,
        criterions=criterions,
        loss_weights=loss_weights,
        mode="multi_head_mps",
        checkpoint_dir="hf_space/weights",
        log_dir="logs",
        device=device,
        amp_dtype=amp_dtype
    )
    logger.info("Trainer initialized.")

    # 5. Execute Training
    if args.smoke_test:
        logger.info("Starting Smoke Test...")
        trainer.train(
            train_loader, 
            val_loader, 
            warmup_epochs=1, 
            finetune_epochs=1, 
            smoke_test=True,
            weight_decay=0.05
        )
    else:
        logger.info("Starting Full Training...")
        resume_path = "hf_space/weights/multi_head_mps/fold0_last_model.pth" if args.resume else None
        
        metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            smoke_test=False,
            resume_path=resume_path,
            warmup_epochs=3,
            warmup_lr=5e-5, 
            finetune_epochs=20,
            backbone_lr=5e-6,
            head_lr=5e-5,
            weight_decay=0.05,
            patience=5
        )

if __name__ == "__main__":
    main()
