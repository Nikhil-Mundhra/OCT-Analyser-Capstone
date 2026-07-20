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

import torch.nn.functional as F

class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        if self.alpha is not None:
            # alpha is used like pos_weight: multiplier for positive targets
            alpha_t = self.alpha * targets + (1 - targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8, weight=None):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        loss *= one_sided_w
        
        if self.weight is not None:
            loss *= self.weight.to(loss.device)
            
        return -loss.mean()

PATHOLOGY_CLASSES = [
    'CNV', 'DRUSEN', 'AMD', 'General_AMD', 
    'DME', 'DR', 'MH', 'RVO', 'RAO', 
    'CSR', 'ERM', 'VID'
]

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
        
        # Initialize flat multi-label vector for 12 classes
        h2 = torch.zeros(len(PATHOLOGY_CLASSES), dtype=torch.float32)
        
        if pd.notna(row['head3_labels']) and row['head3_labels'] != "":
            labels = [l.strip() for l in str(row['head3_labels']).split(',')]
            for label in labels:
                if label == 'Generic_AMD': # Backwards compatibility
                    h2[PATHOLOGY_CLASSES.index('AMD')] = 1
                elif label in PATHOLOGY_CLASSES:
                    h2[PATHOLOGY_CLASSES.index(label)] = 1

        labels_dict = {
            'normal_abnormal': torch.tensor([h1], dtype=torch.float32),
            'pathology': h2
        }
        return image, labels_dict

def compute_loss_weights(df, device):
    logger.info("Calculating dynamic loss weights to handle class imbalance...")
    
    # Mathematical Multiplier to force the network toward 0 False Negatives
    FALSE_NEGATIVE_PENALTY_MULTIPLIER = 5.0
    logger.info(f"Applying False Negative Penalty Multiplier: {FALSE_NEGATIVE_PENALTY_MULTIPLIER}x")

    # Head 1
    h1_pos = df['head1_label'].sum()
    h1_neg = len(df) - h1_pos
    h1_pos_weight = torch.tensor([float(h1_neg) / max(1, h1_pos) * FALSE_NEGATIVE_PENALTY_MULTIPLIER], dtype=torch.float32).to(device)

    # Head 2 Multi-Label Frequency Weights
    h2_counts = {c: 0 for c in PATHOLOGY_CLASSES}
    for idx, row in df.iterrows():
        if pd.notna(row['head3_labels']) and str(row['head3_labels']).strip() != "":
            labels = [l.strip() for l in str(row['head3_labels']).split(',')]
            for label in labels:
                if label == 'Generic_AMD':
                    h2_counts['AMD'] += 1
                elif label in h2_counts:
                    h2_counts[label] += 1
                    
    max_count = max(h2_counts.values()) if h2_counts else 1
    h2_weights = torch.ones(len(PATHOLOGY_CLASSES), dtype=torch.float32)
    for i, cls_name in enumerate(PATHOLOGY_CLASSES):
        count = max(1, h2_counts[cls_name])
        # Scale so the most frequent class has weight 1.0, and rarer classes have higher weights
        # We cap it at 10.0 to prevent gradient explosions on extremely rare classes
        weight = min(max_count / count, 10.0)
        h2_weights[i] = weight
        
    return h1_pos_weight, h2_weights.to(device)

class BlackoutCorners(object):
    """Blacks out bottom corners to hide scanner UI compasses/logos."""
    def __init__(self, fraction=0.18, x_offset_frac=0.0, y_offset_frac=0.0):
        self.fraction = fraction
        self.x_offset_frac = x_offset_frac
        self.y_offset_frac = y_offset_frac

    def __call__(self, img):
        from PIL import ImageDraw
        w, h = img.size
        base_dim = max(w, h)
        box_size = int(base_dim * self.fraction)
        x_off = int(base_dim * self.x_offset_frac)
        y_off = int(base_dim * self.y_offset_frac)
        
        draw = ImageDraw.Draw(img)
        # Bottom Left
        x1 = x_off
        y1 = h - box_size - y_off
        x2 = x_off + box_size
        y2 = h - y_off
        draw.rectangle([x1, y1, x2, y2], fill="black")
        return img

class LetterboxPad(object):
    """Pads an image to a square aspect ratio using black pixels."""
    def __call__(self, img):
        from torchvision.transforms.functional import pad
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        return pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Head ConvNeXt on MPS")
    parser.add_argument('--manifest', type=str, default='dataset_manifest.csv')
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test on 2 batches')
    parser.add_argument('--resume', action='store_true', help='Resume from latest fold0_last_model.pth checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='hf_space/weights', help='Directory to save checkpoints')
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

    train_transform = transforms.Compose([
        BlackoutCorners(),
        LetterboxPad(),
        transforms.Resize((224, 224)),
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
        BlackoutCorners(),
        LetterboxPad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    logger.info(f"Loading dataset from {args.manifest}...")
    full_dataset = MultiHeadOCTDataset(args.manifest, transform=train_transform)
    logger.info(f"Dataset loaded with {len(full_dataset)} items.")
    
    h1_w, h2_w = compute_loss_weights(full_dataset.df, device)
    
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
        'h1': BinaryFocalLossWithLogits(alpha=h1_w, gamma=2.0),
        'h2': AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05, weight=h2_w)
    }
    logger.info("Criterions initialized.")
    
    loss_weights = {
        'h1': 1.0,
        'h2': 1.0
    }

    # 4. Trainer Initialization
    logger.info("Initializing Trainer...")
    trainer = MultiHeadTrainer(
        model=model,
        criterions=criterions,
        loss_weights=loss_weights,
        mode="multi_head_mps",
        checkpoint_dir=args.checkpoint_dir,
        log_dir="logs",
        device=device,
        amp_dtype=amp_dtype,
        metric_extractors={
            'h2': lambda logits: (torch.sigmoid(logits) > 0.5).int()
        }
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
