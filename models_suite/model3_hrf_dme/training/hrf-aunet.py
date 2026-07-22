#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated Attention U-Net Training Script for HPC
Combines all code files into a single executable script

This script implements Attention U-Net with:
- Attention Gates to focus on HRF regions
- CLAHE preprocessing + Z-Score normalization
- Dice Loss for direct metric optimization
- Cosine Annealing LR scheduler

Usage:
    python3 attention_unet_hpc.py

Author: Generated from Colab notebook
Date: 2026-02-14
"""

import os
import sys

# Fix for OpenCV on headless systems (HPC without display)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import tifffile
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import jaccard_score, roc_curve, auc, confusion_matrix
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Attention U-Net training - EXACT VALUES FROM COLAB"""
    
    # ========== DATA PATHS (CONFIGURED FOR YOUR HPC) ==========
    DATA_DIR = '/home/user11/HRF-DATASET'
    CHECKPOINT_DIR = '/home/user11/HRF-DATASET/checkpoints_attention_unet'
    
    # Data settings
    IMAGE_SIZE = None  # None = keep original size
    BATCH_SIZE = 4
    NUM_WORKERS = 0  # Set to 0 to avoid shared memory issues
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    APPLY_AUGMENTATION = False  # NO AUGMENTATION (as per config)
    APPLY_PREPROCESSING = True  # CLAHE + Z-Score Normalization
    
    # Model settings
    MODEL_NAME = 'AttentionUNet'
    N_CHANNELS = 3
    N_CLASSES = 1
    BILINEAR = False
    BASE_FILTERS = 64
    
    # Loss function settings (Dice Loss as per config)
    LOSS_NAME = 'dice'
    SMOOTH = 1.0
    
    # Optimizer settings
    OPTIMIZER = 'adam'
    LR = 0.001
    WEIGHT_DECAY = 0.0001
    
    # Scheduler settings (Cosine as per config)
    SCHEDULER = 'cosine'
    T_MAX = 100  # num_epochs
    ETA_MIN = 1e-6
    
    # Training settings
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15  # As per config
    USE_AMP = True
    RANDOM_SEED = 42


# ============================================================================
# DATASET CLASS (Enhanced with CLAHE + Z-Score Normalization)
# ============================================================================

class HRFDatasetEnhanced(Dataset):
    """
    Enhanced Dataset class for HRF segmentation.
    
    Includes:
    - CLAHE preprocessing (always applied)
    - Z-Score Normalization with ImageNet stats
    - Optional Data Augmentation
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_files: List[str],
        image_size: Optional[int] = None,
        apply_augmentation: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        
        if self.apply_augmentation:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            ])
        else:
            self.transform = None

    def apply_clahe(self, image):
        """Apply CLAHE to enhance contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final

    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find corresponding mask
        mask_candidates = [
            img_name.replace('.jpg', '_HRF.ome.tiff').replace('.jpeg', '_HRF.ome.tiff').replace('.png', '_HRF.ome.tiff'),
            img_name.replace('.jpg', '.tiff').replace('.jpeg', '.tiff').replace('.png', '.tiff'),
            img_name.replace('.jpg', '_mask.tiff').replace('.jpeg', '_mask.tiff').replace('.png', '_mask.tiff'),
        ]
        
        mask_path = None
        for candidate in mask_candidates:
            candidate_path = os.path.join(self.mask_dir, candidate)
            if os.path.exists(candidate_path):
                mask_path = candidate_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {img_name}")
        
        try:
            mask = tifffile.imread(mask_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not read mask file {mask_path}: {e}")
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] if mask.shape[0] > mask.shape[1] else mask[0]
            
        mask = (mask > 0).astype(np.float32)
        
        # PREPROCESSING: CLAHE
        image = self.apply_clahe(image)
        
        # AUGMENTATION
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # NORMALIZATION
        image = image.astype(np.float32) / 255.0
        
        # Z-Score Normalization (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :]).float()
        
        return image_tensor, mask_tensor


def create_dataloaders(
    data_dir: str,
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: Optional[int] = None,
    apply_augmentation: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    image_dir = os.path.join(data_dir, 'HRF_IMAGES')
    mask_dir = os.path.join(data_dir, 'HRF_MASKS')
    
    train_dataset = HRFDatasetEnhanced(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=train_files,
        image_size=image_size,
        apply_augmentation=apply_augmentation, 
    )
    
    val_dataset = HRFDatasetEnhanced(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=val_files,
        image_size=image_size,
        apply_augmentation=False,
    )
    
    test_dataset = HRFDatasetEnhanced(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_files=test_files,
        image_size=image_size,
        apply_augmentation=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth: float = 1.0, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1.0 - dice


# ============================================================================
# METRICS
# ============================================================================

def calculate_dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = np.sum(pred * target)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    
    return float(dice)


def calculate_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def calculate_precision_recall(pred: np.ndarray, target: np.ndarray) -> tuple:
    pred = pred.flatten()
    target = target.flatten()
    
    TP = np.sum(pred * target)
    FP = np.sum(pred * (1 - target))
    FN = np.sum((1 - pred) * target)
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    
    return float(precision), float(recall)


def calculate_f1_score(pred: np.ndarray, target: np.ndarray) -> float:
    precision, recall = calculate_precision_recall(pred, target)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return float(f1)


def calculate_specificity(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.flatten()
    target = target.flatten()
    
    TN = np.sum((1 - pred) * (1 - target))
    FP = np.sum(pred * (1 - target))
    
    specificity = TN / (TN + FP + 1e-8)
    return float(specificity)


def evaluate_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    if isinstance(predictions, torch.Tensor):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = (predictions > threshold).astype(np.float32)
    
    batch_size = predictions.shape[0]
    metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'specificity': 0.0,
    }
    
    for i in range(batch_size):
        pred = predictions[i, 0]
        target = targets[i, 0]
        
        metrics['dice'] += calculate_dice_score(pred, target)
        metrics['iou'] += calculate_iou(pred, target)
        precision, recall = calculate_precision_recall(pred, target)
        metrics['precision'] += precision
        metrics['recall'] += recall
        metrics['f1'] += calculate_f1_score(pred, target)
        metrics['specificity'] += calculate_specificity(pred, target)
    
    for key in metrics:
        metrics[key] /= batch_size
    
    return metrics


class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'specificity': [],
        }
    
    def update(self, batch_metrics: Dict[str, float]):
        for key, value in batch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_average(self) -> Dict[str, float]:
        return {key: np.mean(values) if values else 0.0 for key, values in self.metrics.items()}
    
    def get_std(self) -> Dict[str, float]:
        return {key: np.std(values) if values else 0.0 for key, values in self.metrics.items()}


# ============================================================================
# U-NET BUILDING BLOCKS
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# ATTENTION U-NET MODEL
# ============================================================================

class AttentionBlock(nn.Module):
    """
    Attention Gate (AG)
    
    Filters the features from the skip connection using the gating signal
    from the coarser scale.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UpAttention(nn.Module):
    """Upscaling with Attention Gate"""
    def __init__(self, in_channels_deeper, in_channels_skip, out_channels, bilinear=False):
        super().__init__()
        
        self.attention = AttentionBlock(F_g=in_channels_deeper, F_l=in_channels_skip, F_int=in_channels_skip // 2)
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_skip + in_channels_deeper, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_deeper, in_channels_deeper // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_skip + in_channels_deeper // 2, out_channels)
            
    def forward(self, x_deeper, x_skip):
        x_upsampled = self.up(x_deeper)
        
        diffY = x_skip.size()[2] - x_upsampled.size()[2]
        diffX = x_skip.size()[3] - x_upsampled.size()[3]

        x_upsampled = F.pad(x_upsampled, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        
        x_skip_attended = self.attention(g=x_deeper, x=x_skip)
        x = torch.cat([x_skip_attended, x_upsampled], dim=1)
        
        return self.conv(x)


class AttentionUNet(nn.Module):
    """Attention U-Net architecture"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, base_filters=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_filters = base_filters

        # ENCODER
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)

        # DECODER WITH ATTENTION
        self.up4 = UpAttention(base_filters * 16, base_filters * 8, base_filters * 8, bilinear)
        self.up3 = UpAttention(base_filters * 8, base_filters * 4, base_filters * 4, bilinear)
        self.up2 = UpAttention(base_filters * 4, base_filters * 2, base_filters * 2, bilinear)
        self.up1 = UpAttention(base_filters * 2, base_filters, base_filters, bilinear)
        
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        logits = self.outc(x)
        return logits


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.scaler = GradScaler() if use_amp else None
        self.best_val_dice = 0.0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics_tracker = MetricsTracker()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            batch_metrics = evaluate_batch(outputs, masks)
            metrics_tracker.update(batch_metrics)
            epoch_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item(), 'dice': batch_metrics['dice']})
        
        avg_metrics = metrics_tracker.get_average()
        avg_metrics['loss'] = epoch_loss / len(self.train_loader)
        
        return avg_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        metrics_tracker = MetricsTracker()
        epoch_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                batch_metrics = evaluate_batch(outputs, masks)
                metrics_tracker.update(batch_metrics)
                epoch_loss += loss.item()
                
                pbar.set_postfix({'loss': loss.item(), 'dice': batch_metrics['dice']})
        
        avg_metrics = metrics_tracker.get_average()
        avg_metrics['loss'] = epoch_loss / len(self.val_loader)
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filename: str = 'checkpoint.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 15,
        resume_from: Optional[str] = None,
    ):
        start_epoch = 0
        patience_counter = 0
        
        if resume_from is not None and os.path.exists(resume_from):
            start_epoch, metrics = self.load_checkpoint(resume_from)
            if metrics is not None and 'dice' in metrics:
                self.best_val_dice = metrics['dice']
                print(f"Restored best validation Dice: {self.best_val_dice:.4f}")
            start_epoch += 1
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        
        for epoch in range(start_epoch, num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"LR: {current_lr:.6f}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")
            
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.save_checkpoint(epoch, val_metrics, 'best_model.pth')
                patience_counter = 0
                print(f"[BEST] Model saved! Dice: {self.best_val_dice:.4f}")
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, 'latest_model.pth')
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {patience_counter} epochs without improvement")
                break
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print("="*50)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_test_set(model, test_loader, device, checkpoint_dir):
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    model.eval()
    metrics_tracker = MetricsTracker()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            batch_metrics = evaluate_batch(outputs, masks)
            metrics_tracker.update(batch_metrics)
            
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())
    
    results = metrics_tracker.get_average()
    std_results = metrics_tracker.get_std()
    
    all_targets_np = np.concatenate(all_targets)
    all_probs_np = np.concatenate(all_probs)
    all_preds_np = (all_probs_np > 0.5).astype(int)
    
    jaccard = jaccard_score(all_targets_np, all_preds_np)
    fpr, tpr, _ = roc_curve(all_targets_np, all_probs_np)
    auc_score = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(all_targets_np, all_preds_np).ravel()
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    for metric, value in results.items():
        print(f"{metric.capitalize():15s}: {value:.4f} +/- {std_results[metric]:.4f}")
    print("-" * 60)
    print(f"{'Jaccard':15s}: {jaccard:.4f}")
    print(f"{'AUC':15s}: {auc_score:.4f}")
    print(f"{'Confusion Matrix':15s}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("="*60)
    
    metrics_file = os.path.join(checkpoint_dir, 'test_metrics.csv')
    results_dict = {
        'timestamp': pd.Timestamp.now(),
        'dice': results['dice'],
        'dice_std': std_results['dice'],
        'iou': results['iou'],
        'iou_std': std_results['iou'],
        'precision': results['precision'],
        'precision_std': std_results['precision'],
        'recall': results['recall'],
        'recall_std': std_results['recall'],
        'f1': results['f1'],
        'f1_std': std_results['f1'],
        'specificity': results['specificity'],
        'specificity_std': std_results['specificity'],
        'jaccard': jaccard,
        'auc': auc_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    df = pd.DataFrame([results_dict])
    df.to_csv(metrics_file, index=False)
    print(f"\nResults saved to: {metrics_file}")
    
    return results, fpr, tpr, auc_score, (tn, fp, fn, tp)


def visualize_predictions(model, test_loader, device, checkpoint_dir, num_samples=5):
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    dataset = test_loader.dataset
    dataset_len = len(dataset)
    
    if dataset_len < num_samples:
        indices = list(range(dataset_len))
    else:
        indices = random.sample(range(dataset_len), num_samples)
    
    print(f"Visualizing samples at indices: {indices}")
    
    selected_images = []
    selected_masks = []
    model.eval()
    
    with torch.no_grad():
        for idx in indices:
            img_tensor, mask_tensor = dataset[idx]
            selected_images.append(img_tensor)
            selected_masks.append(mask_tensor)
        
        batch_images = torch.stack(selected_images).to(device)
        batch_masks = torch.stack(selected_masks).cpu().numpy()
        
        outputs = model(batch_images)
        predictions = torch.sigmoid(outputs) > 0.5
        predictions = predictions.cpu().numpy()
    
    for i in range(len(indices)):
        img_np = batch_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize from ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        mask_np = batch_masks[i, 0]
        pred_np = predictions[i, 0]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img_np)
        axes[0].set_title(f'Sample {indices[i]}: Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title(f'Sample {indices[i]}: Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title(f'Sample {indices[i]}: Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        filename = f'prediction_sample_{indices[i]}.png'
        save_path = os.path.join(checkpoint_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {filename}")
    
    print("[DONE] Visualizations saved")


def plot_metrics(fpr, tpr, auc_score, confusion_mat, checkpoint_dir):
    tn, fp, fn, tp = confusion_mat
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    ax2.set_title('Confusion Matrix')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'metrics_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[DONE] Metrics visualization saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("ATTENTION U-NET TRAINING - HPC VERSION")
    print("="*70)
    print(f"Start time: {pd.Timestamp.now()}")
    
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    print(f"\nCheckpoint directory: {Config.CHECKPOINT_DIR}")
    
    # DATA PREPARATION
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)
    
    image_dir = os.path.join(Config.DATA_DIR, 'HRF_IMAGES')
    
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png', '.tiff'))])
    
    print(f"Total images found: {len(image_files)}")
    
    np.random.seed(Config.RANDOM_SEED)
    all_files = image_files.copy()
    np.random.shuffle(all_files)
    
    n_total = len(all_files)
    n_train = int(n_total * Config.TRAIN_RATIO)
    n_val = int(n_total * Config.VAL_RATIO)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    print(f"Training: {len(train_files)} ({100*len(train_files)/n_total:.1f}%)")
    print(f"Validation: {len(val_files)} ({100*len(val_files)/n_total:.1f}%)")
    print(f"Test: {len(test_files)} ({100*len(test_files)/n_total:.1f}%)")
    
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Config.DATA_DIR,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        image_size=Config.IMAGE_SIZE,
        apply_augmentation=Config.APPLY_AUGMENTATION,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # MODEL SETUP
    print("\n" + "="*70)
    print("MODEL SETUP")
    print("="*70)
    
    model = AttentionUNet(
        n_channels=Config.N_CHANNELS,
        n_classes=Config.N_CLASSES,
        bilinear=Config.BILINEAR,
        base_filters=Config.BASE_FILTERS,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    loss_fn = DiceLoss(smooth=Config.SMOOTH)
    print(f"\nLoss: {Config.LOSS_NAME}")
    print(f"  smooth={Config.SMOOTH}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
    )
    print(f"\nOptimizer: {Config.OPTIMIZER.upper()}")
    print(f"  lr={Config.LR}, weight_decay={Config.WEIGHT_DECAY}")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.T_MAX,
        eta_min=Config.ETA_MIN
    )
    print(f"\nScheduler: {Config.SCHEDULER}")
    print(f"  T_max={Config.T_MAX}, eta_min={Config.ETA_MIN}")
    
    # TRAINING
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Early stopping patience: {Config.EARLY_STOPPING_PATIENCE}")
    print(f"Mixed precision: {Config.USE_AMP}")
    print(f"Preprocessing: CLAHE + Z-Score (ImageNet)")
    print(f"Augmentation: {Config.APPLY_AUGMENTATION}")
    print("="*70)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        use_amp=Config.USE_AMP,
    )
    
    trainer.train(
        num_epochs=Config.NUM_EPOCHS,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
    )
    
    # EVALUATION
    print("\n" + "="*70)
    print("LOADING BEST MODEL FOR EVALUATION")
    print("="*70)
    
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Best validation Dice: {checkpoint['metrics']['dice']:.4f}")
    
    results, fpr, tpr, auc_score, confusion_mat = evaluate_test_set(
        model, test_loader, device, Config.CHECKPOINT_DIR
    )
    
    visualize_predictions(
        model, test_loader, device, Config.CHECKPOINT_DIR, num_samples=5
    )
    
    plot_metrics(fpr, tpr, auc_score, confusion_mat, Config.CHECKPOINT_DIR)
    
    # COMPLETION
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"End time: {pd.Timestamp.now()}")
    print(f"Best validation Dice: {trainer.best_val_dice:.4f}")
    print(f"Test Dice: {results['dice']:.4f}")
    print(f"\nAll results saved to: {Config.CHECKPOINT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()