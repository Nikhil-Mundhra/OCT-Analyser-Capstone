import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from pathlib import Path

import sys
import argparse
import numpy as np
import random
import itertools
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import importlib.util

# Add local path first for segmentation imports
local_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, local_path)

from data.segmentation_dataset import (
    OCT5kSegmentationDataset,
    TransformSubset,
    get_training_transforms,
    get_val_transforms,
)

# Use importlib to bypass sys.modules namespace collision for 'data'
cls_path = Path(__file__).resolve().parent.parent.parent / 'image-classification-model-training' / 'data' / 'dataset.py'
spec = importlib.util.spec_from_file_location("cls_dataset_module", str(cls_path))
cls_module = importlib.util.module_from_spec(spec)
sys.modules["cls_dataset_module"] = cls_module
spec.loader.exec_module(cls_module)
MultiHeadOCTDataset = cls_module.MultiHeadOCTDataset

import torchvision.transforms as transforms
from PIL import Image
from models.unet import HierarchicalUNet


# ---------------------------------------------------------------------------
# Loss Functions  (Issue #1 fix)
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice Loss.

    Unlike CrossEntropy, Dice measures *overlap* rather than per-pixel accuracy.
    It is inherently balanced — a background-dominant dataset cannot minimise
    Dice by predicting all-background, because fluid/lesion classes contribute
    equally to the mean whether or not they are large.

    Args:
        smooth: Laplace smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)                                       # (B, C, H, W)
        
        # Mask out ignore_index (255)
        valid_mask = (targets != 255).unsqueeze(1).float()
        safe_targets = targets.clone()
        safe_targets[targets == 255] = 0
        
        targets_one_hot = (
            F.one_hot(safe_targets, num_classes).permute(0, 3, 1, 2).float()       # (B, C, H, W)
        )
        
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # Sum over batch + spatial dims; keep class axis for per-class Dice
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        cardinality  = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Semantic Segmentation.
    Automatically suppresses gradients from easy, highly confident majority classes
    (e.g., massive background regions) and focuses heavily on hard, ambiguous boundaries.
    """
    def __init__(self, weight=None, ignore_index=255, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.weight is not None:
            safe_targets = targets.clone()
            safe_targets[targets == self.ignore_index] = 0
            pixel_weights = self.weight[safe_targets]
            focal_loss = focal_loss * pixel_weights
            
        valid_mask = (targets != self.ignore_index).float()
        valid_pixels = valid_mask.sum()
        if valid_pixels > 0:
            return (focal_loss * valid_mask).sum() / valid_pixels
        return focal_loss.sum()


class CombinedLoss(nn.Module):
    """
    Dice + Focal Loss hybrid — the industry standard for imbalanced
    medical image segmentation.

    - Focal Loss: sharp, per-pixel gradient signal focused on hard boundaries.
    - Dice Loss: overlap-based, class-imbalance-resistant objective.
    """

    def __init__(self, ce_weight: torch.Tensor = None, alpha: float = 0.5):
        super().__init__()
        self.focal = FocalLoss(weight=ce_weight, ignore_index=255, gamma=2.0)
        self.dice  = DiceLoss()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.focal(logits, targets) + (1.0 - self.alpha) * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Validation Metrics
# ---------------------------------------------------------------------------

def compute_dice_score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Computes mean Dice score across foreground classes (background class 0 excluded),
    averaged over the batch.  Returns 0.0 if no foreground class is present.
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        valid_mask = (targets != 255)
        scores = []
        for cls in range(1, num_classes):      # skip class 0 (background)
            pred_mask   = ((preds == cls) & valid_mask).float()
            target_mask = ((targets == cls) & valid_mask).float()
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            if union == 0:
                continue                       # class absent in this batch
            scores.append((2.0 * intersection / union).item())
        return sum(scores) / len(scores) if scores else 0.0


def load_image_gray(p):
    return Image.open(p).convert('L') if isinstance(p, str) else p

# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def train():
    device = torch.device(
        'cuda'  if torch.cuda.is_available()          else
        'mps'   if torch.backends.mps.is_available()  else
        'cpu'
    )
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Hyperparameters & CLI arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Multi-Task Train Hierarchical UNet")
    parser.add_argument('--oct5k-data', type=str, required=True, help='Path to OCT5k dataset')
    parser.add_argument('--cls-data', type=str, required=True, help='Path to base classification dataset')
    parser.add_argument('--cls-config', type=str, required=True, help='Path to hierarchy.yaml for classification')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--accum-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--warmup-epochs', type=int, default=15, help='Epochs to warmup segmentation loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--log-dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory to save checkpoints (defaults to local checkpoints/)')
    parser.add_argument('--save-batches', type=int, default=0, help='Save a checkpoint every N batches (e.g. 2000)')
    
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Validate dataset path
    oct5k_path = args.oct5k_data
    cls_path = args.cls_data
    cls_config = args.cls_config
    if not Path(oct5k_path).exists():
        raise FileNotFoundError(f"OCT5K Dataset path does not exist: {oct5k_path}")
    if not Path(cls_path).exists():
        raise FileNotFoundError(f"Classification Dataset path does not exist: {cls_path}")

    # Assign hyperparameters from args
    batch_size = args.batch_size
    accum_steps = args.accum_steps
    warmup_epochs = args.warmup_epochs
    epochs = args.epochs
    learning_rate = args.lr

    # Device selection - explicitly mapped
    device = torch.device(
        'cuda'  if torch.cuda.is_available() else
        'mps'   if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Explicitly mapped to device: {device}")

    # -------------------------------------------------------------------------
    # Class Weights
    # -------------------------------------------------------------------------
    coarse_class_weights = torch.tensor(
        [0.3,  1.0,  4.0],                                          # bg, retina, lesion
        dtype=torch.float32,
    ).to(device)

    granular_class_weights = torch.tensor(
        [
            0.3,                                                     # Class 0:  Background
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,               # Classes 1-8: Retinal layers
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0,                          # Classes 9-14: Fluid / Lesions
        ],
        dtype=torch.float32,
    ).to(device)

    # -------------------------------------------------------------------------
    # Datasets (Multi-Task Learning)
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    # 1. Classification Dataset
    
    cls_transform = transforms.Compose([
        transforms.Lambda(load_image_gray),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    cls_dataset = MultiHeadOCTDataset(config_path=cls_config, data_root=cls_path, transform=cls_transform)
    cls_train_size = int(0.9 * len(cls_dataset))
    cls_train_sub, cls_val_sub = random_split(
        cls_dataset, [cls_train_size, len(cls_dataset) - cls_train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    cls_train_loader = DataLoader(
        cls_train_sub, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    cls_val_loader = DataLoader(
        cls_val_sub, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True
    )

    # 2. Segmentation Dataset
    oct5k_dataset = OCT5kSegmentationDataset(root_dir=oct5k_path)
    oct5k_train_size = int(0.9 * len(oct5k_dataset))
    oct5k_train_sub, oct5k_val_sub = random_split(
        oct5k_dataset, [oct5k_train_size, len(oct5k_dataset) - oct5k_train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    seg_train_sub = TransformSubset(oct5k_train_sub, transform=get_training_transforms())
    seg_val_sub = TransformSubset(oct5k_val_sub, transform=get_val_transforms())

    seg_train_loader = DataLoader(
        seg_train_sub, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    seg_val_loader = DataLoader(
        seg_val_sub, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True
    )

    print(f"  Classification Train: {len(cls_train_sub)}, Val: {len(cls_val_sub)}")
    print(f"  Segmentation   Train: {len(seg_train_sub)}, Val: {len(seg_val_sub)}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # -------------------------------------------------------------------------
    # Loss & Optimizer
    # -------------------------------------------------------------------------
    criterion_coarse   = CombinedLoss(ce_weight=coarse_class_weights,   alpha=0.5).to(device)
    criterion_granular = CombinedLoss(ce_weight=granular_class_weights, alpha=0.5).to(device)
    criterion_cls      = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    # All parameters learn at the same rate in simultaneous MTL, since encoder 
    # receives gradients from both tasks and must adapt to both.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Cosine annealing gradually reduces LR, helping escape local minima late in training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # Mixed precision scaler (if enabled) and TensorBoard writer
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    writer = SummaryWriter(log_dir=args.log_dir)

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    # Setup checkpoint directory
    if args.checkpoint_dir:
        checkpoints_dir = Path(args.checkpoint_dir)
    else:
        checkpoints_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_cls_acc = 0.0
    best_oct5k_granular_dice = 0.0

    start_epoch = 0
    resume_batch_idx = 0
    if args.resume:
        if Path(args.resume).exists():
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            start_epoch = checkpoint.get("epoch", 0)
            resume_batch_idx = checkpoint.get("batch_idx", 0)
            
            # If batch_idx is 0 or not present, it means it's an end-of-epoch checkpoint
            if resume_batch_idx == 0:
                start_epoch += 1
            
            if "best_cls_acc" in checkpoint:
                best_cls_acc = checkpoint["best_cls_acc"]
            if "best_oct5k_granular_dice" in checkpoint:
                best_oct5k_granular_dice = checkpoint["best_oct5k_granular_dice"]
            # Fast-forward scheduler
            for _ in range(start_epoch):
                scheduler.step()
        else:
            print(f"Warning: Checkpoint path not found: {args.resume}. Training from scratch.")

    # Wrap in DataParallel AFTER loading weights if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via nn.DataParallel!")
        model = nn.DataParallel(model)

    # Helper function to save clean state dicts regardless of DataParallel
    get_model_state = lambda m: m.module.state_dict() if hasattr(m, 'module') else m.state_dict()

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("Starting Multi-Task training...\n")
    for epoch in range(start_epoch, epochs):

        # Dynamic Loss Weighting for segmentation
        lambda_seg = min(1.0, 0.01 + (epoch / max(1, warmup_epochs)) * 0.99)
        if epoch % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Segmentation Loss Multiplier (lambda_seg): {lambda_seg:.4f}")

        # ---- Training ----
        model.train()
        train_loss_cls = 0.0
        train_loss_seg = 0.0
        
        # Interleave batches
        cls_iter = iter(cls_train_loader)
        seg_iter = iter(seg_train_loader)
        
        num_batches = max(len(cls_train_loader), len(seg_train_loader))
        optimizer.zero_grad()

        start_batch_idx = resume_batch_idx if epoch == start_epoch else 0
        
        for batch_idx in range(start_batch_idx, num_batches):
            
            loss_accum_c = 0.0
            loss_accum_s = 0.0
            
            # --- Classification Batch ---
            try:
                images_c, targets_c = next(cls_iter)
            except StopIteration:
                cls_iter = iter(cls_train_loader)
                images_c, targets_c = next(cls_iter)
                
            images_c = images_c.to(device)
            # targets_c contains "pathology" (L2 Router classes)
            labels_c = targets_c["pathology"].to(device)

            if args.amp:
                with torch.amp.autocast('cuda'):
                    cls_logits = model(images_c, task="classification")
                    loss_c = criterion_cls(cls_logits, labels_c)
                    loss_c = loss_c / accum_steps
                scaler.scale(loss_c).backward()
                loss_accum_c += loss_c.item() * accum_steps
            else:
                cls_logits = model(images_c, task="classification")
                loss_c = criterion_cls(cls_logits, labels_c)
                loss_c = loss_c / accum_steps
                loss_c.backward()
                loss_accum_c += loss_c.item() * accum_steps

            train_loss_cls += loss_accum_c
            
            # --- Segmentation Batch ---
            try:
                images_s, mask_c, mask_g = next(seg_iter)
            except StopIteration:
                seg_iter = iter(seg_train_loader)
                images_s, mask_c, mask_g = next(seg_iter)
                
            images_s = images_s.to(device)
            mask_c = mask_c.to(device)
            mask_g = mask_g.to(device)

            if args.amp:
                with torch.amp.autocast('cuda'):
                    coarse_logits, granular_logits = model(images_s, task="segmentation")
                    loss_coarse = criterion_coarse(coarse_logits, mask_c)
                    loss_granular = criterion_granular(granular_logits, mask_g)
                    loss_s = (loss_coarse + loss_granular) * lambda_seg
                    loss_s = loss_s / accum_steps
                scaler.scale(loss_s).backward()
                loss_accum_s += loss_s.item() * accum_steps / lambda_seg  # track unscaled loss
            else:
                coarse_logits, granular_logits = model(images_s, task="segmentation")
                loss_coarse = criterion_coarse(coarse_logits, mask_c)
                loss_granular = criterion_granular(granular_logits, mask_g)
                loss_s = (loss_coarse + loss_granular) * lambda_seg
                loss_s = loss_s / accum_steps
                loss_s.backward()
                loss_accum_s += loss_s.item() * accum_steps / lambda_seg

            train_loss_seg += loss_accum_s

            # --- Gradient Accumulation Step ---
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            # --- Mid-Epoch Save ---
            if args.save_batches > 0 and (batch_idx + 1) % args.save_batches == 0:
                print(f"\n[Mid-Epoch Checkpoint] Saving at Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}...")
                mid_path = checkpoints_dir / f"unet_hierarchical_epoch_{epoch+1}_batch_{batch_idx+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx + 1,  # save the NEXT batch index to resume from
                    'model_state_dict': get_model_state(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_cls_acc': best_cls_acc,
                    'best_oct5k_granular_dice': best_oct5k_granular_dice
                }, mid_path)

            if batch_idx % 10 == 0:
                print(
                    f"  Epoch {epoch+1} | Batch {batch_idx}/{num_batches} | "
                    f"Cls Loss: {loss_accum_c:.4f}  Seg Loss (unscaled): {loss_accum_s:.4f}"
                )

        avg_train_loss_cls = train_loss_cls / num_batches
        avg_train_loss_seg = train_loss_seg / num_batches
        scheduler.step()

        # ---- Validation ----
        model.eval()
        
        # ---------------------------------------------------------------------
        # Validation for Classification
        # ---------------------------------------------------------------------
        val_cls_loss = 0.0
        val_cls_correct = 0
        val_cls_total = 0
        
        # FIX: BatchNorm Covariate Shift
        # Because the encoder sees an equal mix of OCTID (Cls) and OCT5K (Seg) images,
        # the BN running stats are a corrupted 50/50 blend. We force BN to use batch 
        # statistics for validation without updating the running stats.
        bn_momentums = {}
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                bn_momentums[name] = m.momentum
                m.momentum = 0.0  # Prevent updating running stats
                m.train()         # Force use of batch statistics
                
        with torch.no_grad():
            for images_c, targets_c in cls_val_loader:
                images_c = images_c.to(device)
                labels_c = targets_c["pathology"].to(device)
                if args.amp:
                    with torch.amp.autocast('cuda'):
                        cls_logits = model(images_c, task="classification")
                        loss_c = criterion_cls(cls_logits, labels_c)
                else:
                    cls_logits = model(images_c, task="classification")
                    loss_c = criterion_cls(cls_logits, labels_c)
                
                val_cls_loss += loss_c.item()
                preds = torch.argmax(cls_logits, dim=1)
                valid_mask = (labels_c != -1)
                val_cls_correct += (preds[valid_mask] == labels_c[valid_mask]).sum().item()
                val_cls_total += valid_mask.sum().item()
                
        avg_val_cls_loss = val_cls_loss / len(cls_val_loader) if len(cls_val_loader) > 0 else 0
        val_cls_acc = val_cls_correct / val_cls_total if val_cls_total > 0 else 0

        # RESTORE: BatchNorm states
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.momentum = bn_momentums[name]
                m.eval()

        # ---------------------------------------------------------------------
        # Validation for Segmentation
        # ---------------------------------------------------------------------
        def evaluate_loader(loader, loader_name):
            val_loss = 0.0
            val_coarse_dice_sum = 0.0
            val_granular_dice_sum = 0.0

            with torch.no_grad():
                for images, mask_coarse, mask_granular in loader:
                    images        = images.to(device)
                    mask_coarse   = mask_coarse.to(device)
                    mask_granular = mask_granular.to(device)

                    if args.amp:
                        with torch.amp.autocast('cuda'):
                            c_logits, g_logits = model(images, task="segmentation")
                            v_loss = criterion_coarse(c_logits, mask_coarse) + criterion_granular(g_logits, mask_granular)
                    else:
                        c_logits, g_logits = model(images, task="segmentation")
                        v_loss = criterion_coarse(c_logits, mask_coarse) + criterion_granular(g_logits, mask_granular)
                        
                    val_loss += v_loss.item()

                    # Track Dice scores
                    val_coarse_dice_sum  += compute_dice_score(c_logits, mask_coarse,   num_classes=3)
                    val_granular_dice_sum += compute_dice_score(g_logits, mask_granular, num_classes=15)

            avg_loss          = val_loss / len(loader) if len(loader) > 0 else 0
            avg_coarse_dice   = val_coarse_dice_sum / len(loader) if len(loader) > 0 else 0
            avg_granular_dice = val_granular_dice_sum / len(loader) if len(loader) > 0 else 0
            
            writer.add_scalar(f'Loss/Val_{loader_name}', avg_loss, epoch)
            writer.add_scalar(f'Dice/Coarse_{loader_name}', avg_coarse_dice, epoch)
            writer.add_scalar(f'Dice/Granular_{loader_name}', avg_granular_dice, epoch)
            
            return avg_loss, avg_coarse_dice, avg_granular_dice

        oct5k_val_loss, oct5k_coarse_dice, oct5k_granular_dice = evaluate_loader(seg_val_loader, "OCT5K")

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Loss/Train_Cls', avg_train_loss_cls, epoch)
        writer.add_scalar('Loss/Train_Seg', avg_train_loss_seg, epoch)
        writer.add_scalar('Loss/Val_Cls', avg_val_cls_loss, epoch)
        writer.add_scalar('Accuracy/Val_Cls', val_cls_acc, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('Hyperparameters/Lambda_Seg', lambda_seg, epoch)
        
        print(
            f"\n--- Epoch {epoch+1}/{epochs} Summary ---\n"
            f"  Train Loss Cls      : {avg_train_loss_cls:.4f}  | Train Loss Seg: {avg_train_loss_seg:.4f}\n"
            f"  LR                  : {current_lr:.6f}\n"
            f"  [CLS]   Val Loss    : {avg_val_cls_loss:.4f} | Val Acc: {val_cls_acc:.4f}\n"
            f"  [OCT5K] Val Loss    : {oct5k_val_loss:.4f} | Coarse Dice: {oct5k_coarse_dice:.4f} | Granular Dice: {oct5k_granular_dice:.4f}\n"
        )

        # Save best Classification model
        if val_cls_acc > best_cls_acc:
            best_cls_acc = val_cls_acc
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    get_model_state(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_cls_acc": best_cls_acc,
                    "best_oct5k_granular_dice": best_oct5k_granular_dice,
                },
                checkpoints_dir / "unet_hierarchical_best_cls.pth",
            )
            print(f"  ✓ New best Classification model saved (Val Acc: {best_cls_acc:.4f})")

        # Save best OCT5K model
        if oct5k_granular_dice > best_oct5k_granular_dice:
            best_oct5k_granular_dice = oct5k_granular_dice
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    get_model_state(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_cls_acc": best_cls_acc,
                    "best_oct5k_granular_dice": best_oct5k_granular_dice,
                },
                checkpoints_dir / "unet_hierarchical_best_oct5k.pth",
            )
            print(f"  ✓ New best OCT5K model saved (Granular Dice: {best_oct5k_granular_dice:.4f})\n")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    get_model_state(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoints_dir / f"unet_hierarchical_epoch_{epoch+1}.pth",
            )
    writer.close()

if __name__ == "__main__":
    train()
