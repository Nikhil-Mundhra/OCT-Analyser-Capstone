import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

import sys
import argparse
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.segmentation_dataset import (
    OCT5kSegmentationDataset,
    TransformSubset,
    get_training_transforms,
    get_val_transforms,
)
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
        targets_one_hot = (
            F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()       # (B, C, H, W)
        )
        # Sum over batch + spatial dims; keep class axis for per-class Dice
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        cardinality  = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """
    Dice + CrossEntropy hybrid loss — the industry standard for imbalanced
    medical image segmentation (confirmed by RETOUCH, MICCAI 2024 benchmarks).

    - CrossEntropyLoss (with class weights): sharp, per-pixel gradient signal.
      Class weights further penalise misclassifying rare lesion/fluid pixels.
    - DiceLoss: overlap-based, class-imbalance-resistant objective.

    The 50/50 split (alpha=0.5) is a well-validated starting point.
    Increase alpha toward 1.0 if the Dice loss destabilises early training.

    Args:
        ce_weight : 1-D tensor of per-class weights for CrossEntropyLoss.
        alpha     : Weight of the CE term; (1-alpha) goes to Dice.
    """

    def __init__(self, ce_weight: torch.Tensor = None, alpha: float = 0.5):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice  = DiceLoss()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(logits, targets) + (1.0 - self.alpha) * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Validation Metrics
# ---------------------------------------------------------------------------

def compute_dice_score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Computes mean Dice score across foreground classes (background class 0 excluded),
    averaged over the batch.  Returns 0.0 if no foreground class is present.

    This is the primary measure of segmentation quality — not loss.
    High-performing models on OCT datasets typically achieve:
        Dice > 0.80 (retinal layers), Dice > 0.70 (fluid/lesion classes).
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        scores = []
        for cls in range(1, num_classes):      # skip class 0 (background)
            pred_mask   = (preds == cls).float()
            target_mask = (targets == cls).float()
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            if union == 0:
                continue                       # class absent in this batch
            scores.append((2.0 * intersection / union).item())
        return sum(scores) / len(scores) if scores else 0.0


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
    parser = argparse.ArgumentParser(description="Train Hierarchical UNet on OCT5k dataset")
    parser.add_argument('--data', type=str, required=True, help='Path to OCT5k dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--log-dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Validate dataset path
    dataset_path = args.data
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Assign hyperparameters from args
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr

    # Device selection
    device = torch.device(
        'cuda'  if torch.cuda.is_available() else
        'mps'   if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Class Weights  (addresses extreme class imbalance — Issue #1)
    #
    # Rationale:
    #   - Class 0 (Background): hugely dominant → down-weighted to 0.3
    #   - Classes 1–8 (Retinal Layers): moderate frequency → neutral 1.0
    #   - Classes 9–14 (Fluid / Lesions): rare, clinically critical → 4.0
    #
    # These are intentionally hand-tuned starting points based on typical OCT5k
    # class frequency distributions. For a more principled approach, compute from
    # the actual dataset before your first full training run:
    #
    #   from data.segmentation_dataset import OCT5kSegmentationDataset
    #   ds = OCT5kSegmentationDataset(root_dir=dataset_path)
    #   counts = torch.zeros(15)
    #   for _, _, mask_g in ds:
    #       for c in range(15): counts[c] += (mask_g == c).sum()
    #   weights = 1.0 / (counts + 1); weights /= weights.sum()
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
    # Dataset  +  Augmentation
    #
    # Augmentation is applied to the TRAINING split only via TransformSubset.
    # The validation split always sees clean, unmodified images so that metrics
    # are stable and comparable across epochs.
    # -------------------------------------------------------------------------
    print("Loading dataset...")
    full_dataset = OCT5kSegmentationDataset(root_dir=dataset_path)
    print(f"Total samples: {len(full_dataset)}")

    val_size   = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_sub, val_sub = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    # Wrap each split with its own transform pipeline
    train_dataset = TransformSubset(train_sub, transform=get_training_transforms())
    val_dataset   = TransformSubset(val_sub,   transform=get_val_transforms())
    print(f"  Train samples: {len(train_dataset)}  (augmented)")
    print(f"  Val   samples: {len(val_dataset)}   (no augmentation)")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # -------------------------------------------------------------------------
    # Loss & Optimizer
    # -------------------------------------------------------------------------
    criterion_coarse   = CombinedLoss(ce_weight=coarse_class_weights,   alpha=0.5)
    criterion_granular = CombinedLoss(ce_weight=granular_class_weights, alpha=0.5)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Cosine annealing gradually reduces LR, helping escape local minima late in training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # Mixed precision scaler (if enabled) and TensorBoard writer
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    writer = SummaryWriter(log_dir=args.log_dir)

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    checkpoints_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    best_val_granular_dice = 0.0

    start_epoch = 0
    if args.resume:
        if Path(args.resume).exists():
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            if "val_granular_dice" in checkpoint:
                best_val_granular_dice = checkpoint["val_granular_dice"]
            # Fast-forward scheduler
            for _ in range(start_epoch):
                scheduler.step()
        else:
            print(f"Warning: Checkpoint path not found: {args.resume}. Training from scratch.")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...\n")
    for epoch in range(start_epoch, epochs):

        # ---- Training ----
        model.train()
        train_loss = 0.0

        for batch_idx, (images, mask_coarse, mask_granular) in enumerate(train_loader):
            images       = images.to(device)
            mask_coarse  = mask_coarse.to(device)
            mask_granular = mask_granular.to(device)

            optimizer.zero_grad()

            if args.amp:
                with torch.cuda.amp.autocast():
                    coarse_logits, granular_logits = model(images)
                    loss_coarse = criterion_coarse(coarse_logits, mask_coarse)
                    loss_granular = criterion_granular(granular_logits, mask_granular)
                    loss = loss_coarse + loss_granular
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                coarse_logits, granular_logits = model(images)
                loss_coarse = criterion_coarse(coarse_logits, mask_coarse)
                loss_granular = criterion_granular(granular_logits, mask_granular)
                loss = loss_coarse + loss_granular
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}  "
                    f"(Coarse: {loss_coarse.item():.4f}, Granular: {loss_granular.item():.4f})"
                )

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_loss             = 0.0
        val_coarse_dice_sum  = 0.0
        val_granular_dice_sum = 0.0

        with torch.no_grad():
            for images, mask_coarse, mask_granular in val_loader:
                images        = images.to(device)
                mask_coarse   = mask_coarse.to(device)
                mask_granular = mask_granular.to(device)

                c_logits, g_logits = model(images)

                v_loss = (
                    criterion_coarse(c_logits, mask_coarse)
                    + criterion_granular(g_logits, mask_granular)
                )
                val_loss += v_loss.item()

                # Track Dice scores — the real measure of segmentation quality
                val_coarse_dice_sum  += compute_dice_score(c_logits, mask_coarse,   num_classes=3)
                val_granular_dice_sum += compute_dice_score(g_logits, mask_granular, num_classes=15)

        avg_val_loss          = val_loss             / len(val_loader)
        avg_coarse_dice       = val_coarse_dice_sum  / len(val_loader)
        avg_granular_dice     = val_granular_dice_sum / len(val_loader)
        current_lr            = scheduler.get_last_lr()[0]
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Dice/Coarse', avg_coarse_dice, epoch)
        writer.add_scalar('Dice/Granular', avg_granular_dice, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        

        print(
            f"\n--- Epoch {epoch+1}/{epochs} Summary ---\n"
            f"  Train Loss      : {avg_train_loss:.4f}\n"
            f"  Val Loss        : {avg_val_loss:.4f}\n"
            f"  Val Coarse Dice : {avg_coarse_dice:.4f}  (bg excluded; target > 0.80)\n"
            f"  Val Granul Dice : {avg_granular_dice:.4f}  (bg excluded; target > 0.70)\n"
            f"  LR              : {current_lr:.6f}\n"
        )

        # Save best model by granular Dice — the harder, more clinically important task
        if avg_granular_dice > best_val_granular_dice:
            best_val_granular_dice = avg_granular_dice
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            avg_val_loss,
                    "val_granular_dice":   avg_granular_dice,
                },
                checkpoints_dir / "unet_hierarchical_best.pth",
            )
            print(f"  ✓ New best model saved  (Granular Dice: {best_val_granular_dice:.4f})\n")

        # Periodic checkpoint every 10 epochs (for recovery / resume)
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            avg_val_loss,
                },
                checkpoints_dir / f"unet_hierarchical_epoch_{epoch+1}.pth",
            )
    writer.close()

if __name__ == "__main__":
    train()
