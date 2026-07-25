"""
scripts/train_convnext.py

Entry point for training the Multi-Head ConvNeXt model.
Multi-task loss weighting Option B: Weights are tunable via arguments.
"""

import argparse
import logging
import os
import sys

# Crucial to prevent PyTorch DataLoader multiprocessing deadlocks with OpenCV/ITK
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import cv2
cv2.setNumThreads(0)

import torch
import torch.nn as nn
import torch.nn.init
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
try:
    import timm.layers.weight_init
    timm.layers.weight_init.trunc_normal_ = lambda tensor, mean=0., std=1., a=-2., b=2.: torch.nn.init.normal_(tensor, mean=mean, std=std)
except ImportError:
    pass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multi_head_convnext import build_multi_head_model
from data.dataset import build_kfold_dataloaders, MultiHeadOCTDataset
from data.transforms import get_transforms
from training.multi_head_trainer import MultiHeadTrainer
from training.losses import FocalLoss
from utils.device import ComputeManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Head ConvNeXt")
    parser.add_argument("--config", type=str, default="config/hierarchy.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-warmup", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=45)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2, help="Set to 0 to bypass shared memory")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch per phase to verify pipeline")
    parser.add_argument("--w-h1", type=float, default=1.0, help="Weight for Head 1 (Binary) loss")
    parser.add_argument("--w-h2", type=float, default=1.0, help="Weight for Head 2 (12-Class) loss")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face Hub repository ID (e.g. username/repo) for real-time cloud backup")
    parser.add_argument("--accum-steps", type=int, default=1, help="Number of gradient accumulation steps (effective batch size = batch_size * accum_steps)")
    parser.add_argument("--save-steps", type=int, default=2250, help="Save a mid-epoch checkpoint every N batches (0 to disable)")
    parser.add_argument("--use-data-parallel", action="store_true", help="Enable PyTorch DataParallel across multi-GPU (disabled by default for single-GPU efficiency)")
    
    args = parser.parse_args()

    compute_manager = ComputeManager(use_data_parallel=args.use_data_parallel)

    # Get class weights from full dataset for FocalLoss alpha
    full_ds = MultiHeadOCTDataset(config_path=args.config, transform=None)
    h2_alpha = full_ds.compute_class_weights("h2")
    logger.info(f"H2 FocalLoss Alpha weights: {h2_alpha.tolist()}")

    criterions = {
        'h1': nn.BCEWithLogitsLoss(),
        'h2': FocalLoss(gamma=2.0, alpha=h2_alpha, reduction="mean", label_smoothing=0.1)
    }
    
    loss_weights = {
        'h1': args.w_h1,
        'h2': args.w_h2
    }

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    fold_loaders = build_kfold_dataloaders(
        config_path=args.config,
        mode="multi_head",
        n_splits=5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transforms,
        val_transform=val_transforms
    )

    for fold_id, (train_loader, val_loader) in enumerate(fold_loaders):
        logger.info(f"=== Starting Fold {fold_id} ===")
        
        model = build_multi_head_model(pretrained=True, warmup=True)
        
        trainer = MultiHeadTrainer(
            model=model,
            criterions=criterions,
            loss_weights=loss_weights,
            compute_manager=compute_manager,
            mode="multi_head"
        )
        
        best_metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            warmup_epochs=args.epochs_warmup,
            warmup_lr=args.lr_head,
            finetune_epochs=args.epochs_finetune,
            head_lr=args.lr_head,
            backbone_lr=args.lr_backbone,
            fold_id=fold_id,
            smoke_test=args.smoke_test,
            resume_path=args.resume,
            hf_repo=args.hf_repo,
            accum_steps=args.accum_steps,
            save_steps=args.save_steps
        )
        
        logger.info(f"Fold {fold_id} Best Metrics: {best_metrics}")
        
        # By default, run just 1 fold for iteration speed unless requested otherwise
        break

if __name__ == "__main__":
    main()
