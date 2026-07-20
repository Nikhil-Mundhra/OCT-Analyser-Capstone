"""
scripts/train_convnext.py

Entry point for training the Multi-Head ConvNeXt model.
Multi-task loss weighting Option B: Weights are tunable via arguments.
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multi_head_convnext import build_multi_head_model
from data.dataset import build_kfold_dataloaders, MultiHeadOCTDataset
from data.transforms import get_transforms
from training.multi_head_trainer import MultiHeadTrainer
from training.losses import FocalLoss

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Head ConvNeXt")
    parser.add_argument("--config", type=str, default="config/hierarchy.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-warmup", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=45)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch per phase to verify pipeline")
    parser.add_argument("--w-h1", type=float, default=1.0, help="Weight for Head 1 loss")
    parser.add_argument("--w-h2", type=float, default=2.0, help="Weight for Head 2 loss (prioritized for imbalance mitigation)")
    parser.add_argument("--w-h3", type=float, default=0.5, help="Weight for Head 3 loss")
    
    args = parser.parse_args()

    # Get class weights from full dataset for FocalLoss alpha
    full_ds = MultiHeadOCTDataset(config_path=args.config, transform=None)
    h2_alpha = full_ds.compute_class_weights("l2")
    logger.info(f"H2 FocalLoss Alpha weights: {h2_alpha.tolist()}")

    criterions = {
        'h1': nn.BCEWithLogitsLoss(),
        'h2': FocalLoss(gamma=2.0, alpha=h2_alpha, reduction="mean", label_smoothing=0.1),
        'h3': nn.BCEWithLogitsLoss()
    }
    
    loss_weights = {
        'h1': args.w_h1,
        'h2': args.w_h2,
        'h3': args.w_h3
    }

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    fold_loaders = build_kfold_dataloaders(
        config_path=args.config,
        mode="multi_head",
        n_splits=5,
        batch_size=args.batch_size,
        num_workers=4,
        train_transform=train_transforms,
        val_transform=val_transforms
    )

    for fold_id, (train_loader, val_loader) in enumerate(fold_loaders):
        logger.info(f"=== Starting Fold {fold_id} ===")
        
        model = build_multi_head_model(pretrained=True, warmup=True)
        
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
            model = nn.DataParallel(model)
        
        trainer = MultiHeadTrainer(
            model=model,
            criterions=criterions,
            loss_weights=loss_weights,
            mode="multi_head",
            metric_extractors={
                'h2': lambda logits: torch.argmax(logits, dim=1)
            }
        )
        
        best_metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            warmup_epochs=args.epochs_warmup,
            finetune_epochs=args.epochs_finetune,
            head_lr=args.lr_head,
            backbone_lr=args.lr_backbone,
            fold_id=fold_id,
            smoke_test=args.smoke_test
        )
        
        logger.info(f"Fold {fold_id} Best Metrics: {best_metrics}")
        
        # By default, run just 1 fold for iteration speed unless requested otherwise
        break

if __name__ == "__main__":
    main()
