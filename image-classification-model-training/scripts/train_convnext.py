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

# Suppress verbose third-party HTTP request & progress bar logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
for noisy_logger in ["httpx", "urllib3", "huggingface_hub"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Head ConvNeXt")
    parser.add_argument("--config", type=str, default="config/hierarchy.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-warmup", type=int, default=10, help="Number of frozen-backbone warmup epochs for head & CBAM tuning")
    parser.add_argument("--epochs-finetune", type=int, default=10, help="Number of gradual stage-wise fine-tuning epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without macro-F1 improvement)")
    parser.add_argument("--lr-head", type=float, default=1e-4, help="Learning rate for classification heads & CBAM attention blocks")
    parser.add_argument("--lr-backbone", type=float, default=1e-6, help="Learning rate for deep backbone stages (early stages receive 0.1x this rate)")
    parser.add_argument("--num-workers", type=int, default=2, help="Set to 0 to bypass shared memory")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch per phase to verify pipeline")
    parser.add_argument("--w-h1", type=float, default=1.0, help="Weight for Head 1 (Binary) loss")
    parser.add_argument("--w-h2", type=float, default=1.0, help="Weight for Head 2 (12-Class) loss")
    default_ckpt_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "checkpoints"
    parser.add_argument("--checkpoint-dir", type=str, default=default_ckpt_dir, help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face Hub repository ID (e.g. username/repo) for real-time cloud backup")
    parser.add_argument("--accum-steps", type=int, default=1, help="Number of gradient accumulation steps (effective batch size = batch_size * accum_steps)")
    parser.add_argument("--save-steps", type=int, default=2250, help="Save a mid-epoch checkpoint every N batches (0 to disable)")
    parser.add_argument("--use-data-parallel", action="store_true", help="Enable PyTorch DataParallel across multi-GPU (disabled by default for single-GPU efficiency)")
    parser.add_argument("--use-ddp", action="store_true", help="Enable PyTorch DistributedDataParallel across multi-GPU via torchrun")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Enable WeightedRandomSampler for inverse class frequency oversampling")
    parser.add_argument("--version", type=str, default="auto", help="Strict model weight version tag (e.g. 'v1', 'v2', or 'auto' to auto-increment)")
    
    args = parser.parse_args()

    is_ddp_env = ("LOCAL_RANK" in os.environ) or args.use_ddp
    if is_ddp_env and torch.cuda.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    compute_manager = ComputeManager(use_data_parallel=args.use_data_parallel, use_ddp=args.use_ddp)

    # Initialize H2 FocalLoss alpha weights
    full_ds = MultiHeadOCTDataset(config_path=args.config, transform=None)
    class_names = full_ds.get_class_names("h2")

    if args.use_weighted_sampler:
        # WeightedRandomSampler already equalizes batch presentation at the input layer.
        # Setting initial alpha to uniform 1.0 prevents double-counting class imbalance,
        # letting the Dynamic Adaptive Class-Weight Controller adjust loss weights purely based on validation F1.
        h2_alpha = torch.ones(len(class_names), dtype=torch.float32)
        if compute_manager.is_main_process:
            logger.info("=== WeightedRandomSampler Active: Initial FocalLoss Alpha set to Uniform 1.0 (Dynamic Controller Active) ===")
    else:
        h2_alpha = full_ds.compute_class_weights("h2")
        if compute_manager.is_main_process:
            logger.info("=== H2 Normalized Bounded FocalLoss Alpha Weights (Inverse Class Frequency) ===")
            for idx, (c_name, w_val) in enumerate(zip(class_names, h2_alpha.tolist())):
                logger.info(f"  {c_name:<15} : {w_val:.2f}")
            logger.info("=================================================================================")

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
        val_transform=val_transforms,
        use_weighted_sampler=args.use_weighted_sampler,
        is_ddp=compute_manager.is_ddp,
        rank=compute_manager.rank,
        world_size=compute_manager.world_size
    )

    for fold_id, (train_loader, val_loader) in enumerate(fold_loaders):
        if compute_manager.is_main_process:
            logger.info(f"=== Starting Fold {fold_id} ===")
        
        model = build_multi_head_model(pretrained=True, warmup=True)
        model = compute_manager.prepare_model(model)
        
        sub_dir = "WeightedRandomSampler" if args.use_weighted_sampler else None
        trainer = MultiHeadTrainer(
            model=model,
            criterions=criterions,
            loss_weights=loss_weights,
            checkpoint_dir=args.checkpoint_dir,
            compute_manager=compute_manager,
            mode="multi_head",
            sub_dir=sub_dir,
            version=args.version,
            args_dict=vars(args)
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
            save_steps=args.save_steps,
            patience=args.patience,
            use_weighted_sampler=args.use_weighted_sampler
        )
        
        if compute_manager.is_main_process:
            logger.info(f"Fold {fold_id} Best-by-Macro-F1 Metrics: {best_metrics}")
        
        # By default, run just 1 fold for iteration speed unless requested otherwise
        break

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    from utils.gpu_mutex import GPUMutex
    with GPUMutex():
        main()
