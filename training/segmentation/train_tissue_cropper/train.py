"""
training/segmentation/train_tissue_cropper/train.py

Training pipeline for the Attention U-Net Dynamic Tissue Cropper on Unified OCT Datasets.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional, Callable

import numpy as np
import cv2

sys.stdout.reconfigure(line_buffering=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

def seed_worker(worker_id):
    import cv2
    cv2.setNumThreads(0)
    np.random.seed(42 + worker_id)

# Path setups
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
TRAINING_SEG_ROOT = WORKSPACE_ROOT / "training" / "segmentation"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(TRAINING_SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_SEG_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader, Subset

from train_cleanup import enforce_single_instance_and_clean_memory, clean_gpu_memory
from models_suite.tissue_cropper.tissue_unet import TissueCroppingUNet
from dataset_unified_tissue import get_train_val_tissue_datasets
from losses import CombinedTissueLoss, compute_dice_and_iou
import config


def train(
    epochs: int = config.EPOCHS,
    learning_rate: Optional[float] = None,
    resume_best: bool = False,
    patience: int = 4,
    min_delta: float = 0.001,
    smoke_test: bool = False,
    progress_callback: Optional[Callable] = None,
    stop_check_fn: Optional[Callable[[], bool]] = None
):
    enforce_single_instance_and_clean_memory("train_tissue_cropper")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("=" * 70, flush=True)
    print(f" TRAINING ATTENTION U-NET TISSUE CROPPER ON [{device}]", flush=True)
    print("=" * 70, flush=True)

    # 1. Dataset Loading
    print("Loading and stratifying multi-source unified dataset...", flush=True)
    train_ds, val_ds = get_train_val_tissue_datasets(
        segmented_root=config.SEGMENTED_ROOT,
        masked_root=config.MASKED_ROOT,
        target_size=config.IMAGE_SIZE,
        val_split=config.VAL_SPLIT,
        seed=config.SEED,
        in_channels=config.IN_CHANNELS,
    )
    print(f"Dataset Discovered: {len(train_ds)} train samples | {len(val_ds)} validation samples", flush=True)

    if smoke_test:
        print(">> SMOKE TEST MODE ACTIVATED (Limiting to 16 train / 8 val samples) <<", flush=True)
        train_ds = Subset(train_ds, list(range(min(16, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(8, len(val_ds)))))
        epochs = 2

    use_autocast = (device.type in ["mps", "cuda"])
    autocast_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    print(f"Mixed Precision (Autocast): {use_autocast} ({autocast_dtype})", flush=True)

    # 2. DataLoaders (Apple Silicon Unified Memory Optimized)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(config.BATCH_SIZE, len(train_ds)),
        shuffle=True,
        num_workers=config.NUM_WORKERS if not smoke_test else 0,
        persistent_workers=(config.NUM_WORKERS > 0 and not smoke_test),
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=seed_worker if config.NUM_WORKERS > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(config.BATCH_SIZE, len(val_ds)),
        shuffle=False,
        num_workers=config.NUM_WORKERS if not smoke_test else 0,
        persistent_workers=(config.NUM_WORKERS > 0 and not smoke_test),
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=seed_worker if config.NUM_WORKERS > 0 else None,
        drop_last=False,
    )

    # 3. Model, Loss, Optimizer
    model = TissueCroppingUNet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        base_c=config.BASE_CHANNELS
    ).to(device)

    best_val_dice = 0.0
    epochs_without_improvement = 0

    # Warm start from previous best checkpoint if requested
    if resume_best and not smoke_test:
        resume_path = None
        if (config.CHECKPOINT_DIR / "best_model.pth").exists():
            resume_path = config.CHECKPOINT_DIR / "best_model.pth"
        elif (config.SUITE_CKPT_DIR / "best_model.pth").exists():
            resume_path = config.SUITE_CKPT_DIR / "best_model.pth"

        if resume_path is not None:
            print(f"[WARM START] Loading pre-trained weights from: {resume_path}", flush=True)
            ckpt = torch.load(resume_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=True)
            best_val_dice = float(ckpt.get("val_dice", 0.0))
            print(f"[WARM START] Resumed baseline Best Val Dice: {best_val_dice:.4f}", flush=True)

    criterion = CombinedTissueLoss(
        ce_weight=config.CE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        focal_weight=config.FOCAL_WEIGHT
    )
    effective_lr = learning_rate if learning_rate is not None else config.LEARNING_RATE
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoint output directories
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.SUITE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    accumulation_steps = config.ACCUMULATION_STEPS if not smoke_test else 1
    start_time = time.time()
    epoch_logs = []
    early_stopped = False
    user_stopped = False

    for epoch in range(1, epochs + 1):
        if stop_check_fn is not None and stop_check_fn():
            user_stopped = True
            print("=" * 70, flush=True)
            print(f"[*] TRAINING STOPPED BY USER: Halting before epoch {epoch}.", flush=True)
            print("=" * 70, flush=True)
            break

        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(train_loader):
            if stop_check_fn is not None and (i % 5 == 0) and stop_check_fn():
                user_stopped = True
                print(f"[*] Stop signal received during batch {i+1}/{total_batches}.", flush=True)
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, masks) / accumulation_steps
            else:
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] Warning: NaN/Inf detected in loss at Epoch {epoch} Batch {i+1}. Skipping backward step.", flush=True)
                optimizer.zero_grad()
                del outputs, loss, images, masks
                continue

            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"[!] Warning: NaN/Inf detected in gradients at Epoch {epoch} Batch {i+1}. Skipping optimizer step.", flush=True)
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            batch_loss = loss.item() * accumulation_steps
            train_loss += batch_loss * images.size(0)

            if (i + 1) % 50 == 0:
                clean_gpu_memory()

            if progress_callback is not None:
                try:
                    running_avg = train_loss / max(1, (i + 1) * images.size(0))
                    progress_callback(
                        phase="training",
                        epoch=epoch,
                        total_epochs=epochs,
                        batch=i + 1,
                        total_batches=total_batches,
                        batch_loss=float(batch_loss),
                        train_loss=float(running_avg),
                        val_loss=0.0,
                        val_dice=float(best_val_dice),
                        val_iou=0.0
                    )
                except Exception:
                    pass

            if (i + 1) % max(1, total_batches // 4) == 0 or (i + 1) == total_batches:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] Batch [{i+1:03d}/{total_batches:03d}] | Loss: {batch_loss:.4f}", flush=True)

            del outputs, loss, images, masks

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_dices = []
        val_ious = []
        val_total_batches = len(val_loader)

        with torch.no_grad():
            for v_i, (val_images, val_masks) in enumerate(val_loader):
                val_images = val_images.to(device, non_blocking=True)
                val_masks = val_masks.to(device, non_blocking=True)

                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        val_outputs = model(val_images)
                        v_loss = criterion(val_outputs, val_masks)
                else:
                    val_outputs = model(val_images)
                    v_loss = criterion(val_outputs, val_masks)

                val_loss += v_loss.item() * val_images.size(0)

                d_score, iou_score = compute_dice_and_iou(val_outputs, val_masks)
                val_dices.append(d_score)
                val_ious.append(iou_score)

                if progress_callback is not None:
                    try:
                        progress_callback(
                            phase="validating",
                            epoch=epoch,
                            total_epochs=epochs,
                            batch=v_i + 1,
                            total_batches=val_total_batches,
                            batch_loss=float(v_loss.item()),
                            train_loss=float(train_loss),
                            val_loss=float(val_loss / max(1, (v_i + 1) * val_images.size(0))),
                            val_dice=float(np.mean(val_dices)) if val_dices else float(best_val_dice),
                            val_iou=float(np.mean(val_ious)) if val_ious else 0.0
                        )
                    except Exception:
                        pass

                del val_images, val_masks, val_outputs, v_loss

        clean_gpu_memory()
        val_loss /= len(val_ds)
        mean_dice = float(np.mean(val_dices)) if val_dices else 0.0
        mean_iou = float(np.mean(val_ious)) if val_ious else 0.0

        print(
            f"--> Epoch [{epoch:02d}/{epochs:02d}] Summary | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}",
            flush=True
        )

        if progress_callback is not None:
            try:
                progress_callback(
                    phase="epoch_complete",
                    epoch=epoch,
                    total_epochs=epochs,
                    batch=total_batches,
                    total_batches=total_batches,
                    batch_loss=0.0,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    val_dice=float(mean_dice),
                    val_iou=float(mean_iou)
                )
            except Exception:
                pass

        # Checkpoint Saving & Early Stopping Verification
        improvement = mean_dice - best_val_dice
        if improvement >= min_delta and not smoke_test:
            best_val_dice = mean_dice
            epochs_without_improvement = 0
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": mean_dice,
                "val_iou": mean_iou,
                "val_loss": val_loss,
                "config": {
                    "image_size": config.IMAGE_SIZE,
                    "in_channels": config.IN_CHANNELS,
                    "num_classes": config.NUM_CLASSES,
                    "base_channels": config.BASE_CHANNELS,
                }
            }
            torch.save(ckpt_data, config.CHECKPOINT_DIR / "best_model.pth")
            torch.save(ckpt_data, config.SUITE_CKPT_DIR / "best_model.pth")
            print(f"*** New Best Model Saved: Epoch {epoch:02d} | Val Dice: {mean_dice:.4f} (+{improvement:.4f}) ***", flush=True)
        else:
            epochs_without_improvement += 1
        # Record epoch metrics for persistent training history
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_dice": round(float(mean_dice), 4),
            "val_iou": round(float(mean_iou), 4),
            "learning_rate": float(scheduler.get_last_lr()[0])
        })

        if user_stopped:
            print("=" * 70, flush=True)
            print(f"[*] TRAINING HALTED: User requested stop during epoch {epoch}. Preserving best model.", flush=True)
            print("=" * 70, flush=True)
            break

        if patience > 0 and epochs_without_improvement >= patience and not smoke_test:
            early_stopped = True
            print("=" * 70, flush=True)
            print(f"[*] EARLY STOPPING TRIGGERED: Validation Dice has not improved for {patience} consecutive epochs. Terminating training.", flush=True)
            print("=" * 70, flush=True)
            if progress_callback is not None:
                try:
                    progress_callback(
                        phase="early_stopped",
                        epoch=epoch,
                        total_epochs=epochs,
                        batch=total_batches,
                        total_batches=total_batches,
                        batch_loss=0.0,
                        train_loss=float(train_loss),
                        val_loss=float(val_loss),
                        val_dice=float(best_val_dice),
                        val_iou=float(mean_iou)
                    )
                except Exception:
                    pass
            break

    elapsed = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"Training Completed in {elapsed/60:.2f} mins. Best Validation Dice: {best_val_dice:.4f}", flush=True)
    print("=" * 70, flush=True)

    # Persist Round to Training History Ledger
    if not smoke_test and len(epoch_logs) > 0:
        try:
            history_file = config.HISTORY_PATH
            suite_history_file = config.SUITE_HISTORY_PATH

            history_data = {"model_name": "Attention U-Net Tissue Cropper", "rounds": []}
            if history_file.exists():
                try:
                    with open(history_file, "r") as f:
                        history_data = json.load(f)
                except Exception:
                    pass

            source_counts = {}
            if hasattr(train_ds, "samples"):
                for s in train_ds.samples:
                    source_counts[s[2]] = source_counts.get(s[2], 0) + 1
            elif hasattr(train_ds, "dataset") and hasattr(train_ds.dataset, "samples"):
                for idx in getattr(train_ds, "indices", []):
                    s = train_ds.dataset.samples[idx]
                    source_counts[s[2]] = source_counts.get(s[2], 0) + 1

            curated_count = source_counts.get("curated", 0)
            base_count = sum(cnt for src, cnt in source_counts.items() if src != "curated")

            round_num = len(history_data.get("rounds", [])) + 1
            start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time))
            end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))
            duration_mins = round(elapsed / 60.0, 2)

            round_entry = {
                "round_id": round_num,
                "mode": "Fine-Tuning (Warm Start)" if resume_best else "From Scratch",
                "started_at": start_iso,
                "completed_at": end_iso,
                "duration_minutes": duration_mins,
                "device": str(device),
                "epochs_planned": epochs,
                "epochs_completed": len(epoch_logs),
                "early_stopped": early_stopped,
                "user_stopped": user_stopped,
                "learning_rate": effective_lr,
                "best_val_dice": round(float(best_val_dice), 4),
                "dataset": {
                    "total_train": len(train_ds),
                    "total_val": len(val_ds),
                    "base_samples": base_count,
                    "curated_samples": curated_count
                },
                "epoch_logs": epoch_logs
            }

            history_data.setdefault("rounds", []).append(round_entry)
            history_data["total_rounds"] = len(history_data["rounds"])
            history_data["current_best_val_dice"] = max(r.get("best_val_dice", 0.0) for r in history_data["rounds"])

            for p in [history_file, suite_history_file]:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w") as f:
                    json.dump(history_data, f, indent=2)
            print(f"[*] Training history ledger updated: Round {round_num} recorded successfully.", flush=True)
        except Exception as e:
            print(f"[!] Warning: Failed to update training history: {e}", flush=True)

    return best_val_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention U-Net Tissue Cropper")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--resume-best", action="store_true", help="Warm-start training from previous best_model.pth checkpoint")
    parser.add_argument("--patience", type=int, default=4, help="Patience for early stopping (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum delta for validation Dice improvement")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch pre-flight smoke test")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        learning_rate=args.lr,
        resume_best=args.resume_best,
        patience=args.patience,
        min_delta=args.min_delta,
        smoke_test=args.smoke_test
    )
