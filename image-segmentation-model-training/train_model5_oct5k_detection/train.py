import os
import sys

sys.stdout.reconfigure(line_buffering=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

WORKSPACE_ROOT = Path("/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone")
TRAINING_ROOT = WORKSPACE_ROOT / "image-segmentation-model-training"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from train_cleanup import enforce_single_instance_and_clean_memory, clean_gpu_memory
from backend.oct_analyzer.checkpoint_versioning import resolve_and_create_version_dir, update_version_metadata_metrics

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models_suite.model5_oct5k_detection.detector import OCTPathologyDetector
from dataset import OCT5KDetectionDataset
import config

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    # HARDWIRED CLEANUP & SINGLE INSTANCE GUARD
    enforce_single_instance_and_clean_memory("train_model5_oct5k_detection")

    # Use CPU for torchvision Faster R-CNN to avoid Metal RPN kernel compatibility issues on macOS
    device = torch.device("cpu")
    print(f"Training Model 5 (OCT5K Pathology Detection) on device: {device}", flush=True)

    full_dataset = OCT5KDetectionDataset(root_dir=config.DATASET_ROOT)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    detector = OCTPathologyDetector(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=config.LEARNING_RATE)

    version_base_dir = WORKSPACE_ROOT / "checkpoints" / "detection" / "model5_oct5k_detection"
    version_dir, version_tag = resolve_and_create_version_dir(
        base_dir=version_base_dir,
        requested_version=getattr(config, "VERSION", "auto"),
        args_dict={
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "num_classes": config.NUM_CLASSES
        }
    )
    suite_ckpt_dir = WORKSPACE_ROOT / "models_suite" / "model5_oct5k_detection" / "checkpoints"
    os.makedirs(suite_ckpt_dir, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        detector.train()
        train_loss = 0.0
        num_batches = 0
        total_batches = len(train_loader)

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            valid_targets = [t for t in targets if len(t["boxes"]) > 0]
            if not valid_targets:
                continue

            valid_images = [img for img, t in zip(images, targets) if len(t["boxes"]) > 0]

            loss_dict = detector(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            num_batches += 1

            if (i + 1) % 25 == 0 or (i + 1) == total_batches:
                print(f"Epoch {epoch:02d}/{config.EPOCHS:02d} | Batch {i+1}/{total_batches} | Step Loss: {losses.item():.4f}", flush=True)

        if num_batches > 0:
            train_loss /= num_batches
        print(f">>> Epoch {epoch:02d}/{config.EPOCHS:02d} COMPLETE | Detector Loss: {train_loss:.4f}", flush=True)

        if train_loss < best_loss:
            best_loss = train_loss
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }
            version_ckpt = version_dir / "best_model.pth"
            suite_ckpt = suite_ckpt_dir / "best_model.pth"
            
            torch.save(ckpt_data, version_ckpt)
            torch.save(ckpt_data, suite_ckpt)
            update_version_metadata_metrics(version_dir, {
                "Best Epoch": epoch,
                "Detector Training Loss": f"{train_loss:.4f}"
            })
            print(f" -> Saved new best detector checkpoint to {version_ckpt} and {suite_ckpt}", flush=True)

        clean_gpu_memory()

if __name__ == "__main__":
    train()
