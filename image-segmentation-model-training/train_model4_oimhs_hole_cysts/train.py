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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models_suite.model4_oimhs_hole_cysts.oimhs_unet import OIMHSUNet
from dataset import OIMHSDataset
import config

def train():
    # HARDWIRED CLEANUP & SINGLE INSTANCE GUARD
    enforce_single_instance_and_clean_memory("train_model4_oimhs_hole_cysts")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training Model 4 (OIMHS Macular Hole & Cysts U-Net 512x512) on device: {device}", flush=True)

    full_dataset = OIMHSDataset(root_dir=config.DATASET_ROOT)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = OIMHSUNet(in_channels=1, num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    suite_ckpt_dir = WORKSPACE_ROOT / "models_suite" / "model4_oimhs_hole_cysts" / "checkpoints"
    os.makedirs(suite_ckpt_dir, exist_ok=True)

    best_loss = float('inf')
    accumulation_steps = getattr(config, "ACCUMULATION_STEPS", 4)

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches:
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item() * accumulation_steps
            train_loss += batch_loss * images.size(0)

            # Periodic memory flush every 25 batches
            if (i + 1) % 25 == 0:
                clean_gpu_memory()

            if (i + 1) % 100 == 0 or (i + 1) == total_batches:
                print(f"Epoch {epoch:02d}/{config.EPOCHS:02d} | Batch {i+1}/{total_batches} | Loss: {batch_loss:.4f}", flush=True)

            del outputs, loss, images, masks

        train_loss /= len(train_set)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                del outputs, loss, images, masks

        val_loss /= len(val_set)
        print(f">>> Epoch {epoch:02d}/{config.EPOCHS:02d} COMPLETE | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }
            local_ckpt = Path(config.CHECKPOINT_DIR) / "model4_oimhs_best.pth"
            suite_ckpt = suite_ckpt_dir / "best_model.pth"
            
            torch.save(ckpt_data, local_ckpt)
            torch.save(ckpt_data, suite_ckpt)
            print(f" -> Saved new best checkpoint to {suite_ckpt}", flush=True)

        # HARDWIRED END-OF-EPOCH MEMORY FLUSH
        clean_gpu_memory()

if __name__ == "__main__":
    train()
