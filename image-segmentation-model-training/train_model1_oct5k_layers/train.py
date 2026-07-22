import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

WORKSPACE_ROOT = Path("/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models_suite.model1_oct5k_layers.unet_layers import RetinalLayersUNet
from dataset import OCT5KLayersDataset
import config

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training Model 1 (OCT5K 6-Retinal Layer U-Net) on device: {device}", flush=True)

    full_dataset = OCT5KLayersDataset(root_dir=config.DATASET_ROOT)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = RetinalLayersUNet(in_channels=1, num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    suite_ckpt_dir = WORKSPACE_ROOT / "models_suite" / "model1_oct5k_layers" / "checkpoints"
    os.makedirs(suite_ckpt_dir, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

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

        val_loss /= len(val_set)
        print(f"Epoch {epoch:02d}/{config.EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }
            local_ckpt = Path(config.CHECKPOINT_DIR) / "model1_oct5k_layers_best.pth"
            suite_ckpt = suite_ckpt_dir / "best_model.pth"
            
            torch.save(ckpt_data, local_ckpt)
            torch.save(ckpt_data, suite_ckpt)
            print(f" -> Saved new best checkpoint to {suite_ckpt}", flush=True)

if __name__ == "__main__":
    train()
