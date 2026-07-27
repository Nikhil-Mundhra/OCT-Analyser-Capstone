import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from models_suite.model1_oct5k_layers.unet_layers import RetinalLayersUNet
from dataset import OCT5KLayersDataset
import config

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training Model 1 (Retinal Layers) on device: {device}")

    full_dataset = OCT5KLayersDataset(root_dir=config.DATASET_ROOT)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = RetinalLayersUNet(in_channels=1, num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_set)

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
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = Path(config.CHECKPOINT_DIR) / "model1_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f" Saved new best checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train()
