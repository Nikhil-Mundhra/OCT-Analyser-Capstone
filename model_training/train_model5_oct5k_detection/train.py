import os
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from models_suite.model5_oct5k_detection.detector import OCTPathologyDetector
from dataset import OCT5KDetectionDataset
import config

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training Model 5 (OCT5K Pathology Detection) on device: {device}")

    full_dataset = OCT5KDetectionDataset(root_dir=config.DATASET_ROOT)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    detector = OCTPathologyDetector(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        detector.train()
        train_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = detector(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch:02d} | Detector Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            ckpt_path = Path(config.CHECKPOINT_DIR) / "model5_best.pth"
            torch.save(detector.state_dict(), ckpt_path)
            print(f" Saved new best detector checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train()
