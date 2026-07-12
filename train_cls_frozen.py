import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import argparse
import importlib.util
import cv2
import numpy as np

# Setup paths
local_path = str(Path(__file__).resolve().parent / "image-segmentation-model-training" / "training")
sys.path.insert(0, local_path)

# Import model
from backend.core_ml.segmentation.models.unet import HierarchicalUNet
import torchvision.transforms as transforms
from PIL import Image

# Load Dataset module
cls_path = Path(__file__).resolve().parent / 'image-classification-model-training' / 'data' / 'dataset.py'
spec = importlib.util.spec_from_file_location("cls_dataset_module", str(cls_path))
cls_module = importlib.util.module_from_spec(spec)
sys.modules["cls_dataset_module"] = cls_module
spec.loader.exec_module(cls_module)
MultiHeadOCTDataset = cls_module.MultiHeadOCTDataset

class FocalLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def load_image_gray(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return Image.fromarray(img)

class CLAHE_Transform_PIL:
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        res = self.clahe.apply(arr)
        return Image.fromarray(res)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-data", type=str, required=True, help="Path to base classification dataset")
    parser.add_argument("--cls-config", type=str, required=True, help="Path to hierarchy.yaml for classification")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained segmentation checkpoint (to freeze the encoder)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Low learning rate for classification heads")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    cls_transform = transforms.Compose([
        transforms.Lambda(load_image_gray),
        CLAHE_Transform_PIL(clip_limit=2.0, tile_grid=(8, 8)),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    print("Loading dataset...")
    cls_dataset = MultiHeadOCTDataset(config_path=args.cls_config, data_root=args.cls_data, transform=cls_transform)
    
    cls_train_size = int(0.9 * len(cls_dataset))
    cls_train_sub, cls_val_sub = random_split(
        cls_dataset, [cls_train_size, len(cls_dataset) - cls_train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(cls_train_sub, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(cls_val_sub, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Loading model and freezing encoder...")
    model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # FREEZE the entire encoder and decoder
    for param in model.parameters():
        param.requires_grad = False
        
    # UNFREEZE the classification heads
    heads_to_train = [
        model.cls_project,
        model.normal_abnormal_head,
        model.pathology_type_head,
        model.macular_head,
        model.diabetic_head,
        model.vascular_head,
        model.fluid_head,
        model.structural_head
    ]
    for head in heads_to_train:
        for param in head.parameters():
            param.requires_grad = True

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable classification parameters: {n_params:,}")

    h2_alpha = cls_dataset.compute_class_weights("l2").to(device)
    criterion_h1 = nn.BCEWithLogitsLoss().to(device)
    criterion_h2 = FocalLoss(weight=h2_alpha, ignore_index=-1, gamma=2.0).to(device)
    criterion_h3 = nn.BCEWithLogitsLoss().to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-3)
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        # Set the entire model to eval mode to freeze BatchNorm running stats in the encoder!
        model.eval()
        
        # Then, only set the classification heads to train mode (for Dropout)
        for head in heads_to_train:
            head.train()
            
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            labels = {
                "normal_abnormal": targets["normal_abnormal"].to(device),
                "pathology": targets["pathology"].to(device),
                "severity": {k: v.to(device) for k, v in targets["severity"].items()}
            }

            optimizer.zero_grad()
            
            # Since the encoder is frozen and requires_grad=False, PyTorch will automatically
            # stop tracking gradients at the classification head inputs.
            logits = model(images, task="classification")
            
            loss_h1 = criterion_h1(logits["normal_abnormal"], labels["normal_abnormal"].float())
            
            valid_h2 = labels["pathology"] != -1
            if valid_h2.sum() > 0:
                loss_h2 = criterion_h2(logits["pathology"][valid_h2], labels["pathology"][valid_h2])
            else:
                loss_h2 = torch.tensor(0.0, device=device, requires_grad=True)
                
            loss_h3 = sum(criterion_h3(logits["severity"][k], labels["severity"][k].float()) for k in ["macular", "diabetic", "vascular", "fluid", "structural"])
            
            loss = (1.0 * loss_h1) + (2.0 * loss_h2) + (0.5 * loss_h3)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Complete. Avg Train Loss: {train_loss / len(train_loader):.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"checkpoints/cls_frozen_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
