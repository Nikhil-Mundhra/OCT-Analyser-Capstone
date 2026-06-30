import os
import sys
from pathlib import Path
import logging

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

from data.transforms import get_transforms
from models.level1_gatekeeper import build_gatekeeper

# Suppress MacOS OpenMP error
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


class OCTTestDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, option=1):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        df = pd.read_csv(csv_path)
        
        # Filter based on Option 2 if requested
        if option == 2:
            df = df[df['Class'] == df['Label']]
            
        self.manifest = df.reset_index(drop=True)
        
        # Level 1 mapping: Normal=0, Abnormal(Drusen, CNV)=1
        self.label_map = {'normal': 0, 'drusen': 1, 'cnv': 1}
        
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img_path = self.data_dir / row['Directory']
        label_str = str(row['Label']).lower().strip()
        
        # Map label to 0 or 1
        label = self.label_map.get(label_str, -1)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {img_path}: {e}")
            image = Image.new("RGB", (224, 224), color=0)
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Level 1 Model on External Test Set")
    parser.add_argument("--data-dir", type=str, 
                        default="/Users/nikhilmundhra/Downloads/Capstone/DataSets/test/Labeled Retinal Optical Coherence Tomography Dataset for Classification of Normal, Drusen, and CNV Cases 2021",
                        help="Path to dataset directory")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/level1/fold0_best_model.pth",
                        help="Path to trained Level 1 checkpoint")
    parser.add_argument("--option", type=int, choices=[1, 2], default=1,
                        help="Read option (1: All images, 2: Worst-case images)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Classification threshold for Abnormal class (default: 0.35)")
    args = parser.parse_args()
    
    csv_path = Path(args.data_dir) / "data_information.csv"
    
    if not csv_path.exists():
        logger.error(f"CSV file not found at: {csv_path}")
        return
        
    # Define device - explicitly CPU as requested by user
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Build Model
    logger.info("Building model...")
    model = build_gatekeeper(num_classes=2, pretrained=False)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        return
        
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 2. Build Dataset & DataLoader
    logger.info(f"Loading dataset from {args.data_dir} (Option {args.option})")
    transform = get_transforms("level1", "val")
    dataset = OCTTestDataset(csv_path, args.data_dir, transform=transform, option=args.option)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False
    )
    
    logger.info(f"Total test images: {len(dataset)}")
    
    # 3. Evaluation Loop
    all_preds = []
    all_probs = []
    all_labels = []
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i % 10 == 0:
                logger.info(f"Evaluating batch {i}/{len(dataloader)}")
            images = images.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            # Apply custom threshold for class 1 (Abnormal)
            preds = (probs[:, 1] >= args.threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Prob of class 1 (Abnormal)
            all_labels.extend(labels.numpy())
            
    # 4. Compute Metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Filter out valid labels (just in case any -1 sneaked in)
    valid_mask = all_labels != -1
    y_true = all_labels[valid_mask]
    y_pred = all_preds[valid_mask]
    y_prob = all_probs[valid_mask]
    
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float('nan')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS - Level 1 Gatekeeper (Normal vs Abnormal)")
    logger.info("="*50)
    logger.info(f"Total Samples Tested : {len(y_true)}")
    logger.info(f"Accuracy             : {acc:.4f}")
    logger.info(f"AUROC                : {auroc:.4f}")
    logger.info(f"Macro F1             : {macro_f1:.4f}")
    logger.info(f"Weighted F1          : {weighted_f1:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))

if __name__ == "__main__":
    main()
