import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.segmentation_dataset import OCT5kSegmentationDataset
from models.unet import HierarchicalUNet

def verify():
    dataset_path = "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Bad data/OCT5k_Segmentation_Subset"
    dataset = OCT5kSegmentationDataset(root_dir=dataset_path)
    
    print(f"Dataset size: {len(dataset)}")
    img, c_mask, g_mask = dataset[0]
    
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Coarse mask shape: {c_mask.shape}, dtype: {c_mask.dtype}")
    print(f"Granular mask shape: {g_mask.shape}, dtype: {g_mask.dtype}")
    
    # Add batch dim
    img = img.unsqueeze(0)
    
    # Model
    model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
    
    c_logits, g_logits = model(img)
    
    print(f"Coarse logits shape: {c_logits.shape}")
    print(f"Granular logits shape: {g_logits.shape}")
    
    assert c_logits.shape == (1, 3, img.shape[2], img.shape[3]), "Coarse logits shape mismatch"
    assert g_logits.shape == (1, 15, img.shape[2], img.shape[3]), "Granular logits shape mismatch"
    
    print("Verification Passed!")

if __name__ == "__main__":
    verify()
