"""
scripts/test_dataloader.py

Sanity harness to test the MultiHeadOCTDataset on the Micro-Dataset.
"""
import sys
import os
import torch

# Ensure image-classification-model-training is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import build_dataloader
from data.transforms import get_transforms

def main():
    print("Testing MultiHeadOCTDataset against Micro-Dataset...")
    
    # Paths relative to the scripts directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(base_dir, "config/hierarchy.yaml")
    data_root = os.path.join(base_dir, "data/micro_dataset")
    
    print(f"Config path: {config_path}")
    print(f"Data root: {data_root}")
    
    transforms = get_transforms("val")
    
    loader = build_dataloader(
        config_path=config_path,
        data_root=data_root,
        batch_size=4,
        num_workers=0, # Use 0 for testing in script
        transform=transforms,
        shuffle=False
    )
    
    print(f"Total batches: {len(loader)}")
    
    for i, (images, targets) in enumerate(loader):
        print(f"\n--- Batch {i+1} ---")
        print(f"Image tensor shape: {images.shape} | dtype: {images.dtype}")
        
        # Check shapes
        assert images.shape[1:] == (3, 384, 384), f"Unexpected image shape: {images.shape}"
        
        # Check targets
        h1 = targets["normal_abnormal"]
        h2 = targets["pathology"]
        h3 = targets["severity"]
        
        print(f"H1 (Binary) shape: {h1.shape} | values: {h1.squeeze().tolist()}")
        print(f"H2 (Router) shape: {h2.shape} | values: {h2.tolist()}")
        
        print("H3 (Severity) sub-tensors:")
        for k, v in h3.items():
            print(f"  - {k}: shape {v.shape}")
            
        if i == 0:
            print("\nTest passed for first batch. Dataset works correctly!")
            break

if __name__ == "__main__":
    main()
