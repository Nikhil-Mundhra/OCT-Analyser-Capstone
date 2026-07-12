import torch
from pathlib import Path
from backend.core_ml.segmentation.models.unet import HierarchicalUNet
import pandas as pd
import numpy as np
import cv2

device = torch.device("cpu")
model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
checkpoint_path = Path("backend/core_ml/segmentation/weights/unet_hierarchical_best_cls.pth")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

manifest = pd.read_csv("image-classification-model-training/dataset_manifest.csv")
for i in range(5):
    row = manifest.iloc[i]
    img_path = Path("image-classification-model-training") / "data" / row["image_path"].split("data/")[-1]
    
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
        
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Division by 255
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    
    with torch.no_grad():
        outputs = model(img_tensor, task="classification")
        l1_prob = torch.sigmoid(outputs['normal_abnormal'][0, 0]).item()
        l2_probs = torch.softmax(outputs['pathology'][0], dim=0).numpy()
        
    print(f"Path: {img_path.name}")
    print(f"  Pred L1 Prob: {l1_prob:.4f}")
    print(f"  Pred L2 Probs: {np.round(l2_probs, 2)}")
