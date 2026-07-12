import torch
from pathlib import Path
from backend.core_ml.segmentation.models.unet import HierarchicalUNet
import pandas as pd
import numpy as np
import cv2
from PIL import Image

device = torch.device("cpu")
model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
checkpoint_path = Path("backend/core_ml/segmentation/weights/unet_hierarchical_best_cls.pth")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

img = cv2.imread("dummy.jpeg", cv2.IMREAD_GRAYSCALE)
img_min, img_max = img.min(), img.max()
img_norm = (img - img_min) / (img_max - img_min)
img_resized = cv2.resize(img_norm, (512, 512), interpolation=cv2.INTER_LINEAR)

# Test [0, 1]
tensor_01 = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()
with torch.no_grad():
    coarse_01, _, _ = model(tensor_01, task="both")
    pred_01 = torch.argmax(coarse_01, dim=1)[0].numpy()
print("0-1 Segmentation unique values:", np.unique(pred_01))

# Test [-1, 1]
tensor_11 = (tensor_01 * 2.0) - 1.0
with torch.no_grad():
    coarse_11, _, _ = model(tensor_11, task="both")
    pred_11 = torch.argmax(coarse_11, dim=1)[0].numpy()
print("-1 to 1 Segmentation unique values:", np.unique(pred_11))
