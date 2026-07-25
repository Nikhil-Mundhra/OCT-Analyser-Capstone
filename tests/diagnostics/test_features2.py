import torch
from pathlib import Path
from backend.core_ml.segmentation.models.unet import HierarchicalUNet
import numpy as np

device = torch.device("cpu")
model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
checkpoint_path = Path("backend/core_ml/segmentation/weights/unet_hierarchical_best_cls.pth")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

for name, x in [("Zeros", torch.zeros(1, 1, 512, 512)), ("Ones", torch.ones(1, 1, 512, 512)), ("Randn", torch.randn(1, 1, 512, 512))]:
    with torch.no_grad():
        outputs = model(x, task="classification")
    print(f"--- {name} ---")
    print("L1:", outputs['normal_abnormal'].numpy())
    print("L2:", outputs['pathology'].numpy())

