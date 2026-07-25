import torch
from pathlib import Path
from backend.core_ml.segmentation.models.unet import HierarchicalUNet
import numpy as np

device = torch.device("cpu")
model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
checkpoint_path = Path("backend/core_ml/segmentation/weights/unet_hierarchical_best_cls.pth")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

l2_bias = model.pathology_type_head[-1].bias.data.numpy()
print("L2 Pathology bias:", np.round(l2_bias, 3))
print("L2 Softmax with 0 features:", np.round(torch.softmax(model.pathology_type_head[-1].bias.data, dim=0).numpy(), 3))

l1_bias = model.normal_abnormal_head[-1].bias.data.numpy()
print("L1 bias:", np.round(l1_bias, 3))
print("L1 Sigmoid with 0 features:", np.round(torch.sigmoid(model.normal_abnormal_head[-1].bias.data).numpy(), 3))
