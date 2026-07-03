import os
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.level1_gatekeeper import build_gatekeeper
from models.level2_router import build_router
from data.transforms import get_transforms
from utils.gradcam import GradCAM

def main():
    device = torch.device("cpu")
    
    # 1. Load Image
    img_path = "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified/Diabetic Complications/Diabetic Macular Edema (DME)/DME/DME-4441781-1.jpeg"
    img = Image.open(img_path).convert("RGB")
    
    # 2. Transform
    transform = get_transforms("level1", "val")
    tensor_224 = transform(img).unsqueeze(0).to(device)
    tensor_224.requires_grad = True
    
    # 3. Build L1
    l1_model = build_gatekeeper(pretrained=True).to(device)
    l1_model.eval()
    
    # 4. Build L2
    l2_model = build_router(pretrained=True).to(device)
    l2_model.eval()
    
    with torch.enable_grad():
        # L1 CAM
        l1_cam_gen = GradCAM(l1_model, l1_model.features[-1])
        
        feats = l1_model.features(tensor_224)
        print(f"Features non-zero count: {torch.count_nonzero(feats)}")
        print(f"Features max: {feats.max().item()}, min: {feats.min().item()}")
        
        logits_l1 = l1_model(tensor_224)
        print(f"Logits: {logits_l1}")
        pred_l1_idx = torch.argmax(logits_l1).item()
        cam1 = l1_cam_gen.generate_cam(tensor_224, pred_l1_idx)
        print(f"L1 CAM - Max: {np.max(cam1)}, Min: {np.min(cam1)}, Sum: {np.sum(cam1)}")
        
        # L2 CAM
        l2_cam_gen = GradCAM(l2_model, l2_model.features[-1])
        logits_l2 = l2_model(tensor_224)
        pred_l2_idx = torch.argmax(logits_l2).item()
        cam2 = l2_cam_gen.generate_cam(tensor_224, pred_l2_idx)
        print(f"L2 CAM - Max: {np.max(cam2)}, Min: {np.min(cam2)}, Sum: {np.sum(cam2)}")
        
        # Are they identical?
        print(f"Are CAM1 and CAM2 exactly equal? {np.array_equal(cam1, cam2)}")
        
if __name__ == "__main__":
    main()
