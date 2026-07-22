import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.multi_head_convnext import build_multi_head_model
from data.transforms import get_transforms
from utils.gradcam import MultiHeadGradCAM
PATHOLOGY_CLASSES = [
    'CNV', 'DRUSEN', 'AMD', 'General_AMD', 
    'DME', 'DR', 'MH', 'RVO', 'RAO', 
    'CSR', 'ERM', 'VID'
]

def main():
    device = torch.device("cpu")
    
    # 1. Load Image
    img_path = "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified/Diabetic Complications/Diabetic Macular Edema (DME)/DME/DME-4441781-1.jpeg"
    img = Image.open(img_path).convert("RGB")
    
    # 2. Transform
    transform = get_transforms("val")
    tensor_224 = transform(img).unsqueeze(0).to(device)
    tensor_224.requires_grad = True
    
    # 3. Build Multi-Head Model
    model = build_multi_head_model(pretrained=True).to(device)
    model.eval()
    
    # Initialize Multi-Head Grad-CAM on the last backbone stage
    # For ConvNeXt, the last feature map is stage 3
    cam_gen = MultiHeadGradCAM(model, model.backbone)
    
    with torch.enable_grad():
        logits = model(tensor_224)
        print("Model Forward Pass Successful")
        print(f"H1 Logits: {logits['normal_abnormal']}")
        print(f"H2 Logits (shape {logits['pathology'].shape}): {logits['pathology'].detach().numpy()}")
        
        # H1 CAM (Triage)
        pred_h1_prob = torch.sigmoid(logits['normal_abnormal']).item()
        print(f"\nH1 Prediction: {'Abnormal' if pred_h1_prob > 0.5 else 'Normal'} ({pred_h1_prob:.4f})")
        
        cam1 = cam_gen.generate_cam(tensor_224, target_head=1)
        print(f"H1 CAM - Max: {np.max(cam1):.4f}, Min: {np.min(cam1):.4f}, Sum: {np.sum(cam1):.4f}")
        
        # H2 CAM (Pathology Router)
        pred_h2_idx = torch.argmax(logits['pathology']).item()
        pred_h2_class = PATHOLOGY_CLASSES[pred_h2_idx]
        print(f"\nH2 Top Prediction: {pred_h2_class}")
        
        cam2 = cam_gen.generate_cam(tensor_224, target_head=2, target_class=pred_h2_idx)
        print(f"H2 CAM - Max: {np.max(cam2):.4f}, Min: {np.min(cam2):.4f}, Sum: {np.sum(cam2):.4f}")
        
        # Overlay and save examples (optional)
        # overlay1 = MultiHeadGradCAM.overlay_cam(img, cam1)
        # overlay2 = MultiHeadGradCAM.overlay_cam(img, cam2)
        
        print("\nMulti-Head Grad-CAM test completed successfully!")

if __name__ == "__main__":
    main()
