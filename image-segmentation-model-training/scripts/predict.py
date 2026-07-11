import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import sys
# Ensure we can import from the training folder
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.unet import HierarchicalUNet

# L2 Class Mapping from hierarchy.yaml
L2_CLASSES = {
    0: "Macular Degeneration",
    1: "Diabetic Complications",
    2: "Vascular Occlusions",
    3: "Fluid Accumulation",
    4: "Structural Issues"
}

def load_image_gray(p):
    """Loads image to grayscale using PIL to match training pipeline."""
    return Image.open(p).convert('L') if isinstance(p, str) else p

def predict(image_path: str, checkpoint_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the Model
    print(f"Loading model from {checkpoint_path}...")
    model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Weights loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
        
    model.to(device)
    
    # FIX COVARIATE SHIFT FOR INFERENCE: 
    # Force BN to use batch statistics (like we did in validation)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False

    model.eval()
    
    # 2. Load and Preprocess the Image
    print(f"Processing image {image_path}...")
    
    # Replicate the exact training transform pipeline
    img_pil = load_image_gray(image_path)
    img_resized = img_pil.resize((256, 256), Image.BILINEAR)
    img_array = np.array(img_resized)
    
    # Normalize to [0, 1]
    img_normalized = img_array.astype(np.float32) / 255.0
    
    # Convert to tensor: (1, 1, 256, 256)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Model Inference
    print("Running forward pass...")
    with torch.no_grad():
        # A. Classification Task (Hierarchical)
        cls_logits = model(img_tensor, task="classification")
        
        # H1: Normal vs Abnormal
        prob_abnormal = torch.sigmoid(cls_logits["normal_abnormal"]).item()
        is_abnormal = prob_abnormal > 0.5
        
        predicted_disease = "Normal (Healthy)"
        biomarkers = []
        
        if is_abnormal:
            # H2: Broad Pathology
            l2_pred_idx = torch.argmax(cls_logits["pathology"], dim=1).item()
            predicted_disease = L2_CLASSES.get(l2_pred_idx, f"Unknown ({l2_pred_idx})")
            
            # H3: Granular Biomarkers
            for key, logits in cls_logits["severity"].items():
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                for i, p in enumerate(probs):
                    if p > 0.5:
                        biomarkers.append(f"{key}_{i} (conf: {p:.2f})")
        
        # B. Segmentation Task
        c_logits, g_logits = model(img_tensor, task="segmentation")
        coarse_preds = torch.argmax(c_logits, dim=1).squeeze(0).cpu().numpy()
        granular_preds = torch.argmax(g_logits, dim=1).squeeze(0).cpu().numpy()
    
    print("\n" + "="*40)
    print("DIAGNOSTIC REPORT")
    print("="*40)
    print(f"H1 (Binary)    : {'Abnormal' if is_abnormal else 'Normal'} (prob: {prob_abnormal:.4f})")
    print(f"H2 (Pathology) : {predicted_disease}")
    if biomarkers:
        print("H3 (Biomarkers):")
        for b in biomarkers:
            print(f"  - {b}")
    else:
        print("H3 (Biomarkers): None detected above 50% confidence.")
    print("="*40 + "\n")
    
    # 4. Visualization
    print("Plotting results...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title_text = f"Diagnosis: {predicted_disease}"
    fig.suptitle(title_text, fontsize=20, fontweight='bold', color='darkred' if is_abnormal else 'darkgreen')
    
    # Plot Original Image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title("Original Image (256x256)")
    axes[0].axis('off')
    
    # Plot Coarse Mask
    # Map classes 0, 1, 2 to colors for better visibility
    axes[1].imshow(coarse_preds, cmap='viridis', interpolation='nearest')
    axes[1].set_title("Coarse Mask (3 Regions)")
    axes[1].axis('off')
    
    # Plot Granular Mask
    axes[2].imshow(granular_preds, cmap='tab20', interpolation='nearest')
    axes[2].set_title("Granular Mask (15 Layers)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Task OCT Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input OCT scan (PNG/TIF)")
    parser.add_argument("--checkpoint", type=str, default="../../unet_hierarchical_best_cls.pth", help="Path to model checkpoint")
    
    args = parser.parse_args()
    predict(args.image, args.checkpoint)
