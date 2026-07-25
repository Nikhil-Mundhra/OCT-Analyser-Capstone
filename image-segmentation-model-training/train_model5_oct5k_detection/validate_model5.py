import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm

WORKSPACE_ROOT = Path("/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models_suite.model5_oct5k_detection.detector import OCTPathologyDetector, OCT5K_DETECTION_CLASSES
from dataset import OCT5KDetectionDataset
import config

def validate_model5(checkpoint_path: str = None, num_samples_visualize: int = 5, confidence_threshold: float = 0.4):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n==========================================================================")
    print(f"  COMPREHENSIVE MODULAR VALIDATION: MODEL 5 (OCT5K OBJECT DETECTOR)       ")
    print(f"==========================================================================")
    print(f"Device: {device}")

    if checkpoint_path is None:
        checkpoint_path = Path(config.CHECKPOINT_DIR) / "model5_detection_best.pth"

    if not Path(checkpoint_path).exists():
        print(f"[Error] Checkpoint not found at: {checkpoint_path}")
        return

    model = OCTPathologyDetector(num_classes=config.NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from Epoch {ckpt.get('epoch', '?')} (Train Loss: {ckpt.get('train_loss', 0.0):.4f})")
    else:
        model.load_state_dict(ckpt)
        print("Loaded model state dict directly.")

    model.eval()
    dataset = OCT5KDetectionDataset(root_dir=config.DATASET_ROOT)
    print(f"Validating on total {len(dataset)} OCT5K Detection images...\n")

    output_dir = SCRIPT_DIR / "validation_outputs"
    output_dir.mkdir(exist_ok=True)
    fig_dir = output_dir / "2x2_grids"
    fig_dir.mkdir(exist_ok=True)

    detection_counts = {cls_name: 0 for cls_name in OCT5K_DETECTION_CLASSES}

    with torch.no_grad():
        for idx in range(min(num_samples_visualize, len(dataset))):
            img_tensor, target = dataset[idx]
            img_input = [img_tensor.to(device)]
            
            predictions = model(img_input)[0]
            boxes = predictions["boxes"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()

            img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            # Draw Ground Truth Boxes
            gt_img = img_rgb.copy()
            gt_boxes = target["boxes"].numpy()
            gt_labels = target["labels"].numpy()

            for box, label_id in zip(gt_boxes, gt_labels):
                xmin, ymin, xmax, ymax = map(int, box)
                cls_name = OCT5K_DETECTION_CLASSES[label_id] if label_id < len(OCT5K_DETECTION_CLASSES) else f"Class_{label_id}"
                cv2.rectangle(gt_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(gt_img, cls_name, (xmin, max(15, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw Predicted Boxes
            pred_img = img_rgb.copy()
            for box, score, label_id in zip(boxes, scores, labels):
                if score >= confidence_threshold:
                    xmin, ymin, xmax, ymax = map(int, box)
                    cls_name = OCT5K_DETECTION_CLASSES[label_id] if label_id < len(OCT5K_DETECTION_CLASSES) else f"Class_{label_id}"
                    detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                    
                    cv2.rectangle(pred_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.putText(pred_img, f"{cls_name}: {score:.2f}", (xmin, max(15, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(gt_img)
            axes[0].set_title(f"1. Ground Truth Bounding Boxes (Sample #{idx+1})", fontsize=10, fontweight='bold')
            axes[0].axis("off")

            axes[1].imshow(pred_img)
            axes[1].set_title(f"2. Predicted Bounding Boxes (Faster R-CNN)", fontsize=10, fontweight='bold')
            axes[1].axis("off")

            plt.suptitle(f"Model 5 Detection Validation - Sample #{idx+1}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(fig_dir / f"model5_val_sample_{idx+1}.png", dpi=120, bbox_inches='tight')
            plt.close()

    print("\n--------------------------------------------------------------------------")
    print("                      MODEL 5 DETECTION SUMMARY                           ")
    print("--------------------------------------------------------------------------")
    for cls_name, count in detection_counts.items():
        print(f"Pathology Biomarker: {cls_name:30s} | Detected Instances: {count}")

    print(f"\nValidation complete! Saved grid visualizer outputs to {fig_dir}")

if __name__ == "__main__":
    validate_model5()
