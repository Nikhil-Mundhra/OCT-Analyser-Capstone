import os
import sys
import torch
import torch.nn.functional as F
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

from models_suite.model1_oct5k_layers.unet_layers import RetinalLayersUNet
from dataset import OCT5KLayersDataset
import config

LAYER_NAMES = [
    "0. Background / Vitreous",
    "1. Inner Retina (ILM -> OPL)",
    "2. ONL / ELM (OPL -> IS-OS)",
    "3. Ellipsoid Zone (IS-OS -> IBRPE)",
    "4. RPE / Bruch's (IBRPE -> OBRPE)",
    "5. Choroid / Sclera (OBRPE & below)"
]

def compute_dice_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 6):
    dice_per_class = []
    iou_per_class = []
    
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = np.logical_and(pred_c, target_c).sum()
        total_pixels = pred_c.sum() + target_c.sum()
        union = np.logical_or(pred_c, target_c).sum()
        
        if total_pixels == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / total_pixels
            
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
            
        dice_per_class.append(dice)
        iou_per_class.append(iou)
        
    return dice_per_class, iou_per_class

def validate_model1(checkpoint_path: str = None, num_samples_visualize: int = 5):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n==========================================================================")
    print(f"   COMPREHENSIVE MODULAR VALIDATION: MODEL 1 (OCT5K 6-LAYER U-NET)       ")
    print(f"==========================================================================")
    print(f"Device: {device}")

    if checkpoint_path is None:
        checkpoint_path = Path(config.CHECKPOINT_DIR) / "model1_oct5k_layers_best.pth"

    if not Path(checkpoint_path).exists():
        print(f"[Error] Checkpoint not found at: {checkpoint_path}")
        return

    model = RetinalLayersUNet(in_channels=1, num_classes=config.NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from Epoch {ckpt.get('epoch', '?')} (Val Loss: {ckpt.get('val_loss', 0.0):.4f})")
    else:
        model.load_state_dict(ckpt)
        print("Loaded model state dict directly.")

    model.eval()
    dataset = OCT5KLayersDataset(root_dir=config.DATASET_ROOT)
    print(f"Validating on total {len(dataset)} OCT5K B-scan pairs...\n")

    output_dir = SCRIPT_DIR / "validation_outputs"
    output_dir.mkdir(exist_ok=True)

    all_dice = []
    all_iou = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating Model 1"):
            img_tensor, mask_tensor = dataset[idx]
            img_input = img_tensor.unsqueeze(0).to(device)
            
            logits = model(img_input)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            targets = mask_tensor.numpy()

            dice_cls, iou_cls = compute_dice_iou(preds, targets, num_classes=config.NUM_CLASSES)
            all_dice.append(dice_cls)
            all_iou.append(iou_cls)

    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)

    mean_dice_per_class = all_dice.mean(axis=0)
    mean_iou_per_class = all_iou.mean(axis=0)

    # Save summary dataframe
    df = pd.DataFrame({
        "Layer_Class": LAYER_NAMES,
        "Mean_Dice": mean_dice_per_class,
        "Mean_IoU": mean_iou_per_class
    })
    
    csv_path = output_dir / "model1_layer_validation_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("\n--------------------------------------------------------------------------")
    print("                      MODEL 1 VALIDATION METRICS SUMMARY                  ")
    print("--------------------------------------------------------------------------")
    for name, dice, iou in zip(LAYER_NAMES, mean_dice_per_class, mean_iou_per_class):
        print(f"Layer: {name:40s} | Dice: {dice * 100:.2f}% | IoU: {iou * 100:.2f}%")

    overall_mdice = mean_dice_per_class.mean()
    overall_miou = mean_iou_per_class.mean()
    print("--------------------------------------------------------------------------")
    print(f"OVERALL MEAN DICE (mDice): {overall_mdice * 100:.2f}% | OVERALL MEAN IOU (mIoU): {overall_miou * 100:.2f}%")
    print("--------------------------------------------------------------------------\n")

    # Render Visualizations
    print(f"Rendering top {num_samples_visualize} 2x2 grid validation samples...")
    fig_dir = output_dir / "2x2_grids"
    fig_dir.mkdir(exist_ok=True)

    for idx in range(min(num_samples_visualize, len(dataset))):
        img_tensor, mask_tensor = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_input)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        mask_np = mask_tensor.numpy()

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Top-Left: Raw B-scan
        axes[0, 0].imshow(img_np, cmap="gray")
        axes[0, 0].set_title(f"1. Raw OCT B-Scan (Sample #{idx+1})", fontsize=10, fontweight='bold')
        axes[0, 0].axis("off")

        # Top-Right: Ground Truth Layer Mask
        im_gt = axes[0, 1].imshow(mask_np, cmap="inferno", vmin=0, vmax=5)
        axes[0, 1].set_title("2. Ground Truth Layer Mask (0-5)", fontsize=10, fontweight='bold')
        axes[0, 1].axis("off")
        fig.colorbar(im_gt, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Bottom-Left: Predicted Layer Mask
        im_pred = axes[1, 0].imshow(preds, cmap="inferno", vmin=0, vmax=5)
        axes[1, 0].set_title("3. Predicted Layer Mask (U-Net)", fontsize=10, fontweight='bold')
        axes[1, 0].axis("off")
        fig.colorbar(im_pred, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Bottom-Right: Overlay
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        color_mask = plt.cm.get_cmap("rainbow")(preds / 5.0)[:, :, :3] * 255
        overlay = (img_rgb * 0.4 + color_mask * 0.6).astype(np.uint8)

        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("4. Multi-Layer Color Overlay", fontsize=10, fontweight='bold')
        axes[1, 1].axis("off")

        plt.suptitle(f"Model 1 Validation Result - Sample #{idx+1}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / f"model1_val_sample_{idx+1}.png", dpi=120, bbox_inches='tight')
        plt.close()

    print(f"Validation complete! Saved metrics to {csv_path}")

if __name__ == "__main__":
    validate_model1()
