import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib.backends.backend_pdf import PdfPages
import torch.nn.functional as F

import torch.nn.init
try:
    import timm.layers.weight_init
    timm.layers.weight_init.trunc_normal_ = lambda tensor, mean=0., std=1., a=-2., b=2.: torch.nn.init.normal_(tensor, mean=mean, std=std)
except ImportError:
    pass

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multi_head_convnext import build_multi_head_model
from data.dataset import MultiHeadOCTDataset

PATHOLOGY_CLASSES = [
    'CNV', 'DRUSEN', 'AMD', 'General_AMD', 
    'DME', 'DR', 'MH', 'RVO', 'RAO', 
    'CSR', 'ERM', 'VID'
]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx, head='pathology'):
        self.model.zero_grad()
        logits = self.model(x)
        
        if head == 'pathology':
            score = logits[head][0, class_idx]
        elif head == 'normal_abnormal':
            score = logits[head][0, 0]
            
        score.backward(retain_graph=True)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        h, w = x.shape[2], x.shape[3]
        cam_resized = F.interpolate(cam_tensor, size=(h, w), mode='bilinear', align_corners=False)
        return cam_resized.squeeze().numpy()

def setup_environment():
    if torch.backends.mps.is_available(): device = torch.device('mps')
    elif torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    print(f"Using device: {device}")
    
    os.makedirs('telemetry_outputs', exist_ok=True)
    pdf_path = 'telemetry_outputs/Full_Evaluation_Report.pdf'
    return device, pdf_path

def get_data_loader(config_path="image-classification-model-training/config/hierarchy.yaml", batch_size=64):
    from data.transforms import get_transforms
    val_transform = get_transforms("val")
    full_dataset = MultiHeadOCTDataset(config_path=config_path, transform=val_transform)
    train_size = int(0.8 * len(full_dataset))
    np.random.seed(42)
    indices = np.random.permutation(len(full_dataset)).tolist()
    val_dataset = Subset(full_dataset, indices[train_size:])
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def load_model(checkpoint_path, device):
    print("Building model...")
    model = build_multi_head_model(pretrained=False, warmup=False)
    
    if not os.path.exists(checkpoint_path):
        # Fallback to local HF hub cache snapshot if available
        hf_cache_snapshot = "/Users/nikhilmundhra/.cache/huggingface/hub/models--NMundhra--OCT-Classification-Model/snapshots/b8b2d5e7347d463a3d5f5d5c671e5e230968a7a6/fold0_best_model.pth"
        if os.path.exists(hf_cache_snapshot):
            checkpoint_path = hf_cache_snapshot
        elif os.path.exists("checkpoints/multi_head/fold0_best_model.pth"):
            checkpoint_path = "checkpoints/multi_head/fold0_best_model.pth"
        else:
            checkpoint_path = "hf_space/weights/multi_head_mps/fold0_best_model.pth"
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    unwrapped = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped)
    model.to(device)
    model.eval()
    
    if hasattr(model.backbone, "stages"):
        target_layer = model.backbone.stages[-1]
    elif hasattr(model.backbone, "stages_3"):
        target_layer = model.backbone.stages_3
    elif hasattr(model.backbone, "body") and hasattr(model.backbone.body, "stages"):
        target_layer = model.backbone.body.stages[-1]
    else:
        target_layer = list(model.backbone.children())[-1]

    grad_cam = GradCAM(model, target_layer)
    return model, grad_cam

def get_overlay(img_tensor, cam):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).numpy().transpose(1, 2, 0)
    
    # Rotate image and CAM 90 degrees clockwise for upright anatomical display
    img = np.rot90(img, -1, (0, 1))
    cam = np.rot90(cam, -1, (0, 1))
    
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.4 * heatmap + 0.6 * img
    return np.clip(overlay, 0, 1), img

def generate_patient_case_study(img_tensor, class_idx, class_name, model, grad_cam, pdf):
    if img_tensor.ndim == 3:
        img = img_tensor.unsqueeze(0)
    elif img_tensor.ndim == 4:
        img = img_tensor
    else:
        img = img_tensor.view(1, 3, img_tensor.shape[-2], img_tensor.shape[-1])
    
    cam_h1 = grad_cam(img, class_idx=0, head='normal_abnormal')
    
    h2_target_class = PATHOLOGY_CLASSES.index(class_name)
    cam_h2 = grad_cam(img, class_idx=h2_target_class, head='pathology')
    
    overlay_h1, orig_img = get_overlay(img[0], cam_h1)
    overlay_h2, _ = get_overlay(img[0], cam_h2)
    
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_img)
    ax1.set_title(f"Original Scan\nTrue: {class_name}")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    with torch.no_grad():
        l = model(img)
        p_h1 = torch.sigmoid(l['normal_abnormal'])[0, 0].item()
        p_h2_probs = torch.softmax(l['pathology'], dim=1)[0].cpu().numpy()
        
        pred_h1 = "Abnormal" if p_h1 > 0.5 else "Normal"
        pred_h2_idx = np.argmax(p_h2_probs)
        pred_class_name = PATHOLOGY_CLASSES[pred_h2_idx]
    
    h2_str = "\n".join([f"  - {c}: {p*100:.1f}%" for c, p in zip(PATHOLOGY_CLASSES, p_h2_probs) if p > 0.1])
    
    text_str = (
        f"PATIENT CASE STUDY\n"
        f"=================================\n\n"
        f"[ H1 - Triage Prediction ]\n"
        f"Abnormal Probability: {p_h1*100:.2f}%\n"
        f"Threshold Used: 0.50\n\n"
        f"[ H2 - Pathology Routing Prediction ]\n"
        f"Top Predicted: {pred_class_name} (True: {class_name})\n\n"
        f"Granular Pathology Probs (>10%):\n{h2_str}"
    )
    ax2.text(0.1, 0.9, text_str, fontsize=12, family='monospace', va='top')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay_h1)
    ax3.set_title("H1 Grad-CAM (Triage)")
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(overlay_h2)
    ax4.set_title(f"H2 Grad-CAM ({class_name})")
    ax4.axis('off')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def run_evaluation_loop(model, val_loader, grad_cam, pdf, device):
    h1_preds, h1_targets, h1_probs_arr = [], [], []
    all_h2_preds, all_h2_targets, all_h2_probs = [], [], []
    
    gradcam_samples = {} # Store 2 representative images per class for Phase 2
    max_cases_per_disease = 2
    
    # MPS has native FP16 (AMX) but emulates bfloat16 -> prefer float16 on MPS
    _amp_dtype = torch.float16 if device.type == "mps" else (torch.bfloat16 if device.type == "cuda" else torch.float16)
    print("Executing Phase 1: Pure GPU Vectorized Evaluation...")
    
    for i, (images, labels) in enumerate(val_loader):
        images_dev = images.to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=_amp_dtype, enabled=device.type in ('mps', 'cuda')):
                logits = model(images_dev)

            # Cast to float32 before sigmoid/softmax to prevent NaN from FP16 overflow
            logits = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in logits.items()}

            h1_prob_batch = torch.sigmoid(logits['normal_abnormal']).cpu().numpy()
            h1_pred_batch = (h1_prob_batch > 0.5).astype(int)
            h1_preds.extend(h1_pred_batch)
            h1_targets.extend(labels['normal_abnormal'].numpy())
            h1_probs_arr.extend(h1_prob_batch)
            
            num_h2_classes = logits['pathology'].size(-1)
            valid_h2_mask = ((labels['normal_abnormal'] == 1).view(-1)) & (labels['pathology'] >= 0) & (labels['pathology'] < num_h2_classes)
            if valid_h2_mask.sum() > 0:
                probs_h2 = torch.softmax(logits['pathology'][valid_h2_mask], dim=1).cpu().numpy()
                preds_h2 = np.argmax(probs_h2, axis=1)
                
                targets_h2 = labels['pathology'][valid_h2_mask].numpy()
                all_h2_targets.extend(targets_h2)
                all_h2_probs.extend(probs_h2)
                all_h2_preds.extend(preds_h2)
                    
        # Select up to 2 high-confidence sample images per class for Grad-CAM phase
        for b_idx in range(images.size(0)):
            is_abnormal = labels['normal_abnormal'][b_idx].item() == 1
            if is_abnormal:
                class_idx = int(labels['pathology'][b_idx].item())
                if 0 <= class_idx < len(PATHOLOGY_CLASSES):
                    class_name = PATHOLOGY_CLASSES[class_idx]
                    current_samples = gradcam_samples.get(class_name, [])
                    if len(current_samples) < max_cases_per_disease:
                        current_samples.append((images[b_idx].clone(), class_idx, class_name))
                        gradcam_samples[class_name] = current_samples
                    
        if i % 20 == 0 or i == len(val_loader) - 1:
            print(f"Evaluation Progress: Batch {i+1}/{len(val_loader)}")

    print(f"\nPhase 1 Complete! Evaluating Phase 2: Targeted Grad-CAM Generation on {sum(len(v) for v in gradcam_samples.values())} Selected Case Studies...")
    for class_name, samples in gradcam_samples.items():
        for img_tensor, class_idx, c_name in samples:
            img_input = img_tensor.unsqueeze(0).to(device)
            generate_patient_case_study(img_input, class_idx, c_name, model, grad_cam, pdf)

    return h1_preds, h1_targets, h1_probs_arr, all_h2_preds, all_h2_targets, all_h2_probs

def compile_population_metrics(h1_preds, h1_targets, h1_probs_arr, h2_preds, h2_targets, h2_probs_arr, pdf):
    print("\n" + "="*50)
    print("H1 (Normal vs Abnormal) Metrics")
    print("="*50)
    print(f"Accuracy: {accuracy_score(h1_targets, h1_preds):.4f} | Macro F1: {f1_score(h1_targets, h1_preds, average='macro'):.4f}")
    print(classification_report(h1_targets, h1_preds, target_names=["Normal", "Abnormal"]))
    
    print("\n" + "="*50)
    print("H2 (Granular Pathology Multi-Class) Metrics")
    print("="*50)
    print(f"Accuracy: {accuracy_score(h2_targets, h2_preds):.4f}")
    print(f"Macro F1: {f1_score(h2_targets, h2_preds, average='macro', zero_division=0):.4f}")
    print(classification_report(h2_targets, h2_preds, target_names=PATHOLOGY_CLASSES, zero_division=0))

    print("\nSaving Telemetry Population Graphs to PDF...")
    
    h1_targets_flat, h1_probs_flat = np.array(h1_targets).flatten(), np.array(h1_probs_arr).flatten()
    precision, recall, thresholds = precision_recall_curve(h1_targets_flat, h1_probs_flat)
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'H1 PR Curve (AP = {average_precision_score(h1_targets_flat, h1_probs_flat):.3f})')
    for t_val in [0.2, 0.5, 0.8]:
        idx = np.argmin(np.abs(thresholds - t_val)) if len(thresholds) > 0 else 0
        if idx < len(recall):
            plt.plot(recall[idx], precision[idx], 'ro')
            plt.annotate(f'T={t_val:.1f}', (recall[idx], precision[idx]), textcoords="offset points", xytext=(-15,-15), ha='center')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('H1 Triage PR Curve'); plt.legend(); plt.grid(True)
    pdf.savefig(fig1); plt.close()
    
    fig3 = plt.figure(figsize=(10, 8))
    h2_targets_arr = np.array(h2_targets)
    h2_probs_mat = np.array(h2_probs_arr)
    for class_idx, class_name in enumerate(PATHOLOGY_CLASSES):
        t = (h2_targets_arr == class_idx).astype(int)
        p = h2_probs_mat[:, class_idx]
        if np.sum(t) > 0:
            prec, rec, _ = precision_recall_curve(t, p)
            plt.plot(rec, prec, lw=2, label=f'{class_name} (AP = {average_precision_score(t, p):.3f})')
            
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('H2 Granular Pathology PR Curves'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True)
    plt.tight_layout()
    pdf.savefig(fig3); plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Best Model and Generate Grad-CAM Visualizations")
    parser.add_argument("--checkpoint", type=str, default="/Users/nikhilmundhra/.cache/huggingface/hub/models--NMundhra--OCT-Classification-Model/snapshots/b8b2d5e7347d463a3d5f5d5c671e5e230968a7a6/fold0_best_model.pth", help="Path to checkpoint .pth file")
    parser.add_argument("--config", type=str, default="image-classification-model-training/config/hierarchy.yaml", help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=32, help="Validation batch size")
    args = parser.parse_args()

    device, pdf_path = setup_environment()
    pdf = PdfPages(pdf_path)
    val_loader = get_data_loader(config_path=args.config, batch_size=args.batch_size)
    model, grad_cam = load_model(args.checkpoint, device)
    
    res = run_evaluation_loop(model, val_loader, grad_cam, pdf, device)
    compile_population_metrics(*res, pdf)
    
    pdf.close()
    print("Telemetry generation complete! Check the telemetry_outputs/ directory.")

if __name__ == "__main__":
    main()
