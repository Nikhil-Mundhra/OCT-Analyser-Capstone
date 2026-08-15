import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from matplotlib.backends.backend_pdf import PdfPages

# 1. Enforce CPU performance core execution
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
num_cpu_threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(num_cpu_threads)
device = torch.device("cpu")
print(f"Executing on CPU using {num_cpu_threads} performance core threads.")

# 2. Add project root to sys.path
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
            
        h, w = x.shape[-2], x.shape[-1]
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(h, w), mode='bilinear', align_corners=False)
        return cam_resized.squeeze().numpy()

def touch_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        os.utime(checkpoint_path, None)
        print(f"Updated timestamp for checkpoint: {checkpoint_path}")
        print(f"Current modification time: {time.ctime(os.path.getmtime(checkpoint_path))}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

def load_model(checkpoint_path, device):
    print("Building Multi-Head ConvNeXt model...")
    model = build_multi_head_model(pretrained=False, warmup=False)
    
    print(f"Loading weights from {checkpoint_path} onto {device}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    unwrapped = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped)
    model.to(device)
    model.eval()
    
    # Target layer for Grad-CAM is final backbone stage (stages_3 for FeatureListNet)
    if hasattr(model.backbone, 'stages_3'):
        target_layer = model.backbone.stages_3
    elif hasattr(model.backbone, 'stages'):
        target_layer = model.backbone.stages[-1]
    else:
        target_layer = list(model.backbone.children())[-1]
    grad_cam = GradCAM(model, target_layer)
    return model, grad_cam

def get_overlay(img_tensor, cam):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).numpy().transpose(1, 2, 0)
    
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.4 * heatmap + 0.6 * img
    return np.clip(overlay, 0, 1), img

def generate_patient_case_study(img_tensor, class_idx, class_name, model, grad_cam, pdf):
    img = img_tensor.unsqueeze(0).to(device)
    
    cam_h1 = grad_cam(img, class_idx=0, head='normal_abnormal')
    
    h2_target_class = class_idx
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
        pred_class_name = PATHOLOGY_CLASSES[pred_h2_idx] if pred_h2_idx < len(PATHOLOGY_CLASSES) else f"Class_{pred_h2_idx}"
    
    h2_str = "\n".join([f"  - {c}: {p*100:.1f}%" for c, p in zip(PATHOLOGY_CLASSES, p_h2_probs) if p > 0.05])
    
    text_str = (
        f"PATIENT CASE STUDY\n"
        f"=================================\n\n"
        f"[ H1 - Triage Prediction ]\n"
        f"Predicted: {pred_h1}\n"
        f"Abnormal Prob: {p_h1*100:.2f}%\n\n"
        f"[ H2 - Pathology Prediction ]\n"
        f"Top Predicted: {pred_class_name}\n"
        f"True Ground Truth: {class_name}\n\n"
        f"Pathology Probs (>5%):\n{h2_str}"
    )
    ax2.text(0.05, 0.95, text_str, fontsize=10, family='monospace', va='top')
    
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

from data.transforms import get_transforms

def get_validation_loader(batch_size=32, num_samples=1000):
    val_transform = get_transforms("val")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "hierarchy.yaml")
    full_dataset = MultiHeadOCTDataset(config_path=config_path, transform=val_transform)
    
    # Deterministic split for reproducible validation set
    np.random.seed(42)
    val_size = min(num_samples, int(0.2 * len(full_dataset)))
    indices = np.random.permutation(len(full_dataset))[:val_size].tolist()
    val_dataset = Subset(full_dataset, indices)
    print(f"Validation dataset initialized with {len(val_dataset)} images.")
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def run_telemetry_eval():
    checkpoint_path = "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/checkpoints/classification/multi_head/Baseline/v1/fold0_epoch_004.pth"
    
    # Touch checkpoint to update timestamp to today
    touch_checkpoint(checkpoint_path)
    
    output_dir = "telemetry_outputs"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "Full_Evaluation_Report.pdf")
    pdf = PdfPages(pdf_path)
    
    model, grad_cam = load_model(checkpoint_path, device)
    val_loader = get_validation_loader(batch_size=32, num_samples=1000)
    
    h1_preds, h1_targets, h1_probs_arr = [], [], []
    h2_preds, h2_targets, h2_probs_arr = [], [], []
    
    gradcam_counts = {k: 0 for k in PATHOLOGY_CLASSES}
    max_cases_per_disease = 2
    
    print("\nRunning Validation Inference & Grad-CAM visual extraction on CPU performance cores...")
    
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        
        with torch.no_grad():
            logits = model(images)
            
            h1_prob_batch = torch.sigmoid(logits['normal_abnormal']).cpu().numpy().flatten()
            h1_pred_batch = (h1_prob_batch > 0.5).astype(int)
            
            h1_preds.extend(h1_pred_batch)
            h1_targets.extend(labels['normal_abnormal'].numpy().flatten())
            h1_probs_arr.extend(h1_prob_batch)
            
            # H2 head processing
            num_h2_classes = logits['pathology'].size(-1)
            valid_h2_mask = ((labels['normal_abnormal'] == 1).view(-1)) & (labels['pathology'] >= 0) & (labels['pathology'] < num_h2_classes)
            if valid_h2_mask.sum() > 0:
                probs_h2 = torch.softmax(logits['pathology'][valid_h2_mask], dim=1).cpu().numpy()
                preds_h2 = np.argmax(probs_h2, axis=1)
                targets_h2 = labels['pathology'][valid_h2_mask].numpy().flatten()
                
                h2_targets.extend(targets_h2)
                h2_probs_arr.extend(probs_h2)
                h2_preds.extend(preds_h2)
                
        # Grad-CAM case studies sampling
        for b_idx in range(images.size(0)):
            is_abnormal = labels['normal_abnormal'][b_idx].item() == 1
            if is_abnormal:
                class_idx = int(labels['pathology'][b_idx].item())
                if 0 <= class_idx < len(PATHOLOGY_CLASSES):
                    class_name = PATHOLOGY_CLASSES[class_idx]
                    if gradcam_counts.get(class_name, 0) < max_cases_per_disease:
                        generate_patient_case_study(images[b_idx], class_idx, class_name, model, grad_cam, pdf)
                        gradcam_counts[class_name] = gradcam_counts.get(class_name, 0) + 1
                        
        if (i + 1) % 5 == 0 or (i + 1) == len(val_loader):
            print(f"Processed Batch {i+1}/{len(val_loader)}", flush=True)
            
    print("\n" + "=" * 60)
    print("SUMMARY TELEMETRY REPORT")
    print("=" * 60)
    
    h1_acc = accuracy_score(h1_targets, h1_preds)
    h1_f1 = f1_score(h1_targets, h1_preds, average='macro', zero_division=0)
    h1_rec = recall_score(h1_targets, h1_preds, average='macro', zero_division=0)
    
    print(f"\n[ H1 Gatekeeper (Triage) Metrics ]")
    print(f"  Accuracy : {h1_acc:.4f}")
    print(f"  Macro F1 : {h1_f1:.4f}")
    print(f"  Macro Rec: {h1_rec:.4f}")
    print("\nClassification Report (H1):")
    print(classification_report(h1_targets, h1_preds, target_names=["Normal", "Abnormal"], zero_division=0))
    
    if len(h2_targets) > 0:
        h2_acc = accuracy_score(h2_targets, h2_preds)
        h2_f1 = f1_score(h2_targets, h2_preds, average='macro', zero_division=0)
        h2_rec = recall_score(h2_targets, h2_preds, average='macro', zero_division=0)
        
        print(f"\n[ H2 Granular Pathology Metrics ]")
        print(f"  Accuracy : {h2_acc:.4f}")
        print(f"  Macro F1 : {h2_f1:.4f}")
        print(f"  Macro Rec: {h2_rec:.4f}")
        print("\nClassification Report (H2):")
        present_classes = sorted(list(set(h2_targets) | set(h2_preds)))
        target_names = [PATHOLOGY_CLASSES[idx] for idx in present_classes]
        print(classification_report(h2_targets, h2_preds, labels=present_classes, target_names=target_names, zero_division=0))
    else:
        print("\nNo abnormal samples found in validation batch for H2 calculation.")
        
    pdf.close()
    print(f"\n✅ Full Evaluation & Grad-CAM Telemetry Report saved to: {pdf_path}")

if __name__ == "__main__":
    run_telemetry_eval()
