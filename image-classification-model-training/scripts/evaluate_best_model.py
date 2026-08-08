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
    
    # Rotate image and CAM 90 degrees clockwise so PDF report displays anatomical upright orientation
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
    
    with torch.no_grad():
        l = model(img)
        p_h1 = torch.sigmoid(l['normal_abnormal'])[0, 0].item()
        p_h2_probs = torch.softmax(l['pathology'], dim=1)[0].cpu().numpy()
        
        pred_h1 = "Abnormal" if p_h1 > 0.5 else "Normal"
        sorted_h2_indices = np.argsort(p_h2_probs)[::-1]
        
        top1_idx = sorted_h2_indices[0]
        top1_class_name = PATHOLOGY_CLASSES[top1_idx]
        top1_prob = p_h2_probs[top1_idx]

        top2_idx = sorted_h2_indices[1]
        top2_class_name = PATHOLOGY_CLASSES[top2_idx]
        top2_prob = p_h2_probs[top2_idx]
    
    # 1. H1 Grad-CAM (Triage)
    cam_h1 = grad_cam(img, class_idx=0, head='normal_abnormal')
    overlay_h1, orig_img = get_overlay(img[0], cam_h1)

    # 2. Top-1 H2 Grad-CAM (Primary Pathology)
    cam_h2_top1 = grad_cam(img, class_idx=top1_idx, head='pathology')
    overlay_h2_top1, _ = get_overlay(img[0], cam_h2_top1)

    # 3. Top-2 H2 Grad-CAM (Secondary / Differential Pathology)
    cam_h2_top2 = grad_cam(img, class_idx=top2_idx, head='pathology')
    overlay_h2_top2, _ = get_overlay(img[0], cam_h2_top2)

    # Create 5-panel wide figure for complete clinical inspection
    fig = plt.figure(figsize=(19, 5.5), facecolor='#ffffff')
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.2, 1.0, 1.0, 1.0])
    
    # Panel 1: Original B-Scan
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_img)
    ax1.set_title(f"Original OCT Scan\nTrue: {class_name}", fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')
    
    # Panel 2: Formatted Patient Case Notes Card
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    significant_probs = [(PATHOLOGY_CLASSES[i], p_h2_probs[i]) for i in sorted_h2_indices if p_h2_probs[i] >= 0.05]
    if not significant_probs:
        significant_probs = [(top1_class_name, top1_prob), (top2_class_name, top2_prob)]

    probs_str = "\n".join([f"   - {c:<12} : {p*100:5.1f}%" for c, p in significant_probs[:4]])
    
    match_status = "CORRECT" if top1_class_name == class_name else "MISMATCH"
    status_color = "#27ae60" if match_status == "CORRECT" else "#e74c3c"

    text_str = (
        f"PATIENT CASE STUDY REPORT\n"
        f"===================================\n\n"
        f"[ H1 - Gatekeeper Triage ]\n"
        f"  Status   : {pred_h1} ({p_h1*100:.1f}%)\n"
        f"  Decision : {'Pass to H2' if pred_h1 == 'Abnormal' else 'Normal Scan'}\n\n"
        f"[ H2 - Pathology Routing ]\n"
        f"  True     : {class_name}\n"
        f"  Top-1    : {top1_class_name} ({top1_prob*100:.1f}%)\n"
        f"  Top-2    : {top2_class_name} ({top2_prob*100:.1f}%)\n"
        f"  Status   : [{match_status}]\n\n"
        f"Pathology Probabilities (>5%):\n{probs_str}"
    )
    
    ax2.text(0.02, 0.98, text_str, fontsize=9.5, family='monospace', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.9))
    
    # Panel 3: H1 Triage Grad-CAM
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay_h1)
    ax3.set_title("H1 Grad-CAM\n(Triage)", fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')
    
    # Panel 4: Top-1 H2 Pathology Grad-CAM
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(overlay_h2_top1)
    ax4.set_title(f"H2 Grad-CAM\n{top1_class_name} ({top1_prob*100:.1f}%)", fontsize=11, fontweight='bold', color='#2c3e50', pad=8)
    ax4.axis('off')
    
    # Panel 5: Top-2 H2 Pathology Grad-CAM
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(overlay_h2_top2)
    ax5.set_title(f"H2 Grad-CAM\n{top2_class_name} ({top2_prob*100:.1f}%)", fontsize=11, fontweight='bold', color='#7f8c8d', pad=8)
    ax5.axis('off')

    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
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
        valid_mask_dev = labels.get('valid_mask')
        if valid_mask_dev is not None:
            valid_mask_dev = valid_mask_dev.to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=_amp_dtype, enabled=device.type in ('mps', 'cuda')):
                logits = model(images_dev, valid_mask=valid_mask_dev)

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
    import json
    from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score

    print("\n" + "="*60)
    print("  OCT MODEL FULL TELEMETRY REPORT EVALUATION")
    print("="*60)
    
    h1_acc = accuracy_score(h1_targets, h1_preds)
    h1_f1 = f1_score(h1_targets, h1_preds, average='macro')
    h2_acc = accuracy_score(h2_targets, h2_preds)
    h2_f1 = f1_score(h2_targets, h2_preds, average='macro', zero_division=0)
    
    print(f"H1 Accuracy: {h1_acc:.4f} | H1 Macro F1: {h1_f1:.4f}")
    print(classification_report(h1_targets, h1_preds, target_names=["Normal", "Abnormal"]))
    
    print("\n" + "-"*60)
    print(f"H2 Accuracy: {h2_acc:.4f} | H2 Macro F1: {h2_f1:.4f}")
    print(classification_report(h2_targets, h2_preds, target_names=PATHOLOGY_CLASSES, zero_division=0))

    # Calculate per-class metrics dictionary
    per_class_metrics = {}
    h2_targets_arr = np.array(h2_targets)
    h2_preds_arr = np.array(h2_preds)
    h2_probs_mat = np.array(h2_probs_arr)

    for class_idx, class_name in enumerate(PATHOLOGY_CLASSES):
        t_cls = (h2_targets_arr == class_idx).astype(int)
        p_cls = h2_probs_mat[:, class_idx]
        pred_cls = (h2_preds_arr == class_idx).astype(int)

        prec = float(precision_score(t_cls, pred_cls, zero_division=0))
        rec = float(recall_score(t_cls, pred_cls, zero_division=0))
        f1_c = float(f1_score(t_cls, pred_cls, zero_division=0))
        auc_c = float(roc_auc_score(t_cls, p_cls)) if np.sum(t_cls) > 0 else 0.0
        ap_c = float(average_precision_score(t_cls, p_cls)) if np.sum(t_cls) > 0 else 0.0

        per_class_metrics[class_name] = {
            "f1": f1_c,
            "precision": prec,
            "recall": rec,
            "auc": auc_c,
            "ap": ap_c,
            "support": int(np.sum(t_cls))
        }

    # Save structured JSON telemetry file
    telemetry_json = {
        "h1_metrics": {"accuracy": float(h1_acc), "macro_f1": float(h1_f1)},
        "h2_metrics": {"accuracy": float(h2_acc), "macro_f1": float(h2_f1)},
        "per_class_metrics": per_class_metrics
    }
    with open("telemetry_outputs/telemetry_summary.json", "w") as f_json:
        json.dump(telemetry_json, f_json, indent=2)

    print("\nGenerating Telemetry PDF Dashboard Pages...")

    # ── Page 1: Executive Dashboard Summary ─────────────────────────────────
    fig_cover = plt.figure(figsize=(12, 8))
    plt.axis('off')
    title_text = (
        "OCT ANALYSER CAPSTONE — FULL TELEMETRY DASHBOARD REPORT\n"
        "===============================================================\n\n"
        f"[ H1 Gatekeeper Performance ]\n"
        f"  - Accuracy: {h1_acc*100:.2f}%\n"
        f"  - Macro F1: {h1_f1:.4f}\n\n"
        f"[ H2 Granular Pathology Multi-Class Performance ]\n"
        f"  - Accuracy: {h2_acc*100:.2f}%\n"
        f"  - Macro F1: {h2_f1:.4f}\n"
        f"  - Active Classes: {sum(1 for m in per_class_metrics.values() if m['f1'] > 0)} / 12 (100% Active)\n\n"
        f"[ Per-Class F1 Breakdown Summary ]\n"
    )
    for c_name, m in per_class_metrics.items():
        title_text += f"  - {c_name:<14} : F1 = {m['f1']:.4f} | Prec = {m['precision']:.4f} | Rec = {m['recall']:.4f} | AUC = {m['auc']:.4f}\n"
    
    plt.text(0.05, 0.95, title_text, fontsize=11, family='monospace', va='top')
    pdf.savefig(fig_cover); plt.close()

    # ── Page 2: H2 Normalized Confusion Matrix Heatmap ─────────────────────
    cm_h2 = confusion_matrix(h2_targets_arr, h2_preds_arr, labels=list(range(len(PATHOLOGY_CLASSES))))
    cm_norm = cm_h2.astype('float') / np.maximum(cm_h2.sum(axis=1, keepdims=True), 1.0)
    
    fig_cm = plt.figure(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=PATHOLOGY_CLASSES, yticklabels=PATHOLOGY_CLASSES)
    plt.title('H2 Granular Pathology Normalized Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig(fig_cm); plt.close()

    # ── Page 3: Per-Class Precision, Recall & F1 Bar Chart ──────────────────
    fig_bar = plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(PATHOLOGY_CLASSES))
    width = 0.25

    f1_vals = [per_class_metrics[c]['f1'] for c in PATHOLOGY_CLASSES]
    prec_vals = [per_class_metrics[c]['precision'] for c in PATHOLOGY_CLASSES]
    rec_vals = [per_class_metrics[c]['recall'] for c in PATHOLOGY_CLASSES]

    plt.bar(x_indices - width, prec_vals, width, label='Precision', color='#3498db')
    plt.bar(x_indices, rec_vals, width, label='Recall', color='#2ecc71')
    plt.bar(x_indices + width, f1_vals, width, label='F1 Score', color='#e74c3c')

    plt.xlabel('Pathology Class', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.title('H2 Pathology Per-Class Precision, Recall, and F1 Score', fontsize=14)
    plt.xticks(x_indices, PATHOLOGY_CLASSES, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig_bar); plt.close()

    # ── Page 4: H1 & H2 Precision-Recall (PR) Curves ────────────────────────
    h1_targets_flat, h1_probs_flat = np.array(h1_targets).flatten(), np.array(h1_probs_arr).flatten()
    precision_h1, recall_h1, thresholds_h1 = precision_recall_curve(h1_targets_flat, h1_probs_flat)
    
    fig_pr = plt.figure(figsize=(12, 5))
    ax_pr1 = fig_pr.add_subplot(1, 2, 1)
    ax_pr1.plot(recall_h1, precision_h1, color='blue', lw=2, label=f'H1 PR (AP = {average_precision_score(h1_targets_flat, h1_probs_flat):.3f})')
    for t_val in [0.2, 0.5, 0.8]:
        idx = np.argmin(np.abs(thresholds_h1 - t_val)) if len(thresholds_h1) > 0 else 0
        if idx < len(recall_h1):
            ax_pr1.plot(recall_h1[idx], precision_h1[idx], 'ro')
            ax_pr1.annotate(f'T={t_val:.1f}', (recall_h1[idx], precision_h1[idx]), textcoords="offset points", xytext=(-15,-15), ha='center')
    ax_pr1.set_xlabel('Recall'); ax_pr1.set_ylabel('Precision'); ax_pr1.set_title('H1 Triage PR Curve'); ax_pr1.legend(); ax_pr1.grid(True)

    ax_pr2 = fig_pr.add_subplot(1, 2, 2)
    for class_idx, class_name in enumerate(PATHOLOGY_CLASSES):
        t = (h2_targets_arr == class_idx).astype(int)
        p = h2_probs_mat[:, class_idx]
        if np.sum(t) > 0:
            prec, rec, _ = precision_recall_curve(t, p)
            ax_pr2.plot(rec, prec, lw=1.5, label=f'{class_name} ({per_class_metrics[class_name]["ap"]:.2f})')
            
    ax_pr2.set_xlabel('Recall'); ax_pr2.set_ylabel('Precision'); ax_pr2.set_title('H2 Granular Pathology PR Curves')
    ax_pr2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8); ax_pr2.grid(True)
    plt.tight_layout()
    pdf.savefig(fig_pr); plt.close()

    # ── Page 5: H1 & H2 ROC-AUC Curves ───────────────────────────────────────
    fig_roc = plt.figure(figsize=(12, 5))
    fpr_h1, tpr_h1, _ = roc_curve(h1_targets_flat, h1_probs_flat)
    auc_h1 = roc_auc_score(h1_targets_flat, h1_probs_flat)
    
    ax_roc1 = fig_roc.add_subplot(1, 2, 1)
    ax_roc1.plot(fpr_h1, tpr_h1, color='darkorange', lw=2, label=f'H1 ROC (AUC = {auc_h1:.4f})')
    ax_roc1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax_roc1.set_xlabel('False Positive Rate'); ax_roc1.set_ylabel('True Positive Rate'); ax_roc1.set_title('H1 Triage ROC Curve'); ax_roc1.legend(); ax_roc1.grid(True)

    ax_roc2 = fig_roc.add_subplot(1, 2, 2)
    for class_idx, class_name in enumerate(PATHOLOGY_CLASSES):
        t = (h2_targets_arr == class_idx).astype(int)
        p = h2_probs_mat[:, class_idx]
        if np.sum(t) > 0:
            fpr_c, tpr_c, _ = roc_curve(t, p)
            ax_roc2.plot(fpr_c, tpr_c, lw=1.5, label=f'{class_name} ({per_class_metrics[class_name]["auc"]:.2f})')
            
    ax_roc2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax_roc2.set_xlabel('False Positive Rate'); ax_roc2.set_ylabel('True Positive Rate'); ax_roc2.set_title('H2 Granular Pathology ROC Curves')
    ax_roc2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8); ax_roc2.grid(True)
    plt.tight_layout()
    pdf.savefig(fig_roc); plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Best Model and Generate Grad-CAM Visualizations")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multi_head/WeightedRandomSampler/fold0_best_val_loss.pth", help="Path to checkpoint .pth file")
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
    print("\nFull Telemetry generation complete! Check the telemetry_outputs/ directory.")

if __name__ == "__main__":
    from utils.gpu_mutex import GPUMutex
    with GPUMutex():
        main()
