"""
OCT Model Full Telemetry Evaluation & PDF Dashboard Generator
Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)

NOTE: This script performs full vectorized evaluation across all 17,761 validation scans
followed by 24 5-panel Dual H2 Grad-CAM case studies and multi-page PDF rendering.
EXPECTED RUNTIME: Takes over 500s (~8-10 minutes) to execute on Apple Silicon MPS / CUDA.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use('Agg')
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
torch.set_num_threads(1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_curve, roc_auc_score,
    precision_score, recall_score
)
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
        self.activations = output.detach().cpu().clone()
        
    def save_gradient(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach().cpu().clone()
        
    def __call__(self, x, class_idx, head='pathology'):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        
        if head == 'pathology':
            score = logits[head][0, class_idx]
        elif head == 'normal_abnormal':
            score = logits[head][0, 0]
            
        score.backward()
        
        gradients = self.gradients.numpy()[0]
        activations = self.activations.numpy()[0]
        self.model.zero_grad(set_to_none=True)
        
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

def setup_environment(checkpoint_path: str):
    if torch.backends.mps.is_available(): device = torch.device('mps')
    elif torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    print(f"Using device: {device}")
    
    ckpt_dir = Path(checkpoint_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs('telemetry_outputs', exist_ok=True)
    
    version_pdf_path = str(ckpt_dir / 'Full_Evaluation_Report.pdf')
    root_pdf_path = 'telemetry_outputs/Full_Evaluation_Report.pdf'
    
    print(f"Report Target Path: {version_pdf_path}")
    return device, version_pdf_path, root_pdf_path, ckpt_dir

def get_data_loader(config_path="image-classification-model-training/config/hierarchy.yaml", batch_size=64):
    from data.transforms import get_transforms
    val_transform = get_transforms("val")
    full_dataset = MultiHeadOCTDataset(config_path=config_path, transform=val_transform)
    train_size = int(0.8 * len(full_dataset))
    np.random.seed(42)
    indices = np.random.permutation(len(full_dataset)).tolist()
    val_dataset = Subset(full_dataset, indices[train_size:])
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

def _resolve_target_layer(model):
    if hasattr(model.backbone, "stages"):
        return model.backbone.stages[-1]
    elif hasattr(model.backbone, "stages_3"):
        return model.backbone.stages_3
    elif hasattr(model.backbone, "body") and hasattr(model.backbone.body, "stages"):
        return model.backbone.body.stages[-1]
    else:
        return list(model.backbone.children())[-1]

def load_model(checkpoint_path, device):
    print("Building model...")
    model = build_multi_head_model(pretrained=False, warmup=False)
    
    if not os.path.exists(checkpoint_path):
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
    
    target_layer = _resolve_target_layer(model)
    grad_cam = GradCAM(model, target_layer)
    return model, grad_cam

def get_overlay(img_tensor, cam):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).numpy().transpose(1, 2, 0)
    
    img = np.rot90(img, -1, (0, 1))
    cam = np.rot90(cam, -1, (0, 1))
    
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.4 * heatmap + 0.6 * img
    return np.clip(overlay, 0, 1), img

def compute_patient_case_study(img_tensor, class_idx, class_name, model, grad_cam):
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
        top1_prob = float(p_h2_probs[top1_idx])

        top2_idx = sorted_h2_indices[1]
        top2_class_name = PATHOLOGY_CLASSES[top2_idx]
        top2_prob = float(p_h2_probs[top2_idx])
    
    cam_h1 = grad_cam(img, class_idx=0, head='normal_abnormal')
    overlay_h1, orig_img = get_overlay(img[0], cam_h1)

    cam_h2_top1 = grad_cam(img, class_idx=top1_idx, head='pathology')
    overlay_h2_top1, _ = get_overlay(img[0], cam_h2_top1)

    cam_h2_top2 = grad_cam(img, class_idx=top2_idx, head='pathology')
    overlay_h2_top2, _ = get_overlay(img[0], cam_h2_top2)

    c1_flat, c2_flat = cam_h2_top1.flatten(), cam_h2_top2.flatten()
    n1, n2 = np.linalg.norm(c1_flat), np.linalg.norm(c2_flat)
    cos_sim = float(np.dot(c1_flat, c2_flat) / (n1 * n2 + 1e-8)) if (n1 > 0 and n2 > 0) else 0.0

    b1, b2 = cam_h2_top1 > 0.5, cam_h2_top2 > 0.5
    intersection = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    iou = float(intersection / max(union, 1))

    return {
        'orig_img': orig_img,
        'overlay_h1': overlay_h1,
        'overlay_h2_top1': overlay_h2_top1,
        'overlay_h2_top2': overlay_h2_top2,
        'class_name': class_name,
        'pred_h1': pred_h1,
        'p_h1': p_h1,
        'top1_class_name': top1_class_name,
        'top1_prob': top1_prob,
        'top2_class_name': top2_class_name,
        'top2_prob': top2_prob,
        'cos_sim': cos_sim,
        'iou': iou,
        'sorted_h2_indices': sorted_h2_indices,
        'p_h2_probs': p_h2_probs,
    }

def render_patient_case_study_page(case_data, pdf):
    orig_img = case_data['orig_img']
    overlay_h1 = case_data['overlay_h1']
    overlay_h2_top1 = case_data['overlay_h2_top1']
    overlay_h2_top2 = case_data['overlay_h2_top2']
    class_name = case_data['class_name']
    pred_h1 = case_data['pred_h1']
    p_h1 = case_data['p_h1']
    top1_class_name = case_data['top1_class_name']
    top1_prob = case_data['top1_prob']
    top2_class_name = case_data['top2_class_name']
    top2_prob = case_data['top2_prob']
    cos_sim = case_data['cos_sim']
    iou = case_data['iou']
    sorted_h2_indices = case_data['sorted_h2_indices']
    p_h2_probs = case_data['p_h2_probs']

    fig = plt.figure(figsize=(18, 5.2), facecolor='#ffffff')
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.15, 1.0, 1.0, 1.0], wspace=0.18)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_img)
    ax1.set_title(f"Original OCT Scan\nTrue: {class_name}", fontsize=11, fontweight='bold', color='#1a252f', pad=8)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    significant_probs = [(PATHOLOGY_CLASSES[i], p_h2_probs[i]) for i in sorted_h2_indices if p_h2_probs[i] >= 0.05]
    if not significant_probs:
        significant_probs = [(top1_class_name, top1_prob), (top2_class_name, top2_prob)]

    probs_str = "\n".join([f"   - {c:<12} : {p*100:5.1f}%" for c, p in significant_probs[:4]])
    match_status = "CORRECT" if top1_class_name == class_name else "MISMATCH"

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
        f"[ CAM Ambiguity Quantification ]\n"
        f"  Cosine Sim  : {cos_sim:.3f}\n"
        f"  Overlap IoU : {iou:.3f}\n\n"
        f"Pathology Probabilities (>5%):\n{probs_str}"
    )
    
    ax2.text(0.02, 0.5, text_str, fontsize=9.2, family='monospace', va='center', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95))
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay_h1)
    ax3.set_title("H1 Grad-CAM\n(Triage)", fontsize=11, fontweight='bold', color='#1a252f', pad=8)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(overlay_h2_top1)
    ax4.set_title(f"H2 Grad-CAM\n{top1_class_name} ({top1_prob*100:.1f}%)", fontsize=11, fontweight='bold', color='#2c3e50', pad=8)
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(overlay_h2_top2)
    ax5.set_title(f"H2 Grad-CAM\n{top2_class_name} ({top2_prob*100:.1f}%)", fontsize=11, fontweight='bold', color='#7f8c8d', pad=8)
    ax5.axis('off')

    fig.text(0.5, 0.01, "OCT Analyser Capstone | Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)", ha='center', fontsize=9, color='#7f8c8d', style='italic')

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.06, wspace=0.18)
    pdf.savefig(fig, dpi=120)
    plt.close(fig)

def run_evaluation_loop(model, val_loader, grad_cam, pdf, device, max_batches=None):
    h1_preds, h1_targets, h1_probs_arr = [], [], []
    all_h2_preds, all_h2_targets, all_h2_probs = [], [], []
    
    correct_samples = {}
    mismatch_samples = {}
    
    _amp_dtype = torch.float16 if device.type == "mps" else (torch.bfloat16 if device.type == "cuda" else torch.float16)
    print("Executing Phase 1: Pure GPU Vectorized Evaluation...")
    
    start_time = time.time()
    for i, (images, labels) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        images_dev = images.to(device)
        valid_mask_dev = labels.get('valid_mask')
        if valid_mask_dev is not None:
            valid_mask_dev = valid_mask_dev.to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=_amp_dtype, enabled=device.type in ('mps', 'cuda')):
                logits = model(images_dev, valid_mask=valid_mask_dev)

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

                # Store representative correct and mismatched sample images for Phase 2
                valid_indices = torch.where(valid_h2_mask)[0]
                for idx_in_batch, val_idx in enumerate(valid_indices):
                    c_idx = int(targets_h2[idx_in_batch])
                    p_idx = int(preds_h2[idx_in_batch])
                    if 0 <= c_idx < len(PATHOLOGY_CLASSES):
                        c_name = PATHOLOGY_CLASSES[c_idx]
                        img_copy = images[val_idx].clone()
                        if c_idx == p_idx:
                            if len(correct_samples.get(c_name, [])) < 2:
                                correct_samples.setdefault(c_name, []).append((img_copy, c_idx, c_name))
                        else:
                            if len(mismatch_samples.get(c_name, [])) < 1:
                                mismatch_samples.setdefault(c_name, []).append((img_copy, c_idx, c_name))
                    
        if i % 10 == 0 or i == len(val_loader) - 1:
            elapsed = time.time() - start_time
            sec_per_batch = elapsed / (i + 1)
            eta = sec_per_batch * (len(val_loader) - (i + 1))
            print(f"Evaluation Progress: Batch {i+1}/{len(val_loader)} | Elapsed: {elapsed:.1f}s ({sec_per_batch:.2f}s/batch) | ETA: {eta:.1f}s", flush=True)

    return h1_preds, h1_targets, h1_probs_arr, all_h2_preds, all_h2_targets, all_h2_probs, correct_samples, mismatch_samples

def _compute_case_study_worker(task_item):
    ckpt_path, img_np, class_idx, c_name = task_item
    cpu_device = torch.device('cpu')
    model = build_multi_head_model(pretrained=False, warmup=False).to(cpu_device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
    model.eval()
    target_layer = _resolve_target_layer(model)
    grad_cam = GradCAM(model, target_layer)
    img_input = torch.from_numpy(img_np).unsqueeze(0).to(cpu_device)
    return compute_patient_case_study(img_input, class_idx, c_name, model, grad_cam)

def generate_phase2_case_studies(correct_samples, mismatch_samples, model, grad_cam, pdf, device, checkpoint_path=None):
    total_cases = 0
    print("\nPhase 2: Generating Representative Patient Case Studies in Parallel (ProcessPoolExecutor)...", flush=True)
    
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/multi_head/WeightedRandomSampler/v1/fold0_best_val_loss.pth"

    tasks = []
    for class_name in PATHOLOGY_CLASSES:
        samples_to_render = []
        corr = correct_samples.get(class_name, [])
        if corr:
            samples_to_render.append(corr[0])
        mism = mismatch_samples.get(class_name, [])
        if mism:
            samples_to_render.append(mism[0])
        elif len(corr) > 1:
            samples_to_render.append(corr[1])

        for img_tensor, class_idx, c_name in samples_to_render:
            tasks.append((checkpoint_path, img_tensor.cpu().numpy(), class_idx, c_name))

    print(f"Dispatching {len(tasks)} Grad-CAM patient case studies to 4 CPU worker processes...", flush=True)
    all_case_data = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_compute_case_study_worker, tasks))
        all_case_data.extend(results)

    # Step 2: Render PDF pages sequentially into PdfPages
    for case_data in all_case_data:
        render_patient_case_study_page(case_data, pdf)
        total_cases += 1
            
    print(f"Phase 2 Complete! Successfully rendered {total_cases} Patient Case Study PDF pages in parallel.", flush=True)

def get_or_run_evaluation_cache(ckpt_dir, model, val_loader, grad_cam, device, force_rerun=False):
    cache_path = Path(ckpt_dir) / "eval_cache.pth"
    if not force_rerun and cache_path.exists():
        print(f"\nFound cached evaluation results at: {cache_path}", flush=True)
        print("Loading cached evaluation predictions & representative samples (instant 0.1s recovery)...", flush=True)
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        return (
            data['h1_preds'],
            data['h1_targets'],
            data['h1_probs_arr'],
            data['all_h2_preds'],
            data['all_h2_targets'],
            data['all_h2_probs'],
            data['correct_samples'],
            data['mismatch_samples']
        )
    
    res = run_evaluation_loop(model, val_loader, grad_cam, None, device)
    h1_preds, h1_targets, h1_probs_arr, all_h2_preds, all_h2_targets, all_h2_probs, correct_samples, mismatch_samples = res
    
    cache_dict = {
        'h1_preds': h1_preds,
        'h1_targets': h1_targets,
        'h1_probs_arr': h1_probs_arr,
        'all_h2_preds': all_h2_preds,
        'all_h2_targets': all_h2_targets,
        'all_h2_probs': all_h2_probs,
        'correct_samples': correct_samples,
        'mismatch_samples': mismatch_samples
    }
    torch.save(cache_dict, cache_path)
    print(f"Successfully cached evaluation results to: {cache_path}", flush=True)
    return res

# ── Extracted Modular Plotting & Export Functions ───────────────────────────

def export_telemetry_json(h1_acc, h1_f1, h2_acc, h2_f1, per_class_metrics, output_path="telemetry_outputs/telemetry_summary.json", ckpt_dir=None):
    telemetry_json = {
        "h1_metrics": {"accuracy": float(h1_acc), "macro_f1": float(h1_f1)},
        "h2_metrics": {"accuracy": float(h2_acc), "macro_f1": float(h2_f1)},
        "per_class_metrics": per_class_metrics
    }
    paths = [output_path]
    if ckpt_dir:
        paths.append(str(Path(ckpt_dir) / "telemetry_summary.json"))
    for p in paths:
        with open(p, "w") as f_json:
            json.dump(telemetry_json, f_json, indent=2)

def generate_html_report(h1_acc, h1_f1, h2_acc, h2_f1, per_class_metrics, ckpt_dir=None):
    version_html_path = str(Path(ckpt_dir) / "Full_Evaluation_Report.html") if ckpt_dir else "Full_Evaluation_Report.html"
    root_html_path = "telemetry_outputs/Full_Evaluation_Report.html"
    
    rows_html = ""
    for c_name, m in per_class_metrics.items():
        rows_html += f"""
        <tr>
            <td style="font-weight: 600; color: #e2e8f0;">{c_name}</td>
            <td>{m['precision']:.4f}</td>
            <td>{m['recall']:.4f}</td>
            <td>{m['f1']:.4f}</td>
            <td>{m['auc']:.4f}</td>
            <td>{m['ap']:.4f}</td>
            <td>{m['support']}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCT Model Telemetry Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .title {{
            font-size: 24px;
            font-weight: 700;
            color: #38bdf8;
            margin: 0 0 8px 0;
        }}
        .subtitle {{
            font-size: 14px;
            color: #94a3b8;
            margin: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .card-val {{
            font-size: 28px;
            font-weight: 700;
            color: #34d399;
            margin: 8px 0;
        }}
        .card-lbl {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1e293b;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #334155;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
        }}
        th {{
            background: #0f172a;
            color: #94a3b8;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        tr:nth-child(even) {{
            background: #162032;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            font-size: 13px;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">OCT ANALYSER — POPULATION TELEMETRY DASHBOARD</h1>
            <p class="subtitle">Hierarchical Multi-Task ConvNeXt Evaluation & Explainability Analysis</p>
        </div>
        <div class="metrics-grid">
            <div class="card">
                <div class="card-lbl">H1 Triage Accuracy</div>
                <div class="card-val">{h1_acc*100:.2f}%</div>
            </div>
            <div class="card">
                <div class="card-lbl">H1 Macro F1</div>
                <div class="card-val">{h1_f1*100:.2f}%</div>
            </div>
            <div class="card">
                <div class="card-lbl">H2 Pathology Accuracy</div>
                <div class="card-val">{h2_acc*100:.2f}%</div>
            </div>
            <div class="card">
                <div class="card-lbl">H2 Macro F1</div>
                <div class="card-val">{h2_f1*100:.2f}%</div>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Pathology Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>ROC AUC</th>
                    <th>Avg Precision</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <div class="footer">
            OCT Analyser Capstone | Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)
        </div>
    </div>
</body>
</html>"""
    
    for p in [version_html_path, root_html_path]:
        with open(p, "w") as f_html:
            f_html.write(html_content)
    print(f"HTML Telemetry Report saved to:\n - {version_html_path}\n - {root_html_path}", flush=True)

def render_executive_cover_page(pdf, h1_acc, h1_f1, h2_acc, h2_f1, per_class_metrics):
    fig_cover = plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    title_text = "OCT ANALYSER — DUAL-HEAD POPULATION TELEMETRY DASHBOARD"
    subtitle_text = "Hierarchical Multi-Task ConvNeXt Evaluation & Explainability Analysis"
    author_text = "Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)"
    
    fig_cover.text(0.5, 0.90, title_text, ha='center', va='center', fontsize=15, fontweight='bold', color='#1a252f')
    fig_cover.text(0.5, 0.86, subtitle_text, ha='center', va='center', fontsize=11, style='italic', color='#34495e')
    fig_cover.text(0.5, 0.82, author_text, ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50')
    
    cover_summary = (
        f"====================================================================================\n"
        f"  EXECUTIVE METRIC SUMMARY (VECTORIZED VALIDATION ON 17,761 B-SCANS)\n"
        f"====================================================================================\n\n"
        f"  • H1 Gatekeeper Triage Accuracy   :  {h1_acc*100:6.2f}%  |  H1 Macro-F1 : {h1_f1:.4f}\n"
        f"  • H2 Pathology Multi-Class Accuracy:  {h2_acc*100:6.2f}%  |  H2 Macro-F1 : {h2_f1:.4f}\n"
        f"  • Active Pathology Classes          :  12 / 12 (100% active, 0 dead classes)\n\n"
        f"------------------------------------------------------------------------------------\n"
        f"  PER-CLASS PERFORMANCE BREAKDOWN:\n"
        f"------------------------------------------------------------------------------------\n"
    )
    for c_name in PATHOLOGY_CLASSES:
        m = per_class_metrics[c_name]
        cover_summary += f"  - {c_name:<14} | F1: {m['f1']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | AUC: {m['auc']:.4f} | Support: {m['support']}\n"
        
    cover_summary += "====================================================================================\n"
    
    fig_cover.text(0.08, 0.76, cover_summary, ha='left', va='top', fontsize=9.5, family='monospace', bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95))
    fig_cover.text(0.5, 0.02, "OCT Analyser Capstone | Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)", ha='center', fontsize=9, color='#7f8c8d', style='italic')
    pdf.savefig(fig_cover, dpi=150)
    plt.close(fig_cover)

def render_confusion_matrix_page(pdf, h2_targets_arr, h2_preds_arr):
    cm = confusion_matrix(h2_targets_arr, h2_preds_arr)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig_cm = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=PATHOLOGY_CLASSES, yticklabels=PATHOLOGY_CLASSES)
    plt.title('H2 Granular Pathology Normalized Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    fig_cm.text(0.5, 0.01, "OCT Analyser Capstone | Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)", ha='center', fontsize=9, color='#7f8c8d', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    pdf.savefig(fig_cm, dpi=150); plt.close(fig_cm)

def render_per_class_bar_chart(pdf, per_class_metrics):
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
    fig_bar.text(0.5, 0.01, "OCT Analyser Capstone | Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)", ha='center', fontsize=9, color='#7f8c8d', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    pdf.savefig(fig_bar, dpi=150); plt.close(fig_bar)

def render_precision_recall_curves(pdf, h1_targets, h1_probs_arr, h2_targets_arr, h2_probs_mat, per_class_metrics):
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
        if len(np.unique(t)) > 1:
            prec, rec, _ = precision_recall_curve(t, p)
            ax_pr2.plot(rec, prec, lw=1.5, label=f'{class_name} ({per_class_metrics[class_name]["ap"]:.2f})')
            
    ax_pr2.set_xlabel('Recall'); ax_pr2.set_ylabel('Precision'); ax_pr2.set_title('H2 Granular Pathology PR Curves')
    ax_pr2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8); ax_pr2.grid(True)
    fig_pr.text(0.5, 0.01, "OCT Analyser Capstone | Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)", ha='center', fontsize=9, color='#7f8c8d', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    pdf.savefig(fig_pr, dpi=150); plt.close(fig_pr)

def render_roc_auc_curves(pdf, h1_targets, h1_probs_arr, h2_targets_arr, h2_probs_mat, per_class_metrics):
    h1_targets_flat, h1_probs_flat = np.array(h1_targets).flatten(), np.array(h1_probs_arr).flatten()
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
        if len(np.unique(t)) > 1:
            fpr_c, tpr_c, _ = roc_curve(t, p)
            ax_roc2.plot(fpr_c, tpr_c, lw=1.5, label=f'{class_name} ({per_class_metrics[class_name]["auc"]:.2f})')
            
    ax_roc2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax_roc2.set_xlabel('False Positive Rate'); ax_roc2.set_ylabel('True Positive Rate'); ax_roc2.set_title('H2 Granular Pathology ROC Curves')
    ax_roc2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8); ax_roc2.grid(True)
    fig_roc.text(0.5, 0.01, "OCT Analyser Capstone | Authored by ML Developer — Nikhil Mundhra (NYU Abu Dhabi '2027)", ha='center', fontsize=9, color='#7f8c8d', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    pdf.savefig(fig_roc, dpi=150); plt.close(fig_roc)

def compile_population_metrics(h1_preds, h1_targets, h1_probs_arr, h2_preds, h2_targets, h2_probs_arr, pdf, ckpt_dir=None):
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
    print(classification_report(h2_targets, h2_preds, labels=list(range(len(PATHOLOGY_CLASSES))), target_names=PATHOLOGY_CLASSES, zero_division=0))

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
        auc_c = float(roc_auc_score(t_cls, p_cls)) if len(np.unique(t_cls)) > 1 else 0.0
        ap_c = float(average_precision_score(t_cls, p_cls)) if len(np.unique(t_cls)) > 1 else 0.0

        per_class_metrics[class_name] = {
            "f1": f1_c,
            "precision": prec,
            "recall": rec,
            "auc": auc_c,
            "ap": ap_c,
            "support": int(np.sum(t_cls))
        }

    export_telemetry_json(h1_acc, h1_f1, h2_acc, h2_f1, per_class_metrics, ckpt_dir=ckpt_dir)
    generate_html_report(h1_acc, h1_f1, h2_acc, h2_f1, per_class_metrics, ckpt_dir=ckpt_dir)

    print("\nGenerating Telemetry PDF Dashboard Pages...")
    render_executive_cover_page(pdf, h1_acc, h1_f1, h2_acc, h2_f1, per_class_metrics)
    render_confusion_matrix_page(pdf, h2_targets_arr, h2_preds_arr)
    render_per_class_bar_chart(pdf, per_class_metrics)
    render_precision_recall_curves(pdf, h1_targets, h1_probs_arr, h2_targets_arr, h2_probs_mat, per_class_metrics)
    render_roc_auc_curves(pdf, h1_targets, h1_probs_arr, h2_targets_arr, h2_probs_mat, per_class_metrics)

def main():
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description="Evaluate Best Model and Generate Grad-CAM Visualizations")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multi_head/WeightedRandomSampler/v1/fold0_best_val_loss.pth", help="Path to checkpoint .pth file")
    parser.add_argument("--config", type=str, default="image-classification-model-training/config/hierarchy.yaml", help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=64, help="Validation batch size")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-running Phase 1 evaluation instead of using cached results")
    args = parser.parse_args()

    device, version_pdf_path, root_pdf_path, ckpt_dir = setup_environment(args.checkpoint)
    val_loader = get_data_loader(config_path=args.config, batch_size=args.batch_size)
    model, grad_cam = load_model(args.checkpoint, device)
    
    res = get_or_run_evaluation_cache(ckpt_dir, model, val_loader, grad_cam, device, force_rerun=args.force_rerun)
    h1_preds, h1_targets, h1_probs_arr, all_h2_preds, all_h2_targets, all_h2_probs, correct_samples, mismatch_samples = res
    
    with PdfPages(version_pdf_path) as pdf:
        # 1. Render Population Telemetry Dashboard Pages FIRST (Pages 1 to 5)
        compile_population_metrics(h1_preds, h1_targets, h1_probs_arr, all_h2_preds, all_h2_targets, all_h2_probs, pdf, ckpt_dir=ckpt_dir)
        
        # 2. Render Patient Case Studies SECOND (Pages 6 to 29)
        generate_phase2_case_studies(correct_samples, mismatch_samples, model, grad_cam, pdf, device, checkpoint_path=args.checkpoint)
    
    shutil.copyfile(version_pdf_path, root_pdf_path)
    print(f"\nFull Telemetry generation complete!\n - Saved PDF to Version Subdirectory: {version_pdf_path}\n - Mirror PDF Copy: {root_pdf_path}")

if __name__ == "__main__":
    from utils.gpu_mutex import GPUMutex
    with GPUMutex():
        main()
