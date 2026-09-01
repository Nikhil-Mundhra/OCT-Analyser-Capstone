"""
evaluation/h1_sensitivity_diagnostics.py

Empirical Diagnostic Benchmark for H1-Conditioning and Failure Mode Analysis:
1. Scalar Sensitivity Sweep: Sweeps P(H1) in [0.0, 1.0] holding visual features constant, computing dP(D_i)/dP(H1).
2. Permutation Importance: Quantifies model reliance on the appended P(H1) scalar.
3. Decoupled Failure Mode Diagnostics: Separately evaluates Gatekeeper False Positives vs Semantic OOD Pathologies.
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / "web-app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "web-app"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.classification.models.multi_head_convnext import MultiHeadConvNeXt, build_multi_head_model
from backend.core_ml.classification.utils.calibration import TriageCalibrationEngine, CalibrationConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PATHOLOGY_CLASSES = [
    'CNV', 'DRUSEN', 'AMD', 'General_AMD', 
    'DME', 'DR', 'MH', 'RVO', 'RAO', 
    'CSR', 'ERM', 'VID'
]


def run_scalar_sweep(
    model: MultiHeadConvNeXt,
    sample_visual_features: torch.Tensor,
    class_names: Optional[List[str]] = None,
    num_steps: int = 101,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Sweeps P(H1) from 0.0 to 1.0 while holding visual features constant.
    Computes numerical gradient dP(D_i)/dP(H1).
    """
    model.eval()
    class_names = class_names or DEFAULT_PATHOLOGY_CLASSES
    p_h1_values = np.linspace(0.0, 1.0, num_steps)
    
    visual_features = sample_visual_features.to(device)
    if visual_features.ndim == 1:
        visual_features = visual_features.unsqueeze(0)
        
    conditional_probs_trajectory = []
    joint_probs_trajectory = []
    
    with torch.no_grad():
        for p_val in p_h1_values:
            h1_prob_tensor = torch.tensor([[p_val]], dtype=torch.float32, device=device)
            if model.condition_h2_on_h1:
                h2_input = torch.cat([visual_features, h1_prob_tensor], dim=1)
            else:
                h2_input = visual_features
                
            raw_logits = model.granular_pathology_head(h2_input)
            cond_p = torch.softmax(raw_logits, dim=1).cpu().numpy()[0]
            joint_p = cond_p * p_val
            
            conditional_probs_trajectory.append(cond_p)
            joint_probs_trajectory.append(joint_p)
            
    cond_matrix = np.array(conditional_probs_trajectory) # [num_steps, num_classes]
    joint_matrix = np.array(joint_probs_trajectory)
    
    # Compute numerical derivative d(P_cond)/d(P_H1)
    d_cond = np.gradient(cond_matrix, p_h1_values, axis=0)
    mean_abs_derivative = float(np.mean(np.abs(d_cond)))
    max_abs_derivative = float(np.max(np.abs(d_cond)))
    
    top_class_idx = int(np.argmax(cond_matrix[-1]))
    top_class_name = class_names[top_class_idx]
    
    sweep_summary = {
        "top_predicted_class": top_class_name,
        "mean_abs_conditional_derivative": mean_abs_derivative,
        "max_abs_conditional_derivative": max_abs_derivative,
        "conditional_prob_at_p0": float(cond_matrix[0, top_class_idx]),
        "conditional_prob_at_p50": float(cond_matrix[50, top_class_idx]),
        "conditional_prob_at_p100": float(cond_matrix[-1, top_class_idx]),
        "joint_prob_at_p0": float(joint_matrix[0, top_class_idx]),
        "joint_prob_at_p50": float(joint_matrix[50, top_class_idx]),
        "joint_prob_at_p100": float(joint_matrix[-1, top_class_idx]),
        "sensitivity_assessment": (
            "High Sensitivity: H2 conditional logits vary significantly with H1 input scalar."
            if mean_abs_derivative > 0.15
            else "Low/Moderate Sensitivity: H2 prediction is governed almost entirely by visual features."
        )
    }
    
    return sweep_summary


def run_permutation_importance(
    model: MultiHeadConvNeXt,
    visual_features_batch: torch.Tensor,
    h1_probs_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Evaluates loss and accuracy degradation when P(H1) is permuted across a batch.
    """
    model.eval()
    visual_features = visual_features_batch.to(device)
    h1_probs = h1_probs_batch.to(device)
    labels = labels_batch.to(device)
    
    with torch.no_grad():
        # Baseline Forward
        if model.condition_h2_on_h1:
            h2_in_orig = torch.cat([visual_features, h1_probs], dim=1)
        else:
            h2_in_orig = visual_features
        logits_orig = model.granular_pathology_head(h2_in_orig)
        preds_orig = torch.argmax(logits_orig, dim=1)
        acc_orig = float((preds_orig == labels).float().mean().item())
        
        # Permute H1 Scalar
        perm_idx = torch.randperm(h1_probs.size(0))
        h1_perm = h1_probs[perm_idx]
        if model.condition_h2_on_h1:
            h2_in_perm = torch.cat([visual_features, h1_perm], dim=1)
        else:
            h2_in_perm = visual_features
        logits_perm = model.granular_pathology_head(h2_in_perm)
        preds_perm = torch.argmax(logits_perm, dim=1)
        acc_perm = float((preds_perm == labels).float().mean().item())
        
    delta_acc = acc_orig - acc_perm
    return {
        "accuracy_baseline": acc_orig,
        "accuracy_permuted_h1": acc_perm,
        "accuracy_drop": float(round(delta_acc, 4)),
        "scalar_importance_conclusion": (
            "H1 scalar is actively utilized by H2 head."
            if delta_acc > 0.03
            else "H1 scalar exhibits near-zero marginal contribution over visual features."
        )
    }


def evaluate_decoupled_failure_modes(
    calibration_engine: TriageCalibrationEngine,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates Tri-State Triage performance on simulated decoupled failure modes:
    1. Normal False Positive: Normal scan mistakenly rated P(Abnormal)=0.85
    2. Semantic OOD Scan: Unfamiliar pathology with high abnormal prob and flat/irregular logits
    """
    class_names = class_names or DEFAULT_PATHOLOGY_CLASSES
    
    # 1. Normal False Positive Simulation
    sim_fp_h1_prob = 0.85
    sim_fp_h2_logits = np.array([1.2, 0.4, -0.2, 0.1, -1.0, 0.0, -0.5, 0.3, -0.1, -0.2, 0.1, -0.3])
    triage_fp = calibration_engine.evaluate_triage(
        prob_abnormal=sim_fp_h1_prob,
        raw_h2_logits=sim_fp_h2_logits,
        class_names=class_names
    )
    
    # 2. Semantic OOD Simulation (Flat high-uncertainty logits)
    sim_ood_h1_prob = 0.98
    sim_ood_h2_logits = np.array([0.1, 0.15, 0.12, 0.08, 0.14, 0.09, 0.11, 0.13, 0.10, 0.07, 0.16, 0.12])
    triage_ood = calibration_engine.evaluate_triage(
        prob_abnormal=sim_ood_h1_prob,
        raw_h2_logits=sim_ood_h2_logits,
        class_names=class_names
    )
    
    # 3. Normal True Negative Simulation
    sim_tn_h1_prob = 0.03
    sim_tn_h2_logits = np.random.normal(0.0, 1.0, size=len(class_names))
    triage_tn = calibration_engine.evaluate_triage(
        prob_abnormal=sim_tn_h1_prob,
        raw_h2_logits=sim_tn_h2_logits,
        class_names=class_names
    )

    return {
        "true_normal_case": {
            "p_abnormal": sim_tn_h1_prob,
            "triage_state": triage_tn["triage_state"],
            "final_diagnosis": triage_tn["final_diagnosis"],
            "passed_safe_gate": triage_tn["triage_state"] == "NORMAL"
        },
        "semantic_ood_case": {
            "p_abnormal": sim_ood_h1_prob,
            "triage_state": triage_ood["triage_state"],
            "review_reason": triage_ood["review_reason"],
            "energy_score": triage_ood["uncertainty_metrics"]["free_energy"],
            "passed_safe_gate": triage_ood["triage_state"] == "REVIEW_REQUIRED"
        },
        "gatekeeper_false_positive_case": {
            "p_abnormal": sim_fp_h1_prob,
            "triage_state": triage_fp["triage_state"],
            "review_reason": triage_fp["review_reason"],
            "passed_safe_gate": triage_fp["triage_state"] in ("REVIEW_REQUIRED", "KNOWN_PATHOLOGY")
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H1 Sensitivity and Triage Diagnostics")
    parser.add_argument("--output", type=str, default="diagnostic_report.json", help="Output report JSON")
    args = parser.parse_args()
    
    logger.info("Initializing Diagnostic Model and Calibration Engine...")
    model = build_multi_head_model(pretrained=False, warmup=False, condition_h2_on_h1=True)
    engine = TriageCalibrationEngine()
    
    # Create synthetic test batch representing 1792 visual features
    np.random.seed(42)
    torch.manual_seed(42)
    
    sample_feat = torch.randn(1, 1792)
    sweep_res = run_scalar_sweep(model, sample_feat)
    
    batch_feat = torch.randn(64, 1792)
    batch_h1 = torch.rand(64, 1)
    batch_labels = torch.randint(0, 12, (64,))
    perm_res = run_permutation_importance(model, batch_feat, batch_h1, batch_labels)
    
    decoupled_res = evaluate_decoupled_failure_modes(engine)
    
    report = {
        "scalar_sweep": sweep_res,
        "permutation_importance": perm_res,
        "decoupled_failure_modes": decoupled_res
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Diagnostic report exported to {args.output}")
    print(json.dumps(report, indent=4))
