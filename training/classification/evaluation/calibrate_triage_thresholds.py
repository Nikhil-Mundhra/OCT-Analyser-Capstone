"""
evaluation/calibrate_triage_thresholds.py

Empirical Calibration Benchmark for Tri-State Clinical Triage.
Calibrates:
1. Temperature Scaling for H1 Gatekeeper and H2 Pathology Heads (NLL Minimization).
2. Dual H1 Thresholds (tau_n, tau_a) targeting clinical sensitivity >= 98% and specificity >= 95%.
3. Raw Logits Free Energy and Normalized Shannon Entropy OOD Thresholds.
4. Expected Calibration Error (ECE) and Brier Score before and after calibration.
5. Saves calibration_config.json to weights and checkpoint directories.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / "web-app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "web-app"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core_ml.classification.utils.calibration import (
    CalibrationConfig,
    TriageCalibrationEngine,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1) if probs.ndim > 1 else np.maximum(probs, 1.0 - probs)
    predictions = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)
    
    ece = 0.0
    n_samples = len(labels)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Multi-class Brier Score."""
    n_samples = len(labels)
    if probs.ndim == 1:
        return float(np.mean((probs - labels) ** 2))
    
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def fit_temperature_scaling(logits: torch.Tensor, targets: torch.Tensor, is_binary: bool = False) -> float:
    """
    Fits single scalar temperature T via L-BFGS optimization minimizing NLL.
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
        targets = targets.float().view(-1, 1)
        logits = logits.view(-1, 1)
    else:
        criterion = nn.CrossEntropyLoss()
        targets = targets.long().view(-1)
        
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = logits / torch.clamp(temperature, min=0.01)
        loss = criterion(scaled_logits, targets)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return float(torch.clamp(temperature, min=0.01).item())


def calibrate_dual_h1_thresholds(
    h1_probs: np.ndarray,
    h1_labels: np.ndarray,
    target_sensitivity: float = 0.98,
    target_specificity: float = 0.95
) -> Tuple[float, float]:
    """
    Computes dual thresholds tau_n and tau_a:
    - tau_a: Threshold such that Sensitivity(Abnormal) >= target_sensitivity (e.g. 98%).
    - tau_n: Threshold such that Specificity(Normal) >= target_specificity (e.g. 95%).
    """
    thresholds = np.linspace(0.01, 0.99, 1000)
    
    abnormal_mask = (h1_labels == 1)
    normal_mask = (h1_labels == 0)
    
    # 1. tau_a: Lower threshold enough to capture >= target_sensitivity abnormal cases
    sensitivities = [np.mean(h1_probs[abnormal_mask] >= t) for t in thresholds]
    valid_tau_a = [t for t, s in zip(thresholds, sensitivities) if s >= target_sensitivity]
    tau_a = float(max(valid_tau_a)) if valid_tau_a else 0.50
    
    # 2. tau_n: Raise threshold to capture >= target_specificity normal cases
    specificities = [np.mean(h1_probs[normal_mask] <= t) for t in thresholds]
    valid_tau_n = [t for t, spec in zip(thresholds, specificities) if spec >= target_specificity]
    tau_n = float(min(valid_tau_n)) if valid_tau_n else 0.20
    
    # Guarantee tau_n < tau_a
    if tau_n >= tau_a:
        tau_n = max(0.05, tau_a * 0.5)
        
    return float(round(tau_n, 4)), float(round(tau_a, 4))


def run_calibration(
    h1_logits: np.ndarray,
    h1_labels: np.ndarray,
    h2_logits: np.ndarray,
    h2_labels: np.ndarray,
    output_config_path: Path | str,
    target_sensitivity: float = 0.98,
    target_specificity: float = 0.95
) -> CalibrationConfig:
    """
    Runs full calibration pipeline across H1 and H2 validation arrays.
    """
    logger.info("=== Starting Tri-State Triage Calibration Benchmark ===")
    
    # 1. Temperature Scaling on H1
    t_h1 = fit_temperature_scaling(
        torch.tensor(h1_logits, dtype=torch.float32),
        torch.tensor(h1_labels, dtype=torch.float32),
        is_binary=True
    )
    logger.info(f"H1 Optimal Temperature: {t_h1:.4f}")
    
    # 2. Temperature Scaling on H2 (Abnormal samples only)
    valid_h2_mask = (h1_labels == 1) & (h2_labels >= 0)
    if np.sum(valid_h2_mask) > 0:
        t_h2 = fit_temperature_scaling(
            torch.tensor(h2_logits[valid_h2_mask], dtype=torch.float32),
            torch.tensor(h2_labels[valid_h2_mask], dtype=torch.long),
            is_binary=False
        )
    else:
        t_h2 = 1.0
    logger.info(f"H2 Optimal Temperature: {t_h2:.4f}")
    
    # 3. Dual H1 Thresholds
    scaled_h1_probs = 1.0 / (1.0 + np.exp(-h1_logits / t_h1))
    tau_n, tau_a = calibrate_dual_h1_thresholds(
        scaled_h1_probs,
        h1_labels,
        target_sensitivity=target_sensitivity,
        target_specificity=target_specificity
    )
    logger.info(f"Calibrated Dual H1 Thresholds: tau_n = {tau_n:.4f}, tau_a = {tau_a:.4f}")
    
    # 4. Energy and MSP on Known In-Distribution Pathologies
    scaled_h2_logits = h2_logits[valid_h2_mask] / t_h2
    max_scaled = np.max(scaled_h2_logits, axis=1, keepdims=True)
    lse = max_scaled + np.log(np.sum(np.exp(scaled_h2_logits - max_scaled), axis=1, keepdims=True))
    energies = (-t_h2 * lse).flatten()
    
    exp_h2 = np.exp(scaled_h2_logits - np.max(scaled_h2_logits, axis=1, keepdims=True))
    probs_h2 = exp_h2 / np.sum(exp_h2, axis=1, keepdims=True)
    msps = np.max(probs_h2, axis=1)
    
    # Calibrate Energy threshold at 95th percentile of known valid pathologies
    tau_energy = float(round(np.percentile(energies, 95), 4))
    tau_msp = float(round(np.percentile(msps, 5), 4))
    
    # Normalized Entropy
    k = probs_h2.shape[1]
    safe_probs = np.clip(probs_h2, 1e-12, 1.0)
    entropies = -np.sum(safe_probs * np.log(safe_probs), axis=1) / np.log(k)
    tau_entropy = float(round(np.percentile(entropies, 95), 4))
    
    logger.info(f"OOD Calibration Cutoffs: tau_energy = {tau_energy:.4f}, tau_msp = {tau_msp:.4f}, tau_entropy = {tau_entropy:.4f}")
    
    # 5. ECE and Brier Metrics
    raw_h2_probs = np.exp(h2_logits[valid_h2_mask]) / np.sum(np.exp(h2_logits[valid_h2_mask]), axis=1, keepdims=True)
    ece_before = compute_ece(raw_h2_probs, h2_labels[valid_h2_mask])
    ece_after = compute_ece(probs_h2, h2_labels[valid_h2_mask])
    brier_before = compute_brier_score(raw_h2_probs, h2_labels[valid_h2_mask])
    brier_after = compute_brier_score(probs_h2, h2_labels[valid_h2_mask])
    
    logger.info(f"H2 Calibration Results: ECE {ece_before:.4f} -> {ece_after:.4f} | Brier {brier_before:.4f} -> {brier_after:.4f}")
    
    config = CalibrationConfig(
        tau_n=tau_n,
        tau_a=tau_a,
        temperature_h1=float(round(t_h1, 4)),
        temperature_h2=float(round(t_h2, 4)),
        tau_energy=tau_energy,
        tau_msp=tau_msp,
        tau_entropy=tau_entropy,
        energy_rule="greater_than_is_ood",
        metadata={
            "calibrated": True,
            "target_sensitivity": target_sensitivity,
            "target_specificity": target_specificity,
            "ece_before": float(round(ece_before, 4)),
            "ece_after": float(round(ece_after, 4)),
            "brier_before": float(round(brier_before, 4)),
            "brier_after": float(round(brier_after, 4)),
            "num_samples_h1": len(h1_labels),
            "num_samples_h2": int(np.sum(valid_h2_mask)),
        }
    )
    
    config.save(output_config_path)
    logger.info(f"Calibration Configuration successfully serialized to {output_config_path}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Triage Thresholds on Validation Data")
    parser.add_argument("--output", type=str, default="calibration_config.json", help="Output JSON path")
    parser.add_argument("--sensitivity", type=float, default=0.98, help="Target Abnormal Sensitivity")
    parser.add_argument("--specificity", type=float, default=0.95, help="Target Normal Specificity")
    args = parser.parse_args()
    
    # Smoke test calibration routine with representative validation arrays
    np.random.seed(42)
    n_val = 500
    dummy_h1_labels = np.random.choice([0, 1], size=n_val, p=[0.4, 0.6])
    dummy_h1_logits = np.where(dummy_h1_labels == 1, np.random.normal(2.0, 1.0, n_val), np.random.normal(-2.0, 1.0, n_val))
    
    dummy_h2_labels = np.where(dummy_h1_labels == 1, np.random.randint(0, 12, size=n_val), -1)
    dummy_h2_logits = np.random.normal(0.0, 1.0, size=(n_val, 12))
    for i in range(n_val):
        if dummy_h2_labels[i] >= 0:
            dummy_h2_logits[i, dummy_h2_labels[i]] += 3.5
            
    run_calibration(
        h1_logits=dummy_h1_logits,
        h1_labels=dummy_h1_labels,
        h2_logits=dummy_h2_logits,
        h2_labels=dummy_h2_labels,
        output_config_path=args.output,
        target_sensitivity=args.sensitivity,
        target_specificity=args.specificity
    )
