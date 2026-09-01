"""
utils/calibration.py

Calibration Engine and Configuration for Multi-Head OCT Classification.
Supports:
1. Temperature Scaling optimization for H1 and H2.
2. Dual H1 Thresholds (tau_n, tau_a) for high-sensitivity screening triage.
3. Raw-logit Free Energy and normalized Shannon Entropy OOD scoring.
4. Tri-State Triage Decision Engine (NORMAL, KNOWN_PATHOLOGY, REVIEW_REQUIRED).
"""

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CalibrationConfig:
    """Configuration holding empirical thresholds and calibration parameters."""
    tau_n: float = 0.20
    tau_a: float = 0.70
    temperature_h1: float = 1.0
    temperature_h2: float = 1.0
    tau_energy: float = -2.0
    tau_msp: float = 0.35
    tau_entropy: float = 0.85
    energy_rule: str = "greater_than_is_ood"  # or "less_than_is_ood"
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "calibrated": False,
        "target_sensitivity": 0.98,
        "notes": "Default uncalibrated baseline priors. Fit via calibrate_triage_thresholds.py"
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationConfig":
        valid_keys = {
            "tau_n", "tau_a", "temperature_h1", "temperature_h2",
            "tau_energy", "tau_msp", "tau_entropy", "energy_rule", "metadata"
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, file_path: Path | str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, file_path: Path | str) -> "CalibrationConfig":
        path = Path(file_path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class TriageCalibrationEngine:
    """
    Evaluates Tri-State Clinical Triage decisions using calibrated statistical thresholds.
    """
    def __init__(self, config: Optional[CalibrationConfig] = None, config_path: Optional[Path | str] = None):
        if config_path and Path(config_path).exists():
            self.config = CalibrationConfig.load(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = CalibrationConfig()

    @staticmethod
    def compute_energy_score(raw_logits: torch.Tensor | np.ndarray, temperature: float = 1.0) -> float:
        """
        Compute Free Energy Score E(z; T) = -T * log(sum(exp(z_i / T))) from raw logits.
        Lower/more negative values typically correspond to high in-distribution confidence.
        """
        if isinstance(raw_logits, np.ndarray):
            scaled = raw_logits / max(temperature, 1e-5)
            max_val = np.max(scaled)
            lse = max_val + np.log(np.sum(np.exp(scaled - max_val)))
            return float(-temperature * lse)
        
        scaled = raw_logits / max(temperature, 1e-5)
        lse = torch.logsumexp(scaled, dim=-1)
        return float((-temperature * lse).item() if lse.ndim == 0 else (-temperature * lse)[0].item())

    @staticmethod
    def compute_normalized_entropy(probs: np.ndarray | torch.Tensor) -> float:
        """
        Compute normalized Shannon entropy: H_norm = - (1 / log(K)) * sum(p_i * log(p_i))
        Range: [0.0, 1.0] where 1.0 is maximum uniform uncertainty.
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        probs = np.clip(probs.flatten(), 1e-12, 1.0)
        probs = probs / np.sum(probs)
        k = len(probs)
        if k <= 1:
            return 0.0
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(k)
        return float(entropy / max_entropy)

    def evaluate_triage(
        self,
        prob_abnormal: float,
        raw_h2_logits: np.ndarray | torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate full tri-state clinical triage and OOD risk metrics.
        
        Decision Rule:
        1. prob_abnormal <= tau_n -> NORMAL
        2. tau_n < prob_abnormal < tau_a -> REVIEW_REQUIRED (H1_AMBIGUOUS)
        3. prob_abnormal >= tau_a:
           - if is_ood -> REVIEW_REQUIRED (H2_OOD_UNRECOGNIZED or H2_LOW_CONFIDENCE)
           - if not is_ood -> KNOWN_PATHOLOGY
        """
        if isinstance(raw_h2_logits, torch.Tensor):
            raw_h2_logits = raw_h2_logits.detach().cpu().numpy()
        raw_h2_logits = raw_h2_logits.flatten()
        k = len(raw_h2_logits)

        # 1. Temperature-scaled probabilities and energy score from raw logits
        t_h2 = max(self.config.temperature_h2, 1e-5)
        scaled_logits = raw_h2_logits / t_h2
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        h2_probs = exp_logits / np.sum(exp_logits)

        energy = self.compute_energy_score(raw_h2_logits, temperature=t_h2)
        norm_entropy = self.compute_normalized_entropy(h2_probs)
        
        pred_idx = int(np.argmax(h2_probs))
        msp = float(h2_probs[pred_idx])
        pred_class = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)

        # 2. Evaluate OOD conditions
        if self.config.energy_rule == "greater_than_is_ood":
            energy_is_ood = energy > self.config.tau_energy
        else:
            energy_is_ood = energy < self.config.tau_energy

        msp_is_low = msp < self.config.tau_msp
        entropy_is_high = norm_entropy > self.config.tau_entropy
        
        is_ood = bool(energy_is_ood or (msp_is_low and entropy_is_high))

        # 3. Apply Tri-State Triage Decision Logic
        if prob_abnormal <= self.config.tau_n:
            triage_state = "NORMAL"
            review_reason = "NONE"
            clinical_action = "Routine screening interval recommended. Strong evidence of healthy retina."
            final_diagnosis = "NORMAL"
        elif self.config.tau_n < prob_abnormal < self.config.tau_a:
            triage_state = "REVIEW_REQUIRED"
            review_reason = "H1_AMBIGUOUS"
            clinical_action = "Ambiguous screening boundary. Clinician review required to confirm normal vs abnormal."
            final_diagnosis = "REVIEW_REQUIRED"
        else:
            # prob_abnormal >= self.config.tau_a (Strong evidence of abnormality)
            if is_ood:
                triage_state = "REVIEW_REQUIRED"
                review_reason = "H2_OOD_UNRECOGNIZED" if energy_is_ood else "H2_LOW_CONFIDENCE"
                clinical_action = "Retinal abnormality detected, but pathology pattern does not match recognized categories with sufficient confidence. Refer to retina specialist for comprehensive evaluation."
                final_diagnosis = "REVIEW_REQUIRED"
            else:
                triage_state = "KNOWN_PATHOLOGY"
                review_reason = "NONE"
                clinical_action = f"Known pathology pattern detected ({pred_class}). Clinical confirmation and staging recommended."
                final_diagnosis = pred_class

        # Strict joint disease probabilities: P(D_i) = P(D_i | Abnormal) * P(Abnormal)
        joint_probs = {
            (class_names[i] if class_names and i < len(class_names) else f"Class_{i}"): float(h2_probs[i] * prob_abnormal)
            for i in range(k)
        }
        conditional_probs = {
            (class_names[i] if class_names and i < len(class_names) else f"Class_{i}"): float(h2_probs[i])
            for i in range(k)
        }

        return {
            "triage_state": triage_state,
            "review_reason": review_reason,
            "final_diagnosis": final_diagnosis,
            "predicted_pathology_candidate": pred_class,
            "predicted_pathology_idx": pred_idx,
            "clinical_action": clinical_action,
            "uncertainty_metrics": {
                "free_energy": energy,
                "normalized_entropy": norm_entropy,
                "max_softmax_probability": msp,
                "is_ood_energy": bool(energy_is_ood),
                "is_low_confidence": bool(msp_is_low),
                "is_high_entropy": bool(entropy_is_high),
                "is_ood_aggregate": is_ood,
            },
            "joint_probabilities": joint_probs,
            "conditional_probabilities": conditional_probs,
            "h1_abnormal_prob": prob_abnormal,
            "thresholds_applied": {
                "tau_n": self.config.tau_n,
                "tau_a": self.config.tau_a,
                "tau_energy": self.config.tau_energy,
                "tau_msp": self.config.tau_msp,
                "tau_entropy": self.config.tau_entropy,
                "temperature_h2": t_h2,
            }
        }
