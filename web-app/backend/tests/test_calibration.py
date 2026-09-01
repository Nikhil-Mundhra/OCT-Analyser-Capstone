import numpy as np
import torch
import tempfile
from pathlib import Path

from backend.core_ml.classification.utils.calibration import (
    CalibrationConfig,
    TriageCalibrationEngine,
)
from backend.core_ml.classification.scripts.inference_pipeline import (
    OCTInferencePipeline,
    DEFAULT_PATHOLOGY_CLASSES,
)


def test_calibration_config_serialization():
    config = CalibrationConfig(
        tau_n=0.15,
        tau_a=0.75,
        temperature_h1=1.1,
        temperature_h2=1.3,
        tau_energy=-1.8,
        tau_msp=0.40,
        tau_entropy=0.80,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        config.save(tmp_path)
        loaded = CalibrationConfig.load(tmp_path)
        assert loaded.tau_n == 0.15
        assert loaded.tau_a == 0.75
        assert loaded.temperature_h2 == 1.3
        assert loaded.tau_energy == -1.8
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_triage_routing_normal():
    config = CalibrationConfig(tau_n=0.20, tau_a=0.70)
    engine = TriageCalibrationEngine(config=config)
    
    # Low abnormal probability
    raw_logits = np.zeros(12)
    res = engine.evaluate_triage(
        prob_abnormal=0.08,
        raw_h2_logits=raw_logits,
        class_names=DEFAULT_PATHOLOGY_CLASSES
    )
    assert res["triage_state"] == "NORMAL"
    assert res["review_reason"] == "NONE"
    assert res["final_diagnosis"] == "NORMAL"


def test_triage_routing_h1_ambiguous():
    config = CalibrationConfig(tau_n=0.20, tau_a=0.70)
    engine = TriageCalibrationEngine(config=config)
    
    # Ambiguous screening probability (0.45 is in (0.20, 0.70))
    raw_logits = np.zeros(12)
    res = engine.evaluate_triage(
        prob_abnormal=0.45,
        raw_h2_logits=raw_logits,
        class_names=DEFAULT_PATHOLOGY_CLASSES
    )
    assert res["triage_state"] == "REVIEW_REQUIRED"
    assert res["review_reason"] == "H1_AMBIGUOUS"
    assert res["final_diagnosis"] == "REVIEW_REQUIRED"


def test_triage_routing_known_pathology():
    config = CalibrationConfig(tau_n=0.20, tau_a=0.70, tau_energy=-1.0, tau_msp=0.30)
    engine = TriageCalibrationEngine(config=config)
    
    # Strong abnormal probability + sharp prominent logit for DME (idx 4)
    raw_logits = np.zeros(12)
    raw_logits[4] = 4.5  # High confidence DME
    res = engine.evaluate_triage(
        prob_abnormal=0.92,
        raw_h2_logits=raw_logits,
        class_names=DEFAULT_PATHOLOGY_CLASSES
    )
    assert res["triage_state"] == "KNOWN_PATHOLOGY"
    assert res["review_reason"] == "NONE"
    assert res["final_diagnosis"] == "DME"
    assert res["predicted_pathology_candidate"] == "DME"


def test_triage_routing_h2_ood():
    config = CalibrationConfig(tau_n=0.20, tau_a=0.70, tau_energy=-2.5, tau_msp=0.40)
    engine = TriageCalibrationEngine(config=config)
    
    # Strong abnormal probability + flat/uncertain H2 logits (novel/unseen pathology)
    raw_logits = np.array([0.1] * 12)
    res = engine.evaluate_triage(
        prob_abnormal=0.95,
        raw_h2_logits=raw_logits,
        class_names=DEFAULT_PATHOLOGY_CLASSES
    )
    assert res["triage_state"] == "REVIEW_REQUIRED"
    assert res["review_reason"] in ("H2_OOD_UNRECOGNIZED", "H2_LOW_CONFIDENCE")
    assert res["final_diagnosis"] == "REVIEW_REQUIRED"
    assert res["uncertainty_metrics"]["is_ood_aggregate"] is True


def test_energy_and_entropy_computations():
    logits = np.array([3.0, 1.0, -1.0])
    energy = TriageCalibrationEngine.compute_energy_score(logits, temperature=1.0)
    assert isinstance(energy, float)
    assert energy < 0.0  # LogSumExp of [3, 1, -1] is positive -> Energy is negative
    
    uniform_probs = np.array([1/3, 1/3, 1/3])
    entropy_uniform = TriageCalibrationEngine.compute_normalized_entropy(uniform_probs)
    assert np.isclose(entropy_uniform, 1.0, atol=1e-3)
    
    sharp_probs = np.array([0.999, 0.0005, 0.0005])
    entropy_sharp = TriageCalibrationEngine.compute_normalized_entropy(sharp_probs)
    assert entropy_sharp < 0.1
