"""
scripts/inference_pipeline.py

End-to-End Inference Pipeline for Multi-Head ConvNeXt OCT Classification with Calibrated Tri-State Clinical Triage.
Architecture:
- H1: Screening Gatekeeper P(Abnormal) with Dual Thresholds (tau_n, tau_a).
- H2: Granular Multi-Class Pathology Head with Raw Logits Free Energy and Entropy OOD Scoring.
- Tri-State Triage:
    1. NORMAL (P(Abnormal) <= tau_n)
    2. REVIEW_REQUIRED (tau_n < P(Abnormal) < tau_a OR OOD / Low Confidence)
    3. KNOWN_PATHOLOGY (P(Abnormal) >= tau_a AND In-Distribution Known Disease)
- State-dependent Grad-CAM interpretability.
- Fully backwards-compatible with Level1, Level2, Level3, Final_Diagnosis schema.
"""

import sys
import os
import tempfile
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from backend.core_ml.classification.models.multi_head_convnext import build_multi_head_model
from backend.core_ml.classification.data.transforms import BlackoutCorners, LetterboxPad
from backend.core_ml.classification.utils.gradcam import MultiHeadGradCAM
from backend.core_ml.classification.utils.calibration import (
    CalibrationConfig,
    TriageCalibrationEngine,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PATHOLOGY_CLASSES = [
    'CNV', 'DRUSEN', 'AMD', 'General_AMD', 
    'DME', 'DR', 'MH', 'RVO', 'RAO', 
    'CSR', 'ERM', 'VID'
]

class OCTInferencePipeline:
    def __init__(
        self,
        *args,
        device: str = "auto",
        calibration_config: Optional[CalibrationConfig] = None,
        calibration_path: Optional[Path | str] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ):
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Initialising OCT Inference Pipeline on device: {self.device}")
        
        self.transform = transforms.Compose([
            BlackoutCorners(),
            LetterboxPad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Build Model
        logger.info("Building Multi-Head ConvNeXt V2...")
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
        multi_head_ckpt = weights_dir / "multi_head.pth"
        if not multi_head_ckpt.exists():
            multi_head_ckpt = weights_dir / "multi_head_mps" / "fold0_best_model.pth"
        
        if not multi_head_ckpt.exists():
            token = os.getenv("HF_TOKEN")
            try:
                from huggingface_hub import hf_hub_download
                logger.info("Downloading classification weights from HF Hub: NMundhra/OCT-Classifier-Model-Weights...")
                cached = hf_hub_download(
                    repo_id="NMundhra/OCT-Classifier-Model-Weights",
                    filename="fold0_best_model.pth",
                    repo_type="model",
                    token=token
                )
                multi_head_ckpt = Path(cached)
                logger.info(f"Successfully cached classification weights to {multi_head_ckpt}")
            except Exception as err:
                logger.warning(f"Could not download weights from HF Hub: {err}")
        
        self.model = build_multi_head_model(pretrained=False, warmup=False).to(self.device)
        
        if multi_head_ckpt.exists():
            state = torch.load(multi_head_ckpt, map_location=self.device)
            state_dict = state.get("model_state_dict", state)
            
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_key = k[7:] if k.startswith('module.') else k
                clean_state_dict[clean_key] = v
                
            self.model.load_state_dict(clean_state_dict)
            logger.info(f"  -> Loaded weights from {multi_head_ckpt}")
        else:
            logger.warning(f"  -> No checkpoint found at {multi_head_ckpt}. Using random initialization!")
            
        self.model.eval()
        self.l1_mapping = {0: "NORMAL", 1: "ABNORMAL"}

        # Dynamic class names resolution
        self.class_names = class_names or DEFAULT_PATHOLOGY_CLASSES

        # Initialize Calibration Engine
        if calibration_path is None:
            default_calib = weights_dir / "calibration_config.json"
            if default_calib.exists():
                calibration_path = default_calib
            elif (weights_dir / "multi_head_mps" / "calibration_config.json").exists():
                calibration_path = weights_dir / "multi_head_mps" / "calibration_config.json"

        self.calibration_engine = TriageCalibrationEngine(
            config=calibration_config,
            config_path=calibration_path
        )
        logger.info(f"  -> Triage Calibration Engine ready (tau_n={self.calibration_engine.config.tau_n}, tau_a={self.calibration_engine.config.tau_a})")

    def _get_heatmap_base64(self, img_pil, cam_array):
        import base64
        import io
        overlay = MultiHeadGradCAM.overlay_cam(img_pil, cam_array, alpha=0.5)
        buffered = io.BytesIO()
        overlay.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def predict(
        self, 
        image_input: Any, 
        gradcam: bool = False, 
        output_dir: str = "output/explanations"
    ) -> Dict[str, Any]:
        if isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            logger.info(f"Processing image: {image_input}")
            try:
                img = Image.open(image_input).convert("RGB")
            except Exception as e:
                return {"error": f"Failed to load image: {e}"}
        else:
            return {"error": f"Unsupported image input type: {type(image_input)}"}
            
        tensor = self.transform(img)
        input_tensor = tensor.unsqueeze(0).to(self.device)
        
        results = {
            "Level1": {},
            "Level2": {},
            "Level3": {},
            "Final_Diagnosis": None,
            "Triage": {},
            "Path": [],
            "gradcams": {}
        }
        
        if gradcam:
            grad_context = torch.enable_grad()
            input_tensor.requires_grad = True
        else:
            grad_context = torch.no_grad()
            
        with grad_context:
            if hasattr(self.model.backbone, "stages"):
                target_layer = self.model.backbone.stages[-1].blocks[-1]
            elif hasattr(self.model.backbone, "stages_3"):
                target_layer = self.model.backbone.stages_3.blocks[-1]
            elif hasattr(self.model, "cbam_s4"):
                target_layer = self.model.cbam_s4
            else:
                target_layer = self.model.granular_pathology_head
            cam_generator = MultiHeadGradCAM(self.model, target_layer)
            
            outputs = self.model(input_tensor)
            out1 = outputs['normal_abnormal']
            out2 = outputs['pathology']  # Raw H2 Logits before activation
            
            # --- Probability Extraction ---
            prob_abnormal = torch.sigmoid(out1[0, 0]).item()
            num_h2_classes = out2.size(-1)
            active_class_names = self.class_names[:num_h2_classes] if len(self.class_names) >= num_h2_classes else [f"Class_{i}" for i in range(num_h2_classes)]

            # --- Calibrated Tri-State Triage & OOD Evaluation ---
            raw_h2_logits_np = out2[0].detach().cpu().numpy()
            triage_eval = self.calibration_engine.evaluate_triage(
                prob_abnormal=prob_abnormal,
                raw_h2_logits=raw_h2_logits_np,
                class_names=active_class_names
            )

            triage_state = triage_eval["triage_state"]
            review_reason = triage_eval["review_reason"]
            final_diagnosis = triage_eval["final_diagnosis"]
            pred_candidate = triage_eval["predicted_pathology_candidate"]
            pred_candidate_idx = triage_eval["predicted_pathology_idx"]

            # --- Populating Standard Hierarchical Output Schema ---
            # Level 1: Gatekeeper
            pred_l1_label = "NORMAL" if prob_abnormal <= 0.50 else "ABNORMAL"
            conf_l1 = prob_abnormal if prob_abnormal > 0.50 else 1.0 - prob_abnormal
            results["Level1"] = {
                "prediction": pred_l1_label,
                "confidence": float(conf_l1),
                "probs": {"NORMAL": float(1.0 - prob_abnormal), "ABNORMAL": float(prob_abnormal)},
                "screening_boundary": "NORMAL" if prob_abnormal <= self.calibration_engine.config.tau_n else ("ABNORMAL" if prob_abnormal >= self.calibration_engine.config.tau_a else "AMBIGUOUS")
            }
            results["Path"].append(f"L1: {pred_l1_label} (P(A)={prob_abnormal:.3f})")

            # Level 2 & 3: Pathology Profile
            h2_conditional_probs = triage_eval["conditional_probabilities"]
            h2_joint_probs = triage_eval["joint_probabilities"]
            conf_l2 = h2_conditional_probs[pred_candidate]

            results["Level2"] = {
                "prediction": pred_candidate if triage_state == "KNOWN_PATHOLOGY" else final_diagnosis,
                "candidate": pred_candidate,
                "confidence": float(conf_l2),
                "probs": h2_conditional_probs,
                "joint_probs": h2_joint_probs,
            }
            results["Level3"] = results["Level2"]
            results["Path"].append(f"Triage: {triage_state} [{review_reason}]")

            results["Final_Diagnosis"] = final_diagnosis
            results["Triage"] = triage_eval

            # --- State-Dependent Grad-CAM Generation ---
            if gradcam:
                if triage_state == "NORMAL":
                    # Level 1 Healthy context
                    heatmap_l1 = cam_generator.generate_cam(input_tensor, target_head=1)
                    results["gradcams"]["L1"] = self._get_heatmap_base64(img, heatmap_l1)
                elif triage_state == "KNOWN_PATHOLOGY":
                    # Level 1 abnormality evidence + Level 2 specific disease activation
                    heatmap_l1 = cam_generator.generate_cam(input_tensor, target_head=1)
                    results["gradcams"]["L1"] = self._get_heatmap_base64(img, heatmap_l1)
                    heatmap_l2 = cam_generator.generate_cam(input_tensor, target_head=2, target_class=pred_candidate_idx)
                    results["gradcams"]["L2"] = self._get_heatmap_base64(img, heatmap_l2)
                elif triage_state == "REVIEW_REQUIRED":
                    # For Review Required cases, prominently render H1 Abnormality evidence
                    heatmap_l1 = cam_generator.generate_cam(input_tensor, target_head=1)
                    results["gradcams"]["L1"] = self._get_heatmap_base64(img, heatmap_l1)
                    # When reason is OOD/Low confidence, we avoid anchoring the clinician on the unconfirmed class,
                    # but provide candidate heatmap under an explicit unconfirmed key if requested
                    if review_reason in ("H2_OOD_UNRECOGNIZED", "H2_LOW_CONFIDENCE"):
                        heatmap_l2_candidate = cam_generator.generate_cam(input_tensor, target_head=2, target_class=pred_candidate_idx)
                        results["gradcams"]["L2_Candidate_Unconfirmed"] = self._get_heatmap_base64(img, heatmap_l2_candidate)

        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run OCT Multi-Head Inference with Tri-State Triage")
    parser.add_argument("--image", type=str, required=True, help="Path to raw OCT image")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmaps")
    parser.add_argument("--output-dir", type=str, default="output/explanations", help="Output directory for heatmaps")
    args = parser.parse_args()
    
    pipeline = OCTInferencePipeline()
    res = pipeline.predict(args.image, gradcam=args.gradcam, output_dir=args.output_dir)
    print("\n--- INFERENCE RESULTS WITH TRI-STATE TRIAGE ---")
    print(json.dumps(res, indent=4))
