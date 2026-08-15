"""
scripts/inference_pipeline.py

End-to-End Inference Pipeline for Multi-Head ConvNeXt OCT Classification.
Flattened architecture (H1 = Triage, H2 = Granular Multi-Label Pathology).
Maintains L1/L2/L3 result dict structure for backend API compatibility.
"""

import sys
import os
import tempfile
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from backend.core_ml.classification.models.multi_head_convnext import build_multi_head_model
from backend.core_ml.classification.data.transforms import BlackoutCorners, LetterboxPad
import torchvision.transforms as transforms
from backend.core_ml.classification.utils.gradcam import MultiHeadGradCAM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PATHOLOGY_CLASSES = [
    'CNV', 'DRUSEN', 'AMD', 'General_AMD', 
    'DME', 'DR', 'MH', 'RVO', 'RAO', 
    'CSR', 'ERM', 'VID'
]

class OCTInferencePipeline:
    def __init__(
        self,
        *args,
        device: str = "auto",
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
            out2 = outputs['pathology']
            
            # --- LEVEL 1: Gatekeeper ---
            prob1 = torch.sigmoid(out1[0, 0]).item()
            pred_l1_idx = 1 if prob1 > 0.5 else 0
            pred_l1_label = self.l1_mapping[pred_l1_idx]
            conf_l1 = prob1 if prob1 > 0.5 else 1.0 - prob1
            
            results["Level1"] = {
                "prediction": pred_l1_label,
                "confidence": conf_l1,
                "probs": {"NORMAL": 1.0 - prob1, "ABNORMAL": prob1}
            }
            results["Path"].append(f"L1: {pred_l1_label}")
            
            if gradcam:
                heatmap = cam_generator.generate_cam(input_tensor, target_head=1)
                results["gradcams"]["L1"] = self._get_heatmap_base64(img, heatmap)

            if pred_l1_label == "NORMAL":
                results["Final_Diagnosis"] = "NORMAL"
                logger.info("Pipeline terminated at Level 1 (NORMAL)")
                return results
                
            # --- LEVEL 2 & 3: Granular Pathology (Multi-Label Flat) ---
            # Using Sigmoid for multi-label probabilities
            probs_h2 = torch.sigmoid(out2[0]).detach().cpu().numpy()
            pred_l2_idx = np.argmax(probs_h2)
            pred_l2_label = PATHOLOGY_CLASSES[pred_l2_idx]
            conf_l2 = probs_h2[pred_l2_idx].item()
            
            # We mock Level 2 and Level 3 to be the same to maintain API compatibility
            results["Level2"] = {
                "prediction": pred_l2_label,
                "confidence": conf_l2,
                "probs": {PATHOLOGY_CLASSES[i]: probs_h2[i].item() for i in range(len(PATHOLOGY_CLASSES))}
            }
            results["Level3"] = results["Level2"]
            results["Path"].append(f"L2: {pred_l2_label}")
            
            if gradcam:
                heatmap = cam_generator.generate_cam(input_tensor, target_head=2, target_class=pred_l2_idx)
                results["gradcams"]["L2"] = self._get_heatmap_base64(img, heatmap)
            
            results["Final_Diagnosis"] = pred_l2_label
            
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run OCT Multi-Head Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to raw OCT image")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmaps")
    parser.add_argument("--output-dir", type=str, default="output/explanations", help="Output directory for heatmaps")
    args = parser.parse_args()
    
    pipeline = OCTInferencePipeline()
    
    res = pipeline.predict(args.image, gradcam=args.gradcam, output_dir=args.output_dir)
    print("\n--- INFERENCE RESULTS ---")
    print(json.dumps(res, indent=4))
