"""
scripts/inference_pipeline.py

End-to-End Inference Pipeline for Multi-Head ConvNeXt OCT Classification.
Mimics the old L1 -> L2 -> L3 interface for compatibility with the backend API.
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

from backend.core_ml.classification.models.multi_head_convnext import build_multi_head_model
from backend.core_ml.classification.data.transforms import get_transforms
from backend.core_ml.classification.utils.gradcam import MultiHeadGradCAM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class OCTInferencePipeline:
    def __init__(
        self,
        *args,
        device: str = "auto",
        **kwargs
    ):
        """
        Initializes the Multi-Head inference pipeline.
        Accepts *args and **kwargs to ignore legacy arguments (l1_ckpt, l2_ckpt, etc).
        
        Args:
            device: 'cuda', 'mps', 'cpu', or 'auto'.
        """
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
        
        self.transform = get_transforms("val")
        
        # Build Model
        logger.info("Building Multi-Head ConvNeXt V2...")
        # Since we just transitioned, we may not have trained weights yet.
        # We load pretrained=False (unless training completed and weights exist).
        # We will check if multi_head.pth exists in weights_dir.
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
        multi_head_ckpt = weights_dir / "multi_head.pth"
        if not multi_head_ckpt.exists():
            multi_head_ckpt = weights_dir / "multi_head_mps" / "fold0_best_model.pth"
        
        self.model = build_multi_head_model(pretrained=False, warmup=False).to(self.device)
        
        if multi_head_ckpt.exists():
            state = torch.load(multi_head_ckpt, map_location=self.device)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            logger.info(f"  -> Loaded weights from {multi_head_ckpt}")
        else:
            logger.warning(f"  -> No checkpoint found at {multi_head_ckpt}. Using random initialization!")
            
        self.model.eval()
            
        # Label Mappings
        self.l1_mapping = {0: "NORMAL", 1: "ABNORMAL"}
        self.l2_mapping = {
            0: "Macular",
            1: "Diabetic",
            2: "Vascular",
            3: "Fluid",
            4: "Structural"
        }
        
        # L3 Mapping depends on L2 result
        self.l3_mappings = {
            "Macular": {0: 'CNV', 1: 'DRUSEN', 2: 'Generic_AMD'},
            "Diabetic": {0: 'DME', 1: 'DR'},
            "Vascular": {0: 'RVO', 1: 'RAO', 2: 'Generic_Vascular'},
            "Fluid": {0: 'CSR'},
            "Structural": {0: 'ERM', 1: 'VID'}
        }

        # Key mapping for model dict
        self.l2_to_dict_key = {
            "Macular": "macular",
            "Diabetic": "diabetic",
            "Vascular": "vascular",
            "Fluid": "fluid",
            "Structural": "structural"
        }

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
        image_path: str, 
        gradcam: bool = False, 
        output_dir: str = "output/explanations"
    ) -> Dict[str, Any]:
        """
        Runs the multi-head inference pipeline on a single image.
        """
        logger.info(f"Processing image: {image_path}")
        
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}
            
        # MONAI Transform
        tensor = self.transform(image_path)
        if hasattr(tensor, "as_tensor"):
            tensor = tensor.as_tensor()
        elif isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
            
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
            target_layer = self.model.backbone.stages[-1].blocks[-1]
            cam_generator = MultiHeadGradCAM(self.model, target_layer)
            
            outputs = self.model(input_tensor)
            out1 = outputs['normal_abnormal']
            out2 = outputs['pathology']
            out3_dict = outputs['severity']
            
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
                
            # --- LEVEL 2: Disease Router ---
            probs_l2 = F.softmax(out2[0], dim=0)
            pred_l2_idx = torch.argmax(probs_l2).item()
            pred_l2_label = self.l2_mapping[pred_l2_idx]
            conf_l2 = probs_l2[pred_l2_idx].item()
            
            results["Level2"] = {
                "prediction": pred_l2_label,
                "confidence": conf_l2,
                "probs": {self.l2_mapping[i]: probs_l2[i].item() for i in range(5)}
            }
            results["Path"].append(f"L2: {pred_l2_label}")
            
            if gradcam:
                heatmap = cam_generator.generate_cam(input_tensor, target_head=2)
                results["gradcams"]["L2"] = self._get_heatmap_base64(img, heatmap)
            
            # --- LEVEL 3: Specialist ---
            dict_key = self.l2_to_dict_key[pred_l2_label]
            l3_logits = out3_dict[dict_key][0]
            l3_probs = torch.sigmoid(l3_logits) # Since it's multi-label
            
            l3_classes_map = self.l3_mappings[pred_l2_label]
            
            # For Final Diagnosis, we pick the one with highest probability
            pred_l3_idx = torch.argmax(l3_probs).item()
            # If the probability is still very low, we might just return the L2 label? 
            # We'll just mimic old behavior and pick the argmax
            pred_l3_label = l3_classes_map.get(pred_l3_idx, pred_l2_label)
            conf_l3 = l3_probs[pred_l3_idx].item()
            
            results["Level3"] = {
                "specialist_used": pred_l2_label,
                "prediction": pred_l3_label,
                "confidence": conf_l3,
                "probs": {l3_classes_map[i]: l3_probs[i].item() for i in range(len(l3_classes_map))}
            }
            results["Path"].append(f"L3: {pred_l3_label}")
            results["Final_Diagnosis"] = pred_l3_label
            
            if gradcam:
                # Target the highest severity class in Head 3 for this pathology
                heatmap = cam_generator.generate_cam(input_tensor, target_head=3, sub_head=dict_key, class_idx=pred_l3_idx)
                results["gradcams"]["L3"] = self._get_heatmap_base64(img, heatmap)
                
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
