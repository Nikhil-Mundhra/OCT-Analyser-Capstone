"""
scripts/inference_pipeline.py

End-to-End Inference Pipeline for Hierarchical OCT Classification.
Connects L1 -> L2 -> L3 into a single callable function.
Takes a raw OCT scan and returns a final diagnosis with confidence scores.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image

# Add parent directory to path so we can import from models and data
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.level1_gatekeeper import build_gatekeeper
from models.level2_router import build_router
from models.level3_specialist import build_specialist, SPECIALIST_CONFIGS
from data.transforms import get_transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class OCTInferencePipeline:
    def __init__(
        self,
        l1_ckpt: Optional[str] = None,
        l2_ckpt: Optional[str] = None,
        l3_ckpts: Optional[Dict[str, str]] = None,
        device: str = "auto",
    ):
        """
        Initializes the entire L1 -> L2 -> L3 inference pipeline.
        
        Args:
            l1_ckpt: Path to Level 1 (Gatekeeper) checkpoint.
            l2_ckpt: Path to Level 2 (Router) checkpoint.
            l3_ckpts: Dict mapping specialist names ('Macular', etc.) to checkpoint paths.
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
        
        # 1. Load Transforms (we use 'val' split for deterministic preprocessing)
        self.transform_l1_l2 = get_transforms("level1", "val")
        # For L3, we can just use level3_macular's val transform as all L3 val transforms are identical (384px)
        self.transform_l3 = get_transforms("level3_macular", "val")
        
        # 2. Build Models
        logger.info("Building Level 1 Gatekeeper...")
        self.l1_model = build_gatekeeper(pretrained=False).to(self.device)
        self._load_ckpt(self.l1_model, l1_ckpt)
        self.l1_model.eval()
        
        logger.info("Building Level 2 Router...")
        self.l2_model = build_router(pretrained=False).to(self.device)
        self._load_ckpt(self.l2_model, l2_ckpt)
        self.l2_model.eval()
        
        self.l3_models = {}
        # L3 models bypassed until Tuesday
        # l3_ckpts = l3_ckpts or {}
        # for spec_name in SPECIALIST_CONFIGS.keys():
        #     logger.info(f"Building Level 3 Specialist: {spec_name}...")
        #     model = build_specialist(spec_name, pretrained=False).to(self.device)
        #     self._load_ckpt(model, l3_ckpts.get(spec_name))
        #     model.eval()
        #     self.l3_models[spec_name] = model
            
        # 3. Label Mappings
        self.l1_mapping = {0: "NORMAL", 1: "ABNORMAL"}
        self.l2_mapping = {
            0: "Macular",
            1: "Diabetic",
            2: "Vascular",
            3: "Fluid",
            4: "Structural"
        }

    def _load_ckpt(self, model: torch.nn.Module, ckpt_path: Optional[str]):
        """Helper to load state dict if path is provided."""
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            # If standard PyTorch Lightning or custom training loop saved dict with 'model_state_dict'
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            logger.info(f"  -> Loaded weights from {ckpt_path}")
        else:
            logger.warning(f"  -> No checkpoint provided for {model.__class__.__name__}. Using random initialization!")

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Runs the end-to-end inference pipeline on a single image.
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load Image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}
            
        # Prepare Tensors
        tensor_224 = self.transform_l1_l2(img).unsqueeze(0).to(self.device)
        tensor_384 = self.transform_l3(img).unsqueeze(0).to(self.device)
        
        results = {
            "Level1": {},
            "Level2": {},
            "Level3": {},
            "Final_Diagnosis": None,
            "Path": []
        }
        
        # --- LEVEL 1: Gatekeeper ---
        logits_l1 = self.l1_model(tensor_224)
        probs_l1 = F.softmax(logits_l1, dim=1).squeeze(0)
        pred_l1_idx = torch.argmax(probs_l1).item()
        pred_l1_label = self.l1_mapping[pred_l1_idx]
        conf_l1 = probs_l1[pred_l1_idx].item()
        
        results["Level1"] = {
            "prediction": pred_l1_label,
            "confidence": conf_l1,
            "probs": {self.l1_mapping[i]: probs_l1[i].item() for i in range(2)}
        }
        results["Path"].append(f"L1: {pred_l1_label}")
        
        if pred_l1_label == "NORMAL":
            results["Final_Diagnosis"] = "NORMAL"
            logger.info("Pipeline terminated at Level 1 (NORMAL)")
            return results
            
        # --- LEVEL 2: Disease Router ---
        logits_l2 = self.l2_model(tensor_224)
        probs_l2 = F.softmax(logits_l2, dim=1).squeeze(0)
        pred_l2_idx = torch.argmax(probs_l2).item()
        pred_l2_label = self.l2_mapping[pred_l2_idx]
        conf_l2 = probs_l2[pred_l2_idx].item()
        
        results["Level2"] = {
            "prediction": pred_l2_label,
            "confidence": conf_l2,
            "probs": {self.l2_mapping[i]: probs_l2[i].item() for i in range(5)}
        }
        results["Path"].append(f"L2: {pred_l2_label}")
        
        # --- LEVEL 3: Specialist (Skipped for now) ---
        # specialist_model = self.l3_models[pred_l2_label]
        # spec_config = SPECIALIST_CONFIGS[pred_l2_label]
        # l3_classes_map = spec_config["classes"]
        # 
        # logits_l3 = specialist_model(tensor_384)
        # probs_l3 = F.softmax(logits_l3, dim=1).squeeze(0)
        # pred_l3_idx = torch.argmax(probs_l3).item()
        # pred_l3_label = l3_classes_map[pred_l3_idx]
        # conf_l3 = probs_l3[pred_l3_idx].item()
        # 
        # results["Level3"] = {
        #     "specialist_used": spec_config["specialist_name"],
        #     "prediction": pred_l3_label,
        #     "confidence": conf_l3,
        #     "probs": {l3_classes_map[i]: probs_l3[i].item() for i in range(len(l3_classes_map))}
        # }
        # results["Path"].append(f"L3: {pred_l3_label}")
        # results["Final_Diagnosis"] = pred_l3_label
        
        logger.info(f"Skipping Level 3 for now. Outputting Level 2 prediction.")
        results["Final_Diagnosis"] = pred_l2_label
        
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run OCT Hierarchical Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to raw OCT image")
    parser.add_argument("--l1_ckpt", type=str, default=None, help="L1 model checkpoint path")
    parser.add_argument("--l2_ckpt", type=str, default=None, help="L2 model checkpoint path")
    args = parser.parse_args()
    
    pipeline = OCTInferencePipeline(
        l1_ckpt=args.l1_ckpt,
        l2_ckpt=args.l2_ckpt,
        # L3 checkpoints can be passed in similarly, skipping for CLI simplicity in this test
    )
    
    res = pipeline.predict(args.image)
    print("\n--- INFERENCE RESULTS ---")
    print(json.dumps(res, indent=4))
