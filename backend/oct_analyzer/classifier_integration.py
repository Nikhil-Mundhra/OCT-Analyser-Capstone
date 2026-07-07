import sys
from pathlib import Path
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Locate the huggingface space directory relative to the backend
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HF_SPACE_DIR = PROJECT_ROOT / "image-classification-model-training" / "hf_space"

# We must add it to path to import inference_pipeline correctly 
if str(HF_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(HF_SPACE_DIR))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from scripts.inference_pipeline import OCTInferencePipeline
except ImportError as e:
    logger.error(f"Could not import OCTInferencePipeline from {HF_SPACE_DIR}. Error: {e}")
    OCTInferencePipeline = None


class ClassifierWrapper:
    _instance: Optional["ClassifierWrapper"] = None

    def __init__(self):
        if OCTInferencePipeline is None:
            raise RuntimeError("OCTInferencePipeline is not available.")
            
        weights_dir = HF_SPACE_DIR / "weights"
        l3_ckpts = {
            "Macular": str(weights_dir / "level3_macular.pth"),
            "Diabetic": str(weights_dir / "level3_diabetic.pth"),
            "Vascular": str(weights_dir / "level3_vascular.pth"),
            "Fluid": str(weights_dir / "level3_fluid.pth"),
            "Structural": str(weights_dir / "level3_structural.pth"),
        }
        
        self.pipeline = OCTInferencePipeline(
            l1_ckpt=str(weights_dir / "level1.pth"),
            l2_ckpt=str(weights_dir / "level2.pth"),
            l3_ckpts=l3_ckpts,
            device="auto"
        )

    @classmethod
    def get_instance(cls) -> "ClassifierWrapper":
        if cls._instance is None:
            logger.info("Initializing ClassifierWrapper singleton...")
            cls._instance = cls()
        return cls._instance

    def predict(self, image_path: str, gradcam: bool = True) -> dict[str, Any]:
        """Runs the prediction on a given image file path."""
        return self.pipeline.predict(image_path, gradcam=gradcam)

def get_classifier() -> ClassifierWrapper:
    return ClassifierWrapper.get_instance()
