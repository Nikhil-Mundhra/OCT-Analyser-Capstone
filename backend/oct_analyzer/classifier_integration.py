import sys
from pathlib import Path
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — locate the HF Space directory (shared between local & remote)
# ---------------------------------------------------------------------------
from .constants import CLASSIFIER_WEIGHTS_PATH, PROJECT_ROOT, get_compute_device
CORE_ML_CLASS_DIR = PROJECT_ROOT / "backend" / "core_ml" / "classification"

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from backend.core_ml.classification.scripts.inference_pipeline import OCTInferencePipeline
except ImportError as e:
    logger.error(f"Could not import OCTInferencePipeline from {CORE_ML_CLASS_DIR}. Error: {e}")
    OCTInferencePipeline = None

class ClassifierWrapper:
    _instance: Optional["ClassifierWrapper"] = None

    def __init__(self):
        if OCTInferencePipeline is None:
            raise RuntimeError("OCTInferencePipeline is not available.")

        weights_dir = CORE_ML_CLASS_DIR / "weights"
        l3_ckpts = {
            "Macular": str(weights_dir / "level3_macular.pth"),
            "Diabetic": str(weights_dir / "level3_diabetic.pth"),
            "Vascular": str(weights_dir / "level3_vascular.pth"),
            "Fluid": str(weights_dir / "level3_fluid.pth"),
            "Structural": str(weights_dir / "level3_structural.pth"),
        }

        self.device = get_compute_device()
        logger.info(f"[ClassifierWrapper] Initialized on device: {self.device}")

        self.pipeline = OCTInferencePipeline(
            l1_ckpt=str(CLASSIFIER_WEIGHTS_PATH),
            l2_ckpt=str(weights_dir / "level2.pth"),
            l3_ckpts=l3_ckpts,
            device=str(self.device),
        )

    @classmethod
    def get_instance(cls) -> "ClassifierWrapper":
        if cls._instance is None:
            logger.info("Initializing ClassifierWrapper singleton...")
            cls._instance = cls()
        return cls._instance

    def predict(self, image_path: str, gradcam: bool = True) -> dict[str, Any]:
        """Runs prediction locally or offloads to HF ZeroGPU Space if enabled."""
        if os.getenv("OCT_REMOTE_OFFLOAD", "false").lower() == "true":
            try:
                from .remote_hf_client import RemoteHFSpaceClient
                remote_client = RemoteHFSpaceClient.get_instance()
                logger.info("Offloading classification to HF ZeroGPU Space (NMundhra/OCT-Image-Classifier-Model)...")
                res = remote_client.predict_classification(image_path)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.warning(f"Remote HF offloading failed ({e}). Falling back to local execution.")

        return self.pipeline.predict(image_path, gradcam=gradcam)


def get_classifier() -> ClassifierWrapper:
    return ClassifierWrapper.get_instance()
