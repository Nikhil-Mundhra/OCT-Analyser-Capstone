import sys
from pathlib import Path
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — locate the HF Space directory (shared between local & remote)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_ML_CLASS_DIR = PROJECT_ROOT / "backend" / "core_ml" / "classification"

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from backend.core_ml.classification.scripts.inference_pipeline import OCTInferencePipeline
except ImportError as e:
    logger.error(f"Could not import OCTInferencePipeline from {CORE_ML_CLASS_DIR}. Error: {e}")
    OCTInferencePipeline = None

# ---------------------------------------------------------------------------
# Device override
#
# Set OCT_LOCAL_DEVICE=cpu  to force CPU (good for low-spec dev machines or
#   CI where no GPU is present).
# Set OCT_LOCAL_DEVICE=mps  to force Apple Silicon MPS.
# Set OCT_LOCAL_DEVICE=cuda to force a specific CUDA GPU.
# Leave unset to let the pipeline auto-detect the best available device.
# ---------------------------------------------------------------------------
_LOCAL_DEVICE = os.environ.get("OCT_LOCAL_DEVICE", "auto").strip().lower()
logger.info(f"[ClassifierWrapper] OCT_LOCAL_DEVICE='{_LOCAL_DEVICE}' "
            f"(set to 'cpu', 'mps', or 'cuda' to override auto-detection)")


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

        self.pipeline = OCTInferencePipeline(
            l1_ckpt=str(weights_dir / "level1.pth"),
            l2_ckpt=str(weights_dir / "level2.pth"),
            l3_ckpts=l3_ckpts,
            device=_LOCAL_DEVICE,
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
