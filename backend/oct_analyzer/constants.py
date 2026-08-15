import os
import tempfile
from pathlib import Path
import torch

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Storage Directories
RUNTIME_DIR = Path(os.getenv("OCT_RUNTIME_DIR", Path(tempfile.gettempdir()) / "runtime_uploads"))
UPLOAD_DIR = Path(os.getenv("OCT_UPLOAD_DIR", RUNTIME_DIR / "uploads"))
PREVIEW_DIR = Path(os.getenv("OCT_PREVIEW_DIR", RUNTIME_DIR / "previews"))

# Supported Formats
SUPPORTED_SUFFIXES = {
    ".vol", ".dcm", ".zip", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"
}

# CORS & Network Configuration
RAW_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [origin.strip() for origin in RAW_CORS_ORIGINS.split(",") if origin.strip()] if RAW_CORS_ORIGINS else ["*"]
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"https?://.*")

# Service Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Model Checkpoints & Scripts
SEGMENT_PREDICT_SCRIPT = Path(
    os.getenv("SEGMENT_PREDICT_SCRIPT", PROJECT_ROOT / "image-segmentation-model-training" / "scripts" / "predict.py")
)
UNET_CHECKPOINT_PATH = Path(
    os.getenv("UNET_CHECKPOINT_PATH", PROJECT_ROOT / "models_suite" / "model1_oct5k_layers" / "checkpoints" / "best_model.pth")
)
CLASSIFIER_WEIGHTS_PATH = Path(
    os.getenv(
        "CLASSIFIER_WEIGHTS_PATH",
        PROJECT_ROOT / "checkpoints" / "classification" / "multi_head" / "WeightedRandomSampler" / "v1" / "fold0_best_val_loss.pth"
    )
)

# Global Compute Device Switch
OCT_LOCAL_DEVICE = os.getenv("OCT_LOCAL_DEVICE", "cpu").strip().lower()

def get_compute_device(device_override: str | None = None) -> torch.device:
    """
    Global device selector for all ML components (classifier, segmenter, preprocessor).
    Priority order:
      1. Explicit argument `device_override` (if passed)
      2. Environment variable `OCT_LOCAL_DEVICE` ('cpu', 'mps', 'cuda', 'auto')
      3. Default fallback: 'cpu' (prevents GPU memory contention during training)
    """
    target = (device_override or os.getenv("OCT_LOCAL_DEVICE", "cpu")).strip().lower()
    
    if target in ("cpu", "cuda", "mps"):
        return torch.device(target)
    elif target == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    return torch.device("cpu")
