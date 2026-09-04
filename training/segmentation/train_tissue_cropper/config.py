"""
training/segmentation/train_tissue_cropper/config.py

Configuration parameters for the Attention U-Net Tissue Cropping training pipeline.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEGMENTED_ROOT = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Segmented")
MASKED_ROOT = PROJECT_ROOT / "data" / "Classified-masked"

# Architecture & Image Dimensions
IMAGE_SIZE = (384, 384)
IN_CHANNELS = 1
NUM_CLASSES = 2
BASE_CHANNELS = 32

# Training Hyperparameters
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
NUM_WORKERS = 4
PIN_MEMORY = False
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
VAL_SPLIT = 0.15
SEED = 42

# Loss Weights
DICE_WEIGHT = 1.0
CE_WEIGHT = 1.0
FOCAL_WEIGHT = 0.5

# Versioning & Checkpoints
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "segmentation" / "tissue_cropper"
SUITE_CKPT_DIR = PROJECT_ROOT / "models_suite" / "tissue_cropper" / "checkpoints"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.json"
SUITE_HISTORY_PATH = SUITE_CKPT_DIR / "training_history.json"
VERSION = "v1.0"
