import os
from pathlib import Path

# Paths
DATASET_ROOT = os.getenv("OCT5K_SEGMENTATION_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Segmented/OCT5K_Semantic_Segmentation")
CHECKPOINT_DIR = "./checkpoints/model1_oct5k_layers"

# Hyperparameters
NUM_CLASSES = 6
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
IMAGE_SIZE = (512, 512)
