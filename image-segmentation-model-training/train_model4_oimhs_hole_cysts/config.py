import os

DATASET_ROOT = os.getenv("OIMHS_DATASET_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Segmented/OIMHS_Formatted")
CHECKPOINT_DIR = "./checkpoints/model4_oimhs"

# Hyperparameters (512x512 resolution with Batch Size 4 + Gradient Accumulation to prevent RAM/Swap ballooning)
NUM_CLASSES = 5  # 0: Background, 1: Macular Hole, 2: Choroid, 3: Retina, 4: Intraretinal Cysts
EPOCHS = 8
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-4
IMAGE_SIZE = (512, 512)
