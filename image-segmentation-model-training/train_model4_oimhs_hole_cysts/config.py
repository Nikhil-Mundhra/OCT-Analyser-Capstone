import os

DATASET_ROOT = os.getenv("OIMHS_DATASET_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Segmented/OIMHS_Formatted")
CHECKPOINT_DIR = "./checkpoints/model4_oimhs"

NUM_CLASSES = 5 # 0: Background, 1: Macular Hole, 2: Choroid, 3: Retina, 4: Intraretinal Cysts
EPOCHS = 35
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
