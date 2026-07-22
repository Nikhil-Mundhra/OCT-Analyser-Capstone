import os

DATASET_ROOT = os.getenv("OCT5K_DETECTION_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Segmented/OCT5K_Object_Detection")
CHECKPOINT_DIR = "./checkpoints/model5_detection"

NUM_CLASSES = 10 # Background + 9 Pathology Bounding Box Classes
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
