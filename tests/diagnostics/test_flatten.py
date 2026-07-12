import numpy as np
from PIL import Image
import torch
import cv2

from backend.oct_analyzer.data_loader import load_normalized_scan
from backend.oct_analyzer.mvp_pipeline import process_scan
from backend.oct_analyzer.segmentation import UnifiedOCTAnalyzer
from backend.oct_analyzer.anatomical_flattener import flatten_volume_to_rpe

# Create dummy image: a horizontal white line on black background
img = np.zeros((512, 512), dtype=np.uint8)
img[250:260, :] = 255 # horizontal line
Image.fromarray(img).save("dummy_horiz.png")

scan = load_normalized_scan("dummy_horiz.png")
flattened = flatten_volume_to_rpe(scan.volume)

Image.fromarray(flattened[0].astype(np.uint8)).save("dummy_flattened.png")
print("Done flattening")
