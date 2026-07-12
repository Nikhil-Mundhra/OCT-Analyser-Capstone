import numpy as np
from PIL import Image
import torch

from backend.oct_analyzer.data_loader import load_normalized_scan
from backend.oct_analyzer.mvp_pipeline import process_scan
from backend.oct_analyzer.segmentation import UnifiedOCTAnalyzer

# Create dummy image
img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
Image.fromarray(img).save("dummy.png")

scan = load_normalized_scan("dummy.png")
print("Scan shape:", scan.volume.shape)

try:
    res = process_scan(scan)
    print("Process success, gradcam shape:", len(res["gradcams"]))
except Exception as e:
    import traceback
    traceback.print_exc()

