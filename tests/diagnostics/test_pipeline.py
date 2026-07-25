import numpy as np
from PIL import Image
import torch
import cv2
import traceback
import sys

from backend.oct_analyzer.data_loader import load_normalized_scan
from backend.oct_analyzer.mvp_pipeline import process_scan

# Create dummy image: a horizontal white line on black background
img = np.zeros((400, 500), dtype=np.uint8)
img[200:250, :] = 255 # horizontal line (retina)
Image.fromarray(img).save("dummy.jpeg")

try:
    scan = load_normalized_scan("dummy.jpeg")
    print("Scan shape:", scan.volume.shape)
    
    # We will trace the shapes inside process_scan manually to see what happens
    print("\n--- TRACING PIPELINE ---")
    
    from backend.oct_analyzer.mvp_pipeline import validate_volume, crop_black_padding, center_crop_volume
    
    print("1. crop_black_padding")
    cropped, _ = crop_black_padding(scan.volume)
    print("   Cropped shape:", cropped.shape)
    
    print("2. center_crop_volume")
    foveal_crop, _ = center_crop_volume(cropped, scan.spacing_mm)
    print("   Foveal crop shape:", foveal_crop.shape)
    
    from backend.oct_analyzer.pre_processing import get_preprocessing_pipeline
    pipeline = get_preprocessing_pipeline()
    tensor = pipeline(foveal_crop)
    print("3. Preprocessing")
    print("   Tensor shape:", tensor.shape)
    
    from backend.oct_analyzer.anatomical_flattener import flatten_volume_to_rpe
    if tensor.shape[1] > 1:
        print("4. Flattening (should skip for 2D, but let's see)")
        flattened = flatten_volume_to_rpe(tensor)
        flattened_volume = flattened.detach().cpu().numpy()[0]
    else:
        print("4. Skipping flattening")
        flattened_volume = tensor.detach().cpu().numpy()[0]
    print("   Flattened volume shape:", flattened_volume.shape)
    
    from backend.oct_analyzer.segmentation import segment_retinal_layers
    print("5. Segmentation")
    print("   Spacing mm:", scan.spacing_mm)
    segmentation_result, pipeline_results = segment_retinal_layers(flattened_volume, scan.spacing_mm)
    print("   Segmentation mask shape:", segmentation_result.labels.shape)
    print("   Pipeline best_slice_idx:", pipeline_results.get("best_slice_idx"))

except Exception as e:
    traceback.print_exc()
