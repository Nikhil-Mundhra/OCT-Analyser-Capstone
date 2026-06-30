import os
import shutil
import numpy as np
import imageio.v2 as imageio
from scipy import stats
from pathlib import Path
from tqdm import tqdm

def extract_segmentation_data(
    base_dir="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Bad data/OCT5k",
    output_dir="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Bad data/OCT5k_Segmentation_Subset"
):
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    # Input directories
    images_dir = base_path / "Images" / "Images_Manual"
    masks_g1 = base_path / "Masks" / "Masks_Manual" / "Grading_1"
    masks_g2 = base_path / "Masks" / "Masks_Manual" / "Grading_2"
    masks_g3 = base_path / "Masks" / "Masks_Manual" / "Grading_3"
    
    # Output directories
    out_images = output_path / "Images"
    out_masks = output_path / "Masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    
    # Gather all mask relative paths from Grading_1
    all_mask_paths = list(masks_g1.rglob("*.png"))
    
    print(f"Found {len(all_mask_paths)} segmentation scans to process.")
    
    success_count = 0
    missing_count = 0
    
    for mask_path in tqdm(all_mask_paths, desc="Processing masks"):
        rel_path = mask_path.relative_to(masks_g1)
        
        m1_path = masks_g1 / rel_path
        m2_path = masks_g2 / rel_path
        m3_path = masks_g3 / rel_path
        img_path = images_dir / rel_path
        
        # Verify all files exist
        if not (m1_path.exists() and m2_path.exists() and m3_path.exists() and img_path.exists()):
            print(f"Missing one of the files for {rel_path}. Skipping.")
            missing_count += 1
            continue
            
        # Create safe flattened filename
        safe_filename = str(rel_path).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        
        # 1. Copy image
        shutil.copy2(img_path, out_images / safe_filename)
        
        # 2. Compute majority vote mask
        # Masks are grayscale PNGs where pixel values are class IDs (0-14)
        m1 = imageio.imread(m1_path)
        m2 = imageio.imread(m2_path)
        m3 = imageio.imread(m3_path)
        
        stacked = np.stack([m1, m2, m3], axis=0)
        # scipy.stats.mode returns (mode_array, count_array)
        # If all 3 are different (count=1), scipy keeps the smallest value. 
        mode_result = stats.mode(stacked, axis=0, keepdims=False)
        voted_mask = mode_result.mode # Extract the mode array
        voted_mask = np.squeeze(voted_mask) # Ensure 2D (H, W)
        
        # Ensure it is uint8 to save as PNG (classes are 0-14, so it fits perfectly in grayscale PNG)
        voted_mask = voted_mask.astype(np.uint8)
        imageio.imwrite(out_masks / safe_filename, voted_mask)
        
        success_count += 1

    print(f"Extraction complete! Successfully processed {success_count} image-mask pairs.")
    print(f"Missing pairs: {missing_count}")

if __name__ == "__main__":
    extract_segmentation_data()
