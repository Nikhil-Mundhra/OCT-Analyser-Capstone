import os
import pandas as pd
import shutil
from pathlib import Path

def extract_detection_data(
    base_dir="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Bad data/OCT5k",
    output_dir="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Bad data/OCT5k_Detection_Subset",
    target_classes=None
):
    """
    Extracts the 566 scans and their bounding boxes from the OCT5k dataset.
    
    Args:
        base_dir: Path to the root of the extracted OCT5k dataset.
        output_dir: Where to save the filtered images and annotations.
        target_classes: A list of class names to keep. If None, keeps all classes.
                        (e.g. ['Fluid', 'Harddrusen', 'Softdrusen', 'Geographicatrophy', 'Reticulardrusen', 'Choroidalfolds'])
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    csv_path = base_path / "Detection" / "all_bounding_boxes.csv"
    images_src_dir = base_path / "Detection" / "Images"
    
    if not csv_path.exists():
        print(f"Error: Could not find annotations at {csv_path}")
        return
        
    print(f"Loading annotations from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # The CSV structure is: idx, image, xmin, ymin, xmax, ymax, class
    # Example image path: 'AMD Part1/AMD (3)/Image (14).png'
    
    # 1. Filter by target classes if specified
    if target_classes:
        print(f"Filtering down to {len(target_classes)} specific classes...")
        # Ensure exact string matching (removing any accidental trailing spaces)
        df['class'] = df['class'].str.strip()
        df = df[df['class'].isin(target_classes)]
        
    # 2. Find unique images
    unique_images = df['image'].unique()
    print(f"Found {len(unique_images)} unique scans containing the targeted annotations.")
    
    # 3. Create output directories
    out_images_dir = output_path / "Images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Copy the images over
    print(f"Copying images to {out_images_dir}...")
    copied_count = 0
    missing_count = 0
    
    for img_rel_path in unique_images:
        src_img = images_src_dir / img_rel_path
        
        # We will flatten the directory structure for easier use in training pipelines (e.g. YOLO/FasterRCNN)
        # Convert "AMD Part1/AMD (3)/Image (14).png" -> "AMD_Part1_AMD_3_Image_14.png"
        safe_filename = str(img_rel_path).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        dst_img = out_images_dir / safe_filename
        
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            # Update the dataframe with the new simplified image path
            df.loc[df['image'] == img_rel_path, 'new_image_path'] = safe_filename
            copied_count += 1
        else:
            missing_count += 1
            print(f"Warning: Image not found {src_img}")
            
    print(f"Successfully copied {copied_count} images. ({missing_count} missing)")
    
    # 5. Save the filtered and path-updated CSV
    output_csv = output_path / "filtered_annotations.csv"
    
    # Clean up the dataframe for the output
    if 'new_image_path' in df.columns:
        df['image'] = df['new_image_path']
        df = df.drop(columns=['new_image_path', df.columns[0]]) # Drop the original index column if it exists
        
    df.to_csv(output_csv, index=False)
    print(f"Saved filtered annotations to {output_csv}")
    print("\n--- Summary ---")
    print("Class distribution in the extracted subset:")
    print(df['class'].value_counts())

if __name__ == "__main__":
    # You mentioned 6 classes. The paper has 9. 
    # If you only want 6 specific classes, uncomment the list below and edit it to match the 6 you want.
    
    desired_6_classes = [
        'Fluid', 
        'Harddrusen', 
        'Softdrusen', 
        'Geographicatrophy', 
        'PRlayerdisruption', 
        'Choroidalfolds'
    ]
    
    # To extract all 9 classes (all 566 scans), set target_classes=None
    extract_detection_data(target_classes=None) 
    
    # To extract specifically 6 classes, change to:
    # extract_detection_data(target_classes=desired_6_classes)
