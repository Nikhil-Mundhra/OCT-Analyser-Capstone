import os
import csv
from pathlib import Path

# Mapping Dataset folders to Model Labels
H2_DIR_MAP = {
    "Macular Degeneration Spectrum": "Macular_Degeneration",
    "Diabetic Complications": "Diabetic_Complications",
    "Vascular Occlusions (Blockages)": "Vascular_Occlusions",
    "Fluid Accumulation": "Fluid_Accumulation",
    "Vitreomacular and Structural Disorders": "Structural_Issues"
}

H3_DIR_MAP = {
    "Choroidal Neovascularization": "CNV",
    "DRUSEN": "DRUSEN",
    "AMD": "Generic_AMD",
    "General": "Generic_AMD", # For Macular General
    "Diabetic Macular Edema (DME)": "DME",
    "Central Serous Retinopathy": "CSR",
    "RVO": "RVO",
    "RAO": "RAO",
    "ERM": "ERM",
    "VID": "VID"
}

def generate_manifest(root_dir, output_csv):
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"CRITICAL: Dataset root '{root_dir}' does not exist! Please check your Kaggle input paths.")
    
    rows = []
    
    for dirpath, _, filenames in os.walk(root_path):
        rel_path = Path(dirpath).relative_to(root_path)
        parts = rel_path.parts
        
        if not parts:
            continue
            
        l1_folder = parts[0]
        
        # Head 1 logic
        head1_label = 0 if l1_folder == "Normal (Healthy)" else 1
        
        head2_label = ""
        head3_labels = ""
        
        if head1_label == 1:
            head2_label = H2_DIR_MAP.get(l1_folder, "")
            
            if len(parts) >= 2:
                l2_folder = parts[1]
                
                # Special fix for Macular Hole misplaced in Vascular
                if l2_folder == "Macular-Hole-Retinal-OCT-images":
                    head2_label = "Structural_Issues"
                    head3_labels = "MH"
                elif l1_folder == "Diabetic Complications" and l2_folder == "General":
                    head3_labels = "DR"
                elif l1_folder == "Macular Degeneration Spectrum" and l2_folder == "General":
                    head3_labels = "Generic_AMD"
                else:
                    head3_labels = H3_DIR_MAP.get(l2_folder, "")
                    
        for f in filenames:
            ext = Path(f).suffix.lower()
            if ext in valid_exts:
                img_path = str(Path(dirpath) / f)
                rows.append({
                    "image_path": img_path,
                    "head1_label": head1_label,
                    "head2_label": head2_label,
                    "head3_labels": head3_labels
                })
                
    if len(rows) == 0:
        raise ValueError(f"CRITICAL: Found 0 images in {root_dir}! Please check that this path contains your .jpg/.png images.")
                
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'head1_label', 'head2_label', 'head3_labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            
    print(f"Manifest generated at {output_csv} with {len(rows)} images.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified")
    parser.add_argument('--output_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_manifest.csv"))
    args = parser.parse_args()
    
    generate_manifest(args.dataset_root, args.output_path)
