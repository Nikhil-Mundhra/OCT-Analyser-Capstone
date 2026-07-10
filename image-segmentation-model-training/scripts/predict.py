import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.unet import HierarchicalUNet
from src.inference.analyzer import SegmentationAnalyzer
from src.inference.export import InferenceExporter

def predict(image_path: str, checkpoint_path: str, output_json: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the Model
    print(f"Loading model from {checkpoint_path}...")
    model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with untrained weights for demonstration.")
        
    model.to(device)
    model.eval()
    
    # 2. Load and Preprocess the Image
    print(f"Processing image {image_path}...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    original_h, original_w = img.shape
    
    # Resize to model's expected input (e.g., 512x512)
    img_resized = cv2.resize(img, (512, 512))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor: (1, 1, 512, 512)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Model Inference
    print("Running forward pass...")
    with torch.no_grad():
        coarse_logits, granular_logits = model(img_tensor)
        
    # Get the predicted class for each pixel from the granular head
    granular_preds = torch.argmax(granular_logits, dim=1).squeeze(0).cpu().numpy()
    
    # 4. Object-Oriented Post-Processing
    print("Extracting vector instances and metrics...")
    analyzer = SegmentationAnalyzer()
    analysis = analyzer.analyze(granular_preds)
    
    # Optional: If you need to map coordinates back to original image size (e.g. 1024x1024)
    # the frontend can do this scaling as documented, or you could do it here before export.
    
    # 5. Export
    print(f"Exporting results to {output_json}...")
    InferenceExporter.to_json_file(analysis, output_json)
    
    # 6. Visualization
    if hasattr(args, 'output_image') and args.output_image:
        print(f"Drawing segmentation mask and saving to {args.output_image}...")
        # Convert resized grayscale to BGR for drawing colors
        img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        # Color palette for different classes
        colors = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 128, 0), (255, 0, 128), (0, 255, 128)
        ]
        
        # analysis is an OCTAnalysisResult, containing layers
        for layer in analysis.layers:
            pts = np.array([[pt.x, pt.y] for pt in layer.boundary_points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Select color based on class_id
            color = colors[layer.class_id % len(colors)]
            
            cv2.polylines(img_color, [pts], isClosed=False, color=color, thickness=2)
            
        cv2.imwrite(args.output_image, img_color)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCT Segmentation Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input OCT scan (PNG/TIF)")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/unet_hierarchical_epoch_10.pth", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="output_analysis.json", help="Path to save the JSON output")
    parser.add_argument("--output-image", type=str, default=None, help="Path to save the visually annotated image (PNG)")
    
    args = parser.parse_args()
    predict(args.image, args.checkpoint, args.output)
