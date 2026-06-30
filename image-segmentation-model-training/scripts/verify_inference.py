import sys
from pathlib import Path
import numpy as np
import cv2

# Ensure we can import from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.analyzer import SegmentationAnalyzer
from src.inference.export import InferenceExporter

def verify_inference():
    print("Generating a dummy segmented mask...")
    # Create a 512x512 mask
    mask = np.zeros((512, 512), dtype=np.uint8)
    
    # 1. Add a continuous layer (e.g. ILM, class 1)
    # A curved line across the image
    for x in range(512):
        y = int(200 + 20 * np.sin(x / 50.0))
        cv2.circle(mask, (x, y), 3, 1, -1)
        
    # 2. Add another continuous layer (e.g. RPE, class 8)
    for x in range(512):
        y = int(300 + 10 * np.cos(x / 60.0))
        cv2.circle(mask, (x, y), 3, 8, -1)
        
    # 3. Add a lesion instance (e.g. Fluid, class 9)
    # Draw two separate fluid pockets
    cv2.circle(mask, (150, 250), 30, 9, -1)
    cv2.rectangle(mask, (350, 240), (400, 280), 9, -1)
    
    # Analyze
    print("Running SegmentationAnalyzer...")
    analyzer = SegmentationAnalyzer()
    analysis = analyzer.analyze(mask)
    
    # Verify outputs
    print(f"Detected {len(analysis.layers)} layers.")
    print(f"Detected {len(analysis.lesions)} lesions.")
    print(f"Total fluid area: {analysis.clinical_metrics.total_fluid_area}")
    print(f"Avg Retinal Thickness: {analysis.clinical_metrics.average_retinal_thickness}")
    
    assert len(analysis.layers) == 2, "Should detect exactly 2 layers."
    assert len(analysis.lesions) == 2, "Should detect exactly 2 fluid pockets."
    
    # Export
    out_file = "inference_test_output.json"
    print(f"Exporting to {out_file}...")
    InferenceExporter.to_json_file(analysis, out_file)
    
    print("Verification Passed successfully!")

if __name__ == "__main__":
    verify_inference()
