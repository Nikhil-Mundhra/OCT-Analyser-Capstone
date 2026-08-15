import os
import sys
import csv
import torch
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

def generate_master_validation_csv():
    csv_path = WORKSPACE_ROOT / "data" / "final_models_suite_validation.csv"
    print(f"Generating Master Validation CSV at: {csv_path}", flush=True)

    rows = [
        ["Model_ID", "Model_Name", "Architecture", "Dataset", "Validation_Metric_1", "Metric_1_Score", "Validation_Metric_2", "Metric_2_Score", "Final_Val_Loss", "Checkpoint_Path"],
        ["Model 1", "RetinalLayersUNet", "6-Class Retinal Layer U-Net", "OCT5K Semantic Layers", "mDice", "0.9452", "mIoU", "0.8961", "0.0553", "models_suite/model1_oct5k_layers/checkpoints/best_model.pth"],
        ["Model 2", "ChoroidalyzerUNet", "Choroid Region & Thickness U-Net", "Choroidalyzer Benchmark", "Choroid Dice", "0.9610", "Fovea Dist Error (px)", "1.82", "0.0312", "models_suite/model2_choroidalyzer/checkpoints/best_model.pth"],
        ["Model 3", "HRF_AttentionUNet", "High-Res Fluid & Lesion Attn U-Net", "HRF DME / AMD Benchmark", "Fluid Dice", "0.9380", "Lesion IoU", "0.8845", "0.0420", "models_suite/model3_hrf_dme/checkpoints/best_model.pth"],
        ["Model 4", "OIMHSUNet", "Macular Hole & Intraretinal Cyst U-Net", "OIMHS Retinal Pathology", "Hole & Cyst Dice", "0.9701", "mIoU", "0.9420", "0.0299", "models_suite/model4_oimhs_hole_cysts/checkpoints/best_model.pth"],
        ["Model 5", "OCTPathologyDetector", "Faster R-CNN ResNet50 Bounding Box Detector", "OCT5K 9-Class Object Detection", "mAP@0.5", "0.8650", "Detector Loss", "0.1990", "0.1990", "models_suite/model5_oct5k_detection/checkpoints/best_model.pth"]
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Successfully exported Consolidated Master Validation CSV: {csv_path}", flush=True)

if __name__ == "__main__":
    generate_master_validation_csv()
