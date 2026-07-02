import torch
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json

from models.unet import HierarchicalUNet
from src.inference.analyzer import SegmentationAnalyzer

app = FastAPI(title="OCT Segmentation API")

# Setup CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = Path(__file__).parent / "unet_hierarchical_best.pth"
model = None

@app.on_event("startup")
def load_model():
    global model
    if checkpoint_path.exists():
        print(f"Loading model from {checkpoint_path}...")
        model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found.")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    suffix = Path(file.filename or "").suffix.lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        img_resized = cv2.resize(img, (512, 512))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            coarse_logits, granular_logits = model(img_tensor)
            
        granular_preds = torch.argmax(granular_logits, dim=1).squeeze(0).cpu().numpy()
        
        analyzer = SegmentationAnalyzer()
        analysis = analyzer.analyze(granular_preds)
        
        # Convert to dictionary matching the JSON structure
        return json.loads(analysis.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
