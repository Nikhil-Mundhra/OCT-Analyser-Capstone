from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, Depends, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

import base64
import io
from PIL import Image

from .data_loader import load_normalized_scan
from .interfaces import ScanResult
from .mvp_pipeline import process_scan
from .preview import preview_path
from .classifier_integration import get_classifier
from .database import Base, engine, get_db
from .models import ScanRecord
from .tasks import process_scan_task
from .auth import get_api_key
from .audit_log import log_scan_accessed, log_scan_created
from .report_generator import generate_pdf_report
from fastapi.responses import Response

import tempfile
import subprocess
import json
import os

# Initialize database tables
Base.metadata.create_all(bind=engine)

RUNTIME_DIR = Path(tempfile.gettempdir()) / "runtime_uploads"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
PREVIEW_DIR = RUNTIME_DIR / "previews"
SUPPORTED_SUFFIXES = {".vol", ".dcm", ".zip", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

app = FastAPI(title="Local OCT Analyzer MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Local OCT Analyzer MVP API",
        "docs": "/docs",
        "frontend": "Start with the frontend URL printed by make run.",
    }


@app.post("/api/scans")
def create_scan(file: UploadFile, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a .vol, .dcm, .zip OCT export, or a 2D image (.png, .jpg, .tif, .tiff)")

    scan_id = uuid4().hex
    upload_path = UPLOAD_DIR / scan_id / f"scan{suffix}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as handle:
        copyfileobj(file.file, handle)

    scan_record = ScanRecord(
        id=scan_id,
        filename=file.filename,
        status="pending",
        is_demo_model=False
    )
    db.add(scan_record)
    db.commit()
    db.refresh(scan_record)
    
    log_scan_created(scan_id, user_id=api_key or "anonymous")

    # Dispatch Celery task
    task = process_scan_task.delay(scan_id, str(upload_path))
    
    scan_record.task_id = task.id
    db.commit()

    return {
        "scan_id": scan_record.id,
        "task_id": scan_record.task_id,
        "status": scan_record.status,
        "filename": scan_record.filename
    }

@app.post("/api/segment_2d")
def segment_2d(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
        
    scan_id = uuid4().hex
    temp_dir = Path(tempfile.gettempdir()) / "oct_segmentation"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = temp_dir / f"{scan_id}{suffix}"
    out_json = temp_dir / f"{scan_id}_out.json"
    
    with img_path.open("wb") as handle:
        copyfileobj(file.file, handle)
        
    script_path = Path(__file__).resolve().parent.parent.parent / "image-segmentation-model-training" / "scripts" / "predict.py"
    checkpoint_path = Path(__file__).resolve().parent.parent.parent / "image-segmentation-model-training" / "checkpoints" / "unet_hierarchical_best.pth"
    
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    try:
        subprocess.run(
            [
                "python3", str(script_path),
                "--image", str(img_path),
                "--checkpoint", str(checkpoint_path),
                "--output", str(out_json)
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        
        with open(out_json, "r") as f:
            result = json.load(f)
            
        return result
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e.stderr}")
    finally:
        if img_path.exists():
            img_path.unlink()
        if out_json.exists():
            out_json.unlink()

def _pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

@app.post("/api/segment_suite")
async def run_segmentation_suite(
    file: UploadFile = File(...),
    model_id: str = Form("all"),
    score_threshold: float = Form(0.5)
) -> dict:
    """
    Executes models from the 5-Model Segmentation & Detection Suite (models_suite/).
    Supports single model selection ("model1".."model5") or full suite ("all").
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        from hf_space.app import (
            predict_model1,
            predict_model2,
            predict_model3,
            predict_model4,
            predict_model5
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load Segmentation 5-Model Suite: {exc}")

    results = {}

    if model_id in ["model1", "all"]:
        overlay, metrics = predict_model1(image)
        results["model1"] = {
            "name": "Retinal Layers U-Net",
            "overlay": _pil_to_base64(overlay) if overlay else None,
            "details": metrics
        }

    if model_id in ["model2", "all"]:
        overlay, metrics = predict_model2(image)
        results["model2"] = {
            "name": "Choroidalyzer U-Net",
            "overlay": _pil_to_base64(overlay) if overlay else None,
            "details": metrics
        }

    if model_id in ["model3", "all"]:
        overlay, metrics = predict_model3(image)
        results["model3"] = {
            "name": "HRF Attention U-Net",
            "overlay": _pil_to_base64(overlay) if overlay else None,
            "details": metrics
        }

    if model_id in ["model4", "all"]:
        overlay, metrics = predict_model4(image)
        results["model4"] = {
            "name": "OIMHS Hole & Cyst U-Net",
            "overlay": _pil_to_base64(overlay) if overlay else None,
            "details": metrics
        }

    if model_id in ["model5", "all"]:
        overlay, metrics = predict_model5(image, score_threshold=score_threshold)
        results["model5"] = {
            "name": "OCT Pathology Detector",
            "overlay": _pil_to_base64(overlay) if overlay else None,
            "details": metrics
        }

    return {"status": "success", "results": results}

@app.post("/predict")
async def predict_image(file: UploadFile) -> dict:
    """
    Classification endpoint that mirrors the Hugging Face Space /predict contract.

    Returns the pipeline result dict directly (Level1, Level2, Level3,
    Final_Diagnosis, Path, gradcams) so local and remote endpoints are
    shape-identical from the frontend's perspective.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    scan_id = uuid4().hex
    temp_dir = Path(tempfile.gettempdir()) / "oct_classification"
    temp_dir.mkdir(parents=True, exist_ok=True)

    img_path = temp_dir / f"{scan_id}{suffix}"
    with img_path.open("wb") as handle:
        copyfileobj(file.file, handle)

    try:
        classifier = get_classifier()
        results = classifier.predict(str(img_path), gradcam=True)
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        # Return the pipeline dict as-is so the response shape matches what the
        # HF Space returns: {Level1, Level2, Level3, Final_Diagnosis, Path, gradcams}
        return results
    finally:
        if img_path.exists():
            img_path.unlink()


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: str, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)) -> dict:
    scan = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
        
    log_scan_accessed(scan_id, user_id=api_key or "anonymous")
        
    response = {
        "scan_id": scan.id,
        "status": scan.status,
        "filename": scan.filename,
        "is_demo_model": scan.is_demo_model
    }
    
    if scan.task_id:
        response["task_id"] = scan.task_id
        
    if scan.detail:
        response["detail"] = scan.detail
        
    if scan.result:
        # Merge result fields if completed
        response.update(scan.result)
        
    return response


@app.get("/api/scans/{scan_id}/preview/{kind:path}")
def get_preview(scan_id: str, kind: str, db: Session = Depends(get_db)) -> FileResponse:
    scan = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")

    try:
        path = preview_path(PREVIEW_DIR / scan_id, kind)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path, media_type="image/png")

@app.get("/api/scans/{scan_id}/report")
def get_scan_report(scan_id: str, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)) -> Response:
    scan = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    if scan is None or not scan.result:
        raise HTTPException(status_code=404, detail="Scan report not available")
        
    log_scan_accessed(scan_id, user_id=api_key or "anonymous")
    pdf_bytes = generate_pdf_report(scan_id, scan.result)
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=OCT_Report_{scan_id}.pdf"}
    )
