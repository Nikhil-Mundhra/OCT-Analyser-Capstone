import base64
import io
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, Depends, Form, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from sqlalchemy.orm import Session

from .data_loader import load_normalized_scan
from .interfaces import ScanResult
from .mvp_pipeline import process_scan
from .preview import preview_path
from .classifier_integration import get_classifier
from .database import Base, engine, get_db
from .models import ScanRecord
from .tasks import process_scan_task, process_scan_task_direct
from .auth import get_api_key
from .audit_log import log_scan_accessed, log_scan_created
from .report_generator import generate_pdf_report

from .constants import (
    CORS_ORIGINS,
    CORS_ORIGIN_REGEX,
    PREVIEW_DIR,
    RUNTIME_DIR,
    SEGMENT_PREDICT_SCRIPT,
    SUPPORTED_SUFFIXES,
    UNET_CHECKPOINT_PATH,
    UPLOAD_DIR,
)

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Local OCT Analyzer MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
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

    # Dispatch Celery task with graceful background thread fallback if broker is unreachable
    try:
        task = process_scan_task.delay(scan_id, str(upload_path))
        scan_record.task_id = task.id
    except Exception as exc:
        print(f"[API] Celery broker dispatch failed ({exc}). Running task in background thread.")
        t = threading.Thread(target=process_scan_task_direct, args=(scan_id, str(upload_path)), daemon=True)
        t.start()
        scan_record.task_id = f"sync_{scan_id}"

    db.commit()

    return {
        "scan_id": scan_record.id,
        "task_id": scan_record.task_id,
        "status": scan_record.status,
        "filename": scan_record.filename
    }

@app.post("/api/segment_2d")
async def segment_2d(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
        
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        from hf_space.app import predict_model1
        res, info = predict_model1(image)
        if isinstance(res, tuple):
            blended, mask = res[0], res[1]
            ov = _pil_to_base64(blended) if blended else None
            mk = _pil_to_base64(mask) if mask else None
        elif res:
            ov, mk = _pil_to_base64(res), None
        else:
            ov, mk = None, None

        return {
            "status": "success",
            "overlay": ov,
            "mask": mk,
            "info": info
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}")

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
    filename = file.filename or "scan.png"
    suffix = Path(filename).suffix.lower() or ".png"
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'")

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

    def _extract_images(res):
        if isinstance(res, tuple):
            blended, mask = res[0], res[1]
            return _pil_to_base64(blended) if blended else None, _pil_to_base64(mask) if mask else None
        elif res:
            return _pil_to_base64(res), None
        return None, None

    model_registry = [
        ("model1", "Retinal Layers U-Net", lambda: predict_model1(image)),
        ("model2", "Choroidalyzer U-Net", lambda: predict_model2(image)),
        ("model3", "HRF Attention U-Net", lambda: predict_model3(image)),
        ("model4", "OIMHS Hole & Cyst U-Net", lambda: predict_model4(image)),
        ("model5", "OCT Pathology Detector", lambda: predict_model5(image, score_threshold=score_threshold)),
    ]

    for key, name, predict_fn in model_registry:
        if model_id in [key, "all"]:
            res, metrics = predict_fn()
            ov, mk = _extract_images(res)
            results[key] = {
                "name": name,
                "overlay": ov,
                "mask": mk,
                "details": metrics,
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
