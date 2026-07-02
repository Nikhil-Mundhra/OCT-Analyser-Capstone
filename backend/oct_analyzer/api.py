from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .data_loader import load_normalized_scan
from .interfaces import ScanResult
from .mvp_pipeline import process_scan
from .preview import preview_path


import tempfile
import subprocess
import json
import os
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

SCAN_STORE: dict[str, dict] = {}


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Local OCT Analyzer MVP API",
        "docs": "/docs",
        "frontend": "Start with the frontend URL printed by make run.",
    }


@app.post("/api/scans", response_model=ScanResult)
def create_scan(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a .vol, .dcm, .zip OCT export, or a 2D image (.png, .jpg, .tif, .tiff)")

    scan_id = uuid4().hex
    upload_path = UPLOAD_DIR / scan_id / f"scan{suffix}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as handle:
        copyfileobj(file.file, handle)

    SCAN_STORE[scan_id] = {
        "scan_id": scan_id,
        "status": "processing",
        "filename": file.filename,
        "is_demo_model": True,
    }

    try:
        scan = load_normalized_scan(upload_path)
        result = process_scan(scan, preview_dir=PREVIEW_DIR / scan_id)
        _prefix_preview_urls(scan_id, result)
        SCAN_STORE[scan_id] = {
            "scan_id": scan_id,
            "filename": file.filename,
            **result,
        }
    except Exception as exc:
        SCAN_STORE[scan_id] = {
            "scan_id": scan_id,
            "filename": file.filename,
            "status": "failed",
            "detail": str(exc),
            "is_demo_model": True,
        }
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SCAN_STORE[scan_id]


@app.post("/api/segment_2d")
def segment_2d(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
        
    scan_id = uuid4().hex
    # Save the file temporarily
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
        # Cleanup
        if img_path.exists():
            img_path.unlink()
        if out_json.exists():
            out_json.unlink()


@app.get("/api/scans/{scan_id}", response_model=ScanResult)
def get_scan(scan_id: str) -> dict:
    scan = SCAN_STORE.get(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@app.get("/api/scans/{scan_id}/preview/{kind:path}")
def get_preview(scan_id: str, kind: str) -> FileResponse:
    if scan_id not in SCAN_STORE:
        raise HTTPException(status_code=404, detail="Scan not found")

    try:
        path = preview_path(PREVIEW_DIR / scan_id, kind)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path, media_type="image/png")


def _prefix_preview_urls(scan_id: str, result: dict) -> None:
    result["previews"] = {
        key: f"/api/scans/{scan_id}/{url}" if isinstance(url, str) else [f"/api/scans/{scan_id}/{u}" for u in url]
        for key, url in result.get("previews", {}).items()
    }
    if result.get("ipnv2", {}).get("previews"):
        result["ipnv2"]["previews"] = {
            key: f"/api/scans/{scan_id}/{url}"
            for key, url in result["ipnv2"]["previews"].items()
        }
