from pathlib import Path
from celery import shared_task
from sqlalchemy.orm import Session

from .celery_app import celery_app
from .database import SessionLocal
from .models import ScanRecord
from .data_loader import load_normalized_scan
from .mvp_pipeline import process_scan

import tempfile

RUNTIME_DIR = Path(tempfile.gettempdir()) / "runtime_uploads"
PREVIEW_DIR = RUNTIME_DIR / "previews"

@celery_app.task(bind=True)
def process_scan_task(self, scan_id: str, upload_path_str: str):
    db: Session = SessionLocal()
    scan_record = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    
    if not scan_record:
        db.close()
        return {"error": "Scan record not found"}

    try:
        scan_record.status = "processing"
        scan_record.task_id = self.request.id
        db.commit()

        upload_path = Path(upload_path_str)
        
        def update_progress(msg: str):
            scan_record.detail = msg
            db.commit()

        update_progress("Loading scan from disk...")
        scan = load_normalized_scan(upload_path)
        
        # Make sure preview directory exists
        preview_dir = PREVIEW_DIR / scan_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        result = process_scan(scan, preview_dir=preview_dir, progress_cb=update_progress)

        # Prefix preview URLs exactly as API did
        _prefix_preview_urls(scan_id, result)

        scan_record.status = "completed"
        scan_record.result = result
        db.commit()
        return result

    except Exception as exc:
        scan_record.status = "failed"
        scan_record.detail = str(exc)
        db.commit()
        raise exc
    finally:
        db.close()

def _prefix_preview_urls(scan_id: str, result: dict) -> None:
    result["previews"] = {
        key: f"/api/scans/{scan_id}/{url}" if isinstance(url, str) else [f"/api/scans/{scan_id}/{u}" for u in url]
        for key, url in result.get("previews", {}).items()
    }
