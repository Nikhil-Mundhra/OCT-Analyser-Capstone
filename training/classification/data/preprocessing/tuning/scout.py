"""
training/classification/data/preprocessing/tuning/scout.py

Backend Scout Service for Classified-unet-masked dataset.
Handles manifest parsing, folder tree indexing, search and anomaly filtering,
real vs mask file resolution, and dynamic diagnostic health checks.
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import cv2
import numpy as np
import scipy.ndimage as ndi

# Ensure root paths are accessible
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "training" / "classification") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "classification"))

from data.preprocessing.mask_diagnostics import evaluate_mask_health, MaskHealthReport

_DEFAULT_UNET_MASKED = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-unet-masked")
UNET_MASKED_DIR = Path(os.environ.get("UNET_MASKED_DIR", str(_DEFAULT_UNET_MASKED if _DEFAULT_UNET_MASKED.exists() else PROJECT_ROOT / "data" / "Classified-unet-masked")))

# In-memory index cache
_SCOUT_CACHE: Dict[str, Any] = {
    "dir": None,
    "manifest_mtime": 0.0,
    "manifest_data": None,
    "summary_data": None,
    "file_index": {},  # rel_path -> metadata dict
    "tree": [],
    "last_scan": 0.0,
}


def get_unet_masked_dir() -> Path:
    """Resolves active Classified-unet-masked root path, respecting environment or runtime overrides."""
    this_mod = sys.modules.get("data.preprocessing.tuning.scout")
    if this_mod is not None:
        val = getattr(this_mod, "UNET_MASKED_DIR", None)
        if val is not None:
            return Path(val)
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server"):
        if mod_name in sys.modules:
            val = getattr(sys.modules[mod_name], "UNET_MASKED_DIR", None)
            if val is not None:
                return Path(val)
    return UNET_MASKED_DIR


def set_unet_masked_dir(path: Path | str) -> Path:
    """Explicitly sets active UNET_MASKED_DIR and invalidates cache."""
    global UNET_MASKED_DIR
    p = Path(path).resolve()
    UNET_MASKED_DIR = p
    invalidate_scout_cache()
    return p


def invalidate_scout_cache():
    """Invalidates the scout memory cache."""
    _SCOUT_CACHE["dir"] = None
    _SCOUT_CACHE["manifest_mtime"] = 0.0
    _SCOUT_CACHE["manifest_data"] = None
    _SCOUT_CACHE["summary_data"] = None
    _SCOUT_CACHE["file_index"] = {}
    _SCOUT_CACHE["tree"] = []
    _SCOUT_CACHE["last_scan"] = 0.0


def _load_manifest_and_index(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Loads manifest.json and indexes all available images and masks."""
    target_dir = Path(base_dir) if base_dir else get_unet_masked_dir()
    manifest_path = target_dir / "manifest.json"
    summary_path = target_dir / "diagnostics_summary.json"
    images_dir = target_dir / "Images"
    masks_dir = target_dir / "Masks"
    quarantine_dir = target_dir / "quarantine_review"

    mtime = manifest_path.stat().st_mtime if manifest_path.exists() else 0.0

    # Return cached index if valid
    if (
        _SCOUT_CACHE["dir"] == target_dir
        and _SCOUT_CACHE["manifest_mtime"] == mtime
        and _SCOUT_CACHE["manifest_data"] is not None
        and (time.time() - _SCOUT_CACHE["last_scan"]) < 30.0
    ):
        return _SCOUT_CACHE

    manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    summary = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    manifest_samples = manifest.get("samples", {})
    file_index: Dict[str, Dict[str, Any]] = {}

    # Build O(1) quarantine lookup map upfront
    quarantine_by_stem: Dict[str, str] = {}
    if quarantine_dir.exists():
        try:
            for qf in quarantine_dir.iterdir():
                if qf.is_file():
                    q_rel = str(qf.relative_to(target_dir))
                    q_stem = qf.stem
                    quarantine_by_stem[q_stem] = q_rel
                    # Index by sub-tokens for fast match
                    for token in q_stem.split("_"):
                        if len(token) >= 5:
                            quarantine_by_stem[token] = q_rel
        except Exception:
            pass

    # 1. Populate from manifest entries
    for key, item in manifest_samples.items():
        img_rel = item.get("image_path", f"Images/{key}")
        if img_rel.startswith("Images/"):
            clean_rel = img_rel[len("Images/"):]
        else:
            clean_rel = key

        severity = item.get("severity", "CLEAN").upper()
        is_suspicious = item.get("is_suspicious", severity in ("WARNING", "CRITICAL"))
        anomaly_flags = item.get("anomaly_flags", [])

        # O(1) quarantine lookup
        stem = Path(clean_rel).stem
        quarantine_rel = quarantine_by_stem.get(stem) if is_suspicious else None
        has_quarantine = quarantine_rel is not None

        file_index[clean_rel] = {
            "rel_path": clean_rel,
            "filename": Path(clean_rel).name,
            "folder": str(Path(clean_rel).parent) if str(Path(clean_rel).parent) != "." else "",
            "severity": severity,
            "is_suspicious": is_suspicious,
            "anomaly_flags": anomaly_flags,
            "metrics": item.get("metrics", {}),
            "meta": item.get("meta", {}),
            "has_mask": True,
            "has_quarantine": has_quarantine,
            "quarantine_rel": quarantine_rel,
            "source": "manifest",
        }

    # 2. Scan disk only if manifest was missing or empty
    if not manifest_samples and images_dir.exists():
        for img_path in images_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            try:
                rel = str(img_path.relative_to(images_dir))
            except ValueError:
                continue

            if rel not in file_index:
                mask_file = masks_dir / rel
                has_mask = mask_file.exists()
                file_index[rel] = {
                    "rel_path": rel,
                    "filename": img_path.name,
                    "folder": str(Path(rel).parent) if str(Path(rel).parent) != "." else "",
                    "severity": "UNINDEXED",
                    "is_suspicious": False,
                    "anomaly_flags": [],
                    "metrics": {},
                    "meta": {},
                    "has_mask": has_mask,
                    "has_quarantine": False,
                    "quarantine_rel": None,
                    "source": "filesystem",
                }

    # 3. Build hierarchical folder tree
    folder_tree = _build_folder_tree(file_index)

    _SCOUT_CACHE["dir"] = target_dir
    _SCOUT_CACHE["manifest_mtime"] = mtime
    _SCOUT_CACHE["manifest_data"] = manifest
    _SCOUT_CACHE["summary_data"] = summary
    _SCOUT_CACHE["file_index"] = file_index
    _SCOUT_CACHE["tree"] = folder_tree
    _SCOUT_CACHE["last_scan"] = time.time()

    return _SCOUT_CACHE


def _build_folder_tree(file_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Builds folder hierarchy with counts and suspicious counts."""
    folders_map: Dict[str, Dict[str, Any]] = {}

    for rel_path, meta in file_index.items():
        folder = meta["folder"]
        if folder not in folders_map:
            folders_map[folder] = {
                "folder": folder,
                "total": 0,
                "clean": 0,
                "suspicious": 0,
                "warning": 0,
                "critical": 0,
            }
        folders_map[folder]["total"] += 1
        sev = meta.get("severity", "CLEAN")
        if sev == "CLEAN":
            folders_map[folder]["clean"] += 1
        elif sev == "WARNING":
            folders_map[folder]["warning"] += 1
            folders_map[folder]["suspicious"] += 1
        elif sev == "CRITICAL":
            folders_map[folder]["critical"] += 1
            folders_map[folder]["suspicious"] += 1
        elif meta.get("is_suspicious"):
            folders_map[folder]["suspicious"] += 1

    sorted_folders = sorted(folders_map.values(), key=lambda x: x["folder"])
    return sorted_folders


def get_scout_overview(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Returns aggregated overview and health tallies for Classified-unet-masked."""
    cache = _load_manifest_and_index(base_dir)
    file_index = cache["file_index"]
    summary_data = cache.get("summary_data") or {}

    total = len(file_index)
    clean = sum(1 for m in file_index.values() if m.get("severity") == "CLEAN")
    warning = sum(1 for m in file_index.values() if m.get("severity") == "WARNING")
    critical = sum(1 for m in file_index.values() if m.get("severity") == "CRITICAL")
    unindexed = sum(1 for m in file_index.values() if m.get("severity") == "UNINDEXED")
    suspicious = warning + critical

    # Tally anomaly flags
    flags_tally: Dict[str, int] = {}
    for m in file_index.values():
        for flag in m.get("anomaly_flags", []):
            flags_tally[flag] = flags_tally.get(flag, 0) + 1

    return {
        "dataset_path": str(cache["dir"]),
        "dataset_exists": cache["dir"].exists() if cache["dir"] else False,
        "total_scans": total,
        "clean_count": clean,
        "warning_count": warning,
        "critical_count": critical,
        "suspicious_count": suspicious,
        "unindexed_count": unindexed,
        "flags_tally": flags_tally,
        "folder_count": len(cache["tree"]),
        "summary_data": summary_data,
        "updated_at": cache["manifest_mtime"],
    }


def get_scout_tree(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns the list of folders with scan tallies."""
    cache = _load_manifest_and_index(base_dir)
    return cache["tree"]


def query_scout_images(
    base_dir: Optional[Path] = None,
    folder: Optional[str] = None,
    filter_type: str = "all",  # "all", "suspicious", "clean", "warning", "critical"
    query: Optional[str] = None,
    sort_by: str = "name",  # "name", "severity", "thickness"
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """Queries images with filtering, search query, sorting, and pagination."""
    cache = _load_manifest_and_index(base_dir)
    file_index = cache["file_index"]

    results = []
    folder_clean = folder.strip().rstrip("/") if folder else ""

    q_lower = query.strip().lower() if query else ""

    for rel_path, item in file_index.items():
        # Folder filter
        if folder_clean:
            if not (item["folder"] == folder_clean or item["folder"].startswith(f"{folder_clean}/")):
                continue

        # Filter by severity/suspicious
        sev = item.get("severity", "CLEAN")
        is_susp = item.get("is_suspicious", False)
        if filter_type == "suspicious" and not is_susp:
            continue
        elif filter_type == "clean" and sev != "CLEAN":
            continue
        elif filter_type == "warning" and sev != "WARNING":
            continue
        elif filter_type == "critical" and sev != "CRITICAL":
            continue

        # Query search
        if q_lower:
            if q_lower not in rel_path.lower() and q_lower not in item["filename"].lower():
                continue

        # Build payload
        clean_rel = item["rel_path"]
        encoded_rel = urllib.parse.quote(clean_rel)
        results.append({
            "id": clean_rel,
            "rel_path": clean_rel,
            "filename": item["filename"],
            "folder": item["folder"],
            "image_url": f"/api/scout/file?type=image&path={encoded_rel}",
            "mask_url": f"/api/scout/file?type=mask&path={encoded_rel}",
            "quarantine_url": f"/api/scout/file?type=quarantine&path={encoded_rel}" if item["has_quarantine"] else None,
            "severity": sev,
            "is_suspicious": is_susp,
            "anomaly_flags": item.get("anomaly_flags", []),
            "metrics": item.get("metrics", {}),
            "meta": item.get("meta", {}),
            "has_mask": item.get("has_mask", True),
        })

    # Sort results
    if sort_by == "severity":
        sev_rank = {"CRITICAL": 0, "WARNING": 1, "UNINDEXED": 2, "CLEAN": 3}
        results.sort(key=lambda x: (sev_rank.get(x["severity"], 4), x["filename"].lower()))
    elif sort_by == "thickness":
        results.sort(
            key=lambda x: x.get("metrics", {}).get("thickness_mean", 0.0),
            reverse=True,
        )
    else:  # "name"
        results.sort(key=lambda x: x["filename"].lower())

    total = len(results)
    paged = results[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "folder": folder_clean,
        "filter": filter_type,
        "images": paged,
    }


def get_scout_detail(base_dir: Optional[Path] = None, rel_path: str = "") -> Dict[str, Any]:
    """
    Returns full diagnostic detail for an image.
    If the image is not indexed or missing health metrics, evaluates health dynamically.
    """
    target_dir = Path(base_dir) if base_dir else get_unet_masked_dir()
    cache = _load_manifest_and_index(target_dir)
    clean_rel = rel_path.strip().lstrip("/")

    item = cache["file_index"].get(clean_rel)
    img_file = target_dir / "Images" / clean_rel
    mask_file = target_dir / "Masks" / clean_rel

    if not img_file.exists():
        return {"status": "error", "message": f"Image file not found: {clean_rel}"}

    # If item has metrics and flags, return it
    if item and item.get("metrics") and item.get("severity") != "UNINDEXED":
        return {
            "status": "success",
            "rel_path": clean_rel,
            "filename": Path(clean_rel).name,
            "folder": str(Path(clean_rel).parent),
            "severity": item["severity"],
            "is_suspicious": item["is_suspicious"],
            "anomaly_flags": item["anomaly_flags"],
            "metrics": item["metrics"],
            "meta": item.get("meta", {}),
            "has_mask": mask_file.exists(),
        }

    # Otherwise evaluate dynamically on the fly
    if not mask_file.exists():
        return {
            "status": "warning",
            "rel_path": clean_rel,
            "filename": Path(clean_rel).name,
            "folder": str(Path(clean_rel).parent),
            "severity": "WARNING",
            "is_suspicious": True,
            "anomaly_flags": ["MISSING_MASK_FILE"],
            "metrics": {},
            "has_mask": False,
        }

    mask_bgr = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.imread(str(img_file))

    if mask_bgr is None or img_bgr is None:
        return {"status": "error", "message": "Failed to decode image or mask bytes"}

    report: MaskHealthReport = evaluate_mask_health(mask_bgr, raw_img=img_bgr)
    return {
        "status": "success",
        "rel_path": clean_rel,
        "filename": Path(clean_rel).name,
        "folder": str(Path(clean_rel).parent),
        "severity": report.severity,
        "is_suspicious": report.is_suspicious,
        "anomaly_flags": report.anomaly_flags,
        "metrics": report.metrics,
        "recommendations": report.recommendations,
        "has_mask": True,
    }


def resolve_scout_file(
    base_dir: Optional[Path] = None, file_type: str = "image", rel_path: str = ""
) -> Optional[Path]:
    """
    Safely resolves path to an image, mask, or quarantine card preventing directory traversal.
    """
    target_dir = Path(base_dir) if base_dir else get_unet_masked_dir()
    clean_rel = rel_path.strip().lstrip("/")

    if file_type == "image":
        sub_dir = target_dir / "Images"
    elif file_type == "mask":
        sub_dir = target_dir / "Masks"
    elif file_type == "quarantine":
        sub_dir = target_dir / "quarantine_review"
        direct = (sub_dir / clean_rel).resolve()
        if str(direct).startswith(str(sub_dir.resolve())) and direct.exists() and direct.is_file():
            return direct
        stem = Path(clean_rel).stem
        if sub_dir.exists():
            for cand in sub_dir.glob(f"*{stem}*"):
                if cand.is_file():
                    return cand
        return None
    else:
        return None

    target = (sub_dir / clean_rel).resolve()
    # Path traversal protection
    if not str(target).startswith(str(sub_dir.resolve())):
        return None

    if target.exists() and target.is_file():
        return target
    return None


_REMBG_SESSION = None

def get_rembg_session():
    """
    Returns a cached, warm multi-threaded ONNX Runtime session for rembg.
    Configures intra_op and inter_op threads to utilize CPU cores efficiently
    and avoids re-loading the 977MB ONNX model on every click.
    """
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION

    import os
    import rembg
    import onnxruntime as ort

    num_cores = os.cpu_count() or 4
    # Allocate up to 8 threads on Apple Silicon performance/efficiency cores
    intra_threads = max(2, min(8, num_cores - 2))
    inter_threads = 2

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = intra_threads
    sess_opts.inter_op_num_threads = inter_threads
    sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Prefer CPUExecutionProvider configured with optimal multi-threading
    # (falls back gracefully across platforms)
    providers = ["CPUExecutionProvider"]

    try:
        _REMBG_SESSION = rembg.new_session("bria-rmbg", sess_opts=sess_opts, providers=providers)
    except Exception:
        try:
            _REMBG_SESSION = rembg.new_session("birefnet-general", sess_opts=sess_opts, providers=providers)
        except Exception:
            _REMBG_SESSION = rembg.new_session()

_UNET_CROPPER_SESSION = None

def get_unet_cropper_session():
    """Returns a cached, warm UNetTissueCropper instance for interactive window inference."""
    global _UNET_CROPPER_SESSION
    if _UNET_CROPPER_SESSION is not None:
        return _UNET_CROPPER_SESSION

    from data.preprocessing.unet_cropper import UNetTissueCropper
    try:
        _UNET_CROPPER_SESSION = UNetTissueCropper()
    except Exception as e:
        sys.stderr.write(f"[Scout] Failed to initialize UNetTissueCropper: {e}\n")
        _UNET_CROPPER_SESSION = None
    return _UNET_CROPPER_SESSION
def _update_scout_manifest_and_cache(target_dir: Path, clean_rel: str, report: Any):
    """Synchronously syncs updated mask metrics and health status into manifest.json and the scout index cache."""
    manifest_path = target_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            samples = manifest_data.get("samples", {})
            matched_key = None
            for k in [clean_rel, str(Path(clean_rel).with_suffix(".jpg")), str(Path(clean_rel).with_suffix(".jpeg")), str(Path(clean_rel).with_suffix(".png"))]:
                if k in samples:
                    matched_key = k
                    break
            if matched_key:
                samples[matched_key]["severity"] = report.severity
                samples[matched_key]["is_suspicious"] = report.is_suspicious
                samples[matched_key]["anomaly_flags"] = report.anomaly_flags
                samples[matched_key]["metrics"] = report.metrics
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=2)
        except Exception:
            pass

    if clean_rel in _SCOUT_CACHE["file_index"]:
        entry = _SCOUT_CACHE["file_index"][clean_rel]
        entry["severity"] = report.severity
        entry["is_suspicious"] = report.is_suspicious
        entry["anomaly_flags"] = report.anomaly_flags
        entry["metrics"] = report.metrics


def _resolve_image_and_mask_files(target_dir: Path, clean_rel: str) -> tuple[Path, Path]:
    """Resolves image and mask file paths, automatically reconciling .png/.jpg/.jpeg extensions."""
    img_file = target_dir / "Images" / clean_rel
    mask_file = target_dir / "Masks" / clean_rel
    if not img_file.exists() or not mask_file.exists():
        for ext in [".png", ".jpg", ".jpeg"]:
            alt_img = img_file.with_suffix(ext)
            alt_mask = mask_file.with_suffix(ext)
            if alt_img.exists() and alt_mask.exists():
                return alt_img, alt_mask
    return img_file, mask_file


def ablate_scout_background(
    base_dir: Optional[Path] = None,
    rel_path: str = "",
    click_x: int = 0,
    click_y: int = 0,
    radius_x: int = 0,
) -> Dict[str, Any]:
    """
    Surgically ablates background noise (such as vitreous plateau, snout artifact, or floating debris)
    at the user's precision click coordinates (click_x, click_y).
    Supports full horizontal layer span (radius_x=0) to clean noise spanning across the entire scan.
    Creates a backup for full single-click undo.
    """
    import shutil
    import rembg

    target_dir = Path(base_dir) if base_dir else get_unet_masked_dir()
    clean_rel = rel_path.strip().lstrip("/")
    img_file, mask_file = _resolve_image_and_mask_files(target_dir, clean_rel)

    if not img_file.exists() or not mask_file.exists():
        return {"status": "error", "message": f"File not found: {clean_rel}"}

    # Path traversal protection
    if not str(img_file.resolve()).startswith(str(target_dir.resolve())) or not str(mask_file.resolve()).startswith(str(target_dir.resolve())):
        return {"status": "error", "message": "Invalid path"}

    # 1. Create persistent backups for undo
    backup_mask = mask_file.with_name(mask_file.stem + ".mask_backup" + mask_file.suffix)
    backup_img = img_file.with_name(img_file.stem + ".img_backup" + img_file.suffix)
    shutil.copyfile(mask_file, backup_mask)
    shutil.copyfile(img_file, backup_img)

    # 2. Load images
    img_bgr = cv2.imread(str(img_file))
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if img_bgr is None or mask is None:
        return {"status": "error", "message": "Failed to decode image or mask"}

    h, w = mask.shape[:2]
    cx = int(np.clip(click_x, 0, w - 1))
    cy = int(np.clip(click_y, 0, h - 1))

    # 3. Check for isolated connected components / detached artifact islands
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    clicked_label = labels[cy, cx] if mask[cy, cx] > 0 else 0
    if clicked_label == 0:
        sub = labels[max(0, cy - 6):min(h, cy + 7), max(0, cx - 6):min(w, cx + 7)]
        non_zero = sub[sub > 0]
        if len(non_zero) > 0:
            clicked_label = int(np.bincount(non_zero).argmax())

    # If clicked on an isolated component that is distinct from the primary retina:
    if num_labels > 2 and clicked_label > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        if clicked_label != largest_label:
            refined_mask = mask.copy()
            refined_mask[labels == clicked_label] = 0
            pixels_ablated = int(np.sum(mask > 0) - np.sum(refined_mask > 0))

            mask_3c = cv2.merge([refined_mask, refined_mask, refined_mask])
            refined_img = np.where(mask_3c > 0, img_bgr, 0).astype(np.uint8)
            cv2.imwrite(str(mask_file), refined_mask)
            cv2.imwrite(str(img_file), refined_img)
            report = evaluate_mask_health(refined_mask, raw_img=refined_img)
            _update_scout_manifest_and_cache(target_dir, clean_rel, report)
            return {
                "status": "success",
                "rel_path": clean_rel,
                "filename": Path(clean_rel).name,
                "pixels_ablated": pixels_ablated,
                "severity": report.severity,
                "is_suspicious": report.is_suspicious,
                "anomaly_flags": report.anomaly_flags,
                "metrics": report.metrics,
                "layer_scope": "isolated_component",
                "can_undo": True,
                "timestamp": int(time.time()),
            }

    # 4. Horizontal span configuration
    # radius_x <= 0 means FULL LAYER SPAN across all columns (0 to w-1)
    if radius_x > 0:
        x1 = max(0, cx - int(radius_x))
        x2 = min(w, cx + int(radius_x))
    else:
        x1 = 0
        x2 = w

    # 5. Anatomical layer boundary tracking
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    # Determine whether user clicked above the ILM (vitreous) or below the RPE (choroid/sclera)
    cx_ys = np.where(mask[:, cx] > 0)[0]
    if len(cx_ys) > 0:
        pos_peaks = np.where((sobely[cx_ys, cx] > 30) & (blur[cx_ys, cx] > 45))[0]
        cx_ilm = cx_ys[pos_peaks[0]] if len(pos_peaks) > 0 else cx_ys[0]
        sub_ys = cx_ys[cx_ys > cx_ilm + 15]
        neg_peaks = np.where((sobely[sub_ys, cx] < -20) & (blur[sub_ys, cx] > 40))[0]
        if len(neg_peaks) > 0:
            cx_rpe = sub_ys[neg_peaks[-1]]
        else:
            cx_rpe = min(cx_ys[-1], cx_ilm + 60)
    else:
        cx_ilm = h // 3
        cx_rpe = 2 * h // 3

    is_supra_ilm = (cy <= cx_ilm + 20)
    is_sub_rpe = (cy >= cx_rpe - 10)

    refined_mask = mask.copy()
    stage1_ablated = 0

    if is_sub_rpe:
        # Bottom-up sub-RPE removal with continuous anatomical contour tracing
        rpe_curve = np.zeros(w, dtype=np.float32)
        valid_cols = []
        for x in range(w):
            ys = np.where(mask[:, x] > 0)[0]
            if len(ys) < 10:
                continue
            pos_peaks = np.where((sobely[ys, x] > 30) & (blur[ys, x] > 45))[0]
            ilm_y = ys[pos_peaks[0]] if len(pos_peaks) > 0 else ys[0]
            sub_ys = ys[ys > ilm_y + 15]
            if len(sub_ys) == 0:
                continue
            neg_peaks = np.where((sobely[sub_ys, x] < -20) & (blur[sub_ys, x] > 40))[0]
            if len(neg_peaks) > 0:
                rpe_curve[x] = sub_ys[neg_peaks[-1]]
                valid_cols.append(x)

        if len(valid_cols) > 5:
            all_x = np.arange(w)
            rpe_interp = np.interp(all_x, valid_cols, rpe_curve[valid_cols])
            rpe_smooth = ndi.median_filter(rpe_interp, size=15)
            rpe_smooth = ndi.gaussian_filter1d(rpe_smooth, sigma=4.0)

            for x in range(x1, x2):
                cutoff_y = int(round(rpe_smooth[x])) + 12
                refined_mask[cutoff_y:, x] = 0
            stage1_ablated = int(np.sum(mask > 0) - np.sum(refined_mask > 0))

    elif is_supra_ilm:
        # Top-down supra-ILM removal with continuous anatomical contour tracing
        ilm_curve = np.zeros(w, dtype=np.float32)
        valid_cols = []
        for x in range(w):
            ys = np.where(mask[:, x] > 0)[0]
            if len(ys) < 10:
                continue
            peaks = np.where((sobely[ys, x] > 35) & (blur[ys, x] > 50))[0]
            if len(peaks) > 0:
                ilm_curve[x] = ys[peaks[0]]
                valid_cols.append(x)

        if len(valid_cols) > 5:
            all_x = np.arange(w)
            ilm_interp = np.interp(all_x, valid_cols, ilm_curve[valid_cols])
            ilm_smooth = ndi.median_filter(ilm_interp, size=15)
            ilm_smooth = ndi.gaussian_filter1d(ilm_smooth, sigma=4.0)

            for x in range(x1, x2):
                cutoff_y = int(round(ilm_smooth[x])) - 4
                for py in range(cutoff_y):
                    if gray[py, x] < 65:
                        refined_mask[py, x] = 0
            stage1_ablated = int(np.sum(mask > 0) - np.sum(refined_mask > 0))

    # Remove small disconnected specks < 100 px
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
    if num_labels > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        for lbl in range(1, num_labels):
            if lbl != largest_label and stats[lbl, cv2.CC_STAT_AREA] < 100:
                refined_mask[labels == lbl] = 0

    if stage1_ablated > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        pixels_ablated = int(np.sum(mask > 0) - np.sum(refined_mask > 0))
    else:
        # 6. Stage 2: Full-layer AI Model Fallback (bria-rmbg / birefnet)
        session = get_rembg_session()
        rembg_mask = rembg.remove(img_bgr, session=session, only_mask=True)
        rembg_bin = (rembg_mask > 100).astype(np.uint8) * 255
        if radius_x > 0:
            patch_bin = rembg_bin[:, x1:x2]
            refined_mask[:, x1:x2] = np.where(mask[:, x1:x2] > 0, patch_bin, 0)
        else:
            refined_mask = np.where(mask > 0, rembg_bin, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        pixels_ablated = int(np.sum(mask > 0) - np.sum(refined_mask > 0))

    # Update background-suppressed image
    mask_3c = cv2.merge([refined_mask, refined_mask, refined_mask])
    refined_img = np.where(mask_3c > 0, img_bgr, 0).astype(np.uint8)

    # Save to disk
    cv2.imwrite(str(mask_file), refined_mask)
    cv2.imwrite(str(img_file), refined_img)

    # 7. Re-evaluate mask health
    report = evaluate_mask_health(refined_mask, raw_img=refined_img)

    # 8. Update manifest and cache
    _update_scout_manifest_and_cache(target_dir, clean_rel, report)

    return {
        "status": "success",
        "rel_path": clean_rel,
        "filename": Path(clean_rel).name,
        "pixels_ablated": pixels_ablated,
        "severity": report.severity,
        "is_suspicious": report.is_suspicious,
        "anomaly_flags": report.anomaly_flags,
        "metrics": report.metrics,
        "layer_scope": "full_layer" if radius_x <= 0 else f"col_{x1}_{x2}",
        "patch_bounds": [x1, x2, 0, h],
        "can_undo": True,
        "timestamp": int(time.time()),
    }


def undo_scout_ablation(
    base_dir: Optional[Path] = None,
    rel_path: str = "",
) -> Dict[str, Any]:
    """
    Restores the previous mask and image state from backup.
    """
    import shutil

    target_dir = Path(base_dir) if base_dir else get_unet_masked_dir()
    clean_rel = rel_path.strip().lstrip("/")
    img_file, mask_file = _resolve_image_and_mask_files(target_dir, clean_rel)

    backup_mask = mask_file.with_name(mask_file.stem + ".mask_backup" + mask_file.suffix)
    backup_img = img_file.with_name(img_file.stem + ".img_backup" + img_file.suffix)

    if not backup_mask.exists() or not backup_img.exists():
        return {"status": "error", "message": "No backup available to undo."}

    shutil.copyfile(backup_mask, mask_file)
    shutil.copyfile(backup_img, img_file)

    # Re-evaluate restored mask health
    restored_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    restored_img = cv2.imread(str(img_file))
    report = evaluate_mask_health(restored_mask, raw_img=restored_img)

    # Update manifest & cache
    _update_scout_manifest_and_cache(target_dir, clean_rel, report)

    return {
        "status": "success",
        "message": "Ablation undone successfully.",
        "rel_path": clean_rel,
        "filename": Path(clean_rel).name,
        "severity": report.severity,
        "is_suspicious": report.is_suspicious,
        "anomaly_flags": report.anomaly_flags,
        "metrics": report.metrics,
        "can_undo": False,
        "timestamp": int(time.time()),
    }


def rerun_unet_on_window(
    base_dir: Optional[Path] = None,
    rel_path: str = "",
    x1: int = 0,
    y1: int = 0,
    x2: int = 0,
    y2: int = 0,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Reruns the Attention U-Net neural tissue segmenter on a targeted user-drawn box window
    directly on the cropped image itself. Merges predicted binary mask into the scan mask.
    Creates a backup for full single-click undo.
    """
    import shutil

    target_dir = Path(base_dir) if base_dir else get_unet_masked_dir()
    clean_rel = rel_path.strip().lstrip("/")
    img_file, mask_file = _resolve_image_and_mask_files(target_dir, clean_rel)

    if not img_file.exists() or not mask_file.exists():
        return {"status": "error", "message": f"File not found: {clean_rel}"}

    if not str(img_file.resolve()).startswith(str(target_dir.resolve())) or not str(mask_file.resolve()).startswith(str(target_dir.resolve())):
        return {"status": "error", "message": "Invalid path"}

    # 1. Backups for full undo
    backup_mask = mask_file.with_name(mask_file.stem + ".mask_backup" + mask_file.suffix)
    backup_img = img_file.with_name(img_file.stem + ".img_backup" + img_file.suffix)
    shutil.copyfile(mask_file, backup_mask)
    shutil.copyfile(img_file, backup_img)

    img_bgr = cv2.imread(str(img_file))
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if img_bgr is None or mask is None:
        return {"status": "error", "message": "Failed to decode image or mask"}

    h, w = mask.shape[:2]

    # Normalize window bounds
    wx1 = int(np.clip(min(x1, x2), 0, w - 1))
    wx2 = int(np.clip(max(x1, x2), 0, w))
    wy1 = int(np.clip(min(y1, y2), 0, h - 1))
    wy2 = int(np.clip(max(y1, y2), 0, h))

    if wx2 - wx1 < 10 or wy2 - wy1 < 10:
        return {"status": "error", "message": "Selected window is too small (must be at least 10x10 px)"}

    cropper = get_unet_cropper_session()
    if cropper is None or not cropper.has_weights:
        return {"status": "error", "message": "Attention U-Net model weights not loaded"}

    # 2. Extract window sub-image and predict with U-Net
    window_crop_bgr = img_bgr[wy1:wy2, wx1:wx2]
    window_gray = cv2.cvtColor(window_crop_bgr, cv2.COLOR_BGR2GRAY)

    probs, y_t, y_b = cropper.predict_mask_and_vectors(window_gray, threshold=threshold)
    pred_mask_window = (probs > threshold).astype(np.uint8) * 255

    # Refine borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_mask_window = cv2.morphologyEx(pred_mask_window, cv2.MORPH_CLOSE, kernel)

    old_window_mask = mask[wy1:wy2, wx1:wx2]
    pixels_changed = int(np.sum(old_window_mask != pred_mask_window))

    # 3. Splice predicted window mask into the full mask
    refined_mask = mask.copy()
    refined_mask[wy1:wy2, wx1:wx2] = pred_mask_window

    # 4. Update background-suppressed image
    mask_3c = cv2.merge([refined_mask, refined_mask, refined_mask])
    refined_img = np.where(mask_3c > 0, img_bgr, 0).astype(np.uint8)

    cv2.imwrite(str(mask_file), refined_mask)
    cv2.imwrite(str(img_file), refined_img)

    # 5. Diagnostic health check
    report = evaluate_mask_health(refined_mask, raw_img=refined_img)

    # 6. Update manifest & cache
    _update_scout_manifest_and_cache(target_dir, clean_rel, report)

    return {
        "status": "success",
        "rel_path": clean_rel,
        "filename": Path(clean_rel).name,
        "pixels_changed": pixels_changed,
        "severity": report.severity,
        "is_suspicious": report.is_suspicious,
        "anomaly_flags": report.anomaly_flags,
        "metrics": report.metrics,
        "window_bounds": [wx1, wy1, wx2, wy2],
        "can_undo": True,
        "timestamp": int(time.time()),
    }


