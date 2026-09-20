"""
data/preprocessing/tuning/processor.py

Dataset subfolder discovery, sample caching, scan image preprocessing,
and interactive dataset curation / export pipeline for U-Net training.
"""

import csv
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Optional
import urllib.parse
import cv2
import numpy as np

from data.preprocessing.masking import (
    generate_tissue_mask,
    generate_tissue_mask_custom,
)
from data.preprocessing.choroid import (
    detect_choroidal_caverns,
    detect_choroidal_holes,
)
from data.preprocessing.geometry import (
    letterbox_pad_and_resize,
    project_and_downsample_vectors,
    render_boundary_overlay,
)
from data.preprocessing.white_bars import (
    detect_and_process_white_bars,
    detect_and_remove_compass_artifacts,
)

# Project root resolution
_REPO_ROOT = Path(__file__).resolve().parents[5]
_LOCAL_SRC = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified")
_LOCAL_OUT = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed-Otsu")
_LOCAL_MASKED = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-masked")

SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", str(_LOCAL_SRC if _LOCAL_SRC.exists() else _REPO_ROOT / "data" / "Classified")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(_LOCAL_OUT if _LOCAL_OUT.exists() else _REPO_ROOT / "data" / "Classified-preprocessed-Otsu")))
MASKED_DATASET_DIR = Path(os.environ.get("MASKED_DATASET_DIR", str(_LOCAL_MASKED if _LOCAL_MASKED.exists() else _REPO_ROOT / "data" / "Classified-masked")))

SFCM_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
FOLDER_SAMPLES_CACHE: dict[str, list[Path]] = {}


def get_source_dir() -> Path:
    """Resolves SOURCE_DIR, respecting runtime/test attribute overrides."""
    this_mod = sys.modules.get("data.preprocessing.tuning.processor")
    if this_mod is not None:
        val = getattr(this_mod, "SOURCE_DIR", None)
        if val is not None:
            return Path(val)
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server"):
        if mod_name in sys.modules:
            val = getattr(sys.modules[mod_name], "SOURCE_DIR", None)
            if val is not None:
                return Path(val)
    return SOURCE_DIR


def get_output_dir() -> Path:
    """Resolves OUTPUT_DIR, respecting runtime/test attribute overrides."""
    this_mod = sys.modules.get("data.preprocessing.tuning.processor")
    if this_mod is not None:
        val = getattr(this_mod, "OUTPUT_DIR", None)
        if val is not None:
            return Path(val)
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server"):
        if mod_name in sys.modules:
            val = getattr(sys.modules[mod_name], "OUTPUT_DIR", None)
            if val is not None:
                return Path(val)
    return OUTPUT_DIR


def get_masked_dataset_dir() -> Path:
    """Resolves MASKED_DATASET_DIR, respecting runtime/test attribute overrides."""
    this_mod = sys.modules.get("data.preprocessing.tuning.processor")
    if this_mod is not None:
        val = getattr(this_mod, "MASKED_DATASET_DIR", None)
        if val is not None:
            return Path(val)
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server"):
        if mod_name in sys.modules:
            val = getattr(sys.modules[mod_name], "MASKED_DATASET_DIR", None)
            if val is not None:
                return Path(val)
    return MASKED_DATASET_DIR


def get_folder_samples_cache() -> dict[str, list[Path]]:
    """Resolves FOLDER_SAMPLES_CACHE, respecting runtime/test attribute overrides."""
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server", "data.preprocessing.tuning.processor"):
        if mod_name in sys.modules:
            val = getattr(sys.modules[mod_name], "FOLDER_SAMPLES_CACHE", None)
            if val is not None and isinstance(val, dict) and val is not FOLDER_SAMPLES_CACHE:
                return val
    return FOLDER_SAMPLES_CACHE


def get_sfcm_cache() -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Resolves SFCM_CACHE, respecting runtime/test attribute overrides."""
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server", "data.preprocessing.tuning.processor"):
        if mod_name in sys.modules:
            val = getattr(sys.modules[mod_name], "SFCM_CACHE", None)
            if val is not None and isinstance(val, dict) and val is not SFCM_CACHE:
                return val
    return SFCM_CACHE


def find_folder_path(folder_name: str | Path, source_dir: Optional[Path] = None) -> Path | None:
    """Finds the absolute path of a dataset subfolder, prioritizing directories with image files."""
    if isinstance(folder_name, Path):
        if folder_name.exists():
            return folder_name
        folder_name = folder_name.name

    active_source = Path(source_dir) if source_dir else get_source_dir()
    if not active_source.exists():
        return None

    direct = active_source / folder_name
    if direct.is_dir():
        direct_files = list(direct.glob("*.jp*g")) + list(direct.glob("*.png"))
        if direct_files:
            return direct

    matches = [d for d in active_source.rglob(folder_name) if d.is_dir() and not d.name.startswith(".")]
    if not matches:
        return None

    matches_with_counts = []
    for d in matches:
        count = len(list(d.glob("*.jp*g")) + list(d.glob("*.png")))
        if count == 0:
            count = len(list(d.rglob("*.jp*g")) + list(d.rglob("*.png")))
        matches_with_counts.append((d, count))

    matches_with_counts.sort(key=lambda x: x[1], reverse=True)
    return matches_with_counts[0][0]


def find_image_path(folder_name: str | Path, filename: str, source_dir: Optional[Path] = None) -> Path | None:
    """Finds the path of a specific image by filename inside a folder."""
    folder_path = find_folder_path(folder_name, source_dir=source_dir)
    if folder_path:
        direct = folder_path / filename
        if direct.exists():
            return direct
        matches = list(folder_path.rglob(filename))
        return matches[0] if matches else None
    return None


def get_available_subfolders(source_dir: Optional[Path] = None) -> list[str]:
    """Finds all leaf directories containing valid scan images."""
    active_source = Path(source_dir) if source_dir else get_source_dir()
    if not active_source.exists():
        return []
    subfolders = []
    try:
        for p in sorted(active_source.rglob("*")):
            if p.is_dir() and not p.name.startswith("."):
                valid = list(p.glob("*.jp*g")) + list(p.glob("*.png"))
                if valid:
                    subfolders.append(p.name)
    except (PermissionError, OSError):
        return []
    return subfolders


def process_and_save_image(src_p: Path, out_folder: Path, folder_name: str, params: dict) -> dict | None:
    """Processes a single scan image, generates overlays, and writes outputs to disk."""
    compass_enabled = params.get("compass_ui_enabled", None)
    compass_location = params.get("compass_location", "auto")
    margin_bottom = params.get("margin_bottom", params.get("margin", 15))

    img_bgr = cv2.imread(str(src_p))
    if img_bgr is None:
        return None

    img_bgr, compass_bbox = detect_and_remove_compass_artifacts(
        img_bgr,
        src_path=str(src_p),
        enabled=compass_enabled,
        location=compass_location,
        margin=margin_bottom,
        return_bbox=True
    )

    img_bgr = detect_and_process_white_bars(img_bgr, white_thresh=190, dark_bg_thresh=70, gap_pixels=3)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    is_unet_mode = bool(params.get("unet_mode", False))
    cropper = get_unet_cropper() if is_unet_mode else None

    if is_unet_mode and cropper is not None and cropper.has_weights:
        try:
            _, y_top_outer, y_bottom_outer = cropper.predict_mask_and_vectors(gray, threshold=0.50)
            orig_h, orig_w = gray.shape
            margin_top = int(params.get("margin_top", 15))
            margin_bot = int(params.get("margin_bottom", 15))
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for x in range(orig_w):
                t_y = max(0, int(y_top_outer[x] - margin_top))
                b_y = min(orig_h - 1, int(y_bottom_outer[x] + margin_bot))
                if b_y > t_y:
                    mask[t_y:b_y + 1, x] = 255
            y_rpe = None
            y_bottom_sfcm = None
            y_bottom_sfcm_raw = None
        except Exception as e:
            sys.stderr.write(f"[UNetMode] Inference fallback to clustering: {e}\n")
            active_sfcm_cache = get_sfcm_cache()
            mask, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = generate_tissue_mask_custom(
                gray, params, compass_bbox=compass_bbox, return_sfcm=True, src_path=str(src_p), sfcm_cache=active_sfcm_cache
            )
    else:
        active_sfcm_cache = get_sfcm_cache()
        mask, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = generate_tissue_mask_custom(
            gray, params, compass_bbox=compass_bbox, return_sfcm=True, src_path=str(src_p), sfcm_cache=active_sfcm_cache
        )

    mask_3c = cv2.merge([mask, mask, mask])
    processed = np.where(mask_3c > 0, img_bgr, 0).astype(np.uint8)

    processed_resized, scale, pad_t, pad_l, h, w = letterbox_pad_and_resize(processed, target_dim=384)
    raw_resized, _, _, _, _, _ = letterbox_pad_and_resize(img_bgr, target_dim=384)

    proc_out = out_folder / f"{src_p.stem}_proc.jpg"
    raw_out = out_folder / f"{src_p.stem}_raw.jpg"

    cv2.imwrite(str(proc_out), processed_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(raw_out), raw_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

    projected_tuple = project_and_downsample_vectors(
        w,
        y_top_outer=y_top_outer,
        y_bottom_outer=y_bottom_outer,
        y_rpe=y_rpe,
        y_sfcm=y_bottom_sfcm_raw if y_bottom_sfcm_raw is not None else y_bottom_sfcm,
        y_slack=y_bottom_sfcm if y_bottom_sfcm_raw is not None else None,
        pad_t=pad_t,
        pad_l=pad_l,
        scale=scale,
        n_pts=64
    )
    if len(projected_tuple) == 5:
        top_pts, bot_pts, rpe_pts, sfcm_raw_pts, slack_pts = projected_tuple
    else:
        top_pts, bot_pts, rpe_pts, sfcm_raw_pts = projected_tuple
        slack_pts = None

    holes_data = []
    caverns_data = []
    if y_rpe is not None and y_bottom_sfcm is not None:
        raw_holes = detect_choroidal_holes(gray, y_rpe, y_bottom_sfcm, params=params) if params.get("holes_enabled", True) else []
        for h_item in raw_holes:
            cnt = h_item["contour"]
            svg_pts = []
            for pt in cnt:
                px, py = pt[0]
                sx = (float(px) + pad_l) * scale
                sy = (float(py) + pad_t) * scale
                svg_pts.append(f"{sx:.1f},{sy:.1f}")
            path_d = "M " + " L ".join(svg_pts) + " Z"
            bx, by, bw, bh = h_item["bbox"]
            holes_data.append({
                "path_d": path_d,
                "bbox": [
                    float(bx + pad_l) * scale,
                    float(by + pad_t) * scale,
                    float(bw) * scale,
                    float(bh) * scale
                ],
                "area": h_item["area"],
                "circularity": h_item["circularity"]
            })

        raw_caverns = detect_choroidal_caverns(gray, y_rpe, y_bottom_sfcm, params=params)
        for c in raw_caverns:
            bx, by, bw, bh = c["bbox"]
            caverns_data.append({
                "bbox": [
                    float(bx + pad_l) * scale,
                    float(by + pad_t) * scale,
                    float(bw) * scale,
                    float(bh) * scale
                ],
                "area": c["area"],
                "circularity": c["circularity"],
                "transmission_ratio": c["transmission_ratio"]
            })

    return {
        "filename": src_p.name,
        "filepath": str(src_p),
        "raw_url": f"/preprocessed/{folder_name}/{raw_out.name}",
        "proc_url": f"/preprocessed/{folder_name}/{proc_out.name}",
        "processed_url": f"/preprocessed/{folder_name}/{proc_out.name}",
        "top_vector": top_pts,
        "bot_vector": bot_pts,
        "bottom_vector": bot_pts,
        "rpe_vector": rpe_pts,
        "sfcm_vector": sfcm_raw_pts,
        "slack_vector": slack_pts,
        "holes": holes_data,
        "caverns": caverns_data
    }


def reprocess_folder_sample(folder_name: str, params: dict, random_sample: bool = False) -> list:
    """Reprocesses a sample batch for a given folder and returns JSON payload."""
    folder_path = find_folder_path(folder_name)
    if not folder_path:
        return []

    cache = get_folder_samples_cache()
    if folder_name not in cache or random_sample:
        files = sorted(
            [f for f in folder_path.glob("*.jp*g") if not f.name.startswith(".")] +
            [f for f in folder_path.glob("*.png") if not f.name.startswith(".")]
        )
        if not files:
            files = sorted(
                [f for f in folder_path.rglob("*.jp*g") if not f.name.startswith(".")] +
                [f for f in folder_path.rglob("*.png") if not f.name.startswith(".")]
            )
        if not files:
            return []

        if len(files) <= 4:
            sample_files = files
        else:
            if random_sample:
                sample_files = sorted(random.sample(files, 4))
            else:
                indices = np.linspace(0, len(files) - 1, 4, dtype=int)
                sample_files = [files[i] for i in indices]
        cache[folder_name] = sample_files
        FOLDER_SAMPLES_CACHE[folder_name] = sample_files
    else:
        sample_files = cache[folder_name]

    active_output = get_output_dir()
    out_folder = active_output / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)

    results = []
    for src_p in sample_files:
        res = process_and_save_image(src_p, out_folder, folder_name, params)
        if res is not None:
            results.append(res)
    return results


def reprocess_single_image(folder_name: str, filename: str, params: dict) -> dict | None:
    """Reprocesses a specific single image given its filename."""
    img_path = find_image_path(folder_name, filename)
    if not img_path:
        return None

    active_output = get_output_dir()
    out_folder = active_output / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)
    return process_and_save_image(img_path, out_folder, folder_name, params)


def process_single_image_cli(
    image_path_or_filename: str | Path,
    folder_name: Optional[str] = None,
    params: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
    save_overlay: bool = True,
    save_mask: bool = False,
) -> dict:
    """
    Direct CLI entrypoint to process an individual OCT scan image, compute layer metrics,
    and save preprocessed, raw, overlay, and mask outputs.
    """
    from data.preprocessing.params import get_folder_params, DEFAULT_PARAMS

    src_p = Path(image_path_or_filename)
    if not src_p.exists() or not src_p.is_file():
        resolved = None
        if folder_name:
            resolved = find_image_path(folder_name, str(image_path_or_filename))
        if not resolved:
            s_dir = get_source_dir()
            matches = list(s_dir.rglob(str(image_path_or_filename)))
            if matches:
                resolved = matches[0]
        if not resolved or not resolved.exists():
            raise FileNotFoundError(f"Image not found: {image_path_or_filename}")
        src_p = resolved

    resolved_folder = folder_name or src_p.parent.name
    active_params = get_folder_params(resolved_folder).copy() if resolved_folder else DEFAULT_PARAMS.copy()
    if params:
        active_params.update(params)

    img_bgr = cv2.imread(str(src_p))
    if img_bgr is None:
        raise ValueError(f"Could not decode image at {src_p}")

    orig_h, orig_w = img_bgr.shape[:2]

    compass_bbox = None
    if active_params.get("compass_ui_enabled", False):
        img_bgr, compass_bbox = detect_and_remove_compass_artifacts(
            img_bgr, location=active_params.get("compass_location", "auto")
        )

    img_bgr = detect_and_process_white_bars(
        img_bgr, white_thresh=190, dark_bg_thresh=70, gap_pixels=3
    )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = generate_tissue_mask_custom(
        gray, active_params, compass_bbox=compass_bbox, return_sfcm=True, src_path=str(src_p)
    )

    processed = np.where(mask[:, :, None] > 0, img_bgr, 0)
    processed_resized, scale, pad_t, pad_l, _, _ = letterbox_pad_and_resize(processed, target_dim=384)
    raw_resized, _, _, _, _, _ = letterbox_pad_and_resize(img_bgr, target_dim=384)

    top_pts, bot_pts, rpe_pts, sfcm_raw_pts, slack_pts = project_and_downsample_vectors(
        orig_w=orig_w,
        y_top=y_top_outer,
        y_bottom=y_bottom_outer,
        y_rpe=y_rpe,
        y_sfcm=y_bottom_sfcm_raw,
        y_slack=y_bottom_sfcm,
        pad_t=pad_t,
        pad_l=pad_l,
        scale=scale,
        n_pts=64
    )

    raw_holes = []
    holes_data = []
    if active_params.get("holes_enabled", True):
        raw_holes = detect_choroidal_holes(gray, y_rpe, y_bottom_sfcm, params=active_params)
        for h_item in raw_holes:
            cnt = h_item["contour"]
            pts = cnt.reshape(-1, 2)
            svg_pts = []
            for px, py in pts:
                sx = (float(px) + pad_l) * scale
                sy = (float(py) + pad_t) * scale
                svg_pts.append(f"{sx:.1f},{sy:.1f}")
            path_d = "M " + " L ".join(svg_pts) + " Z"
            bx, by, bw, bh = h_item["bbox"]
            holes_data.append({
                "path_d": path_d,
                "bbox": [
                    float(bx + pad_l) * scale,
                    float(by + pad_t) * scale,
                    float(bw) * scale,
                    float(bh) * scale
                ],
                "area": h_item["area"],
                "circularity": h_item["circularity"]
            })

    raw_caverns = []
    caverns_data = []
    if active_params.get("detect_caverns", False):
        raw_caverns = detect_choroidal_caverns(gray, y_rpe, y_bottom_sfcm, params=active_params)
        for c in raw_caverns:
            bx, by, bw, bh = c["bbox"]
            caverns_data.append({
                "bbox": [
                    float(bx + pad_l) * scale,
                    float(by + pad_t) * scale,
                    float(bw) * scale,
                    float(bh) * scale
                ],
                "area": c["area"],
                "circularity": c["circularity"],
                "transmission_ratio": c["transmission_ratio"]
            })

    if out_dir:
        destination_dir = Path(out_dir)
    else:
        destination_dir = get_output_dir() / resolved_folder
    destination_dir.mkdir(parents=True, exist_ok=True)

    proc_file = destination_dir / f"{src_p.stem}_proc.jpg"
    raw_file = destination_dir / f"{src_p.stem}_raw.jpg"
    cv2.imwrite(str(proc_file), processed_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(raw_file), raw_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    overlay_file = None
    if save_overlay:
        overlay_bgr = render_boundary_overlay(img_bgr, y_top_outer, y_rpe, y_bottom_sfcm, raw_holes)
        overlay_file = destination_dir / f"{src_p.stem}_overlay.png"
        cv2.imwrite(str(overlay_file), overlay_bgr)

    mask_file = None
    if save_mask:
        mask_file = destination_dir / f"{src_p.stem}_mask.png"
        cv2.imwrite(str(mask_file), mask)

    retinal_thick = y_rpe - y_top_outer
    choroid_thick = y_bottom_sfcm - y_rpe

    return {
        "status": "success",
        "filename": src_p.name,
        "filepath": str(src_p.resolve()),
        "folder": resolved_folder,
        "dimensions": {"height": orig_h, "width": orig_w},
        "metrics": {
            "ilm_y": {"min": round(float(np.min(y_top_outer)), 1), "mean": round(float(np.mean(y_top_outer)), 1), "max": round(float(np.max(y_top_outer)), 1)},
            "rpe_y": {"min": round(float(np.min(y_rpe)), 1), "mean": round(float(np.mean(y_rpe)), 1), "max": round(float(np.max(y_rpe)), 1)},
            "choroid_y": {"min": round(float(np.min(y_bottom_sfcm)), 1), "mean": round(float(np.mean(y_bottom_sfcm)), 1), "max": round(float(np.max(y_bottom_sfcm)), 1)},
            "retinal_thickness_px": {"min": round(float(np.min(retinal_thick)), 1), "mean": round(float(np.mean(retinal_thick)), 1), "max": round(float(np.max(retinal_thick)), 1)},
            "choroid_thickness_px": {"min": round(float(np.min(choroid_thick)), 1), "mean": round(float(np.mean(choroid_thick)), 1), "max": round(float(np.max(choroid_thick)), 1)},
            "holes_count": len(holes_data),
            "caverns_count": len(caverns_data),
        },
        "saved_files": {
            "processed": str(proc_file),
            "raw": str(raw_file),
            "overlay": str(overlay_file) if overlay_file else None,
            "mask": str(mask_file) if mask_file else None,
        },
        "vectors_projected_384": {
            "top": top_pts,
            "bottom": bot_pts,
            "rpe": rpe_pts,
            "sfcm": sfcm_raw_pts,
            "slack": slack_pts,
        }
    }


def process_folder_cli(
    folder_name: str,
    params: Optional[dict] = None,
    sample_count: Optional[int] = 4,
    out_dir: Optional[str | Path] = None,
    save_overlay: bool = True,
    save_mask: bool = False,
) -> list[dict]:
    """Direct CLI entrypoint to process a batch of sample scans from a dataset subfolder."""
    folder_path = find_folder_path(folder_name)
    if not folder_path:
        raise FileNotFoundError(f"Subfolder not found in Classified dataset: {folder_name}")

    files = sorted(
        [f for f in folder_path.glob("*.jp*g") if not f.name.startswith(".")] +
        [f for f in folder_path.glob("*.png") if not f.name.startswith(".")]
    )
    if not files:
        files = sorted(
            [f for f in folder_path.rglob("*.jp*g") if not f.name.startswith(".")] +
            [f for f in folder_path.rglob("*.png") if not f.name.startswith(".")]
        )
    if not files:
        raise FileNotFoundError(f"No scan images found in folder: {folder_name}")

    if sample_count and len(files) > sample_count:
        indices = np.linspace(0, len(files) - 1, sample_count, dtype=int)
        selected_files = [files[i] for i in indices]
    else:
        selected_files = files

    results = []
    for f in selected_files:
        res = process_single_image_cli(
            f, folder_name=folder_name, params=params, out_dir=out_dir,
            save_overlay=save_overlay, save_mask=save_mask
        )
        results.append(res)
    return results


# -----------------------------------------------------------------------------
# Interactive Dataset Curation & Export to Classified-masked/ for U-Net Training
# -----------------------------------------------------------------------------

def _sync_manifest_files(target_dir: Path, manifest_data: dict):
    """Writes curated_manifest.json and regenerates manifest.csv atomically."""
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_json_p = target_dir / "curated_manifest.json"
    manifest_csv_p = target_dir / "manifest.csv"

    with open(manifest_json_p, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    samples = manifest_data.get("samples", {})
    with open(manifest_csv_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "filename", "image_path", "mask_path", "vis_path", "height", "width", "timestamp"])
        for s in samples.values():
            dim = s.get("dimensions", {})
            writer.writerow([
                s.get("folder", ""),
                s.get("filename", ""),
                s.get("image_path", ""),
                s.get("mask_path", ""),
                s.get("vis_path", ""),
                dim.get("height", ""),
                dim.get("width", ""),
                s.get("timestamp", "")
            ])


def get_curated_manifest(masked_dir: Optional[Path] = None) -> dict:
    """Returns all currently curated samples and picked lookup dictionary."""
    target_masked_dir = Path(masked_dir) if masked_dir else get_masked_dataset_dir()
    manifest_p = target_masked_dir / "curated_manifest.json"
    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data.get("samples", {})
                picked_keys = {k: True for k in samples.keys()}
                return {
                    "status": "success",
                    "masked_dir": str(target_masked_dir),
                    "total_count": len(samples),
                    "samples": list(samples.values()),
                    "picked_keys": picked_keys,
                }
        except Exception:
            pass

    return {
        "status": "success",
        "masked_dir": str(target_masked_dir),
        "total_count": 0,
        "samples": [],
        "picked_keys": {},
    }


def save_curated_mask_sample(
    folder_name: str,
    filename: str,
    params: dict,
    masked_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None
) -> dict:
    """
    Generates and saves a high-quality raw image, binary mask, and visualization overlay
    into the Classified-masked/ dataset directory for U-Net training.
    """
    img_path = find_image_path(folder_name, filename, source_dir=source_dir)
    if not img_path or not img_path.exists():
        raise FileNotFoundError(f"Image '{filename}' not found in folder '{folder_name}'")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image at {img_path}")

    orig_h, orig_w = img_bgr.shape[:2]

    compass_enabled = params.get("compass_ui_enabled", None)
    compass_location = params.get("compass_location", "auto")
    margin_bottom = params.get("margin_bottom", params.get("margin", 15))

    img_bgr_clean, compass_bbox = detect_and_remove_compass_artifacts(
        img_bgr,
        src_path=str(img_path),
        enabled=compass_enabled,
        location=compass_location,
        margin=margin_bottom,
        return_bbox=True
    )
    img_bgr_clean = detect_and_process_white_bars(img_bgr_clean, white_thresh=190, dark_bg_thresh=70, gap_pixels=3)

    gray = cv2.cvtColor(img_bgr_clean, cv2.COLOR_BGR2GRAY)
    active_sfcm_cache = get_sfcm_cache()
    mask, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = generate_tissue_mask_custom(
        gray, params, compass_bbox=compass_bbox, return_sfcm=True, src_path=str(img_path), sfcm_cache=active_sfcm_cache
    )

    target_masked_dir = Path(masked_dir) if masked_dir else get_masked_dataset_dir()
    images_dir = target_masked_dir / "Images" / folder_name
    masks_dir = target_masked_dir / "Masks" / folder_name
    vis_dir = target_masked_dir / "Visualizations" / folder_name

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_out_p = images_dir / f"{img_path.stem}.png"
    mask_out_p = masks_dir / f"{img_path.stem}.png"
    vis_out_p = vis_dir / f"{img_path.stem}_overlay.png"

    cv2.imwrite(str(img_out_p), img_bgr)
    cv2.imwrite(str(mask_out_p), mask)

    raw_holes = []
    if y_rpe is not None and y_bottom_sfcm is not None and params.get("holes_enabled", True):
        raw_holes = detect_choroidal_holes(gray, y_rpe, y_bottom_sfcm, params=params)

    overlay_bgr = render_boundary_overlay(
        img_bgr,
        y_top=y_top_outer,
        y_rpe=y_rpe,
        y_sfcm=y_bottom_sfcm if y_bottom_sfcm is not None else y_bottom_outer,
        holes=raw_holes
    )
    cv2.imwrite(str(vis_out_p), overlay_bgr)

    manifest_p = target_masked_dir / "curated_manifest.json"
    manifest_data = {"version": "1.0", "samples": {}}
    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                if "samples" not in manifest_data or not isinstance(manifest_data["samples"], dict):
                    manifest_data["samples"] = {}
        except Exception:
            manifest_data = {"version": "1.0", "samples": {}}

    sample_key = f"{folder_name}/{img_path.name}"
    record = {
        "folder": folder_name,
        "filename": img_path.name,
        "stem": img_path.stem,
        "source_path": str(img_path.resolve()),
        "image_path": str(img_out_p.relative_to(target_masked_dir)),
        "mask_path": str(mask_out_p.relative_to(target_masked_dir)),
        "vis_path": str(vis_out_p.relative_to(target_masked_dir)),
        "dimensions": {"height": orig_h, "width": orig_w},
        "timestamp": time.time(),
        "params": params,
    }
    manifest_data["samples"][sample_key] = record
    manifest_data["total_count"] = len(manifest_data["samples"])

    _sync_manifest_files(target_masked_dir, manifest_data)

    return {
        "status": "success",
        "message": f"Successfully curated '{img_path.name}' into Classified-masked/",
        "sample": record,
        "total_count": len(manifest_data["samples"])
    }


def remove_curated_mask_sample(
    folder_name: str,
    filename: str,
    masked_dir: Optional[Path] = None
) -> dict:
    """Removes a previously curated sample from the Classified-masked/ dataset."""
    target_masked_dir = Path(masked_dir) if masked_dir else get_masked_dataset_dir()
    manifest_p = target_masked_dir / "curated_manifest.json"
    sample_key = f"{folder_name}/{filename}"
    stem = Path(filename).stem

    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                samples = manifest_data.get("samples", {})
        except Exception:
            samples = {}
    else:
        samples = {}

    removed = False
    if sample_key in samples:
        del samples[sample_key]
        removed = True

    for subdir, fname in [
        ("Images", f"{stem}.png"),
        ("Masks", f"{stem}.png"),
        ("Visualizations", f"{stem}_overlay.png")
    ]:
        p = target_masked_dir / subdir / folder_name / fname
        if p.exists():
            try:
                p.unlink()
                removed = True
            except OSError:
                pass

    manifest_data = {"version": "1.0", "samples": samples, "total_count": len(samples)}
    _sync_manifest_files(target_masked_dir, manifest_data)

    return {
        "status": "success",
        "removed": removed,
        "folder": folder_name,
        "filename": filename,
        "total_count": len(samples)
    }


def curate_folder_batch(
    folder_name: str,
    filenames: list[str],
    params: dict,
    masked_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None
) -> dict:
    """Curates a batch of images into Classified-masked/."""
    results = []
    errors = []
    for fname in filenames:
        try:
            res = save_curated_mask_sample(folder_name, fname, params, masked_dir=masked_dir, source_dir=source_dir)
            results.append(res["sample"])
        except Exception as e:
            errors.append({"filename": fname, "error": str(e)})

    manifest = get_curated_manifest(masked_dir=masked_dir)
    return {
        "status": "success" if not errors else "partial",
        "curated_count": len(results),
        "total_count": manifest["total_count"],
        "curated_samples": results,
        "errors": errors
    }


def reset_curated_dataset(masked_dir: Optional[Path] = None) -> dict:
    """Completely clears all curated masks and resets curated_manifest.json to 0."""
    import shutil
    msk_dir = Path(masked_dir) if masked_dir else get_masked_dataset_dir()
    if msk_dir.exists():
        for sub in ["Images", "Masks", "Visualizations"]:
            target_sub = msk_dir / sub
            if target_sub.exists():
                shutil.rmtree(target_sub)
                target_sub.mkdir(parents=True, exist_ok=True)
        manifest_p = msk_dir / "curated_manifest.json"
        manifest_p.write_text(json.dumps({"version": "1.0", "samples": {}}, indent=2), encoding="utf-8")
        csv_p = msk_dir / "curated_manifest.csv"
        if csv_p.exists():
            try:
                csv_p.unlink()
            except Exception:
                pass
    return {"status": "success", "message": "Curated dataset reset to 0", "total_curated": 0}


# ==============================================================================
# ATTENTION U-NET DYNAMIC CROP FILTERING & ACTIVE RETRAINING ENGINE
# ==============================================================================

_UNET_CROPPER_INSTANCE = None
_UNET_CROPPER_CPU_INSTANCE = None
_UNET_PREDICT_CACHE = {}
_PREFETCH_LOCK = threading.Lock()
_RETRAIN_LOCK = threading.Lock()
_RETRAIN_STATE = {
    "running": False,
    "epoch": 0,
    "total_epochs": 10,
    "batch": 0,
    "total_batches": 0,
    "batch_loss": 0.0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "val_dice": 0.0,
    "val_iou": 0.0,
    "phase": "idle",
    "message": "Ready to train.",
    "started_at": 0.0,
    "updated_at": 0.0
}


def is_retraining_active() -> bool:
    """Returns True if U-Net fine-tuning is currently executing in the background."""
    return bool(_RETRAIN_STATE.get("running", False))


def clear_unet_predict_cache():
    """Clears the pre-computed in-memory boundary vector predictions cache."""
    global _UNET_PREDICT_CACHE
    with _PREFETCH_LOCK:
        _UNET_PREDICT_CACHE.clear()


def get_unet_cropper(force_reload: bool = False, prefer_cpu_if_training: bool = True):
    """
    Retrieves or instantiates the UNetTissueCropper.

    If retraining is actively running on MPS/CUDA GPU, dynamically routes interactive
    curation inference to CPU to prevent GPU context switching, resource contention,
    and command buffer stalls. When retraining is idle, uses high-speed GPU acceleration.
    """
    global _UNET_CROPPER_INSTANCE, _UNET_CROPPER_CPU_INSTANCE

    use_cpu = prefer_cpu_if_training and is_retraining_active()

    if use_cpu:
        if _UNET_CROPPER_CPU_INSTANCE is None or force_reload:
            try:
                from data.preprocessing.unet_cropper import UNetTissueCropper
                _UNET_CROPPER_CPU_INSTANCE = UNetTissueCropper(device="cpu", target_size=(384, 384))
            except Exception as e:
                sys.stderr.write(f"[UNetCropper CPU] Failed to load UNetTissueCropper on CPU: {e}\n")
                _UNET_CROPPER_CPU_INSTANCE = None
        return _UNET_CROPPER_CPU_INSTANCE
    else:
        if _UNET_CROPPER_INSTANCE is None or force_reload:
            try:
                from data.preprocessing.unet_cropper import UNetTissueCropper
                _UNET_CROPPER_INSTANCE = UNetTissueCropper(target_size=(384, 384))
            except Exception as e:
                sys.stderr.write(f"[UNetCropper GPU] Failed to load UNetTissueCropper on GPU: {e}\n")
                _UNET_CROPPER_INSTANCE = None
        return _UNET_CROPPER_INSTANCE


def detect_tissue_lateral_bounds(
    probs_orig: np.ndarray,
    y_top: np.ndarray,
    y_bot: np.ndarray,
    threshold: float = 0.50
) -> tuple[int, int, int, int]:
    """
    Analytically derives 2-point slanted lateral crop bounds [crop_left_top, crop_left_bot,
    crop_right_top, crop_right_bot] targeting the anatomical Optic Nerve Head (ONH) /
    Bruch's Membrane Opening (BMO) margins and peripheral tissue boundaries.
    """
    h, w = probs_orig.shape
    bin_mask = (probs_orig >= threshold).astype(np.uint8)

    # Measure column tissue thickness
    col_thickness = np.sum(bin_mask, axis=0).astype(float)
    thick = y_bot - y_top

    # Reference macular thickness in the central scan region
    center_start, center_end = int(w * 0.30), int(w * 0.70)
    macular_thick_ref = float(np.median(thick[center_start:center_end])) if center_end > center_start else 100.0

    # Smoothed ILM slope (dy_top / dx): positive dy means downward plunge into cup
    smooth_top = cv2.GaussianBlur(y_top.reshape(1, -1), (1, 15), sigmaX=3.0).flatten()
    dy_top = np.gradient(smooth_top)

    clt, clb = 0, 0
    crt, crb = w, w

    # -------------------------------------------------------------
    # 1. Check RIGHT Lateral Edge for Optic Nerve Head (ONH) / BMO
    # -------------------------------------------------------------
    search_r_start = int(w * 0.60)
    zero_cols = np.where(col_thickness[search_r_start:] < 15)[0]
    collapse_cols = np.where(thick[search_r_start:] < (0.35 * macular_thick_ref))[0]
    r_zero = (search_r_start + zero_cols[0]) if len(zero_cols) > 0 else w
    r_collapse = (search_r_start + collapse_cols[0]) if len(collapse_cols) > 0 else w
    r_tissue_end = min(r_zero, r_collapse)

    if r_tissue_end < (w - 10):
        # Look backwards from tissue termination for the optic disc rim / ILM cup inflection
        sub_range = range(max(search_r_start, r_tissue_end - 60), r_tissue_end)
        steep_ilm_candidates = [x for x in sub_range if dy_top[x] > 0.40]

        if len(steep_ilm_candidates) > 0:
            # Rim crest / cup inflection where ILM starts descending into the nerve
            rim_crest_x = steep_ilm_candidates[0]
            # BMO: Bruch's membrane opening where outer layer terminates
            bmo_x = min(w, r_tissue_end)
            crt = int(rim_crest_x)
            crb = int(bmo_x)
        else:
            crt = int(r_tissue_end)
            crb = int(r_tissue_end)

    # -------------------------------------------------------------
    # 2. Check LEFT Lateral Edge for Optic Nerve Head (ONH) / BMO
    # -------------------------------------------------------------
    search_l_end = int(w * 0.40)
    zero_cols_l = np.where(col_thickness[:search_l_end] < 15)[0]
    collapse_cols_l = np.where(thick[:search_l_end] < (0.35 * macular_thick_ref))[0]
    l_zero = zero_cols_l[-1] if len(zero_cols_l) > 0 else 0
    l_collapse = collapse_cols_l[-1] if len(collapse_cols_l) > 0 else 0
    l_tissue_end = max(l_zero, l_collapse)

    if l_tissue_end > 10:
        # On the left, descending into the cup toward the left means dy_top < -0.40
        sub_range_l = range(l_tissue_end, min(search_l_end, l_tissue_end + 60))
        steep_ilm_l = [x for x in sub_range_l if dy_top[x] < -0.40]

        if len(steep_ilm_l) > 0:
            rim_crest_x_l = steep_ilm_l[-1]
            bmo_x_l = max(0, l_tissue_end)
            clt = int(rim_crest_x_l)
            clb = int(bmo_x_l)
        else:
            clt = int(l_tissue_end)
            clb = int(l_tissue_end)

    return clt, clb, crt, crb


def sample_adaptive_contour_nodes(y_curve: np.ndarray, target_count: int = 48, min_spacing: int = 4) -> np.ndarray:
    """
    Samples target_count node indices across scan width W weighted by arc-length, slope gradient, and curvature.
    Densely allocates nodes (3-5px spacing) across steep spikes, pits, and sharp tissue transitions,
    while spacing nodes more widely across smooth, flat plateaus.
    """
    w = len(y_curve)
    if w <= target_count:
        return np.arange(w, dtype=int)

    y_arr = np.nan_to_num(y_curve, nan=0.0).astype(np.float32)
    y_smooth = cv2.GaussianBlur(y_arr.reshape(1, -1), (7, 1), 1.5).squeeze()

    dy = np.gradient(y_smooth)
    d2y = np.gradient(dy)

    dy_scale = float(np.std(dy)) + 1e-4
    d2y_scale = float(np.std(d2y)) + 1e-4

    # Combined density: baseline uniform spacing + slope gradient + local curvature
    w_x = 1.0 + 1.8 * np.clip(np.abs(dy) / dy_scale, 0.0, 8.0) + 2.5 * np.clip(np.abs(d2y) / d2y_scale, 0.0, 10.0)

    cumsum = np.cumsum(w_x)
    denom = cumsum[-1] - cumsum[0]
    if denom <= 0:
        return np.linspace(0, w - 1, target_count, dtype=int)

    cdf = (cumsum - cumsum[0]) / denom

    targets = np.linspace(0.0, 1.0, target_count)
    sampled_x = np.interp(targets, cdf, np.arange(w))
    sampled_indices = np.round(sampled_x).astype(int)
    sampled_indices[0] = 0
    sampled_indices[-1] = w - 1

    # Filter with minimum spacing to prevent handle overlap
    filtered = [0]
    for x in sampled_indices[1:-1]:
        if x - filtered[-1] >= min_spacing and (w - 1 - x) >= min_spacing:
            filtered.append(int(x))
    filtered.append(w - 1)

    while len(filtered) < target_count:
        diffs = np.diff(filtered)
        max_gap_idx = int(np.argmax(diffs))
        if diffs[max_gap_idx] <= min_spacing:
            break
        mid = (filtered[max_gap_idx] + filtered[max_gap_idx + 1]) // 2
        filtered.insert(max_gap_idx + 1, mid)

    return np.array(filtered, dtype=int)


def enrich_nodes_for_ablation(
    curve: np.ndarray,
    base_indices: np.ndarray,
    click_x: int,
    modified_mask: np.ndarray,
    max_extra_nodes: int = 6,
    min_spacing: int = 2
) -> tuple[np.ndarray, int]:
    """
    Evaluates an OCT boundary contour following background ablation to identify
    important anatomical and surgical features (sharp peaks, troughs, ablated notches,
    or significant spline approximation residuals) that are not covered by the base node set.
    Inserts targeted anchor nodes at critical extrema to preserve fine-grained contour fidelity.
    """
    w = len(curve)
    current_nodes = set(int(x) for x in base_indices)

    # Detect local extrema (peaks and valleys) on smoothed curve
    y_smooth = cv2.GaussianBlur(curve.astype(np.float32).reshape(1, -1), (5, 1), 1.0).squeeze()

    from scipy.signal import find_peaks
    from scipy.interpolate import PchipInterpolator

    peaks, _ = find_peaks(y_smooth, distance=4, prominence=0.7)
    valleys, _ = find_peaks(-y_smooth, distance=4, prominence=0.7)
    all_extrema = sorted(list(set(np.concatenate([peaks, valleys])))) if (len(peaks) > 0 or len(valleys) > 0) else []

    added_count = 0

    # Priority 1: Check all local extrema in or near the ablated region or click_x
    for ext_x in all_extrema:
        if added_count >= max_extra_nodes:
            break
        is_near_edit = bool(modified_mask[ext_x] or abs(ext_x - click_x) <= 35)
        min_dist = min(abs(ext_x - nx) for nx in current_nodes)

        if is_near_edit and min_dist >= min_spacing:
            sorted_nodes = sorted(list(current_nodes))
            pchip = PchipInterpolator(sorted_nodes, curve[sorted_nodes])
            err_at_peak = abs(float(curve[ext_x]) - float(pchip(ext_x)))
            if err_at_peak >= 0.6:
                current_nodes.add(int(ext_x))
                added_count += 1

    # Priority 2: Greedy residual insertion for any remaining uncovered high-error regions
    for _ in range(max_extra_nodes - added_count):
        sorted_nodes = sorted(list(current_nodes))
        pchip = PchipInterpolator(sorted_nodes, curve[sorted_nodes])
        residuals = np.abs(curve - pchip(np.arange(w)))

        weights = 1.0 + 2.5 * modified_mask.astype(float)
        click_dist = np.abs(np.arange(w) - click_x)
        weights += 2.0 * np.exp(-(click_dist / 25.0) ** 2)

        weighted_res = residuals * weights

        for nx in current_nodes:
            weighted_res[max(0, nx - min_spacing) : min(w, nx + min_spacing + 1)] = 0.0

        worst_x = int(np.argmax(weighted_res))
        worst_err = residuals[worst_x]

        if worst_err < 1.0:
            break

        current_nodes.add(worst_x)
        added_count += 1

    final_nodes = np.array(sorted(list(current_nodes)), dtype=int)
    return final_nodes, added_count


def _compute_scan_vectors(cropper, p: Path) -> tuple[int, int, list[dict], list[dict], dict, int, int, int, int]:
    """Computes high-speed 48-point U-Net boundary vectors, bounding box, and automatic 2-point slanted lateral bounds."""
    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        return 0, 0, None, None, {"ymin": 0, "ymax": 0, "xmin": 0, "xmax": 0}, 0, 0, 0, 0

    # Pre-process white scanner banners & corner artifacts to pure black background
    img_bgr = detect_and_process_white_bars(img_bgr)
    h, w = img_bgr.shape[:2]

    y_top_vec = None
    y_bot_vec = None
    crop_box = {"ymin": 0, "ymax": h, "xmin": 0, "xmax": w}
    crop_left_top = 0
    crop_left_bot = 0
    crop_right_top = w
    crop_right_bot = w

    if cropper is not None and cropper.has_weights:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            probs_orig, y_t, y_b = cropper.predict_mask_and_vectors(gray, threshold=0.50)

            # Sample adaptive anchor points based on curvature and arc-length
            top_indices = sample_adaptive_contour_nodes(y_t, target_count=48)
            bot_indices = sample_adaptive_contour_nodes(y_b, target_count=48)
            y_top_vec = [{"x": int(x), "y": float(y_t[x])} for x in top_indices]
            y_bot_vec = [{"x": int(x), "y": float(y_b[x])} for x in bot_indices]

            margin = 15
            ymin = max(0, int(np.min(y_t)) - margin)
            ymax = min(h, int(np.max(y_b)) + margin)
            crop_box = {"ymin": ymin, "ymax": ymax, "xmin": 0, "xmax": w}

            # Automatically derive exact 2-point slanted lateral bounds
            crop_left_top, crop_left_bot, crop_right_top, crop_right_bot = detect_tissue_lateral_bounds(probs_orig, y_t, y_b)
        except Exception:
            pass

    if y_top_vec is None or y_bot_vec is None:
        sample_indices = np.linspace(0, w - 1, 48, dtype=int)
        y_top_vec = [{"x": int(x), "y": float(h * 0.25)} for x in sample_indices]
        y_bot_vec = [{"x": int(x), "y": float(h * 0.70)} for x in sample_indices]

    return w, h, y_top_vec, y_bot_vec, crop_box, crop_left_top, crop_left_bot, crop_right_top, crop_right_bot


def _background_prefetch_scans(scan_files: list[tuple[str, Path]], start_idx: int, count: int = 40):
    """Background worker that asynchronously pre-computes U-Net vectors for the next scans."""
    def worker():
        cropper = get_unet_cropper()
        if not cropper or not cropper.has_weights:
            return
        target_slice = scan_files[start_idx : start_idx + count]
        for sub, p in target_slice:
            key = f"{sub}/{p.name}"
            with _PREFETCH_LOCK:
                already_cached = key in _UNET_PREDICT_CACHE
            if not already_cached:
                try:
                    w, h, yt, yb, cb, clt, clb, crt, crb = _compute_scan_vectors(cropper, p)
                    with _PREFETCH_LOCK:
                        _UNET_PREDICT_CACHE[key] = {
                            "width": w, "height": h,
                            "y_top_points": yt, "y_bot_points": yb,
                            "crop_box": cb,
                            "crop_left_top": clt, "crop_left_bot": clb,
                            "crop_right_top": crt, "crop_right_bot": crb
                        }
                except Exception:
                    pass
    threading.Thread(target=worker, daemon=True).start()


def extract_vectors_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Extracts top/bottom boundary vectors and lateral crop bounds from an existing 8-bit binary mask.
    Returns (y_top, y_bot, crop_left_top, crop_left_bot, crop_right_top, crop_right_bot).
    """
    h, w = mask.shape[:2]
    y_t = np.full(w, h * 0.25, dtype=np.float32)
    y_b = np.full(w, h * 0.75, dtype=np.float32)

    cols_with_signal = []
    for x in range(w):
        col = mask[:, x]
        nonzero = np.where(col > 0)[0]
        if len(nonzero) > 0:
            y_t[x] = float(nonzero[0])
            y_b[x] = float(nonzero[-1])
            cols_with_signal.append(x)

    if cols_with_signal:
        min_x = cols_with_signal[0]
        max_x = cols_with_signal[-1]
        if min_x > 0:
            y_t[:min_x] = y_t[min_x]
            y_b[:min_x] = y_b[min_x]
        if max_x < w - 1:
            y_t[max_x + 1:] = y_t[max_x]
            y_b[max_x + 1:] = y_b[max_x]
        clt, clb = min_x, min_x
        crt, crb = max_x + 1, max_x + 1
    else:
        clt, clb = 0, 0
        crt, crb = w, w

    return y_t, y_b, clt, clb, crt, crb


def get_crop_filter_queue(
    folder_name: Optional[str] = None,
    offset: int = 0,
    limit: int = 30,
    filter_mode: str = "all",
    source_dir: Optional[Path] = None,
    masked_dir: Optional[Path] = None
) -> dict:
    """
    Retrieves an ordered queue of scans from Classified/ with cached/prefetched U-Net boundary predictions.
    Supports filter_mode='curated' for reviewing and fixing already curated masks.
    """
    src_dir = Path(source_dir) if source_dir else get_source_dir()
    msk_dir = Path(masked_dir) if masked_dir else get_masked_dataset_dir()
    cropper = get_unet_cropper()

    manifest = get_curated_manifest(masked_dir=msk_dir)
    curated_keys = set(manifest.get("picked_keys", {}).keys())

    # Discover images across leaf disease folders
    scan_files: list[tuple[str, Path]] = []
    if folder_name and folder_name.upper() != "ALL":
        target_f = find_folder_path(folder_name, source_dir=src_dir)
        if target_f and target_f.exists():
            for p in sorted(target_f.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("."):
                    scan_files.append((folder_name, p))
    else:
        for sub_name in get_available_subfolders(source_dir=src_dir):
            target_f = find_folder_path(sub_name, source_dir=src_dir)
            if target_f and target_f.exists():
                for p in sorted(target_f.iterdir()):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("."):
                        scan_files.append((sub_name, p))

    # Prioritize or isolate scans based on filter_mode: 'all', 'curated', or 'uncurated'
    uncurated_scans = [s for s in scan_files if f"{s[0]}/{s[1].name}" not in curated_keys]
    curated_scans = [s for s in scan_files if f"{s[0]}/{s[1].name}" in curated_keys]

    if filter_mode == "curated":
        ordered_scans = curated_scans
    elif filter_mode == "uncurated":
        ordered_scans = uncurated_scans
    else:
        ordered_scans = uncurated_scans + curated_scans

    total_scans = len(ordered_scans)
    paged_scans = ordered_scans[offset : offset + limit]

    queue_items = []
    for sub, p in paged_scans:
        sample_key = f"{sub}/{p.name}"
        is_curated = sample_key in curated_keys

        with _PREFETCH_LOCK:
            cached_data = _UNET_PREDICT_CACHE.get(sample_key)

        if cached_data is not None:
            w = cached_data["width"]
            h = cached_data["height"]
            y_top_vec = cached_data["y_top_points"]
            y_bot_vec = cached_data["y_bot_points"]
            crop_box = cached_data["crop_box"]
            clt = cached_data.get("crop_left_top", 0)
            clb = cached_data.get("crop_left_bot", 0)
            crt = cached_data.get("crop_right_top", w)
            crb = cached_data.get("crop_right_bot", w)
        elif is_curated and (msk_dir / "Masks" / sub / f"{p.stem}.png").exists():
            # Load vectors directly from the previously approved curated ground truth mask
            mask_p = msk_dir / "Masks" / sub / f"{p.stem}.png"
            cur_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if cur_mask is not None:
                h, w = cur_mask.shape[:2]
                y_t, y_b, clt, clb, crt, crb = extract_vectors_from_mask(cur_mask)
                top_indices = sample_adaptive_contour_nodes(y_t, target_count=48)
                bot_indices = sample_adaptive_contour_nodes(y_b, target_count=48)
                y_top_vec = [{"x": int(x), "y": round(float(y_t[x]), 2)} for x in top_indices]
                y_bot_vec = [{"x": int(x), "y": round(float(y_b[x]), 2)} for x in bot_indices]
                margin = 15
                ymin = max(0, int(np.min(y_t)) - margin)
                ymax = min(h, int(np.max(y_b)) + margin)
                crop_box = {"ymin": ymin, "ymax": ymax, "xmin": 0, "xmax": w}
                with _PREFETCH_LOCK:
                    _UNET_PREDICT_CACHE[sample_key] = {
                        "width": w, "height": h,
                        "y_top_points": y_top_vec, "y_bot_points": y_bot_vec,
                        "crop_box": crop_box,
                        "crop_left_top": clt, "crop_left_bot": clb,
                        "crop_right_top": crt, "crop_right_bot": crb
                    }
            else:
                w, h, y_top_vec, y_bot_vec, crop_box, clt, clb, crt, crb = _compute_scan_vectors(cropper, p)
        else:
            w, h, y_top_vec, y_bot_vec, crop_box, clt, clb, crt, crb = _compute_scan_vectors(cropper, p)
            with _PREFETCH_LOCK:
                _UNET_PREDICT_CACHE[sample_key] = {
                    "width": w, "height": h,
                    "y_top_points": y_top_vec, "y_bot_points": y_bot_vec,
                    "crop_box": crop_box,
                    "crop_left_top": clt, "crop_left_bot": clb,
                    "crop_right_top": crt, "crop_right_bot": crb
                }

        queue_items.append({
            "folder": sub,
            "filename": p.name,
            "sample_key": sample_key,
            "width": w,
            "height": h,
            "is_curated": is_curated,
            "y_top_points": y_top_vec,
            "y_bot_points": y_bot_vec,
            "crop_box": crop_box,
            "crop_left": int((clt + clb) / 2),
            "crop_right": int((crt + crb) / 2),
            "crop_left_top": clt,
            "crop_left_bot": clb,
            "crop_right_top": crt,
            "crop_right_bot": crb,
            "image_url": f"/api/image_raw_direct?subfolder={urllib.parse.quote(sub)}&filename={urllib.parse.quote(p.name)}",
        })

    # Trigger background pre-computation for the next 40 scans ahead
    _background_prefetch_scans(scan_files, offset + limit, count=40)

    return {
        "folder": folder_name or "ALL",
        "total": total_scans,
        "offset": offset,
        "limit": limit,
        "items": queue_items,
        "curated_total": len(curated_keys),
        "filter_mode": filter_mode
    }


def save_curated_crop_from_vectors(
    folder_name: str,
    filename: str,
    y_top_points: list[dict],
    y_bot_points: list[dict],
    crop_left: int = 0,
    crop_right: Optional[int] = None,
    crop_left_top: Optional[int] = None,
    crop_left_bot: Optional[int] = None,
    crop_right_top: Optional[int] = None,
    crop_right_bot: Optional[int] = None,
    source_dir: Optional[Path] = None,
    masked_dir: Optional[Path] = None
) -> dict:
    """
    Constructs a binary ground-truth mask from user-verified top/bottom boundary vectors
    and 2-point slanted lateral crop boundaries [crop_left_top, crop_left_bot, crop_right_top, crop_right_bot],
    committing the pair into Classified-masked/.
    """
    src_dir = Path(source_dir) if source_dir else get_source_dir()
    msk_dir = Path(masked_dir) if masked_dir else get_masked_dataset_dir()

    img_path = find_image_path(folder_name, filename, source_dir=src_dir)
    if not img_path or not img_path.exists():
        raise FileNotFoundError(f"Source scan not found for '{folder_name}/{filename}'")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not read source scan: {img_path}")

    # Ensure saved curated scan has white scanner borders zeroed to black background
    img_bgr = detect_and_process_white_bars(img_bgr)

    h, w = img_bgr.shape[:2]

    # Resolve 2-Point Slanted Lateral Crop Boundaries
    c_lt = max(0, min(w - 2, int(crop_left_top if crop_left_top is not None else crop_left)))
    c_lb = max(0, min(w - 2, int(crop_left_bot if crop_left_bot is not None else crop_left)))
    c_rt = min(w, max(c_lt + 2, int(crop_right_top if crop_right_top is not None else (crop_right if crop_right is not None else w))))
    c_rb = min(w, max(c_lb + 2, int(crop_right_bot if crop_right_bot is not None else (crop_right if crop_right is not None else w))))

    # Sort and interpolate vectors smoothly across scan width W using Monotonic Cubic PCHIP
    from scipy.interpolate import PchipInterpolator

    # Ensure strictly sorted unique X points
    top_map = {}
    for pt in y_top_points:
        top_map[int(pt["x"])] = float(pt["y"])
    bot_map = {}
    for pt in y_bot_points:
        bot_map[int(pt["x"])] = float(pt["y"])

    top_sorted = sorted(top_map.items())
    bot_sorted = sorted(bot_map.items())

    top_xs = [p[0] for p in top_sorted]
    top_ys = [p[1] for p in top_sorted]
    bot_xs = [p[0] for p in bot_sorted]
    bot_ys = [p[1] for p in bot_sorted]

    full_x = np.arange(w)

    if len(top_xs) >= 2:
        pchip_top = PchipInterpolator(top_xs, top_ys, extrapolate=True)
        interp_top = pchip_top(full_x)
    else:
        interp_top = np.interp(full_x, top_xs, top_ys)

    if len(bot_xs) >= 2:
        pchip_bot = PchipInterpolator(bot_xs, bot_ys, extrapolate=True)
        interp_bot = pchip_bot(full_x)
    else:
        interp_bot = np.interp(full_x, bot_xs, bot_ys)

    # Enforce non-intersecting ordering and valid boundaries
    interp_top = np.clip(interp_top, 0, h - 1)
    interp_bot = np.clip(interp_bot, interp_top + 10, h - 1)

    # Generate 8-bit binary mask (0 = background, 255 = retinal tissue within slanted lateral bounds)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Forward top boundary points
    poly_pts = []
    for x in range(w):
        poly_pts.append([x, int(round(interp_top[x]))])
    # 2. Reverse bottom boundary points
    for x in range(w - 1, -1, -1):
        poly_pts.append([x, int(round(interp_bot[x]))])

    cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 255)

    # 3. Apply Left Slanted Curtain (Zero uncropped region left of line (c_lt, 0) -> (c_lb, h))
    if c_lt > 0 or c_lb > 0:
        left_poly = np.array([[0, 0], [c_lt, 0], [c_lb, h], [0, h]], dtype=np.int32)
        cv2.fillPoly(mask, [left_poly], 0)

    # 4. Apply Right Slanted Curtain (Zero uncropped region right of line (c_rt, 0) -> (c_rb, h))
    if c_rt < w or c_rb < w:
        right_poly = np.array([[c_rt, 0], [w, 0], [w, h], [c_rb, h]], dtype=np.int32)
        cv2.fillPoly(mask, [right_poly], 0)

    # 5. Pitch-Black Border & Non-Acquisition Mask Cleansing
    # Trims entire columns on lateral margins that have zero optical signal (<= 8),
    # while preserving internal fluid cysts (DME) and vessel shadows.
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr

    col_max = np.max(gray_img, axis=0)
    # Detect dead columns from left
    left_dead_idx = 0
    while left_dead_idx < w and col_max[left_dead_idx] <= 8:
        left_dead_idx += 1
    if left_dead_idx > 0:
        mask[:, :left_dead_idx] = 0

    # Detect dead columns from right
    right_dead_idx = w - 1
    while right_dead_idx >= 0 and col_max[right_dead_idx] <= 8:
        right_dead_idx -= 1
    if right_dead_idx < w - 1:
        mask[:, right_dead_idx + 1:] = 0

    # Directory structures
    img_out_dir = msk_dir / "Images" / folder_name
    mask_out_dir = msk_dir / "Masks" / folder_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    out_img_p = img_out_dir / f"{stem}.png"
    out_mask_p = mask_out_dir / f"{stem}.png"

    cv2.imwrite(str(out_img_p), img_bgr)
    cv2.imwrite(str(out_mask_p), mask)

    # Update manifest
    manifest_p = msk_dir / "curated_manifest.json"
    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                samples = manifest_data.get("samples", {})
        except Exception:
            samples = {}
    else:
        samples = {}

    sample_key = f"{folder_name}/{filename}"
    samples[sample_key] = {
        "folder": folder_name,
        "filename": filename,
        "image_path": str(out_img_p.relative_to(msk_dir)),
        "mask_path": str(out_mask_p.relative_to(msk_dir)),
        "width": w,
        "height": h,
        "timestamp": time.time(),
        "source": "swiping_studio"
    }

    manifest_data = {"version": "1.0", "samples": samples, "total_count": len(samples)}
    _sync_manifest_files(msk_dir, manifest_data)

    return {
        "status": "success",
        "sample_key": sample_key,
        "total_curated": len(samples),
        "message": f"Saved curated mask for {filename}"
    }


def get_curation_statistics(
    source_dir: Optional[Path] = None,
    masked_dir: Optional[Path] = None
) -> dict:
    """Aggregates curated vs total scan counts per disease folder."""
    src_dir = Path(source_dir) if source_dir else get_source_dir()
    msk_dir = Path(masked_dir) if masked_dir else get_masked_dataset_dir()

    manifest = get_curated_manifest(masked_dir=msk_dir)
    curated_keys = set(manifest.get("picked_keys", {}).keys())

    folder_stats = {}
    if src_dir.exists():
        for sub_name in get_available_subfolders(source_dir=src_dir):
            target_f = find_folder_path(sub_name, source_dir=src_dir)
            if target_f and target_f.exists():
                total_in_sub = sum(1 for p in target_f.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("."))
                curated_in_sub = sum(1 for k in curated_keys if k.startswith(f"{sub_name}/"))
                folder_stats[sub_name] = {
                    "total": total_in_sub,
                    "curated": curated_in_sub,
                    "pct": (curated_in_sub / total_in_sub * 100) if total_in_sub > 0 else 0.0
                }

    total_scans = sum(v["total"] for v in folder_stats.values())
    total_curated = len(curated_keys)

    return {
        "total_scans": total_scans,
        "total_curated": total_curated,
        "overall_pct": (total_curated / total_scans * 100) if total_scans > 0 else 0.0,
        "folders": folder_stats
    }


_RETRAIN_STOP_FLAG = False


def stop_unet_retraining() -> dict:
    """Signals background retraining worker to halt gracefully."""
    global _RETRAIN_STOP_FLAG, _RETRAIN_STATE
    with _RETRAIN_LOCK:
        if not _RETRAIN_STATE["running"]:
            return {"status": "not_running", "message": "No training job is currently running."}
        _RETRAIN_STOP_FLAG = True
        _RETRAIN_STATE["message"] = "Stopping training gracefully... Preserving current best checkpoint."
        _RETRAIN_STATE["phase"] = "stopping"
        _RETRAIN_STATE["updated_at"] = time.time()
        return {"status": "stopping", "message": "Stop signal sent to retraining worker.", "state": _RETRAIN_STATE}


def get_training_history() -> dict:
    """Reads the persistent training history ledger from checkpoints directory."""
    history_file = _REPO_ROOT / "checkpoints" / "segmentation" / "tissue_cropper" / "training_history.json"
    if not history_file.exists():
        history_file = _REPO_ROOT / "models_suite" / "tissue_cropper" / "checkpoints" / "training_history.json"

    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed to read training history: {e}", "rounds": []}
    return {"model_name": "Attention U-Net Tissue Cropper", "total_rounds": 0, "rounds": []}


def trigger_unet_retraining_async(epochs: int = 10, lr: float = 2e-4) -> dict:
    """Starts U-Net fine-tuning loop in a dedicated background worker thread."""
    global _RETRAIN_STATE, _RETRAIN_STOP_FLAG
    with _RETRAIN_LOCK:
        if _RETRAIN_STATE["running"]:
            return {"status": "already_running", "message": "Training is already in progress.", "state": _RETRAIN_STATE}

        _RETRAIN_STOP_FLAG = False
        _RETRAIN_STATE["running"] = True
        _RETRAIN_STATE["epoch"] = 0
        _RETRAIN_STATE["total_epochs"] = epochs
        _RETRAIN_STATE["batch"] = 0
        _RETRAIN_STATE["total_batches"] = 0
        _RETRAIN_STATE["batch_loss"] = 0.0
        _RETRAIN_STATE["train_loss"] = 0.0
        _RETRAIN_STATE["val_loss"] = 0.0
        _RETRAIN_STATE["val_dice"] = 0.0
        _RETRAIN_STATE["val_iou"] = 0.0
        _RETRAIN_STATE["phase"] = "initializing"
        _RETRAIN_STATE["message"] = "Initializing training pipeline..."
        _RETRAIN_STATE["started_at"] = time.time()
        _RETRAIN_STATE["updated_at"] = time.time()

    def _train_worker():
        global _RETRAIN_STATE
        try:
            seg_dir = str(_REPO_ROOT / "training" / "segmentation")
            if seg_dir not in sys.path:
                sys.path.insert(0, seg_dir)
            if str(_REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(_REPO_ROOT))

            from train_tissue_cropper.train import train as run_training

            def _progress_cb(phase="training", epoch=0, total_epochs=0, batch=0, total_batches=0,
                             batch_loss=0.0, train_loss=0.0, val_loss=0.0, val_dice=0.0, val_iou=0.0, **kwargs):
                with _RETRAIN_LOCK:
                    _RETRAIN_STATE["phase"] = phase
                    _RETRAIN_STATE["epoch"] = epoch
                    _RETRAIN_STATE["total_epochs"] = total_epochs
                    _RETRAIN_STATE["batch"] = batch
                    _RETRAIN_STATE["total_batches"] = total_batches
                    _RETRAIN_STATE["batch_loss"] = batch_loss
                    _RETRAIN_STATE["train_loss"] = train_loss
                    _RETRAIN_STATE["val_loss"] = val_loss
                    _RETRAIN_STATE["val_dice"] = val_dice
                    _RETRAIN_STATE["val_iou"] = val_iou
                    _RETRAIN_STATE["updated_at"] = time.time()

                    if phase == "training":
                        pct = (batch / max(1, total_batches)) * 100
                        _RETRAIN_STATE["message"] = f"Epoch {epoch}/{total_epochs} | Training batch {batch}/{total_batches} ({pct:.0f}%) | Batch Loss: {batch_loss:.4f}"
                    elif phase == "validating":
                        _RETRAIN_STATE["message"] = f"Epoch {epoch}/{total_epochs} | Validating batch {batch}/{total_batches}..."
                    elif phase == "epoch_complete":
                        _RETRAIN_STATE["message"] = f"Epoch {epoch}/{total_epochs} complete | Val Dice: {val_dice:.4f} ({val_dice*100:.1f}%)"
                    elif phase == "early_stopped":
                        _RETRAIN_STATE["message"] = f"Epoch {epoch}/{total_epochs} | Early stopped. Best Val Dice: {val_dice:.4f}"

            best_d = run_training(
                epochs=epochs,
                learning_rate=lr,
                resume_best=True,
                patience=4,
                min_delta=0.001,
                smoke_test=False,
                progress_callback=_progress_cb,
                stop_check_fn=lambda: _RETRAIN_STOP_FLAG
            )

            # Reload updated weights in singleton croppers and invalidate prediction cache
            global _UNET_CROPPER_INSTANCE, _UNET_CROPPER_CPU_INSTANCE
            _UNET_CROPPER_INSTANCE = None
            _UNET_CROPPER_CPU_INSTANCE = None
            clear_unet_predict_cache()
            get_unet_cropper(force_reload=True, prefer_cpu_if_training=False)

            with _RETRAIN_LOCK:
                _RETRAIN_STATE["running"] = False
                _RETRAIN_STATE["phase"] = "complete"
                _RETRAIN_STATE["message"] = f"Training completed successfully! Best Val Dice: {best_d:.4f} ({best_d*100:.1f}%)"
                _RETRAIN_STATE["updated_at"] = time.time()

        except Exception as e:
            sys.stderr.write(f"[RetrainWorker] Error: {e}\n")
            with _RETRAIN_LOCK:
                _RETRAIN_STATE["running"] = False
                _RETRAIN_STATE["phase"] = "error"
                _RETRAIN_STATE["message"] = f"Training failed: {str(e)}"
                _RETRAIN_STATE["updated_at"] = time.time()

    th = threading.Thread(target=_train_worker, daemon=True)
    th.start()

    return {"status": "started", "message": "U-Net retraining started in background.", "state": _RETRAIN_STATE}


def get_unet_retrain_status() -> dict:
    """Returns the live status of the background U-Net retraining process."""
    with _RETRAIN_LOCK:
        return dict(_RETRAIN_STATE)


def ablate_tuning_background(
    folder: str,
    filename: str,
    click_x: int,
    click_y: int,
    y_top_points: list[dict],
    y_bot_points: list[dict],
    radius_x: int = 0,
    source_dir: Optional[Path] = None
) -> dict:
    """
    Surgically ablates background noise (such as vitreous haze, supra-ILM artifacts, or choroidal plateaus)
    on an active tuning scan from Classified/ by adjusting its boundary vectors.
    Updates the prediction cache and returns updated vector coordinates and ablated pixel estimate.
    """
    import scipy.ndimage as ndi

    src_dir = Path(source_dir) if source_dir else get_source_dir()
    img_path = find_image_path(folder, filename, source_dir=src_dir)
    if not img_path or not img_path.exists():
        return {"status": "error", "message": f"Source scan not found for {folder}/{filename}"}

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return {"status": "error", "message": "Failed to decode scan image"}

    img_bgr = detect_and_process_white_bars(img_bgr)
    h, w = img_bgr.shape[:2]
    cx = int(np.clip(click_x, 0, w - 1))
    cy = int(np.clip(click_y, 0, h - 1))

    # Reconstruct interpolated top and bottom vector depths across scan width
    top_map = {int(pt["x"]): float(pt["y"]) for pt in y_top_points}
    bot_map = {int(pt["x"]): float(pt["y"]) for pt in y_bot_points}
    top_xs = sorted(top_map.keys())
    bot_xs = sorted(bot_map.keys())
    full_x = np.arange(w)
    interp_top = np.interp(full_x, top_xs, [top_map[x] for x in top_xs])
    interp_bot = np.interp(full_x, bot_xs, [bot_map[x] for x in bot_xs])

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    is_supra_ilm = (cy <= interp_top[cx] + 25)
    is_sub_rpe = (cy >= interp_bot[cx] - 25)

    if radius_x > 0:
        x1 = max(0, cx - int(radius_x))
        x2 = min(w, cx + int(radius_x))
    else:
        x1 = 0
        x2 = w

    new_top = np.copy(interp_top)
    new_bot = np.copy(interp_bot)
    pixels_ablated = 0

    if is_sub_rpe:
        # Bottom-up sub-RPE noise removal: detect negative gradient peak of RPE
        rpe_curve = np.zeros(w, dtype=np.float32)
        valid_cols = []
        for x in range(w):
            col_ilm = int(np.clip(interp_top[x], 0, h - 1))
            col_bot = int(np.clip(interp_bot[x], col_ilm + 5, h))
            sub_ys = np.arange(col_ilm + 10, min(h, col_bot + 40))
            if len(sub_ys) == 0:
                continue
            neg_peaks = np.where((sobely[sub_ys, x] < -20) & (blur[sub_ys, x] > 35))[0]
            if len(neg_peaks) > 0:
                rpe_curve[x] = sub_ys[neg_peaks[-1]]
                valid_cols.append(x)

        if len(valid_cols) > 5:
            rpe_interp = np.interp(full_x, valid_cols, rpe_curve[valid_cols])
            rpe_smooth = ndi.median_filter(rpe_interp, size=15)
            rpe_smooth = ndi.gaussian_filter1d(rpe_smooth, sigma=4.0)

            for x in range(x1, x2):
                cutoff_y = float(np.clip(rpe_smooth[x] + 12, interp_top[x] + 10, h - 1))
                if cutoff_y < new_bot[x]:
                    pixels_ablated += int(new_bot[x] - cutoff_y)
                    new_bot[x] = cutoff_y
        else:
            # Fallback: snap bottom boundary upward towards click level
            for x in range(x1, x2):
                target_y = float(np.clip(cy - 5, interp_top[x] + 10, h - 1))
                if target_y < new_bot[x]:
                    pixels_ablated += int(new_bot[x] - target_y)
                    new_bot[x] = target_y

    elif is_supra_ilm:
        # Top-down supra-ILM noise removal: detect positive gradient peak of ILM
        ilm_curve = np.zeros(w, dtype=np.float32)
        valid_cols = []
        for x in range(w):
            col_ilm = int(np.clip(interp_top[x], 0, h - 1))
            search_ys = np.arange(max(0, col_ilm - 30), min(h - 1, col_ilm + 40))
            if len(search_ys) == 0:
                continue
            peaks = np.where((sobely[search_ys, x] > 30) & (blur[search_ys, x] > 45))[0]
            if len(peaks) > 0:
                ilm_curve[x] = search_ys[peaks[0]]
                valid_cols.append(x)

        if len(valid_cols) > 5:
            ilm_interp = np.interp(full_x, valid_cols, ilm_curve[valid_cols])
            ilm_smooth = ndi.median_filter(ilm_interp, size=15)
            ilm_smooth = ndi.gaussian_filter1d(ilm_smooth, sigma=4.0)

            for x in range(x1, x2):
                cutoff_y = float(np.clip(ilm_smooth[x] - 2, 0, interp_bot[x] - 10))
                if cutoff_y > new_top[x]:
                    pixels_ablated += int(cutoff_y - new_top[x])
                    new_top[x] = cutoff_y
        else:
            for x in range(x1, x2):
                target_y = float(np.clip(cy + 5, 0, interp_bot[x] - 10))
                if target_y > new_top[x]:
                    pixels_ablated += int(target_y - new_top[x])
                    new_top[x] = target_y
    else:
        # User clicked inside the retina; snap closer boundary to eliminate edge cyst/snout
        dist_to_top = abs(cy - interp_top[cx])
        dist_to_bot = abs(cy - interp_bot[cx])
        scoop_rad = int(radius_x) if radius_x > 0 else 40
        sx1 = max(0, cx - scoop_rad)
        sx2 = min(w, cx + scoop_rad)
        if dist_to_top <= dist_to_bot:
            for x in range(sx1, sx2):
                falloff = np.cos((x - cx) / scoop_rad * (np.pi / 2)) ** 2 if radius_x <= 0 else 1.0
                target_y = float(np.clip(cy + 2, 0, interp_bot[x] - 10))
                delta = (target_y - new_top[x]) * falloff
                if delta > 0:
                    pixels_ablated += int(delta)
                    new_top[x] = np.clip(new_top[x] + delta, 0, interp_bot[x] - 10)
        else:
            for x in range(sx1, sx2):
                falloff = np.cos((x - cx) / scoop_rad * (np.pi / 2)) ** 2 if radius_x <= 0 else 1.0
                target_y = float(np.clip(cy - 2, interp_top[x] + 10, h - 1))
                delta = (new_bot[x] - target_y) * falloff
                if delta > 0:
                    pixels_ablated += int(delta)
                    new_bot[x] = np.clip(new_bot[x] - delta, interp_top[x] + 10, h - 1)

    # Detect which boundaries were modified
    top_mod_mask = (np.abs(new_top - interp_top) > 0.5)
    bot_mod_mask = (np.abs(new_bot - interp_bot) > 0.5)
    top_modified = bool(np.any(top_mod_mask))
    bot_modified = bool(np.any(bot_mod_mask))

    top_added = 0
    bot_added = 0

    if top_modified:
        base_top = sample_adaptive_contour_nodes(new_top, target_count=max(48, len(y_top_points)))
        final_top, top_added = enrich_nodes_for_ablation(
            new_top, base_top, click_x=cx, modified_mask=top_mod_mask, max_extra_nodes=6, min_spacing=2
        )
        updated_top_points = [{"x": int(x), "y": round(float(new_top[x]), 2)} for x in final_top]
    else:
        updated_top_points = y_top_points

    if bot_modified:
        base_bot = sample_adaptive_contour_nodes(new_bot, target_count=max(48, len(y_bot_points)))
        final_bot, bot_added = enrich_nodes_for_ablation(
            new_bot, base_bot, click_x=cx, modified_mask=bot_mod_mask, max_extra_nodes=6, min_spacing=2
        )
        updated_bot_points = [{"x": int(x), "y": round(float(new_bot[x]), 2)} for x in final_bot]
    else:
        updated_bot_points = y_bot_points

    total_added = (top_added if top_modified else 0) + (bot_added if bot_modified else 0)

    # Update in-memory cache
    sample_key = f"{folder}/{filename}"
    with _PREFETCH_LOCK:
        if sample_key in _UNET_PREDICT_CACHE:
            _UNET_PREDICT_CACHE[sample_key]["y_top_points"] = updated_top_points
            _UNET_PREDICT_CACHE[sample_key]["y_bot_points"] = updated_bot_points

    return {
        "status": "success",
        "folder": folder,
        "filename": filename,
        "pixels_ablated": max(pixels_ablated, 1),
        "nodes_added": total_added,
        "y_top_points": updated_top_points,
        "y_bot_points": updated_bot_points,
    }


def rerun_tuning_unet_on_window(
    folder: str,
    filename: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    y_top_points: list[dict],
    y_bot_points: list[dict],
    threshold: float = 0.50,
    source_dir: Optional[Path] = None
) -> dict:
    """
    Reruns Attention U-Net inference on a targeted sub-window [x1:x2, y1:y2] of the raw scan.
    Splices the predicted top and bottom vector depths in that horizontal window into the existing vectors.
    Updates the prediction cache and returns updated vector coordinates and changed pixel count.
    """
    src_dir = Path(source_dir) if source_dir else get_source_dir()
    img_path = find_image_path(folder, filename, source_dir=src_dir)
    if not img_path or not img_path.exists():
        return {"status": "error", "message": f"Source scan not found for {folder}/{filename}"}

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return {"status": "error", "message": "Failed to decode scan image"}

    img_bgr = detect_and_process_white_bars(img_bgr)
    h, w = img_bgr.shape[:2]

    wx1 = int(np.clip(min(x1, x2), 0, w - 1))
    wx2 = int(np.clip(max(x1, x2), 0, w))
    wy1 = int(np.clip(min(y1, y2), 0, h - 1))
    wy2 = int(np.clip(max(y1, y2), 0, h))

    if wx2 - wx1 < 10 or wy2 - wy1 < 10:
        return {"status": "error", "message": "Window too small (must be at least 10x10 px)"}

    cropper = get_unet_cropper()
    if cropper is None or not cropper.has_weights:
        return {"status": "error", "message": "Attention U-Net model weights not available"}

    window_crop = img_bgr[wy1:wy2, wx1:wx2]
    window_gray = cv2.cvtColor(window_crop, cv2.COLOR_BGR2GRAY)

    probs, win_yt, win_yb = cropper.predict_mask_and_vectors(window_gray, threshold=threshold)

    # Reconstruct interpolated top and bottom vector depths across scan width
    top_map = {int(pt["x"]): float(pt["y"]) for pt in y_top_points}
    bot_map = {int(pt["x"]): float(pt["y"]) for pt in y_bot_points}
    top_xs = sorted(top_map.keys())
    bot_xs = sorted(bot_map.keys())
    full_x = np.arange(w)
    interp_top = np.interp(full_x, top_xs, [top_map[x] for x in top_xs])
    interp_bot = np.interp(full_x, bot_xs, [bot_map[x] for x in bot_xs])

    new_top = np.copy(interp_top)
    new_bot = np.copy(interp_bot)
    pixels_changed = 0

    win_w = wx2 - wx1
    for i in range(win_w):
        gx = wx1 + i
        pred_top_y = float(wy1 + win_yt[i])
        pred_bot_y = float(wy1 + win_yb[i])

        # Ensure valid non-inverted bounds
        pred_top_y = float(np.clip(pred_top_y, 0, h - 1))
        pred_bot_y = float(np.clip(pred_bot_y, pred_top_y + 10, h - 1))

        pixels_changed += int(abs(new_top[gx] - pred_top_y) + abs(new_bot[gx] - pred_bot_y))
        new_top[gx] = pred_top_y
        new_bot[gx] = pred_bot_y

    # Resample updated vectors onto adaptive anchor points
    top_indices = sample_adaptive_contour_nodes(new_top, target_count=48)
    bot_indices = sample_adaptive_contour_nodes(new_bot, target_count=48)
    updated_top_points = [{"x": int(x), "y": round(float(new_top[x]), 2)} for x in top_indices]
    updated_bot_points = [{"x": int(x), "y": round(float(new_bot[x]), 2)} for x in bot_indices]

    sample_key = f"{folder}/{filename}"
    with _PREFETCH_LOCK:
        if sample_key in _UNET_PREDICT_CACHE:
            _UNET_PREDICT_CACHE[sample_key]["y_top_points"] = updated_top_points
            _UNET_PREDICT_CACHE[sample_key]["y_bot_points"] = updated_bot_points

    return {
        "status": "success",
        "folder": folder,
        "filename": filename,
        "pixels_changed": pixels_changed,
        "y_top_points": updated_top_points,
        "y_bot_points": updated_bot_points,
        "window_bounds": [wx1, wy1, wx2, wy2]
    }

