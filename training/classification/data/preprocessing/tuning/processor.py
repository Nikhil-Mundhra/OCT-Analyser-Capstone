"""
data/preprocessing/tuning/processor.py

Dataset subfolder discovery, sample caching, and scan image preprocessing pipelines.
"""

import os
from pathlib import Path
import random
import sys
from typing import Optional
import cv2
import numpy as np

from data.preprocessing.tuning.boundaries import (
    generate_tissue_mask_custom,
    letterbox_pad_and_resize,
    project_and_downsample_vectors,
    detect_choroidal_caverns,
    detect_choroidal_holes,
)
from data.preprocessing.white_bars import (
    detect_and_process_white_bars,
    detect_and_remove_compass_artifacts,
)

SOURCE_DIR = Path(
    os.environ.get("SOURCE_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified")
)
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed-Otsu"
    )
)

SFCM_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
FOLDER_SAMPLES_CACHE: dict[str, list[Path]] = {}


def get_source_dir() -> Path:
    """Dynamically resolves SOURCE_DIR, respecting runtime/test monkeypatching."""
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server", "data.preprocessing.tuning.processor"):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            val = getattr(mod, "SOURCE_DIR", None)
            if val is not None and val != SOURCE_DIR:
                return Path(val)
    return SOURCE_DIR


def get_output_dir() -> Path:
    """Dynamically resolves OUTPUT_DIR, respecting runtime/test monkeypatching."""
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server", "data.preprocessing.tuning.processor"):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            val = getattr(mod, "OUTPUT_DIR", None)
            if val is not None and val != OUTPUT_DIR:
                return Path(val)
    return OUTPUT_DIR


def get_folder_samples_cache() -> dict[str, list[Path]]:
    """Dynamically resolves FOLDER_SAMPLES_CACHE, respecting runtime/test monkeypatching."""
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server", "data.preprocessing.tuning.processor"):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            val = getattr(mod, "FOLDER_SAMPLES_CACHE", None)
            if val is not None and isinstance(val, dict) and val is not FOLDER_SAMPLES_CACHE:
                return val
    return FOLDER_SAMPLES_CACHE


def get_sfcm_cache() -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Dynamically resolves SFCM_CACHE, respecting runtime/test monkeypatching."""
    for mod_name in ("tuning_server", "data.preprocessing.tuning.server", "data.preprocessing.tuning.processor"):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            val = getattr(mod, "SFCM_CACHE", None)
            if val is not None and isinstance(val, dict) and val is not SFCM_CACHE:
                return val
    return SFCM_CACHE


def find_folder_path(folder_name: str | Path, source_dir: Optional[Path] = None) -> Path | None:
    """Finds the absolute path of a dataset subfolder, prioritizing directories with image files."""
    if isinstance(folder_name, Path):
        if folder_name.exists():
            return folder_name
        folder_name = folder_name.name

    active_source = source_dir or get_source_dir()
    if not active_source.exists():
        return None

    # Check direct child first if it contains images
    direct = active_source / folder_name
    if direct.is_dir():
        direct_files = list(direct.glob("*.jp*g")) + list(direct.glob("*.png"))
        if direct_files:
            return direct

    # Search all matching directories across source hierarchy
    matches = [d for d in active_source.rglob(folder_name) if d.is_dir() and not d.name.startswith(".")]
    if not matches:
        return None

    # Prioritize matching directories that actually contain valid scan images
    matches_with_counts = []
    for d in matches:
        count = len(list(d.glob("*.jp*g")) + list(d.glob("*.png")))
        if count == 0:
            count = len(list(d.rglob("*.jp*g")) + list(d.rglob("*.png")))
        matches_with_counts.append((d, count))

    matches_with_counts.sort(key=lambda x: x[1], reverse=True)
    return matches_with_counts[0][0]


def find_image_path(folder_name: str | Path, filename: str) -> Path | None:
    """Finds the path of a specific image by filename inside a folder."""
    folder_path = find_folder_path(folder_name)
    if folder_path:
        direct = folder_path / filename
        if direct.exists():
            return direct
        matches = list(folder_path.rglob(filename))
        return matches[0] if matches else None
    return None


def get_available_subfolders(source_dir: Optional[Path] = None) -> list[str]:
    """Finds all leaf directories containing valid scan images."""
    active_source = source_dir or get_source_dir()
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
    """Reprocesses a 4-image batch sample for a given folder and returns JSON payload."""
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


def render_boundary_overlay(
    image_bgr: np.ndarray,
    y_top: np.ndarray,
    y_rpe: np.ndarray,
    y_sfcm: np.ndarray,
    holes: Optional[list] = None,
    caverns: Optional[list] = None,
) -> np.ndarray:
    """
    Renders an anatomical multi-surface diagnostic overlay on an RGB/BGR OCT scan.
    - Cyan (#00FFFF): Inner Limiting Membrane (ILM)
    - Green (#00FF00): Retinal Pigment Epithelium (RPE)
    - Orange (#FFA500): Choroid Scleral Interface (SFCM Floor)
    - Pink (#FF1493): Choroidal Holes
    - Purple (#8A2BE2): Choroidal Caverns
    """
    overlay = image_bgr.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    h, w = overlay.shape[:2]
    for x in range(w):
        if y_top is not None and 0 <= int(y_top[x]) < h:
            cv2.circle(overlay, (x, int(y_top[x])), 1, (255, 255, 0), -1)   # Cyan: ILM
        if y_rpe is not None and 0 <= int(y_rpe[x]) < h:
            cv2.circle(overlay, (x, int(y_rpe[x])), 1, (0, 255, 0), -1)     # Green: RPE
        if y_sfcm is not None and 0 <= int(y_sfcm[x]) < h:
            cv2.circle(overlay, (x, int(y_sfcm[x])), 1, (0, 165, 255), -1) # Orange: Choroid

    if holes:
        for hole in holes:
            if "contour" in hole:
                cv2.drawContours(overlay, [hole["contour"]], -1, (255, 20, 147), 1)
            elif "bbox" in hole:
                bx, by, bw, bh = hole["bbox"]
                cv2.rectangle(overlay, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (255, 20, 147), 1)

    if caverns:
        for c in caverns:
            if "bbox" in c:
                bx, by, bw, bh = c["bbox"]
                cv2.rectangle(overlay, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (226, 43, 138), 1)

    return overlay


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
        # Attempt lookup via folder
        resolved = None
        if folder_name:
            resolved = find_image_path(folder_name, str(image_path_or_filename))
        if not resolved:
            # Search whole source dir
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

    # Compass & White Bar Removal
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

    # Output directory
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
    """
    Direct CLI entrypoint to process a batch of sample scans from a dataset subfolder.
    """
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
