"""
data/preprocessing/pipeline.py

High-level processing pipeline orchestration function (process_image).
Handles image I/O, scanner-provenance compass removal, white bar removal, tissue masking, letterbox framing, and saving.
Automatically loads folder-specific parameters from folder_params.json with configurable compass location.
Integrates 1D compass vector interpolation to eliminate sharp blackout steps.
"""

from pathlib import Path
import traceback
import cv2
import numpy as np

from .white_bars import detect_and_process_white_bars, detect_and_remove_compass_artifacts
from .masking import generate_tissue_mask
from .params import get_folder_params

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def process_image(
    src_path: str,
    dst_path: str,
    quality: int = 95,
    frame: bool = True,
    frame_size: int = 384,
    highlight_red: bool = False,
    save_mask: bool = False,
    overlay_rgb: bool = False,
    clear_corners: bool = True,
    params: dict = None
) -> bool:
    try:
        src_p = Path(src_path)
        folder_name = src_p.parent.name

        # Load folder-specific parameters from folder_params.json if not passed directly
        if params is None:
            params = get_folder_params(folder_name)

        margin_bottom = params.get("margin_bottom", params.get("margin", 15))
        compass_enabled = params.get("compass_ui_enabled", None)
        compass_location = params.get("compass_location", "auto")

        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [SKIP] Cannot open: {src_path}", flush=True)
            return False

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 0. Scanner-Provenance Compass Artifact Detection & Zero-Erasure
        img, compass_bbox = detect_and_remove_compass_artifacts(
            img,
            src_path=src_path,
            enabled=compass_enabled,
            location=compass_location,
            margin=margin_bottom,
            return_bbox=True
        )

        raw_bgr = img.copy()

        # 1. White Bar Removal
        img = detect_and_process_white_bars(
            img, white_thresh=190, dark_bg_thresh=70, gap_pixels=3, highlight_red=highlight_red
        )

        # 2. Morphological Tissue Mask (passing compass_bbox for vector interpolation)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_u8 = np.clip(gray * (255.0 if gray.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8) if gray.dtype != np.uint8 else gray
        mask = generate_tissue_mask(
            gray_u8,
            clear_corners=clear_corners,
            compass_bbox=compass_bbox,
            params=params,
            src_path=src_path
        )

        # 3. Handle RGB Overlay Mode vs Production Masking
        if overlay_rgb:
            overlay = raw_bgr.copy()
            bg_indices = (mask == 0)
            overlay[bg_indices] = (overlay[bg_indices] * 0.80 + np.array([0, 0, 180]) * 0.20).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(overlay, contours, -1, (200, 230, 130), 1)
            img = overlay
        elif not highlight_red:
            mask_3c = cv2.merge([mask, mask, mask])
            img = np.where(mask_3c > 0, img, 0).astype(np.uint8)

        # 4. Framing: Letterbox pad to square + resize to frame_size
        if frame:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            pad_top = (max_dim - h) // 2
            pad_bottom = max_dim - h - pad_top
            pad_left = (max_dim - w) // 2
            pad_right = max_dim - w - pad_left
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
            mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)
            if frame_size is not None:
                img = cv2.resize(img, (frame_size, frame_size), interpolation=cv2.INTER_AREA if not overlay_rgb else cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)

        # 5. Save Output Image
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        ext = dst.suffix.lower()
        if ext in {'.jpg', '.jpeg'}:
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            cv2.imwrite(str(dst), img)

        # 6. Save Optional Binary Tissue Mask
        if save_mask:
            mask_path = dst.parent / f"{dst.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        return True

    except Exception as e:
        print(f"  [ERROR] {src_path}: {e}", flush=True)
        traceback.print_exc()
        return False
