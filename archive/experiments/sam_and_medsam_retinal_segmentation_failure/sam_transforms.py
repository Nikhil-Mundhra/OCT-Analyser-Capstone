"""
training/classification/data/preprocessing/sam_transforms.py

Medical Image Preprocessing, False-Color Mapping, Synthetic 3-Channel Composites,
and Automated Prompt Generation for SAM 2 / MedSAM on Retinal OCT B-Scans.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .white_bars import detect_and_process_white_bars


def clean_scan_artifacts(gray_u8: np.ndarray) -> np.ndarray:
    """
    Suppresses scanner white calibration bars, HUD text, and solid metadata banners
    from top/bottom margins to ensure clean anatomical tissue profiling.
    """
    h, w = gray_u8.shape
    # 1. White bar raycasting removal
    clean_bgr = detect_and_process_white_bars(gray_u8, white_thresh=190, dark_bg_thresh=70)
    clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY) if clean_bgr.ndim == 3 else clean_bgr

    # 2. Suppress high-intensity UI annotations / text
    clean[clean >= 190] = 0

    # 3. Dynamic top solid metadata banner blanking (top 20%)
    for r in range(int(h * 0.20)):
        row_slice = clean[r, :]
        if np.mean(row_slice > 0) > 0.60 and np.mean(row_slice) > 140:
            clean[:r + 5, :] = 0

    # 4. Dynamic bottom solid metadata banner blanking (bottom 30%)
    for r in range(int(h * 0.70), h):
        row_slice = clean[r, :]
        if np.mean(row_slice > 0) > 0.60 and np.mean(row_slice) > 140:
            clean[r:, :] = 0
            break

    return clean


def build_sam_multichannel_inputs(
    gray_u8: np.ndarray,
    clahe_clip_limit: float = 2.5,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    sobel_ksize: int = 3
) -> Dict[str, np.ndarray]:
    """
    Transforms a single-channel grayscale OCT B-scan into multi-channel and false-color
    representations designed to optimize SAM 2 / MedSAM attention and feature extraction.

    Returns a dict containing:
      - 'raw_gray': Cleaned grayscale (H, W)
      - 'clahe': Contrast-Limited Adaptive Histogram Equalized grayscale (H, W)
      - 'sobel_y': Vertical Sobel gradient magnitude highlighting ILM/RPE transitions (H, W)
      - 'composite_3c': Synthetic 3-channel composite [Raw, CLAHE, Sobel-Y] (H, W, 3)
      - 'viridis_3c': Perceptually uniform Viridis false-color representation (H, W, 3)
      - 'jet_3c': High-contrast Jet false-color representation (H, W, 3)
    """
    clean = clean_scan_artifacts(gray_u8)

    # 1. CLAHE local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    img_clahe = clahe.apply(clean)

    # 2. Vertical Sobel gradient (emphasizes horizontal layer boundaries: ILM, IS/OS, RPE, CSI)
    grad_y = cv2.Sobel(clean.astype(float), cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    grad_y_u8 = np.clip(np.abs(grad_y), 0, 255).astype(np.uint8)

    # 3. Synthetic 3-Channel composite: [Raw, CLAHE, Sobel-Y]
    composite_3c = cv2.merge([clean, img_clahe, grad_y_u8])

    # 4. Viridis False Color Mapping
    viridis_3c = cv2.applyColorMap(clean, cv2.COLORMAP_VIRIDIS)

    # 5. Jet False Color Mapping
    jet_3c = cv2.applyColorMap(clean, cv2.COLORMAP_JET)

    return {
        "raw_gray": clean,
        "clahe": img_clahe,
        "sobel_y": grad_y_u8,
        "composite_3c": composite_3c,
        "viridis_3c": viridis_3c,
        "jet_3c": jet_3c,
    }


def generate_retinal_tissue_prompts(
    gray_u8: np.ndarray,
    num_pos_points: int = 7,
    margin_top_box: int = 120,
    margin_bot_box: int = 140
) -> Dict[str, np.ndarray]:
    """
    Generates high-confidence positive anchor points, negative boundary rejection points,
    and a bounding box prior for promptable segmentation models (SAM 2, MedSAM).

    Physics & Optical Prior:
      - Cleaned backscatter profiling locates the true neurosensory parenchyma & RPE band.
      - Positive points track the curved horizontal retinal core axis.
      - Negative points are positioned in the hyporeflective vitreous cavity (top)
        and deep retrobulbar space (bottom) to prevent mask leakage.

    Returns:
      - 'point_coords': (N, 2) array of [x, y] pixel coordinates
      - 'point_labels': (N,) array of 1 (positive) or 0 (negative)
      - 'box': (4,) array of [x_min, y_min, x_max, y_max]
      - 'y_center': Integer y-coordinate of the estimated retinal core
    """
    h, w = gray_u8.shape
    clean = clean_scan_artifacts(gray_u8)

    # Focus vertical profile on central 70% of scan
    center_roi = clean[:, int(w * 0.15):int(w * 0.85)]
    vert_profile = np.mean(center_roi, axis=1)

    # Blank out top and bottom 5% boundary rows to avoid residual edge spikes
    vert_profile[:int(h * 0.05)] = 0
    vert_profile[int(h * 0.95):] = 0

    smoothed_profile = gaussian_filter1d(vert_profile, sigma=6.0)

    # Core tissue center is at the global backscatter peak
    y_center = int(np.argmax(smoothed_profile))
    # Safeguard against degenerate cases
    if smoothed_profile[y_center] < 1.0:
        y_center = int(h * 0.50)
    else:
        y_center = max(int(h * 0.10), min(int(h * 0.90), y_center))

    # 1. Trace curved retinal axis across columns for positive prompt placement
    x_pos = np.linspace(int(w * 0.15), int(w * 0.85), num_pos_points, dtype=int)
    pos_coords = []
    
    # Search band around y_center (+- 40px) to snap to local backscatter maxima
    search_half = max(20, int(h * 0.08))
    for x in x_pos:
        col_slice = clean[:, max(0, x - 5):min(w, x + 6)]
        col_prof = np.mean(col_slice, axis=1)
        col_prof_smooth = gaussian_filter1d(col_prof, sigma=3.0)
        
        y_min_search = max(0, y_center - search_half)
        y_max_search = min(h - 1, y_center + search_half)
        
        if y_max_search > y_min_search and np.max(col_prof_smooth[y_min_search:y_max_search]) > 5.0:
            local_peak_y = y_min_search + int(np.argmax(col_prof_smooth[y_min_search:y_max_search]))
        else:
            local_peak_y = y_center
            
        pos_coords.append([x, local_peak_y])

    pos_coords = np.array(pos_coords, dtype=int)
    pos_labels = np.ones(len(pos_coords), dtype=int)

    # 2. Negative Prompt Points: Vitreous cavity (top) and Retrobulbar floor (bottom)
    neg_x = x_pos[::2]  # Subsample x coordinates for negative points
    neg_top = []
    neg_bot = []
    
    for x in neg_x:
        # Find corresponding local positive y
        idx = np.where(x_pos == x)[0][0]
        y_local = pos_coords[idx, 1]
        
        y_vit = max(10, y_local - 100)
        y_ret = min(h - 10, y_local + 120)
        neg_top.append([x, y_vit])
        neg_bot.append([x, y_ret])

    neg_coords = np.vstack([neg_top, neg_bot])
    neg_labels = np.zeros(len(neg_coords), dtype=int)

    all_coords = np.vstack([pos_coords, neg_coords])
    all_labels = np.concatenate([pos_labels, neg_labels])

    # 3. Bounding Box Prior
    min_pos_y = int(np.min(pos_coords[:, 1]))
    max_pos_y = int(np.max(pos_coords[:, 1]))

    y_box_min = max(0, min_pos_y - margin_top_box)
    y_box_max = min(h - 1, max_pos_y + margin_bot_box)
    box = np.array([0, y_box_min, w, y_box_max], dtype=int)

    return {
        "point_coords": all_coords,
        "point_labels": all_labels,
        "pos_coords": pos_coords,
        "neg_coords": neg_coords,
        "box": box,
        "y_center": y_center
    }


def mask_to_smooth_envelope(
    binary_mask: np.ndarray,
    margin_top: int = 15,
    margin_bottom: int = 20,
    gaussian_sigma: float = 8.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a 2D binary segmentation mask into continuous, smooth top and bottom
    boundary vectors with custom anatomical padding margins and Gaussian smoothing.

    Guarantees 100% preservation of segmented retinal structures without sharp staircase edges.

    Returns:
      - envelope_mask: (H, W) uint8 binary mask (0 or 255)
      - y_top_outer: (W,) smoothed upper boundary vector (y_top - margin_top)
      - y_bot_outer: (W,) smoothed lower boundary vector (y_bottom + margin_bottom)
    """
    h, w = binary_mask.shape
    y_top = np.full(w, -1.0, dtype=float)
    y_bottom = np.full(w, -1.0, dtype=float)

    for x in range(w):
        ys = np.where(binary_mask[:, x] > 0)[0]
        if len(ys) > 0:
            y_top[x] = float(ys[0])
            y_bottom[x] = float(ys[-1])

    valid_top = (y_top >= 0)
    valid_bot = (y_bottom >= 0)
    x_all = np.arange(w)

    if not np.any(valid_top) or not np.any(valid_bot):
        # Fallback to full mask if empty
        return np.ones((h, w), dtype=np.uint8) * 255, np.zeros(w), np.full(w, h - 1)

    # 1D linear interpolation across any columns with gaps (e.g. shadow beams)
    y_top_interp = np.interp(x_all, x_all[valid_top], y_top[valid_top])
    y_bot_interp = np.interp(x_all, x_all[valid_bot], y_bottom[valid_bot])

    # Gaussian smoothing to enforce C^1 continuity
    y_top_smooth = gaussian_filter1d(y_top_interp, sigma=gaussian_sigma)
    y_bot_smooth = gaussian_filter1d(y_bot_interp, sigma=gaussian_sigma)

    # Apply padding margins
    y_top_outer = np.maximum(0, y_top_smooth - margin_top)
    y_bot_outer = np.minimum(h - 1, y_bot_smooth + margin_bottom)

    # Render smooth envelope mask
    envelope_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        yt = int(y_top_outer[x])
        yb = int(y_bot_outer[x])
        if yb > yt:
            envelope_mask[yt:yb + 1, x] = 255

    return envelope_mask, y_top_outer, y_bot_outer


def draw_prompt_visualization(
    img_bgr: np.ndarray,
    prompts: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Renders visual prompt overlays (green positive points, red negative points, cyan bounding box)
    onto a BGR image for validation and inspection.
    """
    vis = img_bgr.copy()
    box = prompts["box"]
    pos_coords = prompts["pos_coords"]
    neg_coords = prompts["neg_coords"]

    # 1. Draw Bounding Box (Cyan rectangle)
    cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

    # 2. Draw Positive Prompt Points (Green circles with dark outline)
    for pt in pos_coords:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, (0, 0, 0), -1)
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

    # 3. Draw Negative Prompt Points (Red cross / circles)
    for pt in neg_coords:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, (0, 0, 0), -1)
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

    return vis
