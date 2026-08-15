"""
data/preprocessing/tuning/server.py

Interactive Folder-Specific Preprocessing Parameter Tuning Local Server.
Runs on http://localhost:8000. Reads and updates folder_params.json directly with
separate top and bottom envelope padding margin parameters and compass_location selection.
Integrates 1D vector interpolation across compass box columns, dynamic beam capping,
and returns pixel-perfect 384x384 letterboxed boundary vectors for interactive SVG overlay rendering.
"""

import json
import os
import random
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    gaussian_filter,
    gaussian_filter1d,
    maximum_filter1d,
    median_filter,
)
from scipy.signal import savgol_filter

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "image-classification-model-training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "image-classification-model-training"))

from data.preprocessing.outliers import reject_outliers_1d
from data.preprocessing.params import (
    DEFAULT_PARAMS,
    get_folder_params,
    initialize_default_params_file,
    load_all_params,
    save_all_params,
)
from data.preprocessing.white_bars import (
    detect_and_process_white_bars,
    detect_and_remove_compass_artifacts,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = PACKAGE_DIR / "dashboard"

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


def suppress_boundary_spikes(
    y: np.ndarray,
    spike_px: float,
    window: int = 80,
    direction: str = "up"
) -> np.ndarray:
    """
    Detect and interpolate isolated spikes in a 1D boundary vector.

    direction='up'  : catches upward spikes (y < rolling_median - spike_px).
                      Use for the top (ILM) boundary — spikes point toward smaller y.
    direction='down': catches downward dips  (y > rolling_median + spike_px).
                      Use for the bottom (choroid) boundary — dips point toward larger y.

    Spike columns are replaced by linear interpolation from valid neighbours so that
    the surrounding retinal curvature is fully preserved.
    """
    if spike_px <= 0:
        return y

    rolling_med = median_filter(y.astype(float), size=window, mode="nearest")
    if direction == "up":
        is_spike = y < (rolling_med - spike_px)
    else:
        is_spike = y > (rolling_med + spike_px)

    if not np.any(is_spike):
        return y

    W = len(y)
    valid = ~is_spike
    x_all = np.arange(W)
    y_fixed = y.copy()
    if np.any(valid):
        y_fixed[is_spike] = np.interp(x_all[is_spike], x_all[valid], y[valid])
    return y_fixed


def detect_rpe_band(gray_u8: np.ndarray, y_top_outer: np.ndarray) -> np.ndarray:
    """
    SOTA Dynamic Programming / Graph-Search RPE Detection Algorithm (Chiu et al. framework).
    When RPE splits into 2 hyper-reflective bands (elevated RPE + Bruch's Membrane),
    this algorithm locks onto the LOWER/BOTTOM band (Bruch's Membrane) where the choroid starts.
    """
    H, W = gray_u8.shape

    grad_y = cv2.Sobel(gray_u8.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    norm_img = gray_u8.astype(float) / 255.0
    norm_grad = np.clip(grad_y / 255.0, 0, None)

    search_min_y = np.maximum(0, y_top_outer.astype(int) + 8)
    search_max_y = np.minimum(H - 1, search_min_y + int(H * 0.42))

    y_grid = np.arange(H)[:, None]
    y_rel = (y_grid - search_min_y[None, :]) / np.maximum(1.0, (search_max_y - search_min_y)[None, :])
    y_rel = np.clip(y_rel, 0.0, 1.0)

    rpe_cost_field = 1.0 - (0.30 * norm_img + 0.30 * norm_grad + 0.40 * (y_rel ** 1.8))

    for x in range(W):
        min_y = search_min_y[x]
        max_y = search_max_y[x]
        if max_y > min_y:
            rpe_cost_field[:min_y, x] = 10.0
            rpe_cost_field[max_y:, x] = 10.0

    dp_cost = np.full((H, W), 1e6, dtype=float)
    backtrack = np.zeros((H, W), dtype=int)

    min_y_0, max_y_0 = search_min_y[0], search_max_y[0]
    if max_y_0 > min_y_0:
        dp_cost[min_y_0:max_y_0, 0] = rpe_cost_field[min_y_0:max_y_0, 0]

    max_step = 3
    smooth_weight = 0.20

    for x in range(1, W):
        y_min_curr, y_max_curr = search_min_y[x], search_max_y[x]
        y_min_prev, y_max_prev = search_min_y[x - 1], search_max_y[x - 1]

        if y_max_curr <= y_min_curr or y_max_prev <= y_min_prev:
            continue

        for y in range(y_min_curr, y_max_curr):
            p_start = max(y_min_prev, y - max_step)
            p_end = min(y_max_prev, y + max_step + 1)

            if p_start < p_end:
                prev_y_coords = np.arange(p_start, p_end)
                transition_penalties = smooth_weight * ((prev_y_coords - y) ** 2)
                candidate_costs = dp_cost[prev_y_coords, x - 1] + transition_penalties

                best_prev_idx = np.argmin(candidate_costs)
                dp_cost[y, x] = rpe_cost_field[y, x] + candidate_costs[best_prev_idx]
                backtrack[y, x] = prev_y_coords[best_prev_idx]

    rpe_path = np.zeros(W, dtype=float)
    last_min, last_max = search_min_y[-1], search_max_y[-1]
    best_last_y = last_min + np.argmin(dp_cost[last_min:max(last_min + 1, last_max), -1])
    rpe_path[-1] = float(best_last_y)

    curr_y = int(best_last_y)
    for x in range(W - 1, 0, -1):
        curr_y = backtrack[curr_y, x]
        rpe_path[x - 1] = float(curr_y)

    rpe_bottom_env = maximum_filter1d(rpe_path, size=15)
    win_len = 31 if W >= 31 else (W - 1 if W % 2 == 0 else W)
    if win_len >= 5:
        smoothed_rpe = savgol_filter(rpe_bottom_env, window_length=win_len, polyorder=2)
    else:
        smoothed_rpe = gaussian_filter1d(rpe_bottom_env, sigma=10.0)

    smoothed_rpe = np.maximum(smoothed_rpe, rpe_path)
    return smoothed_rpe


def compute_sfcm_choroid_boundary(
    gray_u8: np.ndarray,
    y_top_outer: np.ndarray,
    params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spatial Fuzzy C-Means (SFCM) clustering to segment the vascular choroidal layer.
    """
    H, W = gray_u8.shape
    y_rpe = detect_rpe_band(gray_u8, y_top_outer)

    margin_bottom = float(params.get("sfcm_margin_bottom", params.get("margin_bottom", 15)))
    gaussian_sigma = float(params.get("sfcm_gaussian_sigma", params.get("gaussian_sigma", 15)))
    n_clusters = int(params.get("sfcm_n_clusters", 3))
    m = float(params.get("sfcm_fuzziness_m", 2.0))
    max_iter = 10

    y_bottom_sfcm = np.zeros(W, dtype=float)
    sub_pixels = []
    coords = []
    max_depth_px = min(140, int(H * 0.30))

    for x in range(W):
        y_start = min(H - 10, max(0, int(y_rpe[x]) + 2))
        y_end = min(H, y_start + max_depth_px)
        for y in range(y_start, y_end):
            sub_pixels.append(float(gray_u8[y, x]))
            coords.append((y, x))

    if len(sub_pixels) < 100:
        return y_rpe, y_rpe + margin_bottom + 40.0

    sub_pixels = np.array(sub_pixels, dtype=float)
    min_val, max_val = np.percentile(sub_pixels, 5), np.percentile(sub_pixels, 95)
    centroids = np.linspace(min_val, max_val, n_clusters)

    for _ in range(max_iter):
        dists = np.abs(sub_pixels[:, None] - centroids[None, :]) + 1e-5
        inv_dists = (1.0 / dists) ** (2.0 / (m - 1.0))
        U = inv_dists / np.sum(inv_dists, axis=1, keepdims=True)

        u_m = U ** m
        centroids = np.sum(u_m * sub_pixels[:, None], axis=0) / (np.sum(u_m, axis=0) + 1e-5)
        centroids = np.sort(centroids)

    stroma_map = np.zeros((H, W), dtype=float)
    for idx, (y, x) in enumerate(coords):
        stroma_map[y, x] = U[idx, 1]

    sfcm_smooth = gaussian_filter(stroma_map, sigma=(2.0, 3.0))

    for x in range(W):
        y_start = min(H - 10, max(0, int(y_rpe[x]) + 2))
        y_end = min(H, y_start + max_depth_px)
        col_stroma = sfcm_smooth[y_start:y_end, x]

        if len(col_stroma) > 5:
            peak_idx = np.argmax(col_stroma)
            drop_indices = np.where(col_stroma[peak_idx:] < 0.25 * col_stroma[peak_idx])[0]
            if len(drop_indices) > 0:
                csi_offset = peak_idx + drop_indices[0]
            else:
                grad = np.gradient(col_stroma)
                csi_offset = np.argmin(grad)

            csi_offset = max(20, min(110, csi_offset))
            y_bottom_sfcm[x] = y_start + csi_offset + margin_bottom
        else:
            y_bottom_sfcm[x] = y_start + 40.0 + margin_bottom

    y_bottom_sfcm = gaussian_filter1d(y_bottom_sfcm, sigma=gaussian_sigma)
    return y_rpe, y_bottom_sfcm


def _estimate_adaptive_thresholds(gray_u8: np.ndarray, top_mult: float, bot_mult: float) -> tuple[int, int, int]:
    """Computes Pass 1 (ILM) and Pass 2 (Choroid) adaptive intensity thresholds."""
    H, _ = gray_u8.shape
    edge_sample = np.concatenate([gray_u8[:20, :].flatten(), gray_u8[H - 20:, :].flatten()])
    edge_non_zero = edge_sample[edge_sample > 0]
    bg_mean = float(np.mean(edge_non_zero)) if len(edge_non_zero) > 0 else 25.0
    bg_std = float(np.std(edge_non_zero)) if len(edge_non_zero) > 0 else 5.0

    noise_cutoff = max(25, int(bg_mean + top_mult * bg_std))
    above_noise = gray_u8[gray_u8 > noise_cutoff]
    if len(above_noise) > 100:
        otsu_val, _ = cv2.threshold(above_noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_val <= 0:
            otsu_val = float(np.mean(above_noise))
        thresh_top_val = max(int(noise_cutoff + 5), int(otsu_val))
    else:
        otsu_val = 70.0
        thresh_top_val = 70

    thresh_bot_val = max(20, int(bg_mean + bot_mult * bg_std))
    thresh_bot_val = min(thresh_top_val - 5, thresh_bot_val)
    return thresh_top_val, thresh_bot_val, int(otsu_val)


def _extract_raw_boundary_contours(
    gray_u8: np.ndarray,
    thresh_top_val: int,
    thresh_bot_val: int,
    sb_top_pct: int,
    sb_bot_pct: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generates morphological masks and extracts raw 1D column-wise boundary indices."""
    H, W = gray_u8.shape
    _, thresh_top = cv2.threshold(gray_u8, thresh_top_val, 255, cv2.THRESH_BINARY)
    _, thresh_bot = cv2.threshold(gray_u8, thresh_bot_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    eff_sb_top = max(121, int(W * max(0.05, min(0.40, float(sb_top_pct) / 100.0))))
    if eff_sb_top % 2 == 0:
        eff_sb_top += 1
    sk_top = cv2.getStructuringElement(cv2.MORPH_RECT, (eff_sb_top, 1))

    eff_sb_bot = max(121, int(W * max(0.05, min(0.40, float(sb_bot_pct) / 100.0))))
    if eff_sb_bot % 2 == 0:
        eff_sb_bot += 1
    sk_bot = cv2.getStructuringElement(cv2.MORPH_RECT, (eff_sb_bot, 1))

    closed_top = cv2.morphologyEx(thresh_top, cv2.MORPH_CLOSE, kernel)
    closed_top = cv2.morphologyEx(closed_top, cv2.MORPH_CLOSE, sk_top)
    contours_top, _ = cv2.findContours(closed_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_top = [c for c in contours_top if cv2.contourArea(c) >= float(H * W) * 0.0005]
    if not sig_top and contours_top:
        sig_top = [max(contours_top, key=cv2.contourArea)]
    mask_top = np.zeros_like(gray_u8)
    if sig_top:
        cv2.drawContours(mask_top, sig_top, -1, 255, cv2.FILLED)
    mask_top = binary_fill_holes(mask_top.astype(bool)).astype(np.uint8) * 255

    closed_bot = cv2.morphologyEx(thresh_bot, cv2.MORPH_CLOSE, kernel)
    closed_bot = cv2.morphologyEx(closed_bot, cv2.MORPH_CLOSE, sk_bot)
    contours_bot, _ = cv2.findContours(closed_bot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_bot = [c for c in contours_bot if cv2.contourArea(c) >= float(H * W) * 0.0005]
    if not sig_bot and contours_bot:
        sig_bot = [max(contours_bot, key=cv2.contourArea)]
    mask_bot = np.zeros_like(gray_u8)
    if sig_bot:
        cv2.drawContours(mask_bot, sig_bot, -1, 255, cv2.FILLED)
    mask_bot = binary_fill_holes(mask_bot.astype(bool)).astype(np.uint8) * 255

    y_top = np.full(W, -1, dtype=float)
    y_bottom = np.full(W, -1, dtype=float)
    for x in range(W):
        ys_t = np.where(mask_top[:, x] > 0)[0]
        if len(ys_t) > 0:
            y_top[x] = ys_t[0]
        ys_b = np.where(mask_bot[:, x] > 0)[0]
        if len(ys_b) > 0:
            y_bottom[x] = ys_b[-1]

    return y_top, y_bottom


def _interpolate_and_filter_boundaries(
    gray_u8: np.ndarray,
    y_top: np.ndarray,
    y_bottom: np.ndarray,
    compass_bbox: tuple | None,
    margin_bottom: float,
    gaussian_sigma: float,
    top_spike_px: float = 0,
    top_spike_win: int = 80,
    top_dip_px: float = 0,
    top_dip_win: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    """Applies outlier rejection, spike suppression, and 1D Gaussian smoothing."""
    H, W = gray_u8.shape
    valid_t = (y_top >= 0)
    valid_b = (y_bottom >= 0)

    if not np.any(valid_t) or not np.any(valid_b):
        fallback = np.zeros(W, dtype=float)
        return fallback, fallback

    x_t = np.where(valid_t)[0]
    x_b = np.where(valid_b)[0]
    y_top_interp = np.interp(np.arange(W), x_t, y_top[valid_t])
    y_bottom_interp = np.interp(np.arange(W), x_b, y_bottom[valid_b])

    y_top_clean = reject_outliers_1d(y_top_interp, gray_u8=gray_u8, x_all=np.arange(W))
    y_bottom_clean = reject_outliers_1d(y_bottom_interp, gray_u8=gray_u8, x_all=np.arange(W))

    y_top_clean = suppress_boundary_spikes(y_top_clean, top_spike_px, window=top_spike_win, direction="up")
    y_top_clean = suppress_boundary_spikes(y_top_clean, top_dip_px, window=top_dip_win, direction="down")

    if compass_bbox is not None:
        bx0, _, bx1, _ = compass_bbox
        valid_outside = np.ones(W, dtype=bool)
        valid_outside[bx0:min(W, bx1 + 15)] = False
        x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
        if len(x_valid) > 10:
            y_bottom_clean = np.interp(np.arange(W), x_valid, y_bottom_clean[x_valid])
    else:
        if np.mean(gray_u8[int(H * 0.75):, :100] == 0) > 0.85:
            valid_outside = np.ones(W, dtype=bool)
            valid_outside[:int(W * 0.20)] = False
            x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
            if len(x_valid) > 10:
                y_bottom_clean = np.interp(np.arange(W), x_valid, y_bottom_clean[x_valid])
        elif np.mean(gray_u8[int(H * 0.75):, -100:] == 0) > 0.85:
            valid_outside = np.ones(W, dtype=bool)
            valid_outside[int(W * 0.80):] = False
            x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
            if len(x_valid) > 10:
                y_bottom_clean = np.interp(np.arange(W), x_valid, y_bottom_clean[x_valid])

    valid_floor = y_bottom_clean[y_bottom_clean > 0]
    if len(valid_floor) > 20:
        p35_floor = float(np.percentile(valid_floor, 35))
        max_floor_cap = p35_floor + max(45.0, float(margin_bottom) * 2.5)
        y_bottom_clean = np.minimum(y_bottom_clean, max_floor_cap)

    sigma_val = float(gaussian_sigma)
    y_top_final = gaussian_filter1d(y_top_clean, sigma=sigma_val)
    y_bottom_final = gaussian_filter1d(y_bottom_clean, sigma=sigma_val)
    return y_top_final, y_bottom_final


def get_sfcm_cache_key(src_path: str, shape: tuple, params: dict) -> tuple:
    """Generates a stable cache key for SFCM operations."""
    return (
        src_path,
        shape,
        params.get("sfcm_margin_bottom", 15),
        params.get("sfcm_gaussian_sigma", 15),
        params.get("sfcm_n_clusters", 3),
        params.get("sfcm_fuzziness_m", 2.0),
        params.get("top_noise_mult", 1.5),
        params.get("gaussian_sigma", 15)
    )


def generate_tissue_mask_custom(
    gray_u8: np.ndarray,
    params: dict,
    compass_bbox: tuple = None,
    return_sfcm: bool = False,
    src_path: str = None
) -> tuple:
    """Custom fine-tuning tissue mask generator supporting both Otsu and SFCM boundaries."""
    gray_u8 = gray_u8.copy()
    gray_u8[gray_u8 >= 190] = 0
    H, W = gray_u8.shape

    top_mult = float(params.get("top_noise_mult", 1.5))
    bot_mult = float(params.get("bot_noise_mult", 3.0))
    margin_top = int(params.get("margin_top", params.get("margin", 15)))
    margin_bottom = int(params.get("margin_bottom", params.get("margin", 15)))
    sb_top_pct = int(params.get("shadow_bridge_top_pct", params.get("shadow_bridge_pct", 20)))
    sb_bot_pct = int(params.get("shadow_bridge_bot_pct", params.get("shadow_bridge_pct", 20)))
    gaussian_sigma = float(params.get("gaussian_sigma", 15))
    use_sfcm = bool(params.get("use_sfcm", False))

    top_spike_px = float(params.get("top_spike_suppress_px", 0))
    top_spike_win = int(params.get("top_spike_window_px", 80))
    top_dip_px = float(params.get("top_dip_suppress_px", 0))
    top_dip_win = int(params.get("top_dip_window_px", 80))

    thresh_top_val, thresh_bot_val, _ = _estimate_adaptive_thresholds(gray_u8, top_mult, bot_mult)
    y_top, y_bottom = _extract_raw_boundary_contours(
        gray_u8, thresh_top_val, thresh_bot_val, sb_top_pct, sb_bot_pct
    )

    y_top_final, y_bottom_final = _interpolate_and_filter_boundaries(
        gray_u8, y_top, y_bottom, compass_bbox, margin_bottom, gaussian_sigma,
        top_spike_px, top_spike_win, top_dip_px, top_dip_win
    )

    clear_limit = y_bottom_final + margin_bottom

    by0_map = np.full(W, float(H - 1), dtype=float)
    has_blackout = False

    if compass_bbox is not None:
        bx0, by0, bx1, _ = compass_bbox
        c_x0 = max(0, bx0)
        c_x1 = min(W, bx1 + 1)
        by0_map[c_x0:c_x1] = np.minimum(by0_map[c_x0:c_x1], float(by0))
        has_blackout = True

    y_search_start = int(H * 0.60)
    min_run_px = max(15, int(H * 0.04))
    for x in range(W):
        col_slice = gray_u8[y_search_start:, x]
        if len(col_slice) > 0 and col_slice[-1] == 0:
            zero_indices = np.where(col_slice == 0)[0]
            if len(zero_indices) > 0:
                diffs = np.diff(zero_indices)
                non_contig = np.where(diffs > 1)[0]
                if len(non_contig) > 0:
                    run_start_idx = zero_indices[non_contig[-1] + 1]
                else:
                    run_start_idx = zero_indices[0]
                run_length = len(col_slice) - run_start_idx
                if run_length >= min_run_px:
                    top_y = y_search_start + run_start_idx
                    by0_map[x] = min(by0_map[x], float(top_y))
                    has_blackout = True

    if has_blackout:
        clear_limit = np.minimum(clear_limit, by0_map)

    y_rpe, y_bottom_sfcm = None, None
    if use_sfcm:
        cache_key = get_sfcm_cache_key(src_path, gray_u8.shape, params) if src_path else None
        if cache_key and cache_key in SFCM_CACHE:
            y_rpe, y_bottom_sfcm = SFCM_CACHE[cache_key]
        else:
            y_rpe, y_bottom_sfcm = compute_sfcm_choroid_boundary(gray_u8, y_top_final, params)
            if cache_key:
                SFCM_CACHE[cache_key] = (y_rpe.copy(), y_bottom_sfcm.copy())

        if has_blackout:
            y_bottom_sfcm = np.minimum(y_bottom_sfcm, by0_map)

    active_bottom = y_bottom_sfcm if use_sfcm else clear_limit
    envelope_mask = np.zeros_like(gray_u8)
    for x in range(W):
        yt = max(0, int(y_top_final[x]) - margin_top)
        yb = min(H - 1, int(active_bottom[x]))
        if yb >= yt:
            envelope_mask[yt:yb + 1, x] = 255

    y_top_outer = np.maximum(0, y_top_final - margin_top)
    y_bottom_outer = np.minimum(H - 1, clear_limit)

    if return_sfcm:
        return envelope_mask, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm

    return envelope_mask, y_top_outer, y_bottom_outer


def letterbox_pad_and_resize(img: np.ndarray, target_dim: int = 384) -> tuple:
    """Pads an image symmetrically to square letterbox and resizes to target_dim."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    pad_t = (max_dim - h) // 2
    pad_b = max_dim - h - pad_t
    pad_l = (max_dim - w) // 2
    pad_r = max_dim - w - pad_l

    padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized = cv2.resize(padded, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
    scale = target_dim / float(max_dim)
    return resized, scale, pad_t, pad_l, h, w


def project_and_downsample_vectors(
    orig_w: int,
    y_top: np.ndarray = None,
    y_bottom: np.ndarray = None,
    y_rpe: np.ndarray | None = None,
    y_sfcm: np.ndarray | None = None,
    pad_t: float = 0,
    pad_l: float = 0,
    scale: float = 1.0,
    n_pts: int = 64,
    y_top_outer: np.ndarray = None,
    y_bottom_outer: np.ndarray = None,
    y_bottom_sfcm: np.ndarray = None,
    num_points: int = None
) -> tuple:
    """Downsamples and projects boundary vectors into 384x384 letterbox SVG space."""
    if y_top is None and y_top_outer is not None:
        y_top = y_top_outer
    if y_bottom is None and y_bottom_outer is not None:
        y_bottom = y_bottom_outer
    if y_sfcm is None and y_bottom_sfcm is not None:
        y_sfcm = y_bottom_sfcm
    if num_points is not None:
        n_pts = num_points

    indices = np.linspace(0, orig_w - 1, n_pts, dtype=int)

    top_pts = [[float(x + pad_l) * scale, float(y_top[x] + pad_t) * scale] for x in indices]
    bot_pts = [[float(x + pad_l) * scale, float(y_bottom[x] + pad_t) * scale] for x in indices]

    rpe_pts = None
    if y_rpe is not None:
        rpe_pts = [[float(x + pad_l) * scale, float(y_rpe[x] + pad_t) * scale] for x in indices]

    sfcm_pts = None
    if y_sfcm is not None:
        sfcm_pts = [[float(x + pad_l) * scale, float(y_sfcm[x] + pad_t) * scale] for x in indices]

    return top_pts, bot_pts, rpe_pts, sfcm_pts


def find_folder_path(folder_name: str | Path) -> Path | None:
    """Finds the absolute path of a dataset subfolder."""
    if isinstance(folder_name, Path):
        if folder_name.exists():
            return folder_name
        folder_name = folder_name.name

    from data.preprocessing.tuning import server as srv
    active_source = getattr(srv, "SOURCE_DIR", SOURCE_DIR)

    subfolder_path = active_source / folder_name
    if not subfolder_path.exists():
        matches = [d for d in active_source.rglob(folder_name) if d.is_dir()]
        if matches:
            subfolder_path = matches[0]
        else:
            return None
    return subfolder_path


def find_image_path(folder_name: str | Path, filename: str) -> Path | None:
    """Finds the path to a specific image inside a folder or globally."""
    if folder_name:
        subfolder_path = find_folder_path(folder_name)
        if subfolder_path:
            target = subfolder_path / filename
            if target.exists():
                return target
        return None

    from data.preprocessing.tuning import server as srv
    active_source = getattr(srv, "SOURCE_DIR", SOURCE_DIR)
    matches = list(active_source.rglob(filename))
    return matches[0] if matches else None


def get_available_subfolders(source_dir: Path) -> list[str]:
    """Finds all leaf directories containing valid scan images."""
    if not source_dir.exists():
        return []
    subfolders = []
    for p in sorted(source_dir.rglob("*")):
        if p.is_dir() and not p.name.startswith("."):
            valid = list(p.glob("*.jp*g")) + list(p.glob("*.png"))
            if valid:
                subfolders.append(p.name)
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
    mask, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm = generate_tissue_mask_custom(
        gray, params, compass_bbox=compass_bbox, return_sfcm=True, src_path=str(src_p)
    )

    mask_3c = cv2.merge([mask, mask, mask])
    processed = np.where(mask_3c > 0, img_bgr, 0).astype(np.uint8)

    processed_resized, scale, pad_t, pad_l, h, w = letterbox_pad_and_resize(processed, target_dim=384)
    raw_resized, _, _, _, _, _ = letterbox_pad_and_resize(img_bgr, target_dim=384)

    proc_out = out_folder / f"{src_p.stem}_proc.jpg"
    raw_out = out_folder / f"{src_p.stem}_raw.jpg"

    cv2.imwrite(str(proc_out), processed_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(raw_out), raw_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

    top_pts, bot_pts, rpe_pts, sfcm_pts = project_and_downsample_vectors(
        w, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm, pad_t, pad_l, scale
    )

    return {
        "filename": src_p.name,
        "filepath": str(src_p),
        "raw_url": f"/preprocessed/{folder_name}/{raw_out.name}",
        "proc_url": f"/preprocessed/{folder_name}/{proc_out.name}",
        "top_vector": top_pts,
        "bottom_vector": bot_pts,
        "rpe_vector": rpe_pts,
        "sfcm_vector": sfcm_pts,
        "scale": round(scale, 6),
        "pad_t": int(pad_t),
        "orig_h": int(h),
        "orig_w": int(w)
    }


def reprocess_folder_sample(folder_name: str, params: dict, random_sample: bool = False) -> list:
    """Reprocesses a batch of sample images for a given folder."""
    subfolder_path = find_folder_path(folder_name)
    if not subfolder_path:
        return []

    imgs = [p for p in subfolder_path.glob("*.jp*g") if not p.name.startswith(".")]
    imgs += [p for p in subfolder_path.glob("*.png") if not p.name.startswith(".")]
    if not imgs:
        return []

    if random_sample or folder_name not in FOLDER_SAMPLES_CACHE:
        sample_files = random.sample(imgs, min(6, len(imgs)))
        FOLDER_SAMPLES_CACHE[folder_name] = sample_files
    else:
        sample_files = FOLDER_SAMPLES_CACHE[folder_name]

    from data.preprocessing.tuning import server as srv
    active_output = getattr(srv, "OUTPUT_DIR", OUTPUT_DIR)
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

    from data.preprocessing.tuning import server as srv
    active_output = getattr(srv, "OUTPUT_DIR", OUTPUT_DIR)
    out_folder = active_output / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)
    return process_and_save_image(img_path, out_folder, folder_name, params)


class FineTuningRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving dashboard UI, static assets, and tuning API endpoints."""

    def _send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file_response(self, file_path: Path, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        with open(file_path, "rb") as f:
            self.wfile.write(f.read())

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html", "/tuning_dashboard.html"):
            html_p = DASHBOARD_DIR / "index.html"
            if html_p.exists():
                self._send_file_response(html_p, "text/html; charset=utf-8")
                return

        elif path.startswith("/css/"):
            rel = path[len("/css/"):]
            css_file = DASHBOARD_DIR / "css" / rel
            if css_file.exists() and css_file.is_file():
                self._send_file_response(css_file, "text/css; charset=utf-8")
                return

        elif path.startswith("/js/"):
            rel = path[len("/js/"):]
            js_file = DASHBOARD_DIR / "js" / rel
            if js_file.exists() and js_file.is_file():
                self._send_file_response(js_file, "application/javascript; charset=utf-8")
                return

        elif path == "/api/folders":
            from data.preprocessing.tuning import server as srv
            active_source = getattr(srv, "SOURCE_DIR", SOURCE_DIR)
            subfolders = get_available_subfolders(active_source)
            saved_params = load_all_params()

            res = {
                "folders": subfolders,
                "saved_params": saved_params,
                "default_params": DEFAULT_PARAMS
            }
            self._send_json_response(res)
            return

        elif path.startswith("/preprocessed/"):
            from data.preprocessing.tuning import server as srv
            active_output = getattr(srv, "OUTPUT_DIR", OUTPUT_DIR)
            rel = path[len("/preprocessed/"):]
            target_file = active_output / rel
            if target_file.exists():
                self._send_file_response(target_file, "image/jpeg")
                return

        super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except Exception:
            self._send_json_response({"status": "error", "message": "Invalid JSON"}, status=400)
            return

        if self.path == "/api/reprocess":
            folder_name = data.get("folder")
            params = data.get("params", DEFAULT_PARAMS)
            random_sample = data.get("random_sample", False)

            all_params = load_all_params()
            all_params[folder_name] = params
            save_all_params(all_params)

            samples = reprocess_folder_sample(folder_name, params, random_sample=random_sample)

            res = {
                "status": "success",
                "folder": folder_name,
                "samples": samples
            }
            self._send_json_response(res)
            return

        elif self.path == "/api/reprocess_single":
            folder_name = data.get("folder")
            filename = data.get("filename")
            params = data.get("params", DEFAULT_PARAMS)

            sample = reprocess_single_image(folder_name, filename, params)
            if sample:
                self._send_json_response({"status": "success", "sample": sample})
            else:
                self._send_json_response({"status": "error", "message": "Image not found"}, status=404)
            return


def run_server(port=8000):
    initialize_default_params_file()
    server_address = ("", port)
    httpd = HTTPServer(server_address, FineTuningRequestHandler)
    print("======================================================================")
    print(f"  Interactive Preprocessing Tuning Server Running on http://localhost:{port}")
    print("======================================================================")
    httpd.serve_forever()


def main():
    run_server()


if __name__ == "__main__":
    main()
