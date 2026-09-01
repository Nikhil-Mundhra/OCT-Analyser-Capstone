"""
data/preprocessing/masking.py

Unified Canonical OCT Retinal Tissue Segmentation & Envelope Generation Engine.
Implements dual-pass adaptive Otsu thresholding, Dynamic Programming ILM tracking,
Spatial Fuzzy C-Means (SFCM) choroidal segmentation, and Autonomous Multi-Surface Graph Optimization.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    gaussian_filter,
    gaussian_filter1d,
    median_filter,
)
from scipy.signal import savgol_filter

from .outliers import reject_outliers_1d
from .choroid import (
    compute_sfcm_choroid_boundary,
    detect_rpe_band,
    get_sfcm_cache_key,
)
from .multisurface import JointMultiSurfaceOptimizer, MultiSurfaceConfig


def suppress_boundary_spikes(
    y: np.ndarray,
    spike_px: float,
    window: int = 80,
    direction: str = "up"
) -> np.ndarray:
    """
    Detect and interpolate isolated spikes in a 1D boundary vector.

    direction='up'  : catches upward spikes (y < rolling_median - spike_px).
                      Use for the top (ILM) boundary - spikes point toward smaller y.
    direction='down': catches downward dips  (y > rolling_median + spike_px).
                      Use for the bottom (choroid) boundary - dips point toward larger y.
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

    w = len(y)
    valid = ~is_spike
    x_all = np.arange(w)
    y_fixed = y.copy()
    if np.any(valid):
        y_fixed[is_spike] = np.interp(x_all[is_spike], x_all[valid], y[valid])
    return y_fixed


def _detect_ilm_dp(
    gray_u8: np.ndarray,
    y_cand_top: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> np.ndarray:
    """
    Gradient-based Dynamic Programming ILM (top boundary) detector.
    Anatomical basis: the ILM is the FIRST significant dark-to-bright edge
    encountered scanning downward from the vitreous into the neurosensory retina.
    """
    params = params or {}
    h, w = gray_u8.shape

    gradient_weight = float(params.get("ilm_gradient_weight", 0.80))
    smooth_weight = float(params.get("ilm_smooth_weight", 0.20))
    band_half = int(params.get("ilm_search_band_px", 50))
    max_step = 6

    if y_cand_top is not None:
        valid_mask = (y_cand_top > 10) & (y_cand_top < int(h * 0.75))
        x_all = np.arange(w)
        if np.any(valid_mask):
            y_guide = np.interp(x_all, x_all[valid_mask], y_cand_top[valid_mask])
        else:
            y_guide = np.full(w, float(h * 0.25))
    else:
        y_guide = np.full(w, float(h * 0.25))

    y_guide_smooth = gaussian_filter1d(y_guide, sigma=6.0)

    search_min_y = np.maximum(5, (y_guide_smooth - band_half).astype(int))
    search_max_y = np.minimum(int(h * 0.70), (y_guide_smooth + band_half).astype(int))

    smoothed = gaussian_filter(gray_u8.astype(float), sigma=1.8)
    grad_down = np.diff(smoothed, axis=0, prepend=smoothed[:1, :])
    grad_down = np.clip(grad_down, 0.0, None)
    g_max = grad_down.max()
    norm_grad = grad_down / (g_max + 1e-6)

    # Vitreous darkness penalty above y
    cum_img = np.cumsum(smoothed, axis=0)
    win = 12
    padded_cum = np.pad(cum_img, ((win, 0), (0, 0)), mode="edge")
    above_mean = (padded_cum[win:h + win, :] - padded_cum[:h, :]) / float(win)
    internal_penalty = np.clip((above_mean - 35.0) / 25.0, 0.0, 1.0)

    cost_field = 1.0 - (gradient_weight * norm_grad) + 0.50 * internal_penalty

    for x in range(w):
        min_y = search_min_y[x]
        max_y = search_max_y[x]
        if max_y > min_y:
            cost_field[:min_y, x] = 10.0
            cost_field[max_y:, x] = 10.0

    dp_cost = np.full((h, w), 1e6, dtype=float)
    backtrack = np.zeros((h, w), dtype=int)

    min_y_0, max_y_0 = search_min_y[0], search_max_y[0]
    if max_y_0 > min_y_0:
        dp_cost[min_y_0:max_y_0, 0] = cost_field[min_y_0:max_y_0, 0]

    for x in range(1, w):
        y_min_curr, y_max_curr = search_min_y[x], search_max_y[x]
        y_min_prev, y_max_prev = search_min_y[x - 1], search_max_y[x - 1]

        if y_max_curr <= y_min_curr or y_max_prev <= y_min_prev:
            continue

        for y in range(y_min_curr, y_max_curr):
            p_start = min(y_max_prev - 1, max(y_min_prev, y - max_step))
            p_end = max(p_start + 1, min(y_max_prev, y + max_step + 1))

            if p_start < p_end:
                prev_ys = np.arange(p_start, p_end)
                penalties = smooth_weight * ((prev_ys - y) ** 2)
                candidates = dp_cost[prev_ys, x - 1] + penalties
                best = int(np.argmin(candidates))
                dp_cost[y, x] = cost_field[y, x] + candidates[best]
                backtrack[y, x] = prev_ys[best]

    ilm_path = np.zeros(w, dtype=float)
    last_min, last_max = search_min_y[-1], search_max_y[-1]
    best_last_y = last_min + int(np.argmin(dp_cost[last_min:max(last_min + 1, last_max), -1]))
    ilm_path[-1] = float(best_last_y)

    curr_y = int(best_last_y)
    for x in range(w - 1, 0, -1):
        curr_y = backtrack[curr_y, x]
        ilm_path[x - 1] = float(curr_y)

    win_len = 21 if w >= 21 else (w - 1 if w % 2 == 0 else w)
    if win_len >= 5:
        ilm_path = savgol_filter(ilm_path, window_length=win_len, polyorder=2)
    else:
        ilm_path = gaussian_filter1d(ilm_path, sigma=6.0)

    return np.clip(ilm_path, 0, h - 1)


def _estimate_adaptive_thresholds(
    gray_u8: np.ndarray,
    top_mult: float = 2.2,
    bot_mult: float = 1.6
) -> Tuple[int, int, int]:
    """
    Computes Pass 1 (ILM Vitreoretinal) and Pass 2 (Choroid) adaptive intensity thresholds.
    Isolates background noise from the lower-quartile distribution of non-zero pixels.
    """
    valid = gray_u8.copy()
    valid[valid >= 190] = 0
    non_zero = valid[(valid > 0) & (valid < 190)]

    if len(non_zero) > 100:
        p20 = float(np.percentile(non_zero, 20))
        bg_pixels = non_zero[non_zero <= p20]
        bg_mean = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 14.0
        bg_std = float(np.std(bg_pixels)) if len(bg_pixels) > 0 else 3.5
    else:
        bg_mean = 14.0
        bg_std = 3.5

    bg_std = max(2.0, min(6.0, bg_std))
    thresh_top_val = max(16, int(bg_mean + top_mult * bg_std))
    thresh_bot_val = max(14, int(bg_mean + bot_mult * bg_std))
    thresh_bot_val = min(thresh_top_val - 2, thresh_bot_val)
    return thresh_top_val, thresh_bot_val, int(thresh_top_val)


def _extract_raw_boundary_contours(
    gray_u8: np.ndarray,
    thresh_top_val: int,
    thresh_bot_val: int,
    sb_top_pct: int,
    sb_bot_pct: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates morphological masks and extracts raw 1D column-wise boundary indices."""
    h, w = gray_u8.shape

    # Pitch-black column exclusion
    tissue_col = np.sum(gray_u8 > 15, axis=0) > int(h * 0.05)
    col_mask = np.zeros((h, w), dtype=np.uint8)
    col_mask[:, tissue_col] = 255

    _, thresh_top = cv2.threshold(gray_u8, thresh_top_val, 255, cv2.THRESH_BINARY)
    _, thresh_bot = cv2.threshold(gray_u8, thresh_bot_val, 255, cv2.THRESH_BINARY)

    thresh_top = cv2.bitwise_and(thresh_top, col_mask)
    thresh_bot = cv2.bitwise_and(thresh_bot, col_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    eff_sb_top = max(121, int(w * max(0.05, min(0.40, float(sb_top_pct) / 100.0))))
    if eff_sb_top % 2 == 0:
        eff_sb_top += 1
    sk_top = cv2.getStructuringElement(cv2.MORPH_RECT, (eff_sb_top, 1))

    eff_sb_bot = max(121, int(w * max(0.05, min(0.40, float(sb_bot_pct) / 100.0))))
    if eff_sb_bot % 2 == 0:
        eff_sb_bot += 1
    sk_bot = cv2.getStructuringElement(cv2.MORPH_RECT, (eff_sb_bot, 1))

    closed_top = cv2.morphologyEx(thresh_top, cv2.MORPH_CLOSE, kernel)
    closed_top = cv2.morphologyEx(closed_top, cv2.MORPH_CLOSE, sk_top)
    closed_top = cv2.bitwise_and(closed_top, col_mask)
    contours_top, _ = cv2.findContours(closed_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_top = [c for c in contours_top if cv2.contourArea(c) >= float(h * w) * 0.0005]
    if not sig_top and contours_top:
        sig_top = [max(contours_top, key=cv2.contourArea)]

    if sig_top:
        retina_level_contours = [
            c for c in sig_top
            if cv2.moments(c)["m00"] > 0
            and (cv2.moments(c)["m01"] / cv2.moments(c)["m00"]) >= h * 0.20
        ]
        if retina_level_contours:
            sig_top = [max(retina_level_contours, key=cv2.contourArea)]
        else:
            sig_top = [max(sig_top, key=cv2.contourArea)]

    mask_top = np.zeros_like(gray_u8)
    if sig_top:
        cv2.drawContours(mask_top, sig_top, -1, 255, cv2.FILLED)
    mask_top = binary_fill_holes(mask_top.astype(bool)).astype(np.uint8) * 255
    mask_top = cv2.bitwise_and(mask_top, col_mask)

    closed_bot = cv2.morphologyEx(thresh_bot, cv2.MORPH_CLOSE, kernel)
    closed_bot = cv2.morphologyEx(closed_bot, cv2.MORPH_CLOSE, sk_bot)
    closed_bot = cv2.bitwise_and(closed_bot, col_mask)
    contours_bot, _ = cv2.findContours(closed_bot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_bot = [c for c in contours_bot if cv2.contourArea(c) >= float(h * w) * 0.0005]
    if not sig_bot and contours_bot:
        sig_bot = [max(contours_bot, key=cv2.contourArea)]
    mask_bot = np.zeros_like(gray_u8)
    if sig_bot:
        cv2.drawContours(mask_bot, sig_bot, -1, 255, cv2.FILLED)
    mask_bot = binary_fill_holes(mask_bot.astype(bool)).astype(np.uint8) * 255
    mask_bot = cv2.bitwise_and(mask_bot, col_mask)

    y_top = np.full(w, -1, dtype=float)
    y_bottom = np.full(w, -1, dtype=float)
    for x in range(w):
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
    compass_bbox: Optional[tuple],
    margin_bottom: float,
    gaussian_sigma: float,
    top_spike_px: float = 0,
    top_spike_win: int = 80,
    top_dip_px: float = 0,
    top_dip_win: int = 80
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies outlier rejection, spike suppression, and 1D Gaussian smoothing."""
    h, w = gray_u8.shape
    valid_t = (y_top >= 0)
    valid_b = (y_bottom >= 0)

    if not np.any(valid_t) or not np.any(valid_b):
        fallback = np.zeros(w, dtype=float)
        return fallback, fallback

    x_t = np.where(valid_t)[0]
    x_b = np.where(valid_b)[0]
    y_top_interp = np.interp(np.arange(w), x_t, y_top[valid_t])
    y_bottom_interp = np.interp(np.arange(w), x_b, y_bottom[valid_b])

    y_top_clean = reject_outliers_1d(y_top_interp, gray_u8=gray_u8, x_all=np.arange(w))
    y_bottom_clean = reject_outliers_1d(y_bottom_interp, gray_u8=gray_u8, x_all=np.arange(w))

    y_top_clean = suppress_boundary_spikes(y_top_clean, top_spike_px, window=top_spike_win, direction="up")
    y_top_clean = suppress_boundary_spikes(y_top_clean, top_dip_px, window=top_dip_win, direction="down")

    if compass_bbox is not None:
        bx0, _, bx1, _ = compass_bbox
        valid_outside = np.ones(w, dtype=bool)
        valid_outside[bx0:min(w, bx1 + 15)] = False
        x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
        if len(x_valid) > 10:
            y_bottom_clean = np.interp(np.arange(w), x_valid, y_bottom_clean[x_valid])
    else:
        if np.mean(gray_u8[int(h * 0.75):, :100] == 0) > 0.85:
            valid_outside = np.ones(w, dtype=bool)
            valid_outside[:int(w * 0.20)] = False
            x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
            if len(x_valid) > 10:
                y_bottom_clean = np.interp(np.arange(w), x_valid, y_bottom_clean[x_valid])
        elif np.mean(gray_u8[int(h * 0.75):, -100:] == 0) > 0.85:
            valid_outside = np.ones(w, dtype=bool)
            valid_outside[int(w * 0.80):] = False
            x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
            if len(x_valid) > 10:
                y_bottom_clean = np.interp(np.arange(w), x_valid, y_bottom_clean[x_valid])

    valid_floor = y_bottom_clean[y_bottom_clean > 0]
    if len(valid_floor) > 20:
        p35_floor = float(np.percentile(valid_floor, 35))
        max_floor_cap = p35_floor + max(45.0, float(margin_bottom) * 2.5)
        y_bottom_clean = np.minimum(y_bottom_clean, max_floor_cap)

    sigma_val = float(gaussian_sigma)
    y_top_final = gaussian_filter1d(y_top_clean, sigma=sigma_val)
    y_bottom_final = gaussian_filter1d(y_bottom_clean, sigma=sigma_val)
    return y_top_final, y_bottom_final


def detect_boundaries_intelligent_auto(
    gray_u8: np.ndarray,
    params: Optional[dict] = None,
    compass_bbox: Optional[tuple] = None,
    src_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Autonomous Per-Image Intelligent Multi-Surface Segmentation Engine.
    Executes joint coupled graph optimization and B-spline fitting.
    """
    params = params or {}
    h, w = gray_u8.shape
    clean_gray = gray_u8.copy()
    clean_gray[clean_gray >= 190] = 0

    # Auto-detect solid top/bottom metadata banners
    top_start = int(h * 0.15)
    for r in range(top_start):
        row_slice = gray_u8[r, :]
        row_nz = row_slice[row_slice > 0]
        if len(row_nz) > int(w * 0.70) and (np.std(row_nz) < 8.0 or np.mean(row_nz) > 175):
            clean_gray[:r + 10, :] = 0

    bottom_start = int(h * 0.85)
    for r in range(bottom_start, h):
        row_slice = gray_u8[r, :]
        row_nz = row_slice[row_slice > 0]
        if len(row_nz) > int(w * 0.80) and (np.std(row_nz) < 6.0 or np.mean(row_nz) > 180):
            clean_gray[r:, :] = 0
            break

    optimizer = JointMultiSurfaceOptimizer(MultiSurfaceConfig(
        min_retina_thickness_px=float(params.get("min_retina_thickness_px", 25.0)),
        max_retina_thickness_px=float(params.get("max_retina_thickness_px", 270.0)),
        min_choroid_thickness_px=float(params.get("min_choroid_thickness_px", 15.0)),
        max_choroid_thickness_px=float(params.get("max_choroid_thickness_px", 220.0)),
        ilm_gradient_weight=float(params.get("ilm_gradient_weight", 0.85)),
        smooth_weight=float(params.get("smooth_weight", 0.20)),
    ))

    y_top_opt, y_rpe, y_sfcm = optimizer.optimize_surfaces(clean_gray, params=params)
    slack_bottom_px = float(params.get("sfcm_slack_bottom_px", params.get("margin_bottom", 20.0)))
    y_sfcm_raw = np.maximum(y_rpe + 10.0, y_sfcm - slack_bottom_px)

    # 6-Point Anatomical Invariant Safety Gate
    retinal_thickness = y_rpe - y_top_opt
    choroid_thickness = y_sfcm - y_rpe

    inv_order = np.all(y_top_opt < y_rpe) and np.all(y_rpe < y_sfcm)
    inv_ret_thick = (float(np.mean(retinal_thickness)) >= 30.0) and (float(np.mean(retinal_thickness)) <= 350.0)
    inv_cho_thick = (float(np.mean(choroid_thickness)) >= 15.0) and (float(np.mean(choroid_thickness)) <= 250.0)
    inv_border = float(np.mean(y_top_opt)) > 5.0 and float(np.std(y_top_opt)) > 0.1
    inv_continuity = np.max(np.abs(np.diff(y_top_opt))) <= 25.0

    all_passed = inv_order and inv_ret_thick and inv_cho_thick and inv_border and inv_continuity

    if not all_passed:
        vert_profile = np.mean(clean_gray, axis=1)
        mass_y = float(np.argmax(gaussian_filter1d(vert_profile, sigma=8.0)))
        y_top_fallback = np.full(w, max(20.0, mass_y - 60.0))
        y_top_opt = _detect_ilm_dp(clean_gray, y_cand_top=y_top_fallback, params={"ilm_gradient_weight": 0.80, "ilm_smooth_weight": 0.20})
        y_rpe = detect_rpe_band(clean_gray, y_top_opt)
        _, y_sfcm, y_sfcm_raw = compute_sfcm_choroid_boundary(clean_gray, y_top_opt, params, return_raw=True)

    y_top_outer = np.maximum(0, y_top_opt)
    y_bot_outer = np.minimum(h - 1, y_sfcm)

    if compass_bbox is not None:
        bx0, by0, bx1, _ = compass_bbox
        c_x0 = max(0, bx0)
        c_x1 = min(w, bx1 + 1)
        y_bot_outer[c_x0:c_x1] = np.minimum(y_bot_outer[c_x0:c_x1], float(by0))

    envelope_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        t_y = int(y_top_outer[x])
        b_y = int(y_bot_outer[x])
        if b_y > t_y:
            envelope_mask[t_y:b_y + 1, x] = 255

    return envelope_mask, y_top_outer, y_bot_outer, y_rpe, y_sfcm, y_sfcm_raw


def generate_tissue_mask(
    gray_u8: np.ndarray,
    margin_top: Optional[int] = None,
    margin_bottom: Optional[int] = None,
    clear_corners: bool = True,
    compass_bbox: Optional[tuple] = None,
    top_noise_mult: float = 1.5,
    bot_noise_mult: float = 3.0,
    shadow_bridge_pct: int = 20,
    gaussian_sigma: int = 15,
    use_sfcm: bool = False,
    return_sfcm: bool = False,
    return_vectors: bool = False,
    src_path: Optional[str] = None,
    sfcm_cache: Optional[dict] = None,
    params: Optional[dict] = None,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Unified canonical tissue mask generator supporting both Batch Processing and Interactive Tuning.
    """
    cfg = (params or {}).copy()
    cfg.update(kwargs)

    # Resolve parameter priorities
    p_margin_top = margin_top if margin_top is not None else int(cfg.get("margin_top", cfg.get("margin", 15)))
    p_margin_bottom = margin_bottom if margin_bottom is not None else int(cfg.get("margin_bottom", cfg.get("margin", 15)))
    p_top_noise_mult = float(cfg.get("top_noise_mult", top_noise_mult))
    p_bot_noise_mult = float(cfg.get("bot_noise_mult", bot_noise_mult))
    p_sb_top_pct = int(cfg.get("shadow_bridge_top_pct", cfg.get("shadow_bridge_pct", shadow_bridge_pct)))
    p_sb_bot_pct = int(cfg.get("shadow_bridge_bot_pct", cfg.get("shadow_bridge_pct", shadow_bridge_pct)))
    p_gaussian_sigma = float(cfg.get("gaussian_sigma", gaussian_sigma))
    p_use_sfcm = bool(cfg.get("use_sfcm", use_sfcm))
    p_use_dp_ilm = bool(cfg.get("use_dp_ilm", False))
    p_auto_mode = bool(cfg.get("auto_mode", False))

    cfg["margin_top"] = p_margin_top
    cfg["margin_bottom"] = p_margin_bottom
    cfg["top_noise_mult"] = p_top_noise_mult
    cfg["bot_noise_mult"] = p_bot_noise_mult
    cfg["shadow_bridge_top_pct"] = p_sb_top_pct
    cfg["shadow_bridge_bot_pct"] = p_sb_bot_pct
    cfg["gaussian_sigma"] = p_gaussian_sigma
    cfg["use_sfcm"] = p_use_sfcm

    if p_auto_mode:
        ret_auto = detect_boundaries_intelligent_auto(gray_u8, params=cfg, compass_bbox=compass_bbox, src_path=src_path)
        if return_sfcm:
            return ret_auto
        if return_vectors:
            env_m, y_t_out, y_b_out, _, _, _ = ret_auto
            return env_m, y_t_out, y_b_out
        return ret_auto[0]

    clean_gray = gray_u8.copy()
    clean_gray[clean_gray >= 190] = 0
    h, w = clean_gray.shape

    top_spike_px = float(cfg.get("top_spike_suppress_px", 0))
    top_spike_win = int(cfg.get("top_spike_window_px", 80))
    top_dip_px = float(cfg.get("top_dip_suppress_px", 0))
    top_dip_win = int(cfg.get("top_dip_window_px", 80))

    thresh_top_val, thresh_bot_val, _ = _estimate_adaptive_thresholds(clean_gray, p_top_noise_mult, p_bot_noise_mult)
    y_top, y_bottom = _extract_raw_boundary_contours(
        clean_gray, thresh_top_val, thresh_bot_val, p_sb_top_pct, p_sb_bot_pct
    )

    if p_use_dp_ilm:
        y_top = _detect_ilm_dp(clean_gray, y_cand_top=y_top, params=cfg)

    y_top_final, y_bottom_final = _interpolate_and_filter_boundaries(
        clean_gray, y_top, y_bottom, compass_bbox, p_margin_bottom, p_gaussian_sigma,
        top_spike_px, top_spike_win, top_dip_px, top_dip_win
    )

    clear_limit = y_bottom_final + p_margin_bottom

    # Blackout boundary capping
    by0_map = np.full(w, float(h - 1), dtype=float)
    has_blackout = False

    if compass_bbox is not None:
        bx0, by0, bx1, _ = compass_bbox
        c_x0 = max(0, bx0)
        c_x1 = min(w, bx1 + 1)
        by0_map[c_x0:c_x1] = np.minimum(by0_map[c_x0:c_x1], float(by0))
        has_blackout = True

    y_search_start = int(h * 0.60)
    min_run_px = max(15, int(h * 0.04))
    for x in range(w):
        col_slice = clean_gray[y_search_start:, x]
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

    y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = None, None, None
    if p_use_sfcm:
        cache_key = get_sfcm_cache_key(src_path, clean_gray.shape, cfg) if src_path else None
        if cache_key and sfcm_cache is not None and cache_key in sfcm_cache:
            cached_val = sfcm_cache[cache_key]
            if len(cached_val) == 3:
                y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = cached_val
            else:
                y_rpe, y_bottom_sfcm = cached_val
                slack_px = float(cfg.get("sfcm_slack_bottom_px", 20))
                y_bottom_sfcm_raw = np.maximum(0, y_bottom_sfcm - slack_px)
        else:
            y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = compute_sfcm_choroid_boundary(clean_gray, y_top_final, cfg, return_raw=True)
            if cache_key and sfcm_cache is not None:
                sfcm_cache[cache_key] = (y_rpe.copy(), y_bottom_sfcm.copy(), y_bottom_sfcm_raw.copy())

        if has_blackout:
            y_bottom_sfcm = np.minimum(y_bottom_sfcm, by0_map)
            y_bottom_sfcm_raw = np.minimum(y_bottom_sfcm_raw, by0_map)

    active_bottom = y_bottom_sfcm if p_use_sfcm else clear_limit
    envelope_mask = np.zeros_like(clean_gray)
    for x in range(w):
        yt = max(0, int(y_top_final[x]) - p_margin_top)
        yb = min(h - 1, int(active_bottom[x]))
        if yb >= yt:
            envelope_mask[yt:yb + 1, x] = 255

    y_top_outer = np.maximum(0, y_top_final - p_margin_top)
    y_bottom_outer = np.minimum(h - 1, clear_limit)

    if return_sfcm:
        return envelope_mask, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw
    if return_vectors:
        return envelope_mask, y_top_outer, y_bottom_outer

    return envelope_mask


def generate_tissue_mask_custom(
    gray_u8: np.ndarray,
    params: dict,
    compass_bbox: Optional[tuple] = None,
    return_sfcm: bool = False,
    src_path: Optional[str] = None,
    sfcm_cache: Optional[dict] = None
) -> tuple:
    """
    Direct backward-compatible alias for fine-tuning dashboard and test suite.
    """
    res = generate_tissue_mask(
        gray_u8,
        params=params,
        compass_bbox=compass_bbox,
        return_sfcm=return_sfcm,
        return_vectors=True,
        src_path=src_path,
        sfcm_cache=sfcm_cache,
    )
    return res
