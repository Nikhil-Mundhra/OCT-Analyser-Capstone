"""
data/preprocessing/masking.py

Dynamic Dual-Pass Adaptive Tissue Segmentation & Envelope Padding.
Implements separate top (margin_top) and bottom (margin_bottom) envelope padding margins,
tight noise-floor Pass 2 bottom thresholding, 1D vector interpolation across compass box columns,
dynamic choroid floor hypertransmission beam capping (p35 + max_cap),
and 1D Gaussian boundary smoothing (sigma=15.0).
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from .outliers import reject_outliers_1d


def generate_tissue_mask(
    gray_u8: np.ndarray,
    margin_top: int = 15,
    margin_bottom: int = 15,
    clear_corners: bool = True,
    compass_bbox: tuple = None,
    top_noise_mult: float = 1.5,
    bot_noise_mult: float = 3.0,
    shadow_bridge_pct: int = 20,
    gaussian_sigma: int = 15
) -> np.ndarray:
    """
    Generates a continuous, smooth tissue mask using dynamic dual-pass Otsu thresholding
    and 1D boundary vector Gaussian smoothing with compass column interpolation.

    Parameters:
      - gray_u8: Grayscale input image (uint8, 0-255).
      - margin_top: Inner Limiting Membrane (ILM) top padding margin (px).
      - margin_bottom: Choroid floor bottom padding margin (px).
      - clear_corners: Enables organic anatomical corner tapering.
      - compass_bbox: Optional tuple (bx0, by0, bx1, by1) of erased compass box.
      - top_noise_mult: Multiplier k for Pass 1 top cutoff (mu_bg + k*sigma_bg).
      - bot_noise_mult: Multiplier k for Pass 2 bottom floor (mu_bg + k*sigma_bg).
      - shadow_bridge_pct: Width of horizontal closing kernel (% image width).
      - gaussian_sigma: Standard deviation sigma for 1D boundary smoothing.
    """
    gray_u8 = gray_u8.copy()
    
    # 0. Zero out white scanner annotation bars (intensity >= 190)
    gray_u8[gray_u8 >= 190] = 0
    H, W = gray_u8.shape

    # 1. Background Noise Profiling
    edge_sample = np.concatenate([gray_u8[:20, :].flatten(), gray_u8[H-20:, :].flatten()])
    edge_non_zero = edge_sample[edge_sample > 0]
    bg_mean = float(np.mean(edge_non_zero)) if len(edge_non_zero) > 0 else 25.0
    bg_std = float(np.std(edge_non_zero)) if len(edge_non_zero) > 0 else 5.0

    # 2. Dynamic Dual-Pass Thresholding Values
    noise_cutoff = max(25, int(bg_mean + top_noise_mult * bg_std))

    above_noise = gray_u8[gray_u8 > noise_cutoff]
    if len(above_noise) > 100:
        otsu_val, _ = cv2.threshold(above_noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_top_val = max(int(noise_cutoff + 5), int(otsu_val))
    else:
        thresh_top_val = 70

    thresh_bot_val = max(20, int(bg_mean + bot_noise_mult * bg_std))
    thresh_bot_val = min(thresh_top_val - 5, thresh_bot_val)

    # 3. Morphological Pass 1 (Top ILM) & Pass 2 (Bottom Choroid Floor)
    _, thresh_top = cv2.threshold(gray_u8, thresh_top_val, 255, cv2.THRESH_BINARY)
    _, thresh_bot = cv2.threshold(gray_u8, thresh_bot_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    sb_pct_val = max(0.05, min(0.40, float(shadow_bridge_pct) / 100.0))
    effective_sb_px = max(121, int(W * sb_pct_val))
    if effective_sb_px % 2 == 0:
        effective_sb_px += 1
    sk = cv2.getStructuringElement(cv2.MORPH_RECT, (effective_sb_px, 1))

    # Pass 1 Top Mask
    closed_top = cv2.morphologyEx(thresh_top, cv2.MORPH_CLOSE, kernel)
    closed_top = cv2.morphologyEx(closed_top, cv2.MORPH_CLOSE, sk)
    contours_top, _ = cv2.findContours(closed_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_top = [c for c in contours_top if cv2.contourArea(c) >= float(H * W) * 0.0005]
    if not sig_top and contours_top:
        sig_top = [max(contours_top, key=cv2.contourArea)]
    mask_top = np.zeros_like(gray_u8)
    if sig_top:
        cv2.drawContours(mask_top, sig_top, -1, 255, cv2.FILLED)
    mask_top = binary_fill_holes(mask_top.astype(bool)).astype(np.uint8) * 255

    # Pass 2 Bottom Mask
    closed_bot = cv2.morphologyEx(thresh_bot, cv2.MORPH_CLOSE, kernel)
    closed_bot = cv2.morphologyEx(closed_bot, cv2.MORPH_CLOSE, sk)
    contours_bot, _ = cv2.findContours(closed_bot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_bot = [c for c in contours_bot if cv2.contourArea(c) >= float(H * W) * 0.0005]
    if not sig_bot and contours_bot:
        sig_bot = [max(contours_bot, key=cv2.contourArea)]
    mask_bot = np.zeros_like(gray_u8)
    if sig_bot:
        cv2.drawContours(mask_bot, sig_bot, -1, 255, cv2.FILLED)
    mask_bot = binary_fill_holes(mask_bot.astype(bool)).astype(np.uint8) * 255

    # 4. Extract 1D Boundary Vectors
    y_top = np.full(W, -1, dtype=float)
    y_bottom = np.full(W, -1, dtype=float)

    for x in range(W):
        ys_t = np.where(mask_top[:, x] > 0)[0]
        if len(ys_t) > 0:
            y_top[x] = ys_t[0]

        ys_b = np.where(mask_bot[:, x] > 0)[0]
        if len(ys_b) > 0:
            y_bottom[x] = ys_b[-1]

    valid_t = (y_top >= 0)
    valid_b = (y_bottom >= 0)
    if not np.any(valid_t) or not np.any(valid_b):
        return np.ones_like(gray_u8, dtype=np.uint8) * 255

    x_t = np.where(valid_t)[0]
    x_b = np.where(valid_b)[0]
    y_top_interp = np.interp(np.arange(W), x_t, y_top[valid_t])
    y_bottom_interp = np.interp(np.arange(W), x_b, y_bottom[valid_b])

    # Outlier Rejection
    y_top_clean = reject_outliers_1d(y_top_interp, gray_u8=gray_u8, x_all=np.arange(W))
    y_bottom_clean = reject_outliers_1d(y_bottom_interp, gray_u8=gray_u8, x_all=np.arange(W))

    # 5. Interpolate y_bottom across compass box columns to prevent sharp step blackouts
    compass_cap_range = None
    if compass_bbox is not None:
        bx0, by0, bx1, by1 = compass_bbox
        valid_outside = np.ones(W, dtype=bool)
        valid_outside[bx0:min(W, bx1 + 15)] = False
        x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
        if len(x_valid) > 10:
            y_bottom_clean = np.interp(np.arange(W), x_valid, y_bottom_clean[x_valid])
        compass_cap_range = (max(0, bx0 - 5), min(W, bx1 + 5), by0)
    else:
        # Auto-detect pre-existing zeroed black box in bottom corners
        if np.mean(gray_u8[int(H * 0.75):, :100] == 0) > 0.85:
            valid_outside = np.ones(W, dtype=bool)
            valid_outside[:int(W * 0.20)] = False
            x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
            if len(x_valid) > 10:
                y_bottom_clean = np.interp(np.arange(W), x_valid, y_bottom_clean[x_valid])
            black_ys = np.where(gray_u8[int(H * 0.70):, :int(W * 0.20)] == 0)[0]
            if len(black_ys) > 0:
                by0 = int(np.min(black_ys) + int(H * 0.70))
                compass_cap_range = (0, int(W * 0.20), by0)
        elif np.mean(gray_u8[int(H * 0.75):, -100:] == 0) > 0.85:
            valid_outside = np.ones(W, dtype=bool)
            valid_outside[int(W * 0.80):] = False
            x_valid = np.where(valid_outside & (y_bottom_clean >= 0))[0]
            if len(x_valid) > 10:
                y_bottom_clean = np.interp(np.arange(W), x_valid, y_bottom_clean[x_valid])
            black_ys = np.where(gray_u8[int(H * 0.70):, int(W * 0.80):] == 0)[0]
            if len(black_ys) > 0:
                by0 = int(np.min(black_ys) + int(H * 0.70))
                compass_cap_range = (int(W * 0.80), W, by0)

    # 6. Hypertransmission / Shadow Beam Floor Capping Rule
    valid_floor = y_bottom_clean[y_bottom_clean > 0]
    if len(valid_floor) > 20:
        p35_floor = float(np.percentile(valid_floor, 35))
        max_floor_cap = p35_floor + max(45.0, float(margin_bottom) * 2.5)
        y_bottom_clean = np.minimum(y_bottom_clean, max_floor_cap)

    # 7. 1D Gaussian Smoothing
    sigma_val = float(gaussian_sigma)
    y_top_final = gaussian_filter1d(y_top_clean, sigma=sigma_val)
    y_bottom_final = gaussian_filter1d(y_bottom_clean, sigma=sigma_val)

def detect_rpe_band(gray_u8: np.ndarray, y_top_outer: np.ndarray, params: dict = None) -> np.ndarray:
    """
    SOTA Dynamic Programming / Graph-Search RPE Detection Algorithm (Chiu et al. framework).
    When RPE splits into 2 hyper-reflective bands (elevated RPE + Bruch's Membrane),
    this algorithm locks onto the LOWER/BOTTOM band (Bruch's Membrane) where the choroid starts.
    Supports folder-specific tuning parameters when SFCM is active.
    Combines:
    1. Positive vertical intensity gradient (IS/OS / RPE transition).
    2. Peak absolute reflectivity of the RPE hyper-reflective complex.
    3. Depth-rewarding prior to select the bottom split band (Bruch's Membrane).
    4. Dijkstra Dynamic Programming path search with quadratic smoothness penalties (y_{x+1} - y_x)^2.
    5. Savitzky-Golay global ocular curvature fitting.
    """
    H, W = gray_u8.shape
    params = params or {}

    smooth_weight = float(params.get("rpe_smooth_weight", 0.20))
    depth_weight = float(params.get("rpe_depth_weight", 0.40))
    gradient_weight = float(params.get("rpe_gradient_weight", 0.30))
    reflectivity_weight = float(params.get("rpe_reflectivity_weight", 0.30))
    bottom_env_size = int(params.get("rpe_bottom_env_size", 15))
    depth_exponent = float(params.get("rpe_depth_exponent", 1.8))
    max_step = int(params.get("rpe_max_step", 3))

    grad_y = cv2.Sobel(gray_u8.astype(float), cv2.CV_64F, 0, 1, ksize=3)

    norm_img = gray_u8.astype(float) / 255.0
    norm_grad = np.clip(grad_y / 255.0, 0, None)

    search_min_y = np.maximum(0, y_top_outer.astype(int) + 8)
    search_max_y = np.minimum(H - 1, search_min_y + int(H * 0.42))

    y_grid = np.arange(H)[:, None]
    y_rel = (y_grid - search_min_y[None, :]) / np.maximum(1.0, (search_max_y - search_min_y)[None, :])
    y_rel = np.clip(y_rel, 0.0, 1.0)

    # Combined RPE Energy Field (lower cost = higher likelihood of bottom RPE/Bruch's)
    rpe_cost_field = 1.0 - (reflectivity_weight * norm_img + gradient_weight * norm_grad + depth_weight * (y_rel ** depth_exponent))

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

    rpe_bottom_env = maximum_filter1d(rpe_path, size=max(1, bottom_env_size))

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
    margin_bottom: float,
    gaussian_sigma: float,
    n_clusters: int = 3,
    max_iter: int = 10,
    m: float = 2.0,
    params: dict = None
) -> np.ndarray:
    """
    Smart Spatial Fuzzy C-Means (SFCM) clustering algorithm.
    1. Detects the hyper-reflective RPE band using SOTA Dynamic Programming.
    2. Operates strictly below the RPE band (y > y_rpe + 2) to segment the choroid layer.
    3. Returns y_bottom_sfcm.
    """
    H, W = gray_u8.shape
    y_rpe = detect_rpe_band(gray_u8, y_top_outer, params=params)
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
        return y_rpe + margin_bottom + 40.0

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

    y_bottom_sfcm = gaussian_filter1d(y_bottom_sfcm, sigma=float(gaussian_sigma))

    # Apply customizable downward trust buffer / slack margin
    params = params or {}
    slack_bottom_px = float(params.get("sfcm_slack_bottom_px", 20))
    y_bottom_sfcm = np.minimum(H - 1, y_bottom_sfcm + slack_bottom_px)

    return y_bottom_sfcm


def detect_choroidal_caverns(
    gray_u8: np.ndarray,
    y_rpe: np.ndarray,
    y_bottom_sfcm: np.ndarray,
    params: dict = None
) -> list[dict]:
    """
    Detects pathological choroidal caverns / cavitations using multi-feature morphology,
    absence of hyperreflective vessel sheaths, and posterior hypertransmission analysis.
    """
    H, W = gray_u8.shape
    params = params or {}

    min_area = int(params.get("cavern_min_area", 15))
    max_area = int(params.get("cavern_max_area", 900))
    dark_thresh = int(params.get("cavern_dark_threshold", 45))
    trans_thresh = float(params.get("cavern_transmission_threshold", 1.30))
    min_circ = float(params.get("cavern_min_circularity", 0.60))
    slack_px = int(params.get("sfcm_slack_bottom_px", 20))

    choroid_mask = np.zeros((H, W), dtype=np.uint8)
    for x in range(W):
        top_y = min(H - 1, max(0, int(y_rpe[x]) + 2))
        bot_y = min(H, max(top_y + 1, int(y_bottom_sfcm[x]) + slack_px))
        choroid_mask[top_y:bot_y, x] = 1

    core_voids = (gray_u8 < dark_thresh) & (choroid_mask == 1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(core_voids.astype(np.uint8), connectivity=8)

    caverns = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue

        rx = stats[i, cv2.CC_STAT_LEFT]
        ry = stats[i, cv2.CC_STAT_TOP]
        rw = stats[i, cv2.CC_STAT_WIDTH]
        rh = stats[i, cv2.CC_STAT_HEIGHT]

        blob_mask = (labels[ry:ry + rh, rx:rx + rw] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        perimeter = cv2.arcLength(cnts[0], True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * (area / (perimeter ** 2))

        if circularity < min_circ:
            continue

        sub_y_start = min(H - 1, ry + rh)
        sub_y_end = min(H, sub_y_start + max(15, slack_px))

        if sub_y_end > sub_y_start:
            sub_signal = float(np.mean(gray_u8[sub_y_start:sub_y_end, rx:rx + rw]))

            left_w = max(0, rx - rw)
            right_w = min(W, rx + 2 * rw)
            lateral_samples = []
            if rx > left_w:
                lateral_samples.append(np.mean(gray_u8[ry:ry + rh, left_w:rx]))
            if right_w > rx + rw:
                lateral_samples.append(np.mean(gray_u8[ry:ry + rh, rx + rw:right_w]))

            ref_bg = float(np.mean(lateral_samples)) if lateral_samples else max(1.0, sub_signal)
            ref_bg = max(1.0, ref_bg)

            trans_ratio = sub_signal / ref_bg

            if trans_ratio >= trans_thresh:
                caverns.append({
                    "bbox": [int(rx), int(ry), int(rw), int(rh)],
                    "area": int(area),
                    "circularity": round(float(circularity), 3),
                    "transmission_ratio": round(float(trans_ratio), 3),
                    "centroid": [round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)]
                })

    return caverns


def generate_tissue_mask(
    gray_u8: np.ndarray,
    margin_top: int = 15,
    margin_bottom: int = 15,
    clear_corners: bool = True,
    compass_bbox: tuple = None,
    top_noise_mult: float = 1.5,
    bot_noise_mult: float = 3.0,
    shadow_bridge_pct: int = 20,
    gaussian_sigma: int = 15,
    use_sfcm: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Generates a continuous, smooth tissue mask using dynamic dual-pass Otsu thresholding
    or SFCM choroid segmentation, and 1D boundary vector Gaussian smoothing with compass column interpolation.
    """
    gray_u8 = gray_u8.copy()
    gray_u8[gray_u8 >= 190] = 0
    H, W = gray_u8.shape

    # 1. Background Noise Profiling
    edge_sample = np.concatenate([gray_u8[:20, :].flatten(), gray_u8[H - 20:, :].flatten()])
    edge_non_zero = edge_sample[edge_sample > 0]
    bg_mean = float(np.mean(edge_non_zero)) if len(edge_non_zero) > 0 else 25.0
    bg_std = float(np.std(edge_non_zero)) if len(edge_non_zero) > 0 else 5.0

    # 2. Dynamic Dual-Pass Thresholding Values
    noise_cutoff = max(25, int(bg_mean + top_noise_mult * bg_std))

    above_noise = gray_u8[gray_u8 > noise_cutoff]
    if len(above_noise) > 100:
        otsu_val, _ = cv2.threshold(above_noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_top_val = max(int(noise_cutoff + 5), int(otsu_val))
    else:
        thresh_top_val = 70

    thresh_bot_val = max(20, int(bg_mean + bot_noise_mult * bg_std))
    thresh_bot_val = min(thresh_top_val - 5, thresh_bot_val)

    # 3. Morphological Pass 1 (Top ILM) & Pass 2 (Bottom Choroid Floor)
    _, thresh_top = cv2.threshold(gray_u8, thresh_top_val, 255, cv2.THRESH_BINARY)
    _, thresh_bot = cv2.threshold(gray_u8, thresh_bot_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    sb_pct_val = max(0.05, min(0.40, float(shadow_bridge_pct) / 100.0))
    effective_sb_px = max(121, int(W * sb_pct_val))
    if effective_sb_px % 2 == 0:
        effective_sb_px += 1
    sk = cv2.getStructuringElement(cv2.MORPH_RECT, (effective_sb_px, 1))

    # Pass 1 Top Mask
    closed_top = cv2.morphologyEx(thresh_top, cv2.MORPH_CLOSE, kernel)
    closed_top = cv2.morphologyEx(closed_top, cv2.MORPH_CLOSE, sk)
    contours_top, _ = cv2.findContours(closed_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_top = [c for c in contours_top if cv2.contourArea(c) >= float(H * W) * 0.0005]
    if not sig_top and contours_top:
        sig_top = [max(contours_top, key=cv2.contourArea)]
    mask_top = np.zeros_like(gray_u8)
    if sig_top:
        cv2.drawContours(mask_top, sig_top, -1, 255, cv2.FILLED)
    mask_top = binary_fill_holes(mask_top.astype(bool)).astype(np.uint8) * 255

    # Pass 2 Bottom Mask
    closed_bot = cv2.morphologyEx(thresh_bot, cv2.MORPH_CLOSE, kernel)
    closed_bot = cv2.morphologyEx(closed_bot, cv2.MORPH_CLOSE, sk)
    contours_bot, _ = cv2.findContours(closed_bot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_bot = [c for c in contours_bot if cv2.contourArea(c) >= float(H * W) * 0.0005]
    if not sig_bot and contours_bot:
        sig_bot = [max(contours_bot, key=cv2.contourArea)]
    mask_bot = np.zeros_like(gray_u8)
    if sig_bot:
        cv2.drawContours(mask_bot, sig_bot, -1, 255, cv2.FILLED)
    mask_bot = binary_fill_holes(mask_bot.astype(bool)).astype(np.uint8) * 255

    # 4. Extract 1D Boundary Vectors
    y_top = np.full(W, -1, dtype=float)
    y_bottom = np.full(W, -1, dtype=float)

    for x in range(W):
        ys_t = np.where(mask_top[:, x] > 0)[0]
        if len(ys_t) > 0:
            y_top[x] = ys_t[0]

        ys_b = np.where(mask_bot[:, x] > 0)[0]
        if len(ys_b) > 0:
            y_bottom[x] = ys_b[-1]

    valid_t = (y_top >= 0)
    valid_b = (y_bottom >= 0)
    if not np.any(valid_t) or not np.any(valid_b):
        return np.ones_like(gray_u8, dtype=np.uint8) * 255

    x_t = np.where(valid_t)[0]
    x_b = np.where(valid_b)[0]
    y_top_interp = np.interp(np.arange(W), x_t, y_top[valid_t])
    y_bottom_interp = np.interp(np.arange(W), x_b, y_bottom[valid_b])

    # Outlier Rejection
    y_top_clean = reject_outliers_1d(y_top_interp, gray_u8=gray_u8, x_all=np.arange(W))
    y_bottom_clean = reject_outliers_1d(y_bottom_interp, gray_u8=gray_u8, x_all=np.arange(W))

    # 5. Interpolate y_bottom across compass box columns
    if compass_bbox is not None:
        bx0, by0, bx1, by1 = compass_bbox
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

    # 6. Hypertransmission / Shadow Beam Floor Capping Rule
    valid_floor = y_bottom_clean[y_bottom_clean > 0]
    if len(valid_floor) > 20:
        p35_floor = float(np.percentile(valid_floor, 35))
        max_floor_cap = p35_floor + max(45.0, float(margin_bottom) * 2.5)
        y_bottom_clean = np.minimum(y_bottom_clean, max_floor_cap)

    # 7. 1D Gaussian Smoothing
    sigma_val = float(gaussian_sigma)
    y_top_final = gaussian_filter1d(y_top_clean, sigma=sigma_val)
    y_bottom_final = gaussian_filter1d(y_bottom_clean, sigma=sigma_val)

    # 8. Corner Tapering & Clearing Limit (SFCM vs Otsu)
    use_sfcm_val = bool(use_sfcm or kwargs.get("use_sfcm", False))
    if use_sfcm_val:
        sfcm_margin = float(kwargs.get("sfcm_margin_bottom", margin_bottom))
        sfcm_sigma = float(kwargs.get("sfcm_gaussian_sigma", sigma_val))
        n_clusters = int(kwargs.get("sfcm_n_clusters", 3))
        m = float(kwargs.get("sfcm_fuzziness_m", 2.0))
        clear_limit = compute_sfcm_choroid_boundary(
            gray_u8, y_top_final, sfcm_margin, sfcm_sigma, n_clusters=n_clusters, m=m
        )
    else:
        clear_limit = y_bottom_final + margin_bottom
        if clear_corners:
            ch, cw = int(H * 0.28), int(W * 0.28)
            for x_range in (range(min(cw, W)), range(max(0, W - cw), W)):
                for x in x_range:
                    last_t = y_bottom_final[x] + margin_bottom
                    if H - ch > last_t:
                        clear_limit[x] = min(clear_limit[x], H - ch)
            clear_limit = gaussian_filter1d(clear_limit, sigma=sigma_val)

    # 9. Strict Blackout Box Capping
    by0_map = np.full(W, float(H - 1), dtype=float)
    has_blackout = False

    if compass_bbox is not None:
        bx0, by0, bx1, by1 = compass_bbox
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

    # 10. Render Continuous Organic Envelope Mask
    envelope_mask = np.zeros_like(gray_u8)
    for x in range(W):
        yt = max(0, int(y_top_final[x]) - margin_top)
        yb = min(H - 1, int(clear_limit[x]))
        if yb >= yt:
            envelope_mask[yt:yb + 1, x] = 255

    return envelope_mask
