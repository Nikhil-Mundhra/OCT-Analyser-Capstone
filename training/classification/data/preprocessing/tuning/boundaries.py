"""
data/preprocessing/tuning/boundaries.py

Retinal layer boundary detection, spike filtering, Dynamic Programming RPE tracking,
Spatial Fuzzy C-Means (SFCM) choroidal segmentation, and vector projection algorithms.
"""

from typing import Optional
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

from data.preprocessing.outliers import reject_outliers_1d


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

    Preserves steep foveal peaks and sharp upward contours while strictly rejecting
    vitreous floaters and deep fluid cysts.
    """
    params = params or {}
    h, w = gray_u8.shape

    gradient_weight = float(params.get("ilm_gradient_weight", 0.80))
    smooth_weight = float(params.get("ilm_smooth_weight", 0.20))
    band_half = int(params.get("ilm_search_band_px", 50))
    max_step = 6  # Allows climbing steep anatomical foveal peaks

    # 1. Candidate envelope guide (gentle smoothing preserves steep peaks)
    if y_cand_top is not None:
        valid_mask = (y_cand_top > 10) & (y_cand_top < int(h * 0.75))
        x_all = np.arange(w)
        if np.any(valid_mask):
            y_guide = np.interp(x_all, x_all[valid_mask], y_cand_top[valid_mask])
        else:
            y_guide = np.full(w, float(h * 0.25))
    else:
        y_guide = np.full(w, float(h * 0.25))

    # Sigma=6 preserves true anatomical arches and steep foveal peaks without flatlining
    y_guide_smooth = gaussian_filter1d(y_guide, sigma=6.0)

    search_min_y = np.maximum(5, (y_guide_smooth - band_half).astype(int))
    search_max_y = np.minimum(int(h * 0.70), (y_guide_smooth + band_half).astype(int))

    # 2. Speckle suppression via 2D Gaussian blur
    smoothed = gaussian_filter(gray_u8.astype(float), sigma=1.8)

    # 3. Downward dark->bright vertical gradient (vitreous -> ILM)
    grad_down = np.diff(smoothed, axis=0, prepend=smoothed[:1, :])
    grad_down = np.clip(grad_down, 0.0, None)
    g_max = grad_down.max()
    norm_grad = grad_down / (g_max + 1e-6)

    # 4. Vitreous darkness factor above y:
    # A true ILM MUST have dark vitreous above it (mean intensity in y-12..y-1 < 40).
    # If the region above is already bright (>45), this is an internal retinal layer, NOT the ILM!
    cum_img = np.cumsum(smoothed, axis=0)
    win = 12
    padded_cum = np.pad(cum_img, ((win, 0), (0, 0)), mode='edge')
    above_mean = (padded_cum[win:h+win, :] - padded_cum[:h, :]) / float(win)
    internal_penalty = np.clip((above_mean - 35.0) / 25.0, 0.0, 1.0)

    # Cost field: strong dark->bright gradient lowers cost; internal tissue above increases cost
    cost_field = 1.0 - (gradient_weight * norm_grad) + 0.50 * internal_penalty

    for x in range(w):
        min_y = search_min_y[x]
        max_y = search_max_y[x]
        if max_y > min_y:
            cost_field[:min_y, x] = 10.0
            cost_field[max_y:, x] = 10.0

    # 5. Dynamic Programming left -> right
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

    # 6. Backtrack from rightmost column
    ilm_path = np.zeros(w, dtype=float)
    last_min, last_max = search_min_y[-1], search_max_y[-1]
    best_last_y = last_min + int(np.argmin(dp_cost[last_min:max(last_min + 1, last_max), -1]))
    ilm_path[-1] = float(best_last_y)

    curr_y = int(best_last_y)
    for x in range(w - 1, 0, -1):
        curr_y = backtrack[curr_y, x]
        ilm_path[x - 1] = float(curr_y)

    # 7. Post-smooth: Savitzky-Golay
    win_len = 21 if w >= 21 else (w - 1 if w % 2 == 0 else w)
    if win_len >= 5:
        ilm_path = savgol_filter(ilm_path, window_length=win_len, polyorder=2)
    else:
        ilm_path = gaussian_filter1d(ilm_path, sigma=6.0)

    return np.clip(ilm_path, 0, h - 1)


def detect_rpe_band(gray_u8: np.ndarray, y_top_outer: np.ndarray, params: Optional[dict] = None) -> np.ndarray:
    """
    SOTA Dynamic Programming / Graph-Search RPE Detection Algorithm (Chiu et al. framework).
    Locks onto the hyperreflective melanin peak below the neurosensory retina.
    Enforces strict physiological retinal search bounds to prevent falling into deep black space
    or elevating falsely inside cystoid spaces.
    """
    h, w = gray_u8.shape
    params = params or {}

    smooth_weight = float(params.get("rpe_smooth_weight", 0.20))
    gradient_weight = float(params.get("rpe_gradient_weight", 0.35))
    reflectivity_weight = float(params.get("rpe_reflectivity_weight", 0.45))
    max_step = 4

    smoothed = gaussian_filter(gray_u8.astype(float), sigma=2.0)
    norm_img = smoothed / (smoothed.max() + 1e-6)

    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    norm_grad = np.clip(grad_y / (np.abs(grad_y).max() + 1e-6), 0, None)

    # Search window: from ILM + min physiological retinal thickness down to limit
    max_retina_px = min(250, int(h * 0.55))
    min_retina_px = max(18, int(h * 0.10))
    search_min_y = np.maximum(0, (y_top_outer + min_retina_px).astype(int))
    search_max_y = np.minimum(h - 1, (y_top_outer + max_retina_px).astype(int))

    # Cost field: RPE is the hyperreflective peak with sharp downward entry gradient
    rpe_cost_field = 1.0 - (reflectivity_weight * norm_img + gradient_weight * norm_grad)

    for x in range(w):
        min_y = search_min_y[x]
        max_y = search_max_y[x]
        if max_y > min_y:
            rpe_cost_field[:min_y, x] = 10.0
            rpe_cost_field[max_y:, x] = 10.0

    dp_cost = np.full((h, w), 1e6, dtype=float)
    backtrack = np.zeros((h, w), dtype=int)

    min_y_0, max_y_0 = search_min_y[0], search_max_y[0]
    if max_y_0 > min_y_0:
        dp_cost[min_y_0:max_y_0, 0] = rpe_cost_field[min_y_0:max_y_0, 0]

    for x in range(1, w):
        y_min_curr, y_max_curr = search_min_y[x], search_max_y[x]
        y_min_prev, y_max_prev = search_min_y[x - 1], search_max_y[x - 1]

        if y_max_curr <= y_min_curr or y_max_prev <= y_min_prev:
            continue

        for y in range(y_min_curr, y_max_curr):
            p_start = max(y_min_prev, y - max_step)
            p_end = min(y_max_prev, y + max_step + 1)

            if p_start < p_end:
                prev_ys = np.arange(p_start, p_end)
                transition_penalties = smooth_weight * ((prev_ys - y) ** 2)
                candidate_costs = dp_cost[prev_ys, x - 1] + transition_penalties
                best = int(np.argmin(candidate_costs))
                dp_cost[y, x] = rpe_cost_field[y, x] + candidate_costs[best]
                backtrack[y, x] = prev_ys[best]

    rpe_path = np.zeros(w, dtype=float)
    last_min, last_max = search_min_y[-1], search_max_y[-1]
    best_last_y = last_min + int(np.argmin(dp_cost[last_min:max(last_min + 1, last_max), -1]))
    rpe_path[-1] = float(best_last_y)

    curr_y = int(best_last_y)
    for x in range(w - 1, 0, -1):
        curr_y = backtrack[curr_y, x]
        rpe_path[x - 1] = float(curr_y)

    win_len = 25 if w >= 25 else (w - 1 if w % 2 == 0 else w)
    if win_len >= 5:
        smoothed_rpe = savgol_filter(rpe_path, window_length=win_len, polyorder=2)
    else:
        smoothed_rpe = gaussian_filter1d(rpe_path, sigma=8.0)

    return np.clip(smoothed_rpe, 0, h - 1)

    smoothed_rpe = np.maximum(smoothed_rpe, rpe_path)
    return smoothed_rpe


def compute_sfcm_choroid_boundary(
    gray_u8: np.ndarray,
    y_top_outer: np.ndarray,
    params: dict,
    return_raw: bool = False
) -> tuple:
    """
    Spatial Fuzzy C-Means (SFCM) clustering to segment the vascular choroidal layer.
    Returns (y_rpe, y_bottom_sfcm_safe) or (y_rpe, y_bottom_sfcm_safe, y_bottom_sfcm_raw) if return_raw=True.
    """
    h, w = gray_u8.shape
    y_rpe = detect_rpe_band(gray_u8, y_top_outer, params=params)

    margin_bottom = float(params.get("sfcm_margin_bottom", params.get("margin_bottom", 15)))
    gaussian_sigma = float(params.get("sfcm_gaussian_sigma", params.get("gaussian_sigma", 15)))
    n_clusters = int(params.get("sfcm_n_clusters", 3))
    m = float(params.get("sfcm_fuzziness_m", 2.0))
    max_iter = 10

    y_bottom_sfcm = np.zeros(w, dtype=float)
    sub_pixels = []
    coords = []
    max_depth_px = min(140, int(h * 0.30))

    for x in range(w):
        y_start = min(h - 10, max(0, int(y_rpe[x]) + 2))
        y_end = min(h, y_start + max_depth_px)
        for y in range(y_start, y_end):
            sub_pixels.append(float(gray_u8[y, x]))
            coords.append((y, x))

    if len(sub_pixels) < 100:
        raw_fallback = y_rpe + margin_bottom + 40.0
        slack_bottom_px = float(params.get("sfcm_slack_bottom_px", 20))
        safe_fallback = np.minimum(h - 1, raw_fallback + slack_bottom_px)
        if return_raw:
            return y_rpe, safe_fallback, raw_fallback
        return y_rpe, safe_fallback

    sub_pixels_arr = np.array(sub_pixels, dtype=float)
    min_val, max_val = np.percentile(sub_pixels_arr, 5), np.percentile(sub_pixels_arr, 95)
    centroids = np.linspace(min_val, max_val, n_clusters)

    u_matrix = None
    for _ in range(max_iter):
        dists = np.abs(sub_pixels_arr[:, None] - centroids[None, :]) + 1e-5
        inv_dists = (1.0 / dists) ** (2.0 / (m - 1.0))
        u_matrix = inv_dists / np.sum(inv_dists, axis=1, keepdims=True)

        u_m = u_matrix ** m
        centroids = np.sum(u_m * sub_pixels_arr[:, None], axis=0) / (np.sum(u_m, axis=0) + 1e-5)
        centroids = np.sort(centroids)

    stroma_map = np.zeros((h, w), dtype=float)
    if u_matrix is not None:
        for idx, (y, x) in enumerate(coords):
            stroma_map[y, x] = u_matrix[idx, 1]

    sfcm_smooth = gaussian_filter(stroma_map, sigma=(2.0, 3.0))

    for x in range(w):
        y_start = min(h - 10, max(0, int(y_rpe[x]) + 2))
        y_end = min(h, y_start + max_depth_px)
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

    y_bottom_sfcm_raw = gaussian_filter1d(y_bottom_sfcm, sigma=gaussian_sigma)

    # Apply customizable downward trust buffer / slack margin
    slack_bottom_px = float(params.get("sfcm_slack_bottom_px", 20))
    y_bottom_sfcm_safe = np.minimum(h - 1, y_bottom_sfcm_raw + slack_bottom_px)

    if return_raw:
        return y_rpe, y_bottom_sfcm_safe, y_bottom_sfcm_raw
    return y_rpe, y_bottom_sfcm_safe


def chaikin_subdivision(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Applies Chaikin's corner cutting algorithm to smooth closed polygon contours."""
    pts = points.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return points

    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.append(q)
            new_pts.append(r)
        pts = np.array(new_pts)
    return pts.reshape(-1, 1, 2)


def refine_lumen_boundary(
    gray_u8: np.ndarray,
    initial_cnt: np.ndarray,
    choroid_roi: np.ndarray,
    num_rays: int = 36
) -> np.ndarray:
    """
    Refines a lumen contour using sub-pixel radial raycasting to snap to the hyperreflective
    collagen vessel wall, followed by periodic circular Gaussian smoothing and Chaikin subdivision.
    Strictly clamps all vertices inside the choroid boundary envelope.
    """
    h, w = gray_u8.shape
    M = cv2.moments(initial_cnt)
    if M["m00"] == 0:
        return initial_cnt
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    area = cv2.contourArea(initial_cnt)
    if area < 15:
        return initial_cnt

    pts = initial_cnt.reshape(-1, 2)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    radii = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)

    # Sort angular samples
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]

    # Target uniform angular grid
    target_angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    target_radii = np.interp(target_angles, angles_sorted, radii_sorted, period=2 * np.pi)

    # Search outward along each ray for peak gradient (vessel wall)
    blur = cv2.GaussianBlur(gray_u8, (3, 3), 1.0)
    refined_radii = []

    for theta, r_init in zip(target_angles, target_radii):
        r_candidates = np.linspace(max(2.0, r_init * 0.75), r_init * 1.25, 12)
        grad_vals = []
        for r_c in r_candidates:
            px = int(round(cx + r_c * np.cos(theta)))
            py = int(round(cy + r_c * np.sin(theta)))
            if 1 <= px < w - 1 and 1 <= py < h - 1 and choroid_roi[py, px] > 0:
                dx = float(blur[py, px + 1]) - float(blur[py, px - 1])
                dy = float(blur[py + 1, px]) - float(blur[py - 1, px])
                rad_grad = dx * np.cos(theta) + dy * np.sin(theta)
                grad_vals.append(rad_grad)
            else:
                grad_vals.append(0.0)

        best_idx = np.argmax(grad_vals) if max(grad_vals) > 4.0 else len(r_candidates) // 2
        refined_radii.append(r_candidates[best_idx])

    # Periodic 1D Gaussian circular filter
    refined_radii = np.array(refined_radii)
    padded = np.concatenate([refined_radii[-6:], refined_radii, refined_radii[:6]])
    smoothed_padded = gaussian_filter1d(padded, sigma=1.4, mode='wrap')
    smoothed_radii = smoothed_padded[6:-6]

    smooth_x = cx + smoothed_radii * np.cos(target_angles)
    smooth_y = cy + smoothed_radii * np.sin(target_angles)

    # Clamp each vertex to ensure 100% inside valid choroid ROI
    clamped_pts = []
    for sx, sy in zip(smooth_x, smooth_y):
        ix = int(np.clip(round(sx), 0, w - 1))
        iy = int(np.clip(round(sy), 0, h - 1))
        if choroid_roi[iy, ix] > 0:
            clamped_pts.append([sx, sy])
        else:
            # Fallback to centroid direction clamp
            clamped_pts.append([cx + 0.8 * (sx - cx), cy + 0.8 * (sy - cy)])

    smooth_pts = np.array(clamped_pts).reshape(-1, 1, 2).astype(np.float32)
    subdivided = chaikin_subdivision(smooth_pts, iterations=1)
    return subdivided.astype(np.int32)


def detect_choroidal_holes(
    gray_u8: np.ndarray,
    y_rpe: np.ndarray,
    y_bottom_sfcm: np.ndarray,
    params: Optional[dict] = None
) -> list[dict]:
    """
    Segments individual choroidal holes / vascular lumens / cavitations strictly within the choroid layer.
    Extracts exact polygon contours, geometric circularity, area, and bounding boxes.

    Robustness & Boundary Constraints:
    - Strictly bound between y_rpe + 3 and y_bottom_sfcm - 2.
    - Zero pixels allowed outside the choroidal envelope.
    - Local adaptive lumen-to-stroma contrast thresholding.
    - Sub-pixel radial raycasting wall gradient snapping and Chaikin smoothing.
    - Excludes non-tissue image edges and background dropout voids.
    """
    h, w = gray_u8.shape
    params = params or {}

    min_area = int(params.get("hole_min_area", 25))
    max_area = int(params.get("hole_max_area", 15000))

    # 1. Identify tissue presence per column to exclude non-tissue scan borders
    tissue_col_mask = (np.sum(gray_u8 > 35, axis=0) > 25)
    valid_x = np.where(tissue_col_mask)[0]
    if len(valid_x) < 20:
        return []

    x_min, x_max = valid_x[0] + 15, valid_x[-1] - 15

    # 2. Build strict choroid ROI mask (100% bounded between RPE and SFCM bottom)
    choroid_roi = np.zeros((h, w), dtype=np.uint8)
    for x in range(x_min, x_max):
        yt = int(y_rpe[x]) + 3
        yb = int(y_bottom_sfcm[x]) - 2
        if yb > yt + 4:
            choroid_roi[yt:yb, x] = 255

    stroma_pixels = gray_u8[choroid_roi > 0]
    if len(stroma_pixels) < 50 or np.mean(stroma_pixels) < 25:
        return []

    contrast_offset = float(params.get("hole_contrast_offset", 8.0))
    local_window = int(params.get("hole_local_window", 25))
    if local_window % 2 == 0:
        local_window += 1
    local_window = max(5, min(61, local_window))

    # 3. UPSTREAM DESPECKLING: Edge-preserving bilateral filter + median blur
    denoised = cv2.bilateralFilter(gray_u8, d=7, sigmaColor=40, sigmaSpace=5)
    denoised = cv2.medianBlur(denoised, 3)

    # 4. ROBUST LOCAL MEDIAN BASELINE (Immune to speckle baseline drag)
    local_median = cv2.medianBlur(denoised, local_window)

    lumen_mask = (denoised < (local_median - contrast_offset)) & (choroid_roi > 0) & (gray_u8 > 0)
    lumen_u8 = lumen_mask.astype(np.uint8) * 255

    # Morphological opening and closing
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lumen_u8 = cv2.morphologyEx(lumen_u8, cv2.MORPH_OPEN, kernel_open)
    lumen_u8 = cv2.morphologyEx(lumen_u8, cv2.MORPH_CLOSE, kernel_close)

    # CRITICAL: Hard clip strictly to choroid ROI so ZERO pixels can exist outside boundary
    lumen_u8 = cv2.bitwise_and(lumen_u8, choroid_roi)

    max_ar = float(params.get("hole_max_aspect_ratio", 2.8))

    # 5. Connected components & Distance Transform Watershed cluster decomposition
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lumen_u8, connectivity=8)

    candidate_contours = []

    for i in range(1, num_labels):
        comp_area = stats[i, cv2.CC_STAT_AREA]
        if comp_area < min_area or comp_area > max_area:
            continue

        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        ar = float(bw) / max(1.0, float(bh))
        inv_ar = float(bh) / max(1.0, float(bw))

        comp_mask = (labels == i).astype(np.uint8) * 255

        if ar <= max_ar and inv_ar <= max_ar:
            # Clean individual lumen/cavern
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidate_contours.extend(cnts)
        else:
            # Clustered / elongated multi-vessel region: Decompose into constituent lumens via Watershed
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            if dist.max() > 2.0:
                _, sure_fg = cv2.threshold(dist, 0.38 * dist.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)

                n_seeds, markers = cv2.connectedComponents(sure_fg)

                if n_seeds > 2:
                    markers = markers + 1
                    markers[comp_mask == 0] = 0

                    img_ws = cv2.cvtColor(comp_mask, cv2.COLOR_GRAY2BGR)
                    markers = cv2.watershed(img_ws, markers)

                    for m_id in range(2, n_seeds + 1):
                        sub_mask = (markers == m_id).astype(np.uint8) * 255
                        sub_cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for sc in sub_cnts:
                            s_area = cv2.contourArea(sc)
                            if s_area >= min_area:
                                _, _, sbw, sbh = cv2.boundingRect(sc)
                                if (float(sbw) / max(1.0, float(sbh)) <= max_ar) and (float(sbh) / max(1.0, float(sbw)) <= max_ar):
                                    candidate_contours.append(sc)
                else:
                    # Morphological isthmus erosion separation for touching vessel segments
                    eroded = cv2.erode(comp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
                    n_eroded, er_labels = cv2.connectedComponents(eroded)
                    if n_eroded > 2:
                        for eid in range(1, n_eroded):
                            sub_er = (er_labels == eid).astype(np.uint8) * 255
                            sub_dil = cv2.dilate(sub_er, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
                            sub_dil = cv2.bitwise_and(sub_dil, comp_mask)
                            sub_cnts, _ = cv2.findContours(sub_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for sc in sub_cnts:
                                if cv2.contourArea(sc) >= min_area:
                                    candidate_contours.append(sc)

    holes = []
    min_circ = 0.25
    min_solidity = 0.60

    for cnt in candidate_contours:
        c_area = cv2.contourArea(cnt)
        if min_area <= c_area <= max_area:
            # Secondary shape gates: circularity and solidity
            peri = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * c_area / (peri * peri + 1e-5)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(c_area) / max(1.0, hull_area)

            if circ < min_circ or solidity < min_solidity:
                continue

            # Refine boundary using sub-pixel radial raycasting and gradient snapping
            refined_cnt = refine_lumen_boundary(gray_u8, cnt, choroid_roi)
            ref_area = cv2.contourArea(refined_cnt)
            if ref_area >= min_area:
                x, y, bw, bh = cv2.boundingRect(refined_cnt)
                # Guarantee every vertex is strictly within the choroid boundary
                pts = refined_cnt.reshape(-1, 2)
                inside_all = True
                for px, py in pts:
                    if px < 0 or px >= w or py < 0 or py >= h or choroid_roi[py, px] == 0:
                        inside_all = False
                        break
                if inside_all:
                    peri_ref = cv2.arcLength(refined_cnt, True)
                    circ_ref = 4 * np.pi * ref_area / (peri_ref * peri_ref + 1e-5)

                    epsilon = 0.010 * peri_ref
                    approx = cv2.approxPolyDP(refined_cnt, epsilon, True)

                    holes.append({
                        "contour": approx,
                        "area": float(ref_area),
                        "circularity": float(circ_ref),
                        "bbox": (int(x), int(y), int(bw), int(bh))
                    })

    return holes


def detect_choroidal_caverns(
    gray_u8: np.ndarray,
    y_rpe: np.ndarray,
    y_bottom_sfcm: np.ndarray,
    params: Optional[dict] = None
) -> list[dict]:
    """
    Detects pathological choroidal caverns / cavitations using multi-feature morphology,
    absence of hyperreflective vessel sheaths, and posterior hypertransmission analysis.

    Physics & Optical Signature:
    - Pathological caverns are empty, non-vascular pseudocysts with zero internal signal.
    - Unlike normal choroidal blood vessels (which have reflective collagen walls and cast shadows),
      caverns lack structural walls and exhibit posterior hypertransmission ('lighthouse effect')
      because unobstructed OCT laser beams pass freely through the empty cavity into the sclera.

    Parameters:
    - gray_u8: Grayscale OCT B-scan uint8 array (H, W).
    - y_rpe: Detected RPE/Bruch's membrane boundary (ceiling of choroid).
    - y_bottom_sfcm: Detected Choroid-Scleral Interface boundary (floor of choroid).
    - params: Dictionary containing detection thresholds:
        - cavern_min_area: Minimum area in pixels (default: 15).
        - cavern_max_area: Maximum area in pixels (default: 900).
        - cavern_dark_threshold: Maximum grayscale intensity for core void (default: 45).
        - cavern_transmission_threshold: Minimum sub-transmission ratio (default: 1.30).
        - cavern_min_circularity: Minimum roundness score (default: 0.60).
        - sfcm_slack_bottom_px: Downward slack buffer in pixels (default: 20).

    Returns:
    - List of detected cavern dictionaries with bounding box, centroid, circularity, and transmission ratio.
    """
    h, w = gray_u8.shape
    params = params or {}

    min_area = int(params.get("cavern_min_area", 15))
    max_area = int(params.get("cavern_max_area", 900))
    dark_thresh = int(params.get("cavern_dark_threshold", 45))
    trans_thresh = float(params.get("cavern_transmission_threshold", 1.30))
    min_circ = float(params.get("cavern_min_circularity", 0.60))
    slack_px = int(params.get("sfcm_slack_bottom_px", 20))

    # 1. Build augmented choroid ROI mask including downward slack buffer
    choroid_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        top_y = min(h - 1, max(0, int(y_rpe[x]) + 2))
        bot_y = min(h, max(top_y + 1, int(y_bottom_sfcm[x]) + slack_px))
        choroid_mask[top_y:bot_y, x] = 1

    # 2. Extract zero-signal core voids embedded inside the choroid
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

        # Circularity metric: 4 * pi * Area / Perimeter^2
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

        # Posterior Hypertransmission Analysis ("Lighthouse" effect)
        # Sample sub-lesional window directly below the hole inside the slack buffer
        sub_y_start = min(h - 1, ry + rh)
        sub_y_end = min(h, sub_y_start + max(15, slack_px))

        if sub_y_end > sub_y_start:
            sub_signal = float(np.mean(gray_u8[sub_y_start:sub_y_end, rx:rx + rw]))

            # Lateral background reference at matching depth
            left_w = max(0, rx - rw)
            right_w = min(w, rx + 2 * rw)
            lateral_samples = []
            if rx > left_w:
                lateral_samples.append(np.mean(gray_u8[ry:ry + rh, left_w:rx]))
            if right_w > rx + rw:
                lateral_samples.append(np.mean(gray_u8[ry:ry + rh, rx + rw:right_w]))

            ref_bg = float(np.mean(lateral_samples)) if lateral_samples else max(1.0, sub_signal)
            ref_bg = max(1.0, ref_bg)

            trans_ratio = sub_signal / ref_bg

            # If posterior transmission exceeds threshold, confirm as cavern
            if trans_ratio >= trans_thresh:
                caverns.append({
                    "bbox": [int(rx), int(ry), int(rw), int(rh)],
                    "area": int(area),
                    "circularity": round(float(circularity), 3),
                    "transmission_ratio": round(float(trans_ratio), 3),
                    "centroid": [round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)]
                })

    return caverns


def _estimate_adaptive_thresholds(gray_u8: np.ndarray, top_mult: float = 2.2, bot_mult: float = 1.6) -> tuple[int, int, int]:
    """
    Computes Pass 1 (ILM Vitreoretinal) and Pass 2 (Choroid) adaptive intensity thresholds.
    Directly isolates scanner vitreous background noise from the lower-quartile distribution
    of non-zero pixels, invariant to image vertical position, macular elevation, or tilt.
    """
    valid = gray_u8.copy()
    valid[valid >= 190] = 0
    non_zero = valid[(valid > 0) & (valid < 190)]

    if len(non_zero) > 100:
        # Background is strictly the lower 20% of non-zero intensities across the scan
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
) -> tuple[np.ndarray, np.ndarray]:
    """Generates morphological masks and extracts raw 1D column-wise boundary indices."""
    h, w = gray_u8.shape

    # --- Pitch-black column exclusion ---
    # A column is "pitch black" (scan border or signal dropout) if fewer than 5% of its
    # pixels have any intensity > 15.  The shadow-bridge closing kernel must never sweep
    # across these columns, because doing so would bridge real tissue into empty space.
    tissue_col = np.sum(gray_u8 > 15, axis=0) > int(h * 0.05)   # shape (w,)
    col_mask = np.zeros((h, w), dtype=np.uint8)
    col_mask[:, tissue_col] = 255

    _, thresh_top = cv2.threshold(gray_u8, thresh_top_val, 255, cv2.THRESH_BINARY)
    _, thresh_bot = cv2.threshold(gray_u8, thresh_bot_val, 255, cv2.THRESH_BINARY)

    # Enforce: pitch-black pixels and non-tissue columns are always background
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
    # Re-enforce column mask after closing so the kernel expansion is clipped
    closed_top = cv2.bitwise_and(closed_top, col_mask)
    contours_top, _ = cv2.findContours(closed_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_top = [c for c in contours_top if cv2.contourArea(c) >= float(h * w) * 0.0005]
    if not sig_top and contours_top:
        sig_top = [max(contours_top, key=cv2.contourArea)]

    # --- Floater / exudate rejection ---
    # Keep only the single largest contour that is not floating in the upper vitreous.
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
) -> tuple[np.ndarray, np.ndarray]:
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


def get_sfcm_cache_key(src_path: Optional[str], shape: tuple, params: dict) -> tuple:
    """Generates an immutable hashable cache key for SFCM Choroid results."""
    return (
        src_path,
        shape,
        params.get("sfcm_n_clusters", 3),
        params.get("sfcm_fuzziness_m", 2.0),
        params.get("sfcm_gaussian_sigma", 15),
        params.get("sfcm_margin_bottom", 16),
        params.get("sfcm_slack_bottom_px", 20),
        params.get("rpe_smooth_weight", 0.20),
        params.get("rpe_depth_weight", 0.40),
        params.get("rpe_gradient_weight", 0.30),
        params.get("rpe_bottom_env_size", 15),
        params.get("detect_caverns", False),
        params.get("cavern_transmission_threshold", 1.35),
        params.get("hole_min_area", 25),
        params.get("hole_max_area", 15000),
        params.get("hole_contrast_offset", 8),
        params.get("hole_local_window", 15),
        params.get("hole_max_aspect_ratio", 2.8),
        params.get("top_noise_mult", 1.5),
        params.get("gaussian_sigma", 15)
    )


def detect_boundaries_intelligent_auto(
    gray_u8: np.ndarray,
    params: Optional[dict] = None,
    compass_bbox: Optional[tuple] = None,
    src_path: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Autonomous Per-Image Intelligent Multi-Surface Segmentation Engine.

    Physics & Anatomical Grounding:
    - Scans each column as an optical backscatter profile:
        Vitreous (0) -> ILM (dark->bright) -> Parenchyma/Cysts -> RPE (melanin max) -> Choroid (vascular) -> CSI
    - Automatically cleans scanner HUDs/banners without losing retinal tissue.
    - Uses vertical cross-sectional intensity profiling to locate the primary neurosensory retinal mass.
    - Anchors the ILM DP search corridor to the verified physiological entry gradient, immune to vitreous haze.
    - Coupled multi-surface DP: ILM via gradient DP, RPE via Chiu DP, Choroid via SFCM.
    - 6-Point Anatomical Invariant Safety Gate verifies biological validity and self-corrects.
    """
    params = params or {}
    h, w = gray_u8.shape
    clean_gray = gray_u8.copy()

    # 1. Dynamic Scanner HUD & Metadata Banner Blanking
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

    # 2. Joint Simultaneous Multi-Surface Graph Optimization & B-Spline Regularization
    from data.preprocessing.tuning.multisurface import JointMultiSurfaceOptimizer, MultiSurfaceConfig

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

    # 3. 6-Point Anatomical Invariant Safety Gate
    retinal_thickness = y_rpe - y_top_opt
    choroid_thickness = y_sfcm - y_rpe

    inv_order = np.all(y_top_opt < y_rpe) and np.all(y_rpe < y_sfcm)
    inv_ret_thick = (float(np.mean(retinal_thickness)) >= 30.0) and (float(np.mean(retinal_thickness)) <= 350.0)
    inv_cho_thick = (float(np.mean(choroid_thickness)) >= 15.0) and (float(np.mean(choroid_thickness)) <= 250.0)
    inv_border = float(np.mean(y_top_opt)) > 5.0 and float(np.std(y_top_opt)) > 0.1
    inv_continuity = np.max(np.abs(np.diff(y_top_opt))) <= 25.0

    all_passed = inv_order and inv_ret_thick and inv_cho_thick and inv_border and inv_continuity

    if not all_passed:
        # Fallback to guided boundary recovery
        vert_profile = np.mean(clean_gray, axis=1)
        mass_y = float(np.argmax(gaussian_filter1d(vert_profile, sigma=8.0)))
        y_top_fallback = np.full(w, max(20.0, mass_y - 60.0))
        y_top_opt = _detect_ilm_dp(clean_gray, y_cand_top=y_top_fallback, params={"ilm_gradient_weight": 0.80, "ilm_smooth_weight": 0.20})
        y_rpe = detect_rpe_band(clean_gray, y_top_opt)
        _, y_sfcm, y_sfcm_raw = compute_sfcm_choroid_boundary(clean_gray, y_top_opt, params, return_raw=True)

    # 4. Construct Final Tissue Extraction Envelope
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


def generate_tissue_mask_custom(
    gray_u8: np.ndarray,
    params: dict,
    compass_bbox: Optional[tuple] = None,
    return_sfcm: bool = False,
    src_path: Optional[str] = None,
    sfcm_cache: Optional[dict] = None
) -> tuple:
    """Custom fine-tuning tissue mask generator supporting Manual (Otsu/SFCM/DP) and Intelligent Auto Mode."""
    # When Auto Mode is active, delegate directly to the autonomous multi-surface engine
    if bool(params.get("auto_mode", False)):
        ret_auto = detect_boundaries_intelligent_auto(gray_u8, params=params, compass_bbox=compass_bbox, src_path=src_path)
        if return_sfcm:
            return ret_auto
        env_m, y_t_out, y_b_out, _, _, _ = ret_auto
        return env_m, y_t_out, y_b_out

    gray_u8 = gray_u8.copy()
    gray_u8[gray_u8 >= 190] = 0
    h, w = gray_u8.shape

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

    use_dp_ilm = bool(params.get("use_dp_ilm", False))

    thresh_top_val, thresh_bot_val, _ = _estimate_adaptive_thresholds(gray_u8, top_mult, bot_mult)
    y_top, y_bottom = _extract_raw_boundary_contours(
        gray_u8, thresh_top_val, thresh_bot_val, sb_top_pct, sb_bot_pct
    )

    # When DP ILM mode is active, replace the Otsu top path entirely.
    # y_bottom from the Otsu path is still used for Otsu bottom mode.
    if use_dp_ilm:
        y_top = _detect_ilm_dp(gray_u8, y_cand_top=y_top, params=params)

    y_top_final, y_bottom_final = _interpolate_and_filter_boundaries(
        gray_u8, y_top, y_bottom, compass_bbox, margin_bottom, gaussian_sigma,
        top_spike_px, top_spike_win, top_dip_px, top_dip_win
    )

    clear_limit = y_bottom_final + margin_bottom

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

    y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = None, None, None
    if use_sfcm:
        cache_key = get_sfcm_cache_key(src_path, gray_u8.shape, params) if src_path else None
        if cache_key and sfcm_cache is not None and cache_key in sfcm_cache:
            cached_val = sfcm_cache[cache_key]
            if len(cached_val) == 3:
                y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = cached_val
            else:
                y_rpe, y_bottom_sfcm = cached_val
                slack_px = float(params.get("sfcm_slack_bottom_px", 20))
                y_bottom_sfcm_raw = np.maximum(0, y_bottom_sfcm - slack_px)
        else:
            y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw = compute_sfcm_choroid_boundary(gray_u8, y_top_final, params, return_raw=True)
            if cache_key and sfcm_cache is not None:
                sfcm_cache[cache_key] = (y_rpe.copy(), y_bottom_sfcm.copy(), y_bottom_sfcm_raw.copy())

        if has_blackout:
            y_bottom_sfcm = np.minimum(y_bottom_sfcm, by0_map)
            y_bottom_sfcm_raw = np.minimum(y_bottom_sfcm_raw, by0_map)

    active_bottom = y_bottom_sfcm if use_sfcm else clear_limit
    envelope_mask = np.zeros_like(gray_u8)
    for x in range(w):
        yt = max(0, int(y_top_final[x]) - margin_top)
        yb = min(h - 1, int(active_bottom[x]))
        if yb >= yt:
            envelope_mask[yt:yb + 1, x] = 255

    y_top_outer = np.maximum(0, y_top_final - margin_top)
    y_bottom_outer = np.minimum(h - 1, clear_limit)

    if return_sfcm:
        return envelope_mask, y_top_outer, y_bottom_outer, y_rpe, y_bottom_sfcm, y_bottom_sfcm_raw

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
    y_top: Optional[np.ndarray] = None,
    y_bottom: Optional[np.ndarray] = None,
    y_rpe: Optional[np.ndarray] = None,
    y_sfcm: Optional[np.ndarray] = None,
    y_slack: Optional[np.ndarray] = None,
    pad_t: float = 0,
    pad_l: float = 0,
    scale: float = 1.0,
    n_pts: int = 64,
    y_top_outer: Optional[np.ndarray] = None,
    y_bottom_outer: Optional[np.ndarray] = None,
    y_bottom_sfcm: Optional[np.ndarray] = None,
    num_points: Optional[int] = None
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

    if y_slack is not None:
        slack_pts = [[float(x + pad_l) * scale, float(y_slack[x] + pad_t) * scale] for x in indices]
        return top_pts, bot_pts, rpe_pts, sfcm_pts, slack_pts

    return top_pts, bot_pts, rpe_pts, sfcm_pts
