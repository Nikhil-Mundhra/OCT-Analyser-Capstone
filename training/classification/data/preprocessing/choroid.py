"""
data/preprocessing/choroid.py

Choroidal layer analysis, Chiu Dynamic Programming RPE tracking,
Spatial Fuzzy C-Means (SFCM) stroma segmentation, vascular lumen raycasting,
and pathological cavern detection with posterior hypertransmission analysis.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import savgol_filter


def detect_rpe_band(
    gray_u8: np.ndarray,
    y_top_outer: np.ndarray,
    params: Optional[dict] = None
) -> np.ndarray:
    """
    Dynamic Programming / Graph-Search RPE Detection Algorithm (Chiu et al. framework).
    Locks onto the hyperreflective melanin peak below the neurosensory retina.
    Enforces physiological retinal search bounds to prevent falling into deep background space.
    """
    h, w = gray_u8.shape
    params = params or {}

    smooth_weight = float(params.get("rpe_smooth_weight", 0.20))
    gradient_weight = float(params.get("rpe_gradient_weight", 0.35))
    reflectivity_weight = float(params.get("rpe_reflectivity_weight", 0.45))
    max_step = int(params.get("rpe_max_step", 4))

    smoothed = gaussian_filter(gray_u8.astype(float), sigma=2.0)
    norm_img = smoothed / (smoothed.max() + 1e-6)

    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    norm_grad = np.clip(grad_y / (np.abs(grad_y).max() + 1e-6), 0, None)

    # Search window: from ILM + min physiological retinal thickness down to max limit
    max_retina_px = min(250, int(h * 0.55))
    min_retina_px = max(18, int(h * 0.10))
    search_min_y = np.maximum(0, (y_top_outer + min_retina_px).astype(int))
    search_max_y = np.minimum(h - 1, (y_top_outer + max_retina_px).astype(int))

    # Cost field: RPE is the hyperreflective peak with downward entry gradient
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
    Segments individual choroidal holes / vascular lumens strictly within the choroid layer.
    Extracts exact polygon contours, geometric circularity, area, and bounding boxes.
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

    # 2. Build strict choroid ROI mask (bounded between RPE and SFCM bottom)
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

    # 4. ROBUST LOCAL MEDIAN BASELINE
    local_median = cv2.medianBlur(denoised, local_window)

    lumen_mask = (denoised < (local_median - contrast_offset)) & (choroid_roi > 0) & (gray_u8 > 0)
    lumen_u8 = lumen_mask.astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lumen_u8 = cv2.morphologyEx(lumen_u8, cv2.MORPH_OPEN, kernel_open)
    lumen_u8 = cv2.morphologyEx(lumen_u8, cv2.MORPH_CLOSE, kernel_close)

    # Hard clip strictly to choroid ROI
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
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidate_contours.extend(cnts)
        else:
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
            peri = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * c_area / (peri * peri + 1e-5)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(c_area) / max(1.0, hull_area)

            if circ < min_circ or solidity < min_solidity:
                continue

            refined_cnt = refine_lumen_boundary(gray_u8, cnt, choroid_roi)
            ref_area = cv2.contourArea(refined_cnt)
            if ref_area >= min_area:
                x, y, bw, bh = cv2.boundingRect(refined_cnt)
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
    """
    h, w = gray_u8.shape
    params = params or {}

    min_area = int(params.get("cavern_min_area", 15))
    max_area = int(params.get("cavern_max_area", 900))
    dark_thresh = int(params.get("cavern_dark_threshold", 45))
    trans_thresh = float(params.get("cavern_transmission_threshold", 1.30))
    min_circ = float(params.get("cavern_min_circularity", 0.60))
    slack_px = int(params.get("sfcm_slack_bottom_px", 20))

    # Build augmented choroid ROI mask including downward slack buffer
    choroid_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        top_y = min(h - 1, max(0, int(y_rpe[x]) + 2))
        bot_y = min(h, max(top_y + 1, int(y_bottom_sfcm[x]) + slack_px))
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

        sub_y_start = min(h - 1, ry + rh)
        sub_y_end = min(h, sub_y_start + max(15, slack_px))

        if sub_y_end > sub_y_start:
            sub_signal = float(np.mean(gray_u8[sub_y_start:sub_y_end, rx:rx + rw]))

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

            if trans_ratio >= trans_thresh:
                caverns.append({
                    "bbox": [int(rx), int(ry), int(rw), int(rh)],
                    "area": int(area),
                    "circularity": round(float(circularity), 3),
                    "transmission_ratio": round(float(trans_ratio), 3),
                    "centroid": [round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)]
                })

    return caverns


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
