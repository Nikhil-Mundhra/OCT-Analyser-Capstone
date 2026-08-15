"""
data/preprocessing/outliers.py

Forgiving 3-Gate Outlier Rejection & Median-MAD Spike Filtering for 1D boundary traces.
Identifies and eliminates unphysical sharp vertical step outlier spikes before 1D Gaussian smoothing.
"""

import numpy as np
from scipy.ndimage import median_filter


def has_intensity_support(
    gray_u8: np.ndarray,
    col_x: int,
    boundary_y: int,
    half_height: int = 8,
    grad_thresh: float = 12.0,
) -> bool:
    """
    Returns True if a smooth reflectivity gradient supports a real tissue boundary
    at (col_x, boundary_y) in the raw image.
    """
    H, W = gray_u8.shape
    y0 = max(0, boundary_y - half_height)
    y1 = min(H, boundary_y + half_height + 1)
    if y1 - y0 < 2:
        return False
    for cx in range(max(0, col_x - 1), min(W, col_x + 2)):
        strip = gray_u8[y0:y1, cx].astype(np.float32)
        if float(np.abs(np.diff(strip)).max()) >= grad_thresh:
            return True
    return False


def reject_outliers_1d(
    y_vec: np.ndarray,
    gray_u8: np.ndarray = None,
    x_all: np.ndarray = None,
    window_radius: int = 11,
    hampel_scale: float = 3.0,
    neighbor_thresh: float = 14.0,
    max_run_width: int = 4,
    gradient_support_thresh: float = 12.0,
    gradient_half_height: int = 8,
    n_passes: int = 2,
) -> np.ndarray:
    """
    Forgiving Three-Gate Outlier Rejection Filter & Spike Interpolator.

    Gate 1 - Flagging (Hampel MAD + median deviation + bilateral neighbor test).
    Gate 2 - Width filter (allows runs up to max_run_width=4 to be filtered out as noise spikes).
    Gate 3 - Intensity validation (check pixel gradient support in raw image).
    """
    y_current = y_vec.copy()
    N = len(y_current)
    win_size = 2 * window_radius + 1
    use_intensity = (gray_u8 is not None) and (x_all is not None)

    for _ in range(n_passes):
        # Local Median Baseline
        med = median_filter(y_current, size=win_size)
        abs_dev = np.abs(y_current - med)
        mad = median_filter(abs_dev, size=win_size)
        local_sigma = np.maximum(3.0, 1.4826 * mad)
        hampel_mask = abs_dev > (hampel_scale * local_sigma)

        # Bilateral Neighbor Spike Test (detect single/narrow column vertical jumps)
        neighbor_mask = np.zeros(N, dtype=bool)
        for i in range(1, N - 1):
            if (abs(y_current[i] - y_current[i - 1]) > neighbor_thresh and
                    abs(y_current[i] - y_current[i + 1]) > neighbor_thresh):
                neighbor_mask[i] = True

        # Absolute Deviation Spike Test (> 14px deviation from local median)
        dev_mask = abs_dev > 14.0

        outlier_mask = hampel_mask | neighbor_mask | dev_mask
        if not np.any(outlier_mask):
            break

        # Gate 2: Width filter - release runs > max_run_width
        run_starts = np.where(np.diff(np.concatenate([[False], outlier_mask, [False]]).astype(int)) == 1)[0]
        run_ends   = np.where(np.diff(np.concatenate([[False], outlier_mask, [False]]).astype(int)) == -1)[0]
        for rs, re in zip(run_starts, run_ends):
            if (re - rs) > max_run_width:
                outlier_mask[rs:re] = False

        if not np.any(outlier_mask):
            break

        # Gate 3: Intensity validation - preserve points with real pixel gradient support
        if use_intensity:
            for i in np.where(outlier_mask)[0]:
                col_x = int(x_all[i])
                boundary_y = int(round(y_current[i]))
                if has_intensity_support(
                    gray_u8, col_x, boundary_y,
                    half_height=gradient_half_height,
                    grad_thresh=gradient_support_thresh,
                ):
                    outlier_mask[i] = False

        if not np.any(outlier_mask):
            break

        valid_indices = np.where(~outlier_mask)[0]
        outlier_indices = np.where(outlier_mask)[0]
        if len(valid_indices) > 1:
            y_current[outlier_indices] = np.interp(
                outlier_indices, valid_indices, y_current[valid_indices]
            )

    return y_current
