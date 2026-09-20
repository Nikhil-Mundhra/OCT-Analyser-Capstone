"""
data/preprocessing/geometry.py

Aspect-ratio preserving letterboxing, SVG coordinate projection & downsampling,
and multi-surface diagnostic overlay rendering.
"""

from typing import Optional, Tuple, List
import cv2
import numpy as np


def letterbox_pad_and_resize(
    img: np.ndarray,
    target_dim: int = 384
) -> Tuple[np.ndarray, float, int, int, int, int]:
    """
    Pads an image symmetrically to square letterbox and resizes to target_dim.

    Returns:
        (resized_img, scale, pad_top, pad_left, orig_height, orig_width)
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    pad_t = (max_dim - h) // 2
    pad_b = max_dim - h - pad_t
    pad_l = (max_dim - w) // 2
    pad_r = max_dim - w - pad_l

    padded = cv2.copyMakeBorder(
        img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    resized = cv2.resize(padded, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
    scale = target_dim / float(max_dim)
    return resized, scale, pad_t, pad_l, h, w


def center_and_letterbox_tissue(
    img: np.ndarray,
    y_top: Optional[np.ndarray] = None,
    y_bot: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    target_dim: int = 384
) -> Tuple[np.ndarray, Optional[np.ndarray], float, int, int]:
    """
    Performs true anatomical centering of retinal tissue within a square target_dim x target_dim frame.
    
    Rather than symmetrically padding the raw acquisition frame (which leaves retina off-center
    if the scanner captured the eye high or low in the scan window), this calculates the true
    vertical centroid of the retinal tissue:
        y_tissue_center = median((y_top + y_bot) / 2)
    and shifts the canvas so the retina is perfectly centered vertically, preserving physical
    1:1 aspect ratio and scaling.

    Args:
        img: Input image (H, W) or (H, W, 3)
        y_top: Optional 1D ILM boundary array
        y_bot: Optional 1D Choroid boundary array
        mask: Optional binary mask (H, W)
        target_dim: Target square dimension (default: 384)

    Returns:
        Tuple:
            - centered_resized_img: (target_dim, target_dim, 3) or (target_dim, target_dim)
            - centered_resized_mask: (target_dim, target_dim) or None
            - scale: geometric scaling factor
            - pad_top: vertical offset applied
            - pad_left: horizontal offset applied
    """
    h, w = img.shape[:2]
    is_color = (img.ndim == 3)

    # Determine vertical center of tissue
    if y_top is not None and y_bot is not None:
        y_tissue_center = float(np.median((y_top + y_bot) / 2.0))
    elif mask is not None and np.sum(mask > 0) > 0:
        y_indices = np.where(mask > 0)[0]
        y_tissue_center = float(np.median(y_indices))
    else:
        y_tissue_center = h / 2.0

    # The maximum dimension dictates the isotropic scaling factor to preserve aspect ratio
    max_dim = max(h, w)
    
    # We want y_tissue_center to map to max_dim // 2 in the padded square
    target_center = max_dim / 2.0
    desired_pad_t = int(round(target_center - y_tissue_center))
    desired_pad_b = max_dim - h - desired_pad_t

    # In case the shift pushes one edge negative, clamp and adjust
    if desired_pad_t < 0:
        pad_t = 0
        pad_b = max_dim - h
    elif desired_pad_b < 0:
        pad_b = 0
        pad_t = max_dim - h
    else:
        pad_t = desired_pad_t
        pad_b = desired_pad_b

    # Symmetrical horizontal padding
    pad_l = (max_dim - w) // 2
    pad_r = max_dim - w - pad_l

    pad_val = [0, 0, 0] if is_color else 0
    padded_img = cv2.copyMakeBorder(
        img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=pad_val
    )
    resized_img = cv2.resize(padded_img, (target_dim, target_dim), interpolation=cv2.INTER_AREA)

    resized_mask = None
    if mask is not None:
        padded_mask = cv2.copyMakeBorder(
            mask, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0
        )
        resized_mask = cv2.resize(padded_mask, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)

    scale = target_dim / float(max_dim)
    return resized_img, resized_mask, scale, pad_t, pad_l


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
    """
    Downsamples and projects boundary vectors into letterbox SVG display space.
    Accepts both canonical and alias keyword arguments for backward compatibility.
    """
    if y_top is None and y_top_outer is not None:
        y_top = y_top_outer
    if y_bottom is None and y_bottom_outer is not None:
        y_bottom = y_bottom_outer
    if y_sfcm is None and y_bottom_sfcm is not None:
        y_sfcm = y_bottom_sfcm
    if num_points is not None:
        n_pts = num_points

    indices = np.linspace(0, orig_w - 1, n_pts, dtype=int)

    top_pts = [[float(x + pad_l) * scale, float(y_top[x] + pad_t) * scale] for x in indices] if y_top is not None else []
    bot_pts = [[float(x + pad_l) * scale, float(y_bottom[x] + pad_t) * scale] for x in indices] if y_bottom is not None else []

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


def render_boundary_overlay(
    image_bgr: np.ndarray,
    y_top: Optional[np.ndarray] = None,
    y_rpe: Optional[np.ndarray] = None,
    y_sfcm: Optional[np.ndarray] = None,
    holes: Optional[List[dict]] = None,
    caverns: Optional[List[dict]] = None,
) -> np.ndarray:
    """
    Renders an anatomical multi-surface diagnostic overlay on an RGB/BGR OCT scan.
    - Cyan (#00FFFF): Inner Limiting Membrane (ILM)
    - Green (#00FF00): Retinal Pigment Epithelium (RPE)
    - Orange (#FFA500): Choroid Scleral Interface (SFCM Floor)
    - Deep Pink (#FF1493): Choroidal Holes
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
