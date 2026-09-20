"""
data/preprocessing/white_bars.py

Raycast scanner white-bar annotation removal & Scanner-Provenance UI Artifact Detection.
Enforces strict Bottom 30% Frame Height Boundary (y > 0.70*H), At-Most-1 Compass Constraint,
strict wireframe box topological scoring (aspect in [0.78, 1.28], n_children in [1, 10], P^2/A <= 45.0),
tight top and inward-facing edge bounding (13px padding),
and configurable Compass Location ('auto', 'bottom_left', 'bottom_right').
Supports returning compass_bbox for downstream 1D vector boundary interpolation.
"""

import cv2
import numpy as np


def is_spectralis_ui_candidate(src_path: str) -> bool:
    """
    Checks if an image path belongs to dataset subfolders known to contain Heidelberg Spectralis
    orientation compass artifacts (e.g. CHU_MH, MH_surgery_others).
    """
    if not src_path:
        return True
    s = src_path.lower()
    return any(k in s for k in ('chu', 'mh_', 'mh38', 'mh84', 'mh69', 'macular-hole'))


def detect_and_remove_compass_artifacts(
    img: np.ndarray,
    src_path: str = '',
    enabled: bool = None,
    location: str = 'auto',
    margin: int = 5,
    return_bbox: bool = False
):
    """
    Dynamically detects orientation compass box UI artifacts ('S', 'I', 'N', 'T' wireframe boxes)
    using scanner-provenance profiling, strict wireframe topological score ranking, precise top and
    inward-facing edge bounding, and location selection.

    Location Options:
      - 'auto': Dynamic automatic candidate search across bottom-left and bottom-right.
      - 'bottom_left': Restricts compass search exclusively to bottom-left quadrant.
      - 'bottom_right': Restricts compass search exclusively to bottom-right quadrant.

    Strict Spatial Directive: Compass boxes ONLY exist in the bottom 30% of the image frame (y > 0.70*H).
    Search is strictly restricted to y in [0.70*H, H], guaranteeing retinal tissue at y < 0.70*H is never touched.

    Top & Inward-Facing Edge Precision:
      - Top edge (by0) and inward-facing edge (bx1 for left, bx0 for right) fit tightly with 13px padding
        to enclose letters 'S', 'I', 'N', 'T' without expanding upward into background or inward into tissue.
    """
    if enabled is False:
        return (img, None) if return_bbox else img
    if enabled is None and src_path and not is_spectralis_ui_candidate(src_path):
        return (img, None) if return_bbox else img

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        return (img, None) if return_bbox else img

    H, W = gray.shape
    out = img.copy()
    compass_bbox = None

    # Dynamic resolution scaling bounds
    min_sz = max(20, int(min(H, W) * 0.04))
    max_sz = min(250, int(min(H, W) * 0.25))

    # Binarize high-intensity UI lines and letters (intensity >= 70)
    _, high_thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

    # Filter search region based on location parameter
    loc = location.lower() if location else 'auto'
    quadrants = []
    if loc in ('auto', 'bottom_left', 'left'):
        quadrants.append(('bottom_left', int(H * 0.70), H, 0, int(W * 0.35)))
    if loc in ('auto', 'bottom_right', 'right'):
        quadrants.append(('bottom_right', int(H * 0.70), H, int(W * 0.65), W))

    candidates = []

    for q_name, y0, y1, x0, x1 in quadrants:
        crop = high_thresh[y0:y1, x0:x1]
        contours, hierarchy = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours is None or len(contours) == 0 or hierarchy is None:
            continue
            
        hierarchy = hierarchy[0]

        for idx, c in enumerate(contours):
            xc, yc, wc, hc = cv2.boundingRect(c)
            if min_sz <= wc <= max_sz and min_sz <= hc <= max_sz:
                aspect = float(hc) / float(wc)
                if 0.78 <= aspect <= 1.28:
                    first_child = hierarchy[idx][2]
                    n_children = 0
                    child_curr = first_child
                    while child_curr != -1:
                        n_children += 1
                        child_curr = hierarchy[child_curr][0]
                        
                    area = cv2.contourArea(c)
                    perim = cv2.arcLength(c, True)
                    p2a = (perim * perim) / max(1.0, area)

                    # Strict wireframe box verification:
                    # 1. Real compasses have 1 <= n_children <= 10. Tissue noise has 20-100 holes!
                    # 2. Clean wireframe box has p2a <= 45.0. Jagged tissue noise has p2a > 100!
                    if 1 <= n_children <= 10 and p2a <= 45.0:
                        square_score = 10.0 - abs(aspect - 1.0) * 10.0
                        score = square_score + float(n_children) * 5.0 - p2a * 0.1
                        candidates.append({
                            'score': score,
                            'q_name': q_name,
                            'bbox': (x0 + xc, y0 + yc, wc, hc)
                        })

    # Strict At-Most-1 Compass Constraint per image: pick single highest-scoring candidate
    if candidates:
        best = max(candidates, key=lambda item: item['score'])
        bx, by, bw, bh = best['bbox']
        q_name = best['q_name']

        # Dedicated 13px padding for top and inward-facing edges (8px + 5px requested)
        tight_pad = 13
        by0 = max(int(H * 0.70), by - tight_pad)  # Top edge
        by1 = min(H, by + bh + tight_pad)          # Bottom edge

        if 'left' in q_name:
            bx0 = 0
            bx1 = min(W, bx + bw + tight_pad)      # Inward-facing right edge
        else:
            bx0 = max(0, bx - tight_pad)           # Inward-facing left edge
            bx1 = W

        compass_bbox = (bx0, by0, bx1, by1)

        if out.ndim == 2:
            out[by0:by1, bx0:bx1] = 0
        else:
            out[by0:by1, bx0:bx1] = [0, 0, 0]

    return (out, compass_bbox) if return_bbox else out


def detect_and_process_white_bars(
    img: np.ndarray,
    white_thresh: int = 175,
    dark_bg_thresh: int = 70,
    gap_pixels: int = 4,
    pad_pixels: int = 4,
    min_bar_width: int = 35,
    max_search_depth_top: float = 0.35,
    max_search_depth_bot: float = 0.35,
    highlight_red: bool = False
) -> np.ndarray:
    """
    Robust Scanner White-Bar, Banner & Corner Artifact Removal.

    Prevents false-positive notched teeth ("castle battlements") cutting into retinal tissue by:
      1. Requiring Horizontal Structure: Only triggers on connected white components spanning
         at least min_bar_width columns (or corner-connected components >= 20 columns).
         Scans where retinal tissue or noise touches the edge without a wide banner are never notched.
      2. Envelope Boundary Smoothing: For detected banners, applies a smoothed envelope cut
         across connected columns, preventing jagged, single-column vertical spikes.
      3. Controlled Padding: Clamps safety padding to avoid overshooting into underlying tissue.
    """
    if img.ndim == 2:
        gray = img
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgr = img.copy()
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        return img

    H, W = gray.shape
    bar_mask = np.zeros((H, W), dtype=np.uint8)

    max_top_y = int(H * max_search_depth_top)
    min_bot_y = int(H * (1.0 - max_search_depth_bot))

    # 1. Top-Down Banner Detection with Horizontal Structure Validation
    top_row_white = (gray[0, :] >= white_thresh).astype(np.uint8)
    num_labels_t, labels_t, stats_t, _ = cv2.connectedComponentsWithStats(top_row_white[:, None])

    valid_top_cols = set()
    for lbl in range(1, num_labels_t):
        area = stats_t[lbl, cv2.CC_STAT_AREA]
        cols_in_lbl = np.where(labels_t[:, 0] == lbl)[0]
        is_corner = (0 in cols_in_lbl) or ((W - 1) in cols_in_lbl)
        if area >= min_bar_width or (is_corner and area >= 20):
            valid_top_cols.update(cols_in_lbl)

    if valid_top_cols:
        col_end_y = {}
        for x in valid_top_cols:
            dark_count = 0
            last_white_y = 0
            for y in range(max_top_y):
                val = gray[y, x]
                if val >= white_thresh:
                    last_white_y = y
                    dark_count = 0
                elif val < dark_bg_thresh:
                    dark_count += 1
                    if dark_count >= gap_pixels:
                        break
                else:
                    break
            col_end_y[x] = last_white_y

        for lbl in range(1, num_labels_t):
            cols_in_lbl = np.where(labels_t[:, 0] == lbl)[0]
            if len(cols_in_lbl) > 0 and cols_in_lbl[0] in valid_top_cols:
                depths = np.array([col_end_y.get(x, 0) for x in cols_in_lbl])
                smooth_depth = int(np.percentile(depths, 75))
                for idx, x in enumerate(cols_in_lbl):
                    cut_y = min(max_top_y, max(depths[idx], smooth_depth) + pad_pixels)
                    bar_mask[:cut_y, x] = 255

    # 2. Bottom-Up Banner Detection with Horizontal Structure Validation
    bot_row_white = (gray[H - 1, :] >= white_thresh).astype(np.uint8)
    num_labels_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(bot_row_white[:, None])

    valid_bot_cols = set()
    for lbl in range(1, num_labels_b):
        area = stats_b[lbl, cv2.CC_STAT_AREA]
        cols_in_lbl = np.where(labels_b[:, 0] == lbl)[0]
        is_corner = (0 in cols_in_lbl) or ((W - 1) in cols_in_lbl)
        if area >= min_bar_width or (is_corner and area >= 20):
            valid_bot_cols.update(cols_in_lbl)

    if valid_bot_cols:
        col_start_y = {}
        for x in valid_bot_cols:
            dark_count = 0
            first_white_y = H - 1
            for y in range(H - 1, min_bot_y, -1):
                val = gray[y, x]
                if val >= white_thresh:
                    first_white_y = y
                    dark_count = 0
                elif val < dark_bg_thresh:
                    dark_count += 1
                    if dark_count >= gap_pixels:
                        break
                else:
                    break
            col_start_y[x] = first_white_y

        for lbl in range(1, num_labels_b):
            cols_in_lbl = np.where(labels_b[:, 0] == lbl)[0]
            if len(cols_in_lbl) > 0 and cols_in_lbl[0] in valid_bot_cols:
                depths = np.array([col_start_y.get(x, H - 1) for x in cols_in_lbl])
                smooth_depth = int(np.percentile(depths, 25))
                for idx, x in enumerate(cols_in_lbl):
                    cut_y = max(min_bot_y, min(depths[idx], smooth_depth) - pad_pixels)
                    bar_mask[cut_y:, x] = 255

    # Morphological closing along horizontal/vertical to ensure solid uniform removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    bar_mask = cv2.morphologyEx(bar_mask, cv2.MORPH_CLOSE, kernel)

    if img.ndim == 2:
        processed_img = gray.copy()
        if highlight_red:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            processed_img[bar_mask == 255] = [0, 0, 255]
        else:
            processed_img[bar_mask == 255] = 0
        return processed_img

    processed_img = bgr.copy()
    if highlight_red:
        processed_img[bar_mask == 255] = [0, 0, 255]
    else:
        processed_img[bar_mask == 255] = [0, 0, 0]

    return processed_img
