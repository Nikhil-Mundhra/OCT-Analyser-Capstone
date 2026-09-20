"""
training/classification/data/preprocessing/mask_diagnostics.py

Automated Medical-Grade Diagnostic Engine for OCT Retinal Tissue Segmentation Masks.
Evaluates 2D probability maps, boundary curves, and masked tissue outputs across 7 strict
anatomical and geometric invariants to detect anomalous, corrupted, or collapsed masks:
  1. Component Disconnection / Floater Fragmentation
  2. Anatomical Thickness Collapse or Over-Expansion (Bloom)
  3. Boundary Inversion (ILM falling below Choroid/RPE)
  4. Boundary Derivative Spikes / High-Frequency Geometric Traps
  5. Pitch-Black Scanner Letterbox Violations
  6. Model Uncertainty / Ambiguous Probabilities
  7. Tissue Area Fraction Outliers
Generates structured health reports and triaged visual diagnostic panels.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np


@dataclass
class MaskHealthReport:
    """Detailed health status and anomaly metrics for a single segmentation prediction."""
    is_suspicious: bool = False
    severity: str = "CLEAN"  # "CLEAN", "WARNING", "CRITICAL"
    anomaly_flags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_mask_health(
    mask: np.ndarray,
    probs: Optional[np.ndarray] = None,
    y_top: Optional[np.ndarray] = None,
    y_bot: Optional[np.ndarray] = None,
    raw_img: Optional[np.ndarray] = None,
    min_thickness_floor: float = 12.0,
    max_thickness_ratio: float = 0.75,
    max_derivative_jump: float = 30.0,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.65,
) -> MaskHealthReport:
    """
    Evaluates retinal tissue mask and boundary curves against strict clinical invariants.

    Args:
        mask: 2D binary uint8 mask (0 = background, >0 = retinal tissue)
        probs: Optional (H, W) float probability map in [0, 1]
        y_top: Optional 1D boundary curve for ILM (length W)
        y_bot: Optional 1D boundary curve for Choroid/RPE (length W)
        raw_img: Optional original grayscale or BGR image
        min_thickness_floor: Minimum thickness in pixels below which mask is considered collapsing/holey
        max_thickness_ratio: Maximum allowed mean thickness relative to image height
        max_derivative_jump: Maximum permissible vertical step between adjacent columns (px)
        min_area_ratio: Minimum expected tissue area fraction
        max_area_ratio: Maximum expected tissue area fraction

    Returns:
        MaskHealthReport with severity rating, anomaly flags, metrics, and recommendations.
    """
    report = MaskHealthReport()
    h, w = mask.shape[:2]
    total_pixels = h * w
    bin_mask = (mask > 0).astype(np.uint8)
    tissue_pixel_count = int(np.sum(bin_mask))
    area_ratio = float(tissue_pixel_count / total_pixels) if total_pixels > 0 else 0.0

    report.metrics["image_shape"] = [h, w]
    report.metrics["tissue_pixels"] = tissue_pixel_count
    report.metrics["area_ratio"] = round(area_ratio, 4)

    # 1. Zero Mask / Total Dropout Check
    if tissue_pixel_count == 0:
        report.is_suspicious = True
        report.severity = "CRITICAL"
        report.anomaly_flags.append("ZERO_TISSUE_DROPOUT")
        report.recommendations.append("Total tissue dropout: model failed to detect any retinal structure.")
        return report

    # 2. Area Ratio Invariants
    if area_ratio < min_area_ratio:
        report.is_suspicious = True
        report.anomaly_flags.append("ANOMALOUS_MINIMAL_AREA")
        report.recommendations.append(f"Tissue area ratio ({area_ratio:.3f}) is abnormally small (< {min_area_ratio}).")
    elif area_ratio > max_area_ratio:
        report.is_suspicious = True
        report.anomaly_flags.append("ANOMALOUS_BLOOM_AREA")
        report.recommendations.append(f"Tissue area ratio ({area_ratio:.3f}) is abnormally large (> {max_area_ratio}).")

    # 3. Connected Components / Fragmentation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    # Exclude background (label 0)
    foreground_components = num_labels - 1
    report.metrics["foreground_components"] = foreground_components

    if foreground_components > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = float(np.max(areas))
        secondary_areas = [float(a) for a in areas if a < max_area]
        report.metrics["max_component_area"] = max_area
        report.metrics["secondary_component_areas"] = secondary_areas

        # Check if significant secondary bodies exist (> 10% of main body) or multiple tiny fragments
        large_secondary = [a for a in secondary_areas if a >= (0.10 * max_area)]
        small_floaters = [a for a in secondary_areas if a < (0.10 * max_area)]

        if large_secondary:
            report.is_suspicious = True
            report.anomaly_flags.append("SEVERED_TISSUE_FRAGMENTS")
            report.recommendations.append(f"Detected {len(large_secondary)} large detached tissue bodies.")
        elif len(small_floaters) > 3:
            report.is_suspicious = True
            report.anomaly_flags.append("VITREAL_OR_SCLERAL_NOISE_SPECKS")
            report.recommendations.append(f"Detected {len(small_floaters)} small disconnected noise specks.")

    # 4. Boundary Vectors Evaluation (ILM vs Choroid)
    if y_top is not None and y_bot is not None:
        y_top = np.asarray(y_top, dtype=float)
        y_bot = np.asarray(y_bot, dtype=float)
        thickness = y_bot - y_top

        min_thick = float(np.min(thickness))
        max_thick = float(np.max(thickness))
        mean_thick = float(np.mean(thickness))
        median_thick = float(np.median(thickness))

        report.metrics["thickness_min"] = round(min_thick, 2)
        report.metrics["thickness_max"] = round(max_thick, 2)
        report.metrics["thickness_mean"] = round(mean_thick, 2)
        report.metrics["thickness_median"] = round(median_thick, 2)

        # 4a. Boundary Inversion (ILM dipping below Choroid)
        inverted_cols = int(np.sum(thickness <= 0))
        report.metrics["inverted_columns"] = inverted_cols
        if inverted_cols > 0:
            report.is_suspicious = True
            report.severity = "CRITICAL"
            report.anomaly_flags.append("BOUNDARY_INVERSION_TOP_BELOW_BOTTOM")
            report.recommendations.append(f"Top curve dips below bottom curve in {inverted_cols} columns.")

        # 4b. Extreme Thinness / Collapse
        collapsed_cols = int(np.sum(thickness < min_thickness_floor))
        report.metrics["collapsed_columns"] = collapsed_cols
        if collapsed_cols > int(0.08 * w):
            report.is_suspicious = True
            report.anomaly_flags.append("THICKNESS_COLLAPSE_FLOOR_VIOLATION")
            report.recommendations.append(f"Retinal band thickness drops below {min_thickness_floor}px in {collapsed_cols} columns.")

        # 4c. Extreme Thickness / Bloom
        if mean_thick > (max_thickness_ratio * h):
            report.is_suspicious = True
            report.anomaly_flags.append("EXCESSIVE_THICKNESS_BLOOM")
            report.recommendations.append(f"Mean retinal thickness ({mean_thick:.1f}px) exceeds {max_thickness_ratio*100:.0f}% of frame height.")

        # 4d. Derivative Spikes / Jagged Step Jumps
        dy_top = np.abs(np.diff(y_top))
        dy_bot = np.abs(np.diff(y_bot))
        max_dy_top = float(np.max(dy_top)) if len(dy_top) > 0 else 0.0
        max_dy_bot = float(np.max(dy_bot)) if len(dy_bot) > 0 else 0.0
        report.metrics["max_dy_top"] = round(max_dy_top, 2)
        report.metrics["max_dy_bot"] = round(max_dy_bot, 2)

        top_spikes = int(np.sum(dy_top > max_derivative_jump))
        bot_spikes = int(np.sum(dy_bot > max_derivative_jump))
        report.metrics["top_derivative_spikes"] = top_spikes
        report.metrics["bot_derivative_spikes"] = bot_spikes

        if top_spikes > 0 or bot_spikes > 0:
            report.is_suspicious = True
            report.anomaly_flags.append("HIGH_FREQUENCY_BOUNDARY_SPIKE")
            report.recommendations.append(
                f"Severe vertical jump detected (|dy| > {max_derivative_jump}px): top={top_spikes} spikes, bot={bot_spikes} spikes."
            )

    # 5. Pitch-Black Non-Acquisition Violation (Outer Scanner Letterbox Margins)
    if raw_img is not None:
        gray_raw = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY) if raw_img.ndim == 3 else raw_img
        col_max = np.max(gray_raw, axis=0)
        dead_cols = (col_max <= 8)
        # Margin violations occur ONLY if mask invades dead scanner acquisition columns at frame borders
        dead_margin_mask = np.tile(dead_cols, (h, 1))
        dead_violations = int(np.sum((bin_mask > 0) & dead_margin_mask))
        report.metrics["pitch_black_violations"] = dead_violations
        if dead_violations > 50:
            report.is_suspicious = True
            report.anomaly_flags.append("PITCH_BLACK_MARGIN_INVASION")
            report.recommendations.append(f"Mask predicted {dead_violations} pixels on pitch-black scanner margin columns.")

    # 6. Model Uncertainty / Entropy
    if probs is not None:
        # Ambiguity zone between 0.40 and 0.60
        ambiguous_pixels = np.sum((probs >= 0.40) & (probs <= 0.60))
        ambiguity_ratio = float(ambiguous_pixels / max(1, tissue_pixel_count))
        report.metrics["ambiguity_ratio"] = round(ambiguity_ratio, 4)
        if ambiguity_ratio > 0.35:
            report.is_suspicious = True
            report.anomaly_flags.append("HIGH_UNCERTAINTY_PREDICTION")
            report.recommendations.append(f"Model prediction shows high uncertainty (ambiguity ratio: {ambiguity_ratio:.2%}).")

    # Determine Overall Severity Rating
    if not report.anomaly_flags:
        report.severity = "CLEAN"
        report.is_suspicious = False
    elif any(f in ("ZERO_TISSUE_DROPOUT", "BOUNDARY_INVERSION_TOP_BELOW_BOTTOM", "SEVERED_TISSUE_FRAGMENTS") for f in report.anomaly_flags):
        report.severity = "CRITICAL"
        report.is_suspicious = True
    else:
        report.severity = "WARNING"
        report.is_suspicious = True

    return report


def render_diagnostic_panel(
    raw_img: np.ndarray,
    clean_img: np.ndarray,
    mask: np.ndarray,
    y_top: Optional[np.ndarray],
    y_bot: Optional[np.ndarray],
    report: MaskHealthReport,
    probs: Optional[np.ndarray] = None,
    target_dim: int = 384
) -> np.ndarray:
    """
    Renders a multi-view visual triage diagnostic card:
      Panel 1: Raw Original B-Scan with overlaid boundary curves (Cyan=ILM, Orange=Choroid) & anomaly markers
      Panel 2: Background-Suppressed Centered Tissue
      Panel 3: Binary Segmentation Mask with Contours
      Panel 4: Diagnostic Header Banner with severity status and flags
    """
    h_orig, w_orig = raw_img.shape[:2]
    raw_bgr = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR) if raw_img.ndim == 2 else raw_img.copy()
    clean_bgr = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR) if clean_img.ndim == 2 else clean_img.copy()

    # Panel 1: Overlay on Raw
    p1 = raw_bgr.copy()
    if y_top is not None and y_bot is not None:
        for x in range(w_orig):
            yt = int(np.clip(y_top[x], 0, h_orig - 1))
            yb = int(np.clip(y_bot[x], 0, h_orig - 1))
            cv2.circle(p1, (x, yt), 1, (255, 255, 0), -1)  # Cyan
            cv2.circle(p1, (x, yb), 1, (0, 165, 255), -1)  # Orange

    # Panel 2: Cleaned BGR resized
    p2 = cv2.resize(clean_bgr, (target_dim, target_dim), interpolation=cv2.INTER_AREA)

    # Panel 3: Mask Visualization
    mask_vis = cv2.applyColorMap((mask > 0).astype(np.uint8) * 255, cv2.COLORMAP_VIRIDIS)
    p3 = cv2.resize(mask_vis, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)

    p1_resized = cv2.resize(p1, (target_dim, target_dim), interpolation=cv2.INTER_AREA)

    # Combine 3 panels horizontally
    panel_row = np.hstack([p1_resized, p2, p3])

    # Add diagnostic banner at top
    banner_h = 70
    banner = np.zeros((banner_h, panel_row.shape[1], 3), dtype=np.uint8)

    # Color code severity
    if report.severity == "CLEAN":
        bg_col = (20, 80, 20)      # Muted Dark Green
        tag_col = (0, 255, 0)
    elif report.severity == "WARNING":
        bg_col = (20, 60, 120)     # Muted Dark Orange
        tag_col = (0, 165, 255)
    else:
        bg_col = (20, 20, 120)     # Muted Dark Red
        tag_col = (0, 0, 255)

    banner[:] = bg_col

    # Text annotations
    title_text = f"DIAGNOSTIC STATUS: [{report.severity}]"
    flags_text = "Anomalies: " + (", ".join(report.anomaly_flags) if report.anomaly_flags else "None (Optimal)")
    metrics_text = f"Area: {report.metrics.get('area_ratio', 0.0):.1%} | Mean Thick: {report.metrics.get('thickness_mean', 0.0)}px | Comps: {report.metrics.get('foreground_components', 1)}"

    cv2.putText(banner, title_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tag_col, 2, cv2.LINE_AA)
    cv2.putText(banner, flags_text[:110], (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(banner, metrics_text, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # Subtitles under panels
    footer_h = 24
    footer = np.zeros((footer_h, panel_row.shape[1], 3), dtype=np.uint8)
    subtitles = ["1. Raw OCT + U-Net Vectors", "2. Centered & Suppressed (384x384)", "3. Binary Tissue Mask"]
    for i, sub in enumerate(subtitles):
        cv2.putText(footer, sub, (i * target_dim + 15, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    final_card = np.vstack([banner, panel_row, footer])
    return final_card
