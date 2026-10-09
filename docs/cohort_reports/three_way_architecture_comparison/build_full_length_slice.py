"""Recreate one audited, full-depth B-scan comparison from source data and checkpoints.

Reads selectively hydrated Box Drive files through a memory map. The source
files can be evicted after the rendered figure has been checked.
The 2.5D masks use the archived fusion and threshold settings. The exact 3D
evaluation ran on Jubail; its published central-slice overlay is recovered for
the displayed axial window, and is not used for quantitative calculations.
"""

from __future__ import annotations

import json
import gc
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion


HERE = Path(__file__).resolve().parent
CAPSTONE = HERE.parents[3]
TRAIN = CAPSTONE / "train-cnn-models"
sys.path.insert(0, str(TRAIN / "model_training/train_rnfl_volumetric"))
sys.path.insert(0, str(TRAIN / "model_training/train_rnfl_3d"))

from batch_cohort_evaluator import InferenceConfig, VolumetricRNFLPredictor  # noqa: E402
from dataset import DROP, SENTINEL, _ARRAY, _D, _IMG, _TYPE  # noqa: E402


SUBJECT = "BEH0314"
EYE = "OD"
BOX_ROOT = Path.home() / "Library/CloudStorage/Box-Box/OCT_Segmentations_Solix/deidentified-new"
DCM_NAME = "BEH0314_Disc Cube_OD_2025-03-03_15-25-51_OPT.dcm"
XML_NAME = "BEH0314, BEH0314 _OD_Disc Cube_132_330_14_1.xml"
SHAPE = (320, 768, 320)
COLORS = {
    "gold": (0, 195, 226),
    "commercial": (239, 68, 68),
    "biplanar": (34, 197, 94),
    "dense3d": (168, 85, 247),
    "transunet": (236, 72, 153),
}


def read_curves(blob: bytes) -> dict[str, np.ndarray]:
    types = [v.decode().strip() for v in _TYPE.findall(blob)]
    widths = {int(v) for v in _ARRAY.findall(blob)}
    n_img = len(_IMG.findall(blob))
    if len(widths) != 1 or n_img != 320:
        raise ValueError(f"Unexpected XML geometry: widths={widths}, images={n_img}")
    width = widths.pop()
    per = len(types) // n_img
    cube = np.array(_D.findall(blob), dtype=np.int32).reshape(n_img, per, width)
    result = {}
    for j, name in enumerate(types[:per]):
        if name in DROP:
            continue
        values = cube[:, j, :].astype(float)
        values[(values >= SENTINEL) | (values <= 0)] = np.nan
        result[name] = values
    return result


def read_volume(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        head = file.read(1 << 23)
    tag = head.find(b"\xe0\x7f\x10\x00")
    if tag < 0:
        raise ValueError("DICOM PixelData tag not found")
    vr = head[tag + 4 : tag + 6]
    offset = tag + (12 if vr in (b"OW", b"OB", b"UN") else 8)
    nbytes = np.prod(SHAPE) * 2
    if path.stat().st_size < offset + nbytes:
        raise ValueError("Truncated DICOM pixel data")
    return np.memmap(path, dtype="<u2", mode="r", offset=offset, shape=SHAPE)


def rasterize(curves: dict[str, np.ndarray], z: int) -> np.ndarray:
    ilm, nfl = curves["ILM"][z], curves["NFL"][z]
    valid = np.isfinite(ilm) & np.isfinite(nfl) & (nfl > ilm)
    top = np.clip(np.round(np.nan_to_num(ilm, nan=-1)), 0, 768)
    bot = np.clip(np.round(np.nan_to_num(nfl, nan=-1)), 0, 768)
    y = np.arange(768)[:, None]
    return ((y >= top) & (y < bot) & valid).astype(np.uint8)


def recover_archived_dense3d_mask() -> np.ndarray:
    """Digitize only the central axial window shown in the archived 3D gallery."""
    path = HERE.parent / "2026-10-04_rnfl_3d_expanded_18574378/assets/executive_cohort_report/gallery_bscan_BEH0314_OD.png"
    source = Image.open(path).convert("RGB")
    if source.size != (2597, 837):
        raise ValueError(f"Unexpected source gallery size: {source.size}")
    panel = source.crop((1315, 45, 2578, 801)).resize((320, 260), Image.Resampling.BILINEAR)
    rgb = np.asarray(panel).astype(np.int16)
    green = (rgb[:, :, 1] - rgb[:, :, 0] > 7) & (rgb[:, :, 1] - rgb[:, :, 2] > 5)
    result = np.zeros((768, 320), dtype=np.uint8)
    result[180:440] = green.astype(np.uint8)
    return result


def predict_biplanar_slice(checkpoint: Path, raw_path: Path, z: int) -> np.ndarray:
    predictor = VolumetricRNFLPredictor.from_checkpoint(
        str(checkpoint), torch.device("cpu"), batch_size=8, os_orientation_mode="corrected"
    )
    raw = read_volume(raw_path)
    cfg = InferenceConfig()
    stack = np.stack([raw[min(max(z + o, 0), 319)] for o in range(-2, 3)])
    tensor = torch.from_numpy(stack.astype(np.float32)[None] / cfg.norm_divisor)
    h_probs, h_ilm, h_nfl, h_cup = predictor._forward_batch(tensor, eye=EYE)
    v_probs = np.zeros((768, 320), dtype=np.float32)
    v_ilm = np.zeros(320, dtype=np.float32)
    v_nfl = np.zeros(320, dtype=np.float32)
    v_cup = np.zeros(320, dtype=np.float32)
    for start in range(0, 320, 8):
        batch = []
        for x in range(start, min(start + 8, 320)):
            planes = [raw[:, :, min(max(x + o, 0), 319)].T for o in range(-2, 3)]
            batch.append(np.stack(planes).astype(np.float32) / cfg.norm_divisor)
        vp, vi, vn, vc = predictor._forward_batch(torch.from_numpy(np.stack(batch)), eye=EYE)
        for j, x in enumerate(range(start, min(start + 8, 320))):
            v_probs[:, x] = vp[j, :, z]
            v_ilm[x] = vi[j, z]
            v_nfl[x] = vn[j, z]
            v_cup[x] = vc[j, z]
        if start % 64 == 0:
            print(f"  vertical planes {start + len(batch)}/320", flush=True)
    probs = (h_probs[0] + v_probs) * 0.5
    ilm = ((h_ilm[0] + v_ilm) * 0.5)[None]
    nfl = ((h_nfl[0] + v_nfl) * 0.5)[None]
    cup = ((h_cup[0] + v_cup) * 0.5)[None]
    mask = (probs[None] > cfg.mask_threshold).astype(np.uint8)
    mask = predictor.recover_thin_dropouts(mask, ilm, nfl, cup, cfg)
    mask = predictor.clamp_mask_to_surfaces(mask, ilm, nfl, cup, cfg)
    del predictor, raw
    return mask[0]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_overlay(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    inside = mask.astype(bool)
    contour = inside & ~binary_erosion(inside)
    rgb[inside] = 0.70 * rgb[inside] + 0.30 * np.array(color)
    rgb[contour] = np.array(color)
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), "RGB")


def render(raw: np.ndarray, z: int, masks: dict[str, np.ndarray]) -> None:
    scan = raw[z].astype(np.float32)
    lo, hi = np.percentile(scan, [1, 99.7])
    gray = np.uint8(np.clip((scan - lo) / max(hi - lo, 1) * 255, 0, 255))
    labels = [
        ("gold", "1. Human edited reference"),
        ("commercial", "2. Raw Commercial Solix"),
        ("biplanar", "3. Bi-Planar 2.5D"),
        ("dense3d", "4. Dense 3D U-Net"),
        ("transunet", "5. TransUNet"),
    ]
    pad, gap, panel_w, panel_h = 38, 28, 1300, 820
    title_h, footer_h = 160, 85
    width = pad * 2 + 2 * panel_w + gap
    height = title_h + 3 * panel_h + 2 * gap + footer_h
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 22), f"Full-depth RNFL comparison | {SUBJECT} {EYE} | B-scan {z}", font=font(53, True), fill="#0f172a")
    draw.text((pad, 90), "Each arm uses the same 768 x 320 raw B-scan. Left: full axial depth. Right: identical rows 180-440 enlarged for boundary review.", font=font(28), fill="#475569")
    for i, (key, label) in enumerate(labels):
        row, col = divmod(i, 2)
        x, y = pad + col * (panel_w + gap), title_h + row * (panel_h + gap)
        draw.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=18, fill="#ffffff", outline=COLORS[key], width=6)
        draw.text((x + 24, y + 19), label, font=font(39, True), fill="#0f172a")
        overlay = draw_overlay(gray, masks[key], COLORS[key])
        full_x, full_y = x + 24, y + 82
        full = overlay.resize((540, 690), Image.Resampling.BILINEAR)
        canvas.paste(full, (full_x, full_y))
        draw.rectangle((full_x, full_y, full_x + 539, full_y + 689), outline="#334155", width=2)
        roi_top = full_y + round(180 / 768 * 690)
        roi_bot = full_y + round(440 / 768 * 690)
        draw.rectangle((full_x + 2, roi_top, full_x + 537, roi_bot), outline="#ffffff", width=4)
        zoom_x, zoom_y = x + 590, y + 135
        zoom = overlay.crop((0, 180, 320, 440)).resize((685, 560), Image.Resampling.BILINEAR)
        canvas.paste(zoom, (zoom_x, zoom_y))
        draw.rectangle((zoom_x, zoom_y, zoom_x + 684, zoom_y + 559), outline="#334155", width=2)
        draw.text((zoom_x, y + 705), "Peripapillary detail (rows 180-440)", font=font(28), fill="#475569")
    x, y = pad + panel_w + gap, title_h + 2 * (panel_h + gap)
    draw.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=18, fill="#eaf1f8", outline="#cbd5e1", width=4)
    notes = [
        "Why this audited slice?",
        "BEH0314 OD has a human edited boundary",
        "and a large commercial error correction.",
        "Eye-level MABE (um): commercial 10.16,",
        "Bi-Planar 3.92, Dense 3D 4.70,",
        "TransUNet 29.23.",
        "The numbers summarize the entire eye volume;",
        "this figure shows only B-scan 160.",
        "Dense 3D here is the project's anisotropic",
        "3D U-Net checkpoint, not an nnU-Net run.",
        "Its overlay is recovered from the archived",
        "rows 180-440 evaluation view.",
    ]
    for j, line in enumerate(notes):
        draw.text((x + 45, y + 40 + j * 65), line, font=font(37 if j == 0 else 31, j == 0), fill="#0f172a" if j == 0 else "#334155")
    draw.text((pad, height - 63), "Cyan: human edited | Red: commercial | Green: Bi-Planar | Purple: Dense 3D | Pink: TransUNet", font=font(29, True), fill="#475569")
    out = HERE / "assets/full_length_audited_BEH0314_OD_bscan160.png"
    canvas.save(out, optimize=True)
    print(f"Saved {out}", flush=True)


def main() -> None:
    good = read_curves((BOX_ROOT / f"tsv/good/{SUBJECT}/curve/{XML_NAME}").read_bytes())
    bad = read_curves((BOX_ROOT / f"tsv/bad/{SUBJECT}/curve/{XML_NAME}").read_bytes())
    cup_counts = np.isnan(good["NFL"]).sum(axis=1)
    z = int(np.argmax(cup_counts)) if cup_counts.max() > 0 else 160
    if z != 160:
        raise ValueError(f"Archived gallery uses B-scan 160; reconstructed disc slice is {z}")
    raw_path = BOX_ROOT / f"dicom/{SUBJECT}/{DCM_NAME}"
    masks = {"gold": rasterize(good, z), "commercial": rasterize(bad, z)}
    del good, bad
    gc.collect()
    ckpts = {
        "biplanar": TRAIN / "checkpoints/rnfl_biplanar_18563914/best_volumetric_rnfl_net.pt",
        "dense3d": TRAIN / "checkpoints/rnfl_3d_18574378/best_rnfl_3d_net.pt",
        "transunet": TRAIN / "checkpoints/rnfl_transunet_v100_18710145/best_volumetric_rnfl_net.pt",
    }
    cache = HERE / "assets/full_length_audited_BEH0314_OD_bscan160_masks.npz"
    if cache.exists():
        with np.load(cache) as saved:
            for key in ckpts:
                if key in saved:
                    masks[key] = saved[key]
    for key in ("biplanar", "transunet"):
        if key not in masks:
            print(f"Predicting {key}", flush=True)
            masks[key] = predict_biplanar_slice(ckpts[key], raw_path, z)
            np.savez_compressed(cache, **masks)
    masks["dense3d"] = recover_archived_dense3d_mask()
    np.savez_compressed(cache, **masks)
    raw = read_volume(raw_path)
    render(raw, z, masks)
    print(json.dumps({k: int(v.sum()) for k, v in masks.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
