from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image, ImageDraw


PREVIEW_KINDS = {"raw", "cropped", "overlay", "features"}


def write_previews(
    preview_dir: Path,
    raw_volume: np.ndarray,
    cropped_volume: np.ndarray,
    segmentation: np.ndarray,
    layers: list[dict[str, Any]],
) -> dict[str, Union[str, list[str]]]:
    preview_dir.mkdir(parents=True, exist_ok=True)

    raw = _slice_image(_center_slice(raw_volume))
    cropped = _slice_image(_center_slice(cropped_volume))
    overlay = _overlay_image(_center_slice(cropped_volume), _center_slice(segmentation))
    features = _feature_chart(layers)

    files = {
        "raw": preview_dir / "raw.png",
        "cropped": preview_dir / "cropped.png",
        "overlay": preview_dir / "overlay.png",
        "features": preview_dir / "features.png",
    }
    raw.save(files["raw"])
    cropped.save(files["cropped"])
    overlay.save(files["overlay"])
    features.save(files["features"])

    results = {kind: f"preview/{kind}" for kind in files}
    
    frames_dir = preview_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    num_slices = raw_volume.shape[0]
    num_frames = min(20, num_slices)
    indices = np.linspace(0, num_slices - 1, num_frames, dtype=int)
    
    frame_urls = []
    for i, idx in enumerate(indices):
        frame = _slice_image(raw_volume[idx])
        frame_name = f"frame_{i}.png"
        frame.save(frames_dir / frame_name)
        frame_urls.append(f"preview/frames/{frame_name}")
    
    results["frames"] = frame_urls
    return results


def preview_path(preview_dir: Path, kind: str) -> Path:
    if kind.startswith("frames/"):
        return preview_dir / kind
    if kind not in PREVIEW_KINDS:
        raise ValueError(f"Unsupported preview kind: {kind}")
    return preview_dir / f"{kind}.png"


def _center_slice(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    return array[array.shape[0] // 2]


def _slice_image(slice_array: np.ndarray) -> Image.Image:
    array = np.asarray(slice_array, dtype=np.float32)
    low = float(np.percentile(array, 1.0))
    high = float(np.percentile(array, 99.0))
    if high <= low:
        scaled = np.zeros_like(array, dtype=np.uint8)
    else:
        scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
        scaled = (scaled * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def _overlay_image(slice_array: np.ndarray, labels: np.ndarray) -> Image.Image:
    image = _slice_image(slice_array).convert("RGBA")
    label_array = np.asarray(labels)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pixels = overlay.load()
    palette = [
        (13, 124, 124, 90),
        (226, 111, 50, 90),
        (85, 109, 213, 90),
        (190, 70, 116, 90),
    ]

    height, width = label_array.shape
    for y in range(height):
        for x in range(width):
            label = int(label_array[y, x])
            if label > 0:
                pixels[x, y] = palette[(label - 1) % len(palette)]

    return Image.alpha_composite(image, overlay).convert("RGB")


def _feature_chart(layers: list[dict[str, Any]]) -> Image.Image:
    width = 960
    height = 420
    margin = 42
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    scores = [float(layer["score"]) for layer in layers]
    max_score = max(scores + [1.0])
    bar_gap = 8
    bar_width = max(8, (width - margin * 2 - bar_gap * (len(scores) - 1)) // len(scores))

    draw.line((margin, height - margin, width - margin, height - margin), fill=(23, 33, 43), width=2)
    draw.line((margin, margin, margin, height - margin), fill=(23, 33, 43), width=2)

    for index, score in enumerate(scores):
        x0 = margin + index * (bar_width + bar_gap)
        x1 = x0 + bar_width
        bar_height = int((height - margin * 2) * (score / max_score))
        y0 = height - margin - bar_height
        draw.rectangle((x0, y0, x1, height - margin), fill=(13, 124, 124))
        draw.text((x0, height - margin + 6), str(index + 1), fill=(96, 112, 131))

    draw.text((margin, 14), "Layer CDF-derived placeholder scores", fill=(23, 33, 43))
    return image

