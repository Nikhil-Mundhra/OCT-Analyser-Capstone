from .runtime import configure_runtime

configure_runtime()

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import csv
import xml.etree.ElementTree as ET

from .scan_types import NormalizedScan

def load_oct_volume(file_path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Ingests proprietary OCT formats and returns a standard 3D NumPy array 
    (Shape: Z, Y, X) and its voxel spacing.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(path)

    if suffix == ".vol":
        import eyepy as ep

        # Heidelberg format
        ev = ep.Oct.from_heyex_vol(str(path))
        volume = ev.volume  # 3D numpy array
        spacing = (ev.meta['ScaleZ'], ev.meta['ScaleX'], ev.meta['Distance'])
        return volume, spacing
        
    if suffix == ".dcm":
        import SimpleITK as sitk

        # Standard DICOM format
        image = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing() # Returns (X, Y, Z)
        return volume, spacing
    
    raise ValueError("Unsupported file format. Please provide .vol or .dcm")


def load_normalized_scan(file_path: str | Path) -> NormalizedScan:
    """
    Loads supported local OCT exports into the MVP volume contract.

    The returned volume is always shaped (Z, Y, X) and spacing is always in
    millimetres ordered as (Z, Y, X).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(path)

    if suffix == ".vol":
        volume, spacing = load_oct_volume(str(path))
        return NormalizedScan(
            volume=_ensure_zyx_volume(volume),
            spacing_mm=_coerce_spacing(spacing),
            source_format="vol",
            metadata={"loader": "eyepy"},
            source_path=path,
        )

    if suffix == ".dcm":
        return _load_dicom_scan(path)

    if suffix == ".zip":
        return _load_image_stack_zip(path)

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".bmp"}:
        return _load_single_image(path)

    raise ValueError("Unsupported file format. Please provide .vol, .dcm, .zip, or a 2D image")


def _load_dicom_scan(path: Path) -> NormalizedScan:
    import pydicom

    dataset = pydicom.dcmread(str(path))
    volume = np.asarray(dataset.pixel_array, dtype=np.float32)
    volume = _ensure_zyx_volume(volume)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    volume = volume * slope + intercept

    pixel_spacing = getattr(dataset, "PixelSpacing", [1.0, 1.0])
    row_spacing = float(pixel_spacing[0]) if len(pixel_spacing) > 0 else 1.0
    column_spacing = float(pixel_spacing[1]) if len(pixel_spacing) > 1 else row_spacing
    slice_spacing = float(
        getattr(dataset, "SpacingBetweenSlices", getattr(dataset, "SliceThickness", 1.0))
    )

    metadata = {
        "modality": str(getattr(dataset, "Modality", "")),
        "sop_class_uid": str(getattr(dataset, "SOPClassUID", "")),
        "rescale_slope": slope,
        "rescale_intercept": intercept,
    }

    return NormalizedScan(
        volume=volume,
        spacing_mm=(slice_spacing, row_spacing, column_spacing),
        source_format="dicom",
        metadata=metadata,
        source_path=path,
    )


def _load_image_stack_zip(path: Path) -> NormalizedScan:
    image_suffixes = {".tif", ".tiff", ".bmp", ".png"}

    with TemporaryDirectory() as temp_dir:
        with ZipFile(path) as archive:
            image_names = sorted(
                name for name in archive.namelist()
                if Path(name).suffix.lower() in image_suffixes and not name.endswith("/")
            )
            metadata_names = sorted(
                name for name in archive.namelist()
                if Path(name).suffix.lower() in {".xml", ".csv"} and not name.endswith("/")
            )

            if not image_names:
                raise ValueError("ZIP export does not contain TIFF, BMP, or PNG slices")

            archive.extractall(temp_dir)

        slices = [_read_grayscale_image(Path(temp_dir) / name) for name in image_names]
        first_shape = slices[0].shape
        if any(image.shape != first_shape for image in slices):
            raise ValueError("ZIP image slices must all have the same dimensions")

        metadata: dict[str, str] = {"slice_count": str(len(slices))}
        for name in metadata_names:
            metadata.update(_read_stack_metadata(Path(temp_dir) / name))

        spacing = _spacing_from_metadata(metadata)

    return NormalizedScan(
        volume=np.stack(slices).astype(np.float32),
        spacing_mm=spacing,
        source_format="image-stack",
        metadata=metadata,
        source_path=path,
    )


def _load_single_image(path: Path) -> NormalizedScan:
    array = _read_grayscale_image(path)
    return NormalizedScan(
        volume=_ensure_zyx_volume(array),
        spacing_mm=(1.0, 1.0, 1.0),
        source_format="single-image",
        metadata={"loader": "pil"},
        source_path=path,
    )


def _read_grayscale_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("F"), dtype=np.float32)
    return array


def _read_stack_metadata(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".xml":
        root = ET.parse(path).getroot()
        return {
            element.tag.lower(): (element.text or "").strip()
            for element in root.iter()
            if element.text and element.text.strip()
        }

    metadata: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                metadata[row[0].strip().lower()] = row[1].strip()
    return metadata


def _spacing_from_metadata(metadata: dict[str, str]) -> tuple[float, float, float]:
    def read_float(*keys: str, default: float) -> float:
        for key in keys:
            value = metadata.get(key)
            if value:
                try:
                    return float(value)
                except ValueError:
                    continue
        return default

    return (
        read_float("spacing_z_mm", "slice_thickness", "slicethickness", default=1.0),
        read_float("spacing_y_mm", "pixel_spacing_y", "pixelspacingy", default=1.0),
        read_float("spacing_x_mm", "pixel_spacing_x", "pixelspacingx", default=1.0),
    )


def _ensure_zyx_volume(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D OCT volume, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def _coerce_spacing(spacing: tuple[float, ...]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in spacing)
    if len(values) != 3:
        return (1.0, 1.0, 1.0)
    return values
