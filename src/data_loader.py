from src.runtime import configure_runtime

configure_runtime()

import numpy as np
from pathlib import Path

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
