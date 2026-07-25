import logging
from pathlib import Path
import numpy as np

try:
    from oct_converter.readers import FDS, BOCT, E2E
except ImportError:
    logging.warning("oct_converter not installed. Proprietary formats won't be supported.")

from .scan_types import NormalizedScan

logger = logging.getLogger(__name__)

def load_proprietary_volume(path: Path) -> NormalizedScan:
    """
    Parses a proprietary OCT file (e.g. .fds) using oct-converter and 
    returns a NormalizedScan containing the 3D volume.
    """
    ext = path.suffix.lower()
    
    oct_volume = None
    
    if ext == '.fds':
        oct_volume = FDS(str(path)).read_oct_volume()
    elif ext == '.boct':
        oct_volume = BOCT(str(path)).read_oct_volume()
    elif ext == '.e2e':
        volumes = E2E(str(path)).read_oct_volume()
        if volumes and len(volumes) > 0:
            oct_volume = volumes[0]
        
    if not oct_volume:
        raise ValueError(f"Could not read OCT volume from {path.name}")
        
    volume_array = np.asarray(oct_volume.volume, dtype=np.float32)
    
    # Ensure Z, Y, X shape
    if volume_array.ndim == 2:
        volume_array = volume_array[np.newaxis, :, :]
    elif volume_array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D OCT volume, got shape {volume_array.shape}")
        
    # oct-converter does not reliably provide voxel spacing for all formats yet.
    # Defaulting to 1.0mm
    spacing = (1.0, 1.0, 1.0)
    
    return NormalizedScan(
        volume=volume_array,
        spacing_mm=spacing,
        source_format=ext[1:],  # remove dot
        metadata={"loader": "oct-converter"},
        source_path=path,
    )
