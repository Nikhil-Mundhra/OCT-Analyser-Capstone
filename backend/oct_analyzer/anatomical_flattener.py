from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter1d
except ModuleNotFoundError:
    gaussian_filter1d = None

def flatten_volume_to_rpe(volume_tensor):
    """
    Takes a 3D tensor (C, Z, Y, X) and flattens the anatomical curvature.
    """
    # Work on a 3D numpy array temporarily
    vol_np = volume_tensor.detach().cpu().numpy()[0] # Drop channel for calculation
    
    # 1. Apply vertical blur to smooth noise and find the macro-structure.
    if gaussian_filter1d is None:
        blurred = _moving_average_z(vol_np, window_size=7)
    else:
        blurred = gaussian_filter1d(vol_np, sigma=3, axis=0)
    
    # 2. The RPE is highly reflective. Find the Z-index of the brightest pixel in every A-scan
    rpe_indices = np.argmax(blurred, axis=0) # Shape: (Y, X)
    
    # 3. Calculate the median depth of the RPE to use as our target flat line
    target_z = int(np.median(rpe_indices))
    
    # 4. Roll (shift) the columns vertically to align the RPE to the target_z
    flattened_vol = np.zeros_like(vol_np)
    y_dim, x_dim = vol_np.shape[1], vol_np.shape[2]
    
    for y in range(y_dim):
        for x in range(x_dim):
            shift_amount = target_z - rpe_indices[y, x]
            # np.roll wraps pixels around, which is fine as the top/bottom are usually dark
            flattened_vol[:, y, x] = np.roll(vol_np[:, y, x], shift_amount)
            
    # Return as channel-first tensor
    return torch.from_numpy(flattened_vol).unsqueeze(0).to(volume_tensor.device)


def _moving_average_z(volume, window_size):
    pad = window_size // 2
    padded = np.pad(volume, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    blurred = np.empty_like(volume)

    for z in range(volume.shape[0]):
        blurred[z] = np.mean(padded[z:z + window_size], axis=0)

    return blurred
