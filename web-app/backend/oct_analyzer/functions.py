from .runtime import configure_runtime

configure_runtime()

import torch

def spatial_continuity_loss(predictions):
    """
    Penalizes high-frequency changes in the segmentation mask 
    to prevent 'jittery' or broken layer boundaries.
    """
    # Calculate gradients along the X and Y axes
    dy = predictions[:, :, :, 1:, :] - predictions[:, :, :, :-1, :]
    dx = predictions[:, :, :, :, 1:] - predictions[:, :, :, :, :-1]
    
    return torch.mean(dy**2) + torch.mean(dx**2)
