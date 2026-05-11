from src.runtime import configure_runtime

configure_runtime()

import platform
import torch
import torch.nn as nn

def get_3d_relaynet(num_layers=10):
    """
    Returns a 3D U-Net configured for retinal layer segmentation.
    """
    _configure_torch_runtime()

    try:
        from monai.networks.nets import UNet

        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_layers, # One channel per retinal layer + fluid
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
        )
    except ModuleNotFoundError:
        return nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_layers, kernel_size=1),
        )


def _configure_torch_runtime():
    if platform.system() == "Darwin" and torch.get_num_threads() != 1:
        torch.set_num_threads(1)
