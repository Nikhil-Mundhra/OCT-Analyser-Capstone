"""
models_suite/tissue_cropper/tissue_unet.py

Attention U-Net Architecture for Binary Retinal Tissue Segmentation and 1D ILM/Choroid Boundary Extraction.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d, median_filter


class AttentionGate(nn.Module):
    """Attention Gate to highlight tissue boundaries along skip connections."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    """Residual Double 3x3 Convolution Block with BatchNorm and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    """Upscaling with Transposed Conv, Attention Gate, and DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate(
            F_g=in_channels // 2,
            F_l=in_channels // 2,
            F_int=in_channels // 4,
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if odd spatial dimensions occur
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2_att = self.attn(g=x1, x=x2)
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)


class TissueCroppingUNet(nn.Module):
    """
    Attention U-Net for Binary Retinal Tissue Envelope Segmentation.
    Extracts high-resolution continuous tissue boundaries with minimal parameter overhead.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.inc = DoubleConv(in_channels, base_c)          # 32
        self.down1 = Down(base_c, base_c * 2)              # 64
        self.down2 = Down(base_c * 2, base_c * 4)          # 128
        self.down3 = Down(base_c * 4, base_c * 8)          # 256
        self.down4 = Down(base_c * 8, base_c * 16)         # 512 bottleneck

        # Decoder with Attention Gates
        self.up1 = Up(base_c * 16, base_c * 8)             # 256
        self.up2 = Up(base_c * 8, base_c * 4)              # 128
        self.up3 = Up(base_c * 4, base_c * 2)              # 64
        self.up4 = Up(base_c * 2, base_c)                  # 32

        # Output Head
        self.outc = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @staticmethod
    def extract_boundary_vectors(
        mask_prob_2d: np.ndarray,
        threshold: float = 0.50,
        smooth_sigma: float = 3.0,
        min_thickness: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts smooth, continuous 1D boundary curves (y_top, y_bot) from a predicted 2D probability map.

        Args:
            mask_prob_2d: (H, W) float array in [0.0, 1.0] representing tissue probabilities.
            threshold: Probability threshold for binary detection.
            smooth_sigma: Gaussian kernel sigma for boundary curve smoothing.
            min_thickness: Minimum anatomical thickness floor (pixels) between ILM and Choroid.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_top, y_bot) each of length W.
        """
        h, w = mask_prob_2d.shape
        raw_bin = (mask_prob_2d >= threshold).astype(np.uint8)

        # 1. Connected Component Filtering
        # Isolate the main retinal tissue body and suppress small, isolated noise specks
        # in the vitreous or deep sclera that pull boundary vectors down/up.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_bin, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = np.max(areas)
            # Retain components that are at least 5% of the largest body or >= 500 pixels
            min_component_area = max(500, int(0.05 * max_area))
            valid_labels = [i + 1 for i, a in enumerate(areas) if a >= min_component_area]
            bin_mask = np.isin(labels, valid_labels).astype(np.uint8)
        else:
            bin_mask = raw_bin

        y_top = np.full(w, np.nan, dtype=float)
        y_bot = np.full(w, np.nan, dtype=float)

        for x in range(w):
            col = bin_mask[:, x]
            tissue_idx = np.where(col > 0)[0]
            if len(tissue_idx) > 0:
                y_top[x] = float(tissue_idx[0])
                y_bot[x] = float(tissue_idx[-1])

        # 1D Linear Interpolation for missing columns
        valid_mask = ~np.isnan(y_top)
        x_indices = np.arange(w)

        if np.any(valid_mask):
            y_top = np.interp(x_indices, x_indices[valid_mask], y_top[valid_mask])
            y_bot = np.interp(x_indices, x_indices[valid_mask], y_bot[valid_mask])
        else:
            # Fallback default vertical positions
            y_top = np.full(w, h * 0.25)
            y_bot = np.full(w, h * 0.75)

        # 2. Outlier Spike Suppression (Median Filter)
        # Suppress remaining high-frequency vertical dips/spikes (e.g. floater or shadow artifacts)
        window = min(w, 51)
        if window % 2 == 0:
            window += 1
        med_top = median_filter(y_top, size=window, mode="nearest")
        spike_top = np.abs(y_top - med_top) > 25.0
        y_top[spike_top] = med_top[spike_top]

        med_bot = median_filter(y_bot, size=window, mode="nearest")
        spike_bot = np.abs(y_bot - med_bot) > 30.0
        y_bot[spike_bot] = med_bot[spike_bot]

        # 3. Apply 1D Gaussian smoothing
        if smooth_sigma > 0:
            y_top = gaussian_filter1d(y_top, sigma=smooth_sigma)
            y_bot = gaussian_filter1d(y_bot, sigma=smooth_sigma)

        # Enforce anatomical thickness & bounds
        y_bot = np.maximum(y_bot, y_top + min_thickness)
        y_top = np.clip(y_top, 0, h - 1)
        y_bot = np.clip(y_bot, 0, h - 1)

        return y_top, y_bot
