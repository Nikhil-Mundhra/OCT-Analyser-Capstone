import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate from 'Attention U-Net' (Oktay et al., 2018).

    Learns a soft spatial attention mask that suppresses irrelevant background
    activations in each skip connection, directing the decoder to focus on
    clinically meaningful structures (thin retinal layers, small fluid pockets).

    Applied on every skip connection in the decoder, so the model explicitly
    learns *where* in the scan each layer/lesion should appear — addressing
    the vanilla U-Net's inability to model long-range spatial dependencies.

    Args:
        F_g  : Channels in the gating signal  (upsampled decoder feature, x1).
        F_l  : Channels in the skip connection (encoder feature, x2).
        F_int: Intermediate channels for the attention computation (typically F_g // 2).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # Project gating signal to F_int
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        # Project skip connection to F_int
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        # Scalar attention coefficient per spatial location
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g : Gating signal from the decoder path (upsampled to match x spatially).
            x : Skip connection from the encoder path.
        Returns:
            Attention-weighted skip connection: x * attention_map.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)        # (B, 1, H, W) attention map in [0, 1]
        return x * psi             # broadcast multiply across all channels


# ---------------------------------------------------------------------------
# Standard U-Net building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


# ---------------------------------------------------------------------------
# ASPP Bottleneck  (Fix #4 — multi-scale receptive field)
# ---------------------------------------------------------------------------

class _ASPPConv(nn.Module):
    """Single dilated convolution branch inside ASPP."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ASPPPooling(nn.Module):
    """Global average pooling branch inside ASPP."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = self.pool(x)                                    # (B, C, 1, 1)
        return F.interpolate(pooled, size=size,
                             mode='bilinear', align_corners=False)  # upsample back


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (from DeepLab v3).

    Runs five parallel branches over the bottleneck feature map:
      1. 1×1 convolution            — captures pixel-level features
      2–4. 3×3 dilated convs        — dilation rates 6, 12, 18 cover
                                      fields of ~13, ~25, ~37 pixels at 32×32
      5. Global average pooling     — captures the full-image context

    The five outputs are concatenated and projected to `out_channels`.
    This gives the decoder access to multi-scale context without any
    additional spatial compression — addressing the bottleneck's tendency
    to lose thin-layer detail through aggressive downsampling.

    Args:
        in_channels  : Input channel count (512 for down4 output).
        out_channels : Output channel count (1024 for the decoder).
        dilations    : Dilation rates for the three parallel atrous convs.
    """

    _MID = 256  # channels per branch; 5 × 256 = 1280 → projected to out_channels

    def __init__(self, in_channels: int, out_channels: int,
                 dilations: tuple = (6, 12, 18)):
        super().__init__()
        m = self._MID
        self.branch_1x1  = nn.Sequential(
            nn.Conv2d(in_channels, m, kernel_size=1, bias=False),
            nn.BatchNorm2d(m),
            nn.ReLU(inplace=True),
        )
        self.branch_d1   = _ASPPConv(in_channels, m, dilations[0])
        self.branch_d2   = _ASPPConv(in_channels, m, dilations[1])
        self.branch_d3   = _ASPPConv(in_channels, m, dilations[2])
        self.branch_pool = _ASPPPooling(in_channels, m)

        self.project = nn.Sequential(
            nn.Conv2d(5 * m, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # light regularisation on bottleneck features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [
            self.branch_1x1(x),
            self.branch_d1(x),
            self.branch_d2(x),
            self.branch_d3(x),
            self.branch_pool(x),
        ]
        return self.project(torch.cat(branches, dim=1))


class BottleneckASPP(nn.Module):
    """
    Drop-in replacement for the deepest `Down` block.
    Applies MaxPool2d then ASPP (instead of MaxPool2d then DoubleConv),
    so the interface is identical: `forward(x) -> tensor`.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dilations: tuple = (6, 12, 18)):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.aspp    = ASPP(in_channels, out_channels, dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aspp(self.maxpool(x))


class Up(nn.Module):
    """
    Upscaling block with an Attention Gate applied to the skip connection
    before concatenation with the upsampled decoder feature.

    Channel layout after ConvTranspose2d:
        x1 (decoder, upsampled) : in_channels // 2
        x2 (encoder skip)       : in_channels // 2
    The attention gate uses x1 as the gating signal and attends x2.
    """

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
        # Pad x1 if spatial sizes don't divide evenly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        # Attend the skip connection before concatenation
        x2 = self.attn(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class HierarchicalUNet(nn.Module):
    """
    Hierarchical Multi-Head U-Net for retinal layer and lesion segmentation.

    Architecture:
      - Shared U-Net encoder/decoder backbone.
      - ASPP bottleneck (Fix #4): replaces the flat DoubleConv at 32×32 with
        Atrous Spatial Pyramid Pooling to capture multi-scale receptive fields
        without additional spatial compression.
      - Attention Gates on every decoder skip connection (Fix #3).
      - Coarse Head: predicts 3 broad categories (Background, Retina, Fluid/Lesion).
      - Granular Head: predicts 15 fine-grained classes, conditioned on the
        coarse head's softmax probabilities — not raw logits (Fix #2).
    """

    def __init__(self, n_channels: int = 1, n_coarse_classes: int = 3, n_granular_classes: int = 15):
        super(HierarchicalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_coarse_classes = n_coarse_classes
        self.n_granular_classes = n_granular_classes

        # Shared Encoder
        self.inc    = DoubleConv(n_channels, 64)
        self.down1  = Down(64, 128)
        self.down2  = Down(128, 256)
        self.down3  = Down(256, 512)
        # Bottleneck: ASPP replaces plain DoubleConv for multi-scale context
        self.down4  = BottleneckASPP(512, 1024, dilations=(6, 12, 18))

        # Shared Decoder — each Up block includes an AttentionGate on its skip connection
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Multi-Scale Classification Pool
        # We pool from x3 (256), x4 (512), and x5 (1024) to retain spatial textures
        self.pool_x3 = nn.AdaptiveAvgPool2d(1)
        self.pool_x4 = nn.AdaptiveAvgPool2d(1)
        self.pool_x5 = nn.AdaptiveAvgPool2d(1)
        
        self.cls_project = nn.Sequential(
            nn.Linear(256 + 512 + 1024, 1024),
            nn.GELU(),
            nn.Dropout(p=0.3)
        )

        # L1: Normal vs Abnormal (Binary)
        self.normal_abnormal_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        )

        # L2: Broad Pathology Routing (5 Classes)
        # Input: 1024 (pooled) + 1 (L1 prob) = 1025
        self.pathology_type_head = nn.Sequential(
            nn.Linear(1025, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 5)
        )

        # L3: Granular Biomarkers (Multi-Label Specialists)
        # Input: 1024 (pooled) + 1 (L1 prob) + 5 (L2 probs) = 1030
        self.macular_head = nn.Sequential(nn.Linear(1030, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 3))
        self.diabetic_head = nn.Sequential(nn.Linear(1030, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 2))
        self.vascular_head = nn.Sequential(nn.Linear(1030, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 3))
        self.fluid_head = nn.Sequential(nn.Linear(1030, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 1))
        self.structural_head = nn.Sequential(nn.Linear(1030, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 2))

        # Coarse Head  →  3-channel output
        self.coarse_conv = nn.Sequential(
            DoubleConv(64, 64),
            nn.Conv2d(64, n_coarse_classes, kernel_size=1),
        )

        # Granular Head  →  15-channel output
        # Input: shared decoder features (64) + coarse softmax probabilities (n_coarse_classes)
        self.granular_conv = nn.Sequential(
            DoubleConv(64 + n_coarse_classes, 64),
            nn.Conv2d(64, n_granular_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, task: str = "segmentation"):
        """
        Args:
            x: Input tensor (B, 1, H, W)
            task: "segmentation" (default), "classification", or "both"
        """
        # ---- Encoder ----
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if task == "classification" or task == "both":
            # 1. Multi-Scale Aggregation
            p3 = torch.flatten(self.pool_x3(x3), 1)
            p4 = torch.flatten(self.pool_x4(x4), 1)
            p5 = torch.flatten(self.pool_x5(x5), 1)
            multi_scale_features = torch.cat([p3, p4, p5], dim=1)
            pooled = self.cls_project(multi_scale_features)
            
            # 2. Strict Hierarchical Classification Pass
            l1_logits = self.normal_abnormal_head(pooled)
            l1_probs = torch.sigmoid(l1_logits)
            
            l2_input = torch.cat([pooled, l1_probs], dim=1)
            l2_logits = self.pathology_type_head(l2_input)
            l2_probs = torch.softmax(l2_logits, dim=1)
            
            l3_input = torch.cat([pooled, l1_probs, l2_probs], dim=1)
            
            cls_logits = {
                "normal_abnormal": l1_logits,
                "pathology": l2_logits,
                "severity": {
                    "macular": self.macular_head(l3_input),
                    "diabetic": self.diabetic_head(l3_input),
                    "vascular": self.vascular_head(l3_input),
                    "fluid": self.fluid_head(l3_input),
                    "structural": self.structural_head(l3_input)
                }
            }

        if task == "classification":
            return cls_logits

        # ---- Decoder (attention-gated skip connections) ----
        d = self.up1(x5, x4)
        d = self.up2(d,  x3)
        d = self.up3(d,  x2)
        shared_features = self.up4(d, x1)

        # ---- Coarse Head ----
        coarse_logits = self.coarse_conv(shared_features)

        # ---- Granular Head ----
        # FIX (Issue #2): pass softmax PROBABILITIES, not raw logits.
        coarse_probs = torch.softmax(coarse_logits, dim=1)
        granular_input = torch.cat([shared_features, coarse_probs], dim=1)
        granular_logits = self.granular_conv(granular_input)

        if task == "both":
            return coarse_logits, granular_logits, cls_logits

        return coarse_logits, granular_logits
