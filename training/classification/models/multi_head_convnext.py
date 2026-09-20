import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class StripPoolingProjection(nn.Module):
    """
    Encodes spatial-extent information that Global Average Pooling discards.

    Rationale: Geographic atrophy (GA) and subretinal fluid (SRF) are
    horizontally-extended flat regions, not discrete blobs. Their discriminative
    signal is *spatial extent* — the RPE band is absent over a wide lateral
    stretch (GA) or a shallow horizontal pocket spans the scan width (SRF).
    Standard GAP collapses all spatial arrangement into a single vector,
    making GA indistinguishable from two drusen mounds with the same average
    activation. This module preserves that extent signal.

    Approach: Pool the CBAM-attended Stage 2 feature map to thin orthogonal
    strips, flatten, and project each to `out_dim` via a single Linear+GELU.
    The horizontal strip (pooled across H) captures lateral distribution;
    the vertical strip (pooled across W) captures depth-layer distribution.
    Both are concatenated into a (B, 2 * out_dim) extent embedding.

    Args:
        in_channels: Channel count of the input feature map (256 for S2).
        h_strips:    Number of rows to pool to for the vertical profile.
        w_strips:    Number of columns to pool to for the horizontal profile.
        out_dim:     Projected dimension per strip direction (default 128).
    """

    def __init__(
        self,
        in_channels: int,
        h_strips: int = 4,
        w_strips: int = 7,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.h_strips = h_strips
        self.w_strips = w_strips
        # Horizontal profile: collapses H, retains W distribution
        self.h_proj = nn.Sequential(
            nn.Linear(in_channels * w_strips, out_dim),
            nn.GELU(),
        )
        # Vertical profile: collapses W, retains H (layer) distribution
        self.v_proj = nn.Sequential(
            nn.Linear(in_channels * h_strips, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    Attended feature map (B, C, H, W).
            mask: Valid-tissue binary mask (B, 1, H, W).

        Returns:
            Extent embedding of shape (B, 2 * out_dim).
        """
        x_masked = x * mask

        # Horizontal extent: pool to (1, w_strips) — captures lateral spread
        h_strip = F.adaptive_avg_pool2d(x_masked, (1, self.w_strips))  # (B, C, 1, w_strips)
        h_feat = self.h_proj(h_strip.flatten(1))                        # (B, out_dim)

        # Vertical extent: pool to (h_strips, 1) — captures depth-layer spread
        v_strip = F.adaptive_avg_pool2d(x_masked, (self.h_strips, 1))  # (B, C, h_strips, 1)
        v_feat = self.v_proj(v_strip.flatten(1))                        # (B, out_dim)

        return torch.cat([h_feat, v_feat], dim=1)  # (B, 2 * out_dim)


class MultiHeadConvNeXt(nn.Module):
    """
    Multi-Head ConvNeXt V2 Model with Multi-Scale Aggregation and Strict Hierarchical Conditioning
    """
    def __init__(self, num_pathology_classes: int = 12, pretrained: bool = True, backbone_name: str = 'convnextv2_base', condition_h2_on_h1: bool = True):
        super().__init__()
        self.condition_h2_on_h1 = condition_h2_on_h1
        
        # 1. Initialize pre-trained backbone, extracting features from stages 1, 2, 3
        # (Resolutions for 224x224 input with convnextv2_base: Stage 1=28x28, Stage 2=14x14, Stage 3=7x7)
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
        # Enable gradient checkpointing to reduce autograd activation memory on MPS by > 80%
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        
        # 2. Freeze all parameters in the stem and the first three stages (stages 0, 1, 2)
        self.freeze_backbone()
                
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Extract output channels for backbone stages 1, 2, 3 dynamically
        feature_channels = self.backbone.feature_info.channels()
        dim_s2, dim_s3, dim_s4 = feature_channels
        
        # Sanity check: Ensure default convnextv2_base matches expected baseline contract (256, 512, 1024)
        if backbone_name == 'convnextv2_base':
            assert (dim_s2, dim_s3, dim_s4) == (256, 512, 1024), \
                f"Architecture mismatch! Expected (256, 512, 1024), got {feature_channels}"
        
        # normal_abnormal_head (binary -> 1 output)
        # H1 only looks at the global context (Stage 4)
        self.normal_abnormal_head = nn.Sequential(
            nn.Linear(dim_s4, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        )
        
        # Multi-Scale CBAM Attention Modules for H2
        self.cbam_s2 = CBAMBlock(in_planes=dim_s2)
        self.cbam_s3 = CBAMBlock(in_planes=dim_s3)
        self.cbam_s4 = CBAMBlock(in_planes=dim_s4)

        # P3: Strip pooling on Stage 2 for spatial-extent biomarkers (GA, SRF).
        # Produces a 256-dim horizontal+vertical profile embedding.
        # h_strips=4, w_strips=7 chosen to match the S2 spatial resolution (28x28)
        # with 4x compression, giving coarse but discriminative extent signatures.
        self.strip_pool_s2 = StripPoolingProjection(
            in_channels=dim_s2, h_strips=4, w_strips=7, out_dim=128
        )

        # P1: Mean + Max dual-stream at S2 and S3.
        # Max pooling preserves peak-reflectivity signal that mean pooling washes out,
        # separating hyperreflective foci (extreme max response) from moderate SHRM,
        # and dark hyporeflective PED interiors from bright fibrovascular PED.
        # S4 remains mean-only: it feeds H1 and contains global context, not local extrema.
        #
        # h2_in_dim = S2-mean(256) + S2-max(256)
        #           + S3-mean(512) + S3-max(512)
        #           + S4-mean(1024)
        #           + strip(256)
        #           + H1-prob(1 if conditioning)
        #         = 2816 + (1 if condition_h2_on_h1 else 0)
        _gap_dim = (dim_s2 * 2) + (dim_s3 * 2) + dim_s4
        _strip_dim = 128 * 2  # 2 * out_dim from StripPoolingProjection
        multi_scale_dim = _gap_dim + _strip_dim
        h2_in_dim = multi_scale_dim + 1 if condition_h2_on_h1 else multi_scale_dim

        # granular_pathology_head (multi-label / multi-class)
        # h2_in_dim = 2817 (with H1 conditioning) or 2816 (without)
        self.granular_pathology_head = nn.Sequential(
            nn.Linear(h2_in_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_pathology_classes)
        )

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, return_probs: bool = False):
        if x.is_cuda and not torch.is_autocast_enabled():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._forward_impl(x, valid_mask, return_probs)
        return self._forward_impl(x, valid_mask, return_probs)

    def _forward_impl(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, return_probs: bool = False):
        # forward_features with features_only=True returns a list of feature maps
        features_list = self.backbone(x)
        f_s2 = features_list[0] # [B, 256, 28, 28]
        f_s3 = features_list[1] # [B, 512, 14, 14]
        f_s4 = features_list[2] # [B, 1024, 7, 7]
        
        # Build or downsample valid_mask using area interpolation
        if valid_mask is None:
            # Fallback if valid_mask not provided: un-normalized background > 0.05
            valid_mask = (x.mean(dim=1, keepdim=True) > -1.8).float()
        
        mask_s2 = F.interpolate(valid_mask, size=f_s2.shape[-2:], mode='area')
        mask_s3 = F.interpolate(valid_mask, size=f_s3.shape[-2:], mode='area')
        mask_s4 = F.interpolate(valid_mask, size=f_s4.shape[-2:], mode='area')

        # --- Stream 1: H1 Gatekeeper (Masked GAP on Stage 4 Bottleneck) ---
        masked_s4 = f_s4 * mask_s4
        gap_s4 = masked_s4.sum(dim=(2, 3)) / mask_s4.sum(dim=(2, 3)).clamp_min(1.0)
        out_normal = self.normal_abnormal_head(gap_s4)
        
        # --- Stream 2: H2 Granular Pathology (Multi-Scale Masked GAP + Strip Pool) ---
        # Apply CBAM and valid_mask at each scale BEFORE pooling.
        att_s2 = self.cbam_s2(f_s2) * mask_s2
        att_s3 = self.cbam_s3(f_s3) * mask_s3
        att_s4 = self.cbam_s4(f_s4) * mask_s4

        # P1: Mean + Max dual-stream at S2 and S3.
        # mean: global context, avg reflectivity across tissue
        # max:  peak reflectivity, separates hyperreflective foci (extreme) from
        #       moderate SHRM, and dark PED interiors from bright fibrovascular PEDs.
        mask_s2_sum = mask_s2.sum(dim=(2, 3)).clamp_min(1.0)
        mask_s3_sum = mask_s3.sum(dim=(2, 3)).clamp_min(1.0)

        gap_mean_s2 = att_s2.sum(dim=(2, 3)) / mask_s2_sum            # (B, 256)
        gap_max_s2  = (att_s2).amax(dim=(2, 3))                       # (B, 256)
        gap_mean_s3 = att_s3.sum(dim=(2, 3)) / mask_s3_sum            # (B, 512)
        gap_max_s3  = (att_s3).amax(dim=(2, 3))                       # (B, 512)
        gap_att_s4  = att_s4.sum(dim=(2, 3)) / mask_s4.sum(dim=(2, 3)).clamp_min(1.0)  # (B, 1024)

        # P3: Horizontal strip pooling on S2 for spatial-extent biomarkers.
        # att_s2 already has the tissue mask applied; pass mask through again for
        # the strip module's internal masked average.
        strip_feat = self.strip_pool_s2(att_s2, mask_s2)               # (B, 256)

        multi_scale_features = torch.cat(
            [gap_mean_s2, gap_max_s2, gap_mean_s3, gap_max_s3, gap_att_s4, strip_feat],
            dim=1,
        )  # (B, 256+256+512+512+1024+256) = (B, 2816)

        # Hierarchical Feature Conditioning: Append H1 Probability if enabled
        if self.condition_h2_on_h1:
            h1_prob = torch.sigmoid(out_normal).detach()
            h2_input = torch.cat([multi_scale_features, h1_prob], dim=1)  # (B, 2817)
        else:
            h2_input = multi_scale_features
        
        out_pathology = self.granular_pathology_head(h2_input)
        
        if return_probs:
            # Strict Hierarchical Classification Conditioning (Mathematical Constraint)
            p_h1 = torch.sigmoid(out_normal)
            p_h2_given_h1 = torch.softmax(out_pathology, dim=1)
            final_h2_prob = p_h2_given_h1 * p_h1
            return {
                'normal_abnormal': p_h1,
                'pathology': final_h2_prob
            }
            
        return {
            'normal_abnormal': out_normal,
            'pathology': out_pathology
        }

    def freeze_full_backbone(self):
        """Freeze stem and ALL backbone stages (0, 1, 2, 3) completely."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_stage3_only(self):
        """Unfreeze ONLY the deepest backbone stage (stage 3 / stage 4 bottleneck). Keep stem & stages 0-2 frozen."""
        for name, param in self.backbone.named_parameters():
            if name.startswith('stages.3.'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_stage2_and_3(self):
        """Unfreeze Stages 2 and 3 (containing all feature maps feeding multi-scale heads S2, S3, S4) while keeping stem & stage 0 frozen for 3.5s/batch throughput."""
        for name, param in self.backbone.named_parameters():
            if name.startswith(('stages.2.', 'stages.3.')):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the entire backbone for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_full_backbone(self):
        """Alias for unfreeze_backbone."""
        self.unfreeze_backbone()

    # Legacy helper alias for backward compatibility
    def freeze_backbone(self):
        """Freeze stem + stages 0-2 for warmup. Stage 3 stays trainable."""
        for name, param in self.backbone.named_parameters():
            if name.startswith(('stem.', 'stages.0.', 'stages.1.', 'stages.2.')):
                param.requires_grad = False
            elif name.startswith('stages.3.'):
                param.requires_grad = True

    def get_param_groups(self, backbone_lr=2e-6, head_lr=2e-5, weight_decay=1e-4, early_backbone_factor=1.0):
        """
        Discriminative Layer-Wise Learning Rates & Weight Decay Splitting:
        - When early_backbone_factor < 1.0: splits early backbone (stem/stages 0-2) at early_lr (0.1x) from late backbone (stage 3).
        - When early_backbone_factor == 1.0: returns standard 4 groups [backbone_decay, backbone_no_decay, head_decay, head_no_decay].
        Excludes 1D biases and normalization parameters from weight decay.
        """
        if early_backbone_factor < 1.0:
            early_decay, early_no_decay = [], []
            late_decay, late_no_decay = [], []

            for name, param in self.backbone.named_parameters():
                if not param.requires_grad:
                    continue
                is_no_decay = param.ndim <= 1 or name.endswith('.bias') or 'norm' in name.lower() or 'ln' in name.lower()
                is_early = name.startswith(('stem.', 'stages.0.', 'stages.1.', 'stages.2.'))

                if is_early:
                    if is_no_decay:
                        early_no_decay.append(param)
                    else:
                        early_decay.append(param)
                else:
                    if is_no_decay:
                        late_no_decay.append(param)
                    else:
                        late_decay.append(param)

            head_modules = [
                self.normal_abnormal_head,
                self.cbam_s2,
                self.cbam_s3,
                self.cbam_s4,
                self.strip_pool_s2,
                self.granular_pathology_head,
            ]
            head_decay, head_no_decay = [], []
            for module in head_modules:
                for name, param in module.named_parameters():
                    if not param.requires_grad:
                        continue
                    if param.ndim <= 1 or name.endswith('.bias') or 'norm' in name.lower() or 'ln' in name.lower():
                        head_no_decay.append(param)
                    else:
                        head_decay.append(param)

            early_lr = backbone_lr * early_backbone_factor

            groups = []
            if early_decay:
                groups.append({"params": early_decay, "lr": early_lr, "weight_decay": weight_decay})
            if early_no_decay:
                groups.append({"params": early_no_decay, "lr": early_lr, "weight_decay": 0.0})
            if late_decay:
                groups.append({"params": late_decay, "lr": backbone_lr, "weight_decay": weight_decay})
            if late_no_decay:
                groups.append({"params": late_no_decay, "lr": backbone_lr, "weight_decay": 0.0})
            if head_decay:
                groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
            if head_no_decay:
                groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})

            return groups

        backbone_decay, backbone_no_decay = [], []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith('.bias') or 'norm' in name.lower() or 'ln' in name.lower():
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

        head_modules = [
            self.normal_abnormal_head,
            self.cbam_s2,
            self.cbam_s3,
            self.cbam_s4,
            self.strip_pool_s2,
            self.granular_pathology_head,
        ]
        head_decay, head_no_decay = [], []
        for module in head_modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith('.bias') or 'norm' in name.lower() or 'ln' in name.lower():
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)

        return [
            {"params": backbone_decay,    "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay,        "lr": head_lr,     "weight_decay": weight_decay},
            {"params": head_no_decay,     "lr": head_lr,     "weight_decay": 0.0},
        ]

def build_multi_head_model(pretrained=True, warmup=True, condition_h2_on_h1=True, num_pathology_classes=12) -> MultiHeadConvNeXt:
    """
    Factory function for creating the Multi-Head ConvNeXt model.
    """
    model = MultiHeadConvNeXt(
        num_pathology_classes=num_pathology_classes,
        pretrained=pretrained,
        condition_h2_on_h1=condition_h2_on_h1
    )
    if warmup:
        model.freeze_full_backbone()
    return model

