import torch
import torch.nn as nn
import timm

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

class MultiHeadConvNeXt(nn.Module):
    """
    Multi-Head ConvNeXt V2 Model with Multi-Scale Aggregation and Strict Hierarchical Conditioning
    """
    def __init__(self, num_pathology_classes: int = 12, pretrained: bool = True):
        super().__init__()
        
        # 1. Initialize pre-trained convnextv2_base, extracting features from stages 1, 2, 3
        # (Resolutions for 224x224 input: Stage 1=28x28, Stage 2=14x14, Stage 3=7x7)
        self.backbone = timm.create_model('convnextv2_base', pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
        
        # 2. Freeze all parameters in the stem and the first three stages (stages 0, 1, 2)
        self.freeze_backbone()
                
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Output channels for convnextv2_base stages 1, 2, 3
        dim_s2 = 256
        dim_s3 = 512
        dim_s4 = 1024
        
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
        
        multi_scale_dim = dim_s2 + dim_s3 + dim_s4
        
        # granular_pathology_head (multi-label)
        # Input dim is multi_scale_dim + 1 (for H1 probability concatenation)
        self.granular_pathology_head = nn.Sequential(
            nn.Linear(multi_scale_dim + 1, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_pathology_classes)
        )

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        # forward_features with features_only=True returns a list of feature maps
        features_list = self.backbone(x)
        f_s2 = features_list[0] # [B, 256, 28, 28]
        f_s3 = features_list[1] # [B, 512, 14, 14]
        f_s4 = features_list[2] # [B, 1024, 7, 7]
        
        # --- Stream 1: H1 Gatekeeper ---
        # Un-gated Global Average Pooling on the final bottleneck
        gap_s4 = self.gap(f_s4).flatten(1)
        out_normal = self.normal_abnormal_head(gap_s4)
        
        # --- Stream 2: H2 Granular Pathology (Multi-Scale) ---
        # Apply CBAM at each scale BEFORE global pooling
        att_s2 = self.cbam_s2(f_s2)
        att_s3 = self.cbam_s3(f_s3)
        att_s4 = self.cbam_s4(f_s4)
        
        gap_att_s2 = self.gap(att_s2).flatten(1)
        gap_att_s3 = self.gap(att_s3).flatten(1)
        gap_att_s4 = self.gap(att_s4).flatten(1)
        
        multi_scale_features = torch.cat([gap_att_s2, gap_att_s3, gap_att_s4], dim=1)
        
        # Hierarchical Feature Conditioning: Append H1 Probability (Soft constraint for the Linear layer)
        h1_prob = torch.sigmoid(out_normal).detach()
        h2_input = torch.cat([multi_scale_features, h1_prob], dim=1)
        
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

    def freeze_backbone(self):
        """Freeze stem + stages 0-2. Stage 3 (last) stays trainable."""
        for name, param in self.backbone.named_parameters():
            if name.startswith(('stem.', 'stages.0.', 'stages.1.', 'stages.2.')):
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the full backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr=5e-5, head_lr=5e-4):
        """Differential LRs: backbone at backbone_lr, all heads and attention at head_lr."""
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.normal_abnormal_head.parameters()) +
            list(self.cbam_s2.parameters()) +
            list(self.cbam_s3.parameters()) +
            list(self.cbam_s4.parameters()) +
            list(self.granular_pathology_head.parameters())
        )
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ]

def build_multi_head_model(pretrained=True, warmup=True) -> MultiHeadConvNeXt:
    """
    Factory function for creating the Multi-Head ConvNeXt model.
    """
    model = MultiHeadConvNeXt(num_pathology_classes=12, pretrained=pretrained)
    if warmup:
        model.freeze_backbone()
    return model
