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
    Multi-Head ConvNeXt V2 Model
    
    1. Initializes a pre-trained convnextv2_base.
    2. Freezes the stem and first three stages, leaving Stage 4 unfrozen.
    3. Branches into Dual-Streams:
       - Stream 1: Un-gated GAP -> H1 (Gatekeeper: Normal vs Abnormal)
       - Stream 2: CBAM-filtered GAP + H1 Prob Conditioning -> H2 (Granular Pathology: 12 classes)
    """
    def __init__(self, num_pathology_classes: int = 12, pretrained: bool = True):
        super().__init__()
        
        # 1. Initialize pre-trained convnextv2_base from timm
        self.backbone = timm.create_model('convnextv2_base', pretrained=pretrained)
        
        # Remove the original classification head
        self.backbone.reset_classifier(0)
        
        # 2. Freeze all parameters in the stem and the first three stages.
        for name, param in self.backbone.named_parameters():
            if name.startswith('stem.') or \
               name.startswith('stages.0.') or \
               name.startswith('stages.1.') or \
               name.startswith('stages.2.'):
                param.requires_grad = False
                
        self.gap = nn.AdaptiveAvgPool2d(1)
        embed_dim = 1024
        
        # normal_abnormal_head (binary -> 1 output)
        self.normal_abnormal_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        )
        
        # CBAM Attention Module for H2
        self.cbam = CBAMBlock(in_planes=embed_dim)
        
        # granular_pathology_head (multi-label -> 12 outputs)
        # Input dim is embed_dim + 1 (for H1 probability concatenation)
        self.granular_pathology_head = nn.Sequential(
            nn.Linear(embed_dim + 1, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_pathology_classes)
        )

    def forward(self, x: torch.Tensor):
        # forward_features returns the unpooled feature map from Stage 4 [B, 1024, H, W]
        features = self.backbone.forward_features(x)
        
        # --- Stream 1: H1 Gatekeeper ---
        # Un-gated Global Average Pooling
        gap_features = self.gap(features).flatten(1)
        out_normal = self.normal_abnormal_head(gap_features)
        
        # --- Stream 2: H2 Granular Pathology ---
        # CBAM Attention Filtering
        att_features = self.cbam(features)
        gap_att_features = self.gap(att_features).flatten(1)
        
        # Hierarchical Conditioning: Append H1 Probability
        # Detach h1_prob to prevent H2 from changing H1's parameters (optional, but safe)
        h1_prob = torch.sigmoid(out_normal).detach()
        h2_input = torch.cat([gap_att_features, h1_prob], dim=1)
        
        out_pathology = self.granular_pathology_head(h2_input)
        
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
            list(self.cbam.parameters()) +
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
