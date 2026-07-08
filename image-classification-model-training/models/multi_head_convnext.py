import torch
import torch.nn as nn
import timm

class MultiHeadConvNeXt(nn.Module):
    """
    Multi-Head ConvNeXt V2 Model
    
    1. Initializes a pre-trained convnextv2_base.
    2. Freezes the stem and first three stages, leaving Stage 4 unfrozen.
    3. Attaches Global Average Pooling to the Stage 4 output.
    4. Branches into parallel MLP heads for Normal/Abnormal (H1), Pathology (H2).
    5. Branches Head 3 (Severity) into 5 distinct sub-tensors for clinical families.
    """
    def __init__(self, num_pathology_classes: int = 5, pretrained: bool = True):
        super().__init__()
        
        # 1. Initialize pre-trained convnextv2_base from timm
        self.backbone = timm.create_model('convnextv2_base', pretrained=pretrained)
        
        # Remove the original classification head
        self.backbone.reset_classifier(0)
        
        # 2. Freeze all parameters in the stem and the first three stages.
        # ConvNeXt stages are 0-indexed in timm (stages.0 to stages.3).
        # We leave Stage 4 (stages.3) unfrozen.
        for name, param in self.backbone.named_parameters():
            if name.startswith('stem.') or \
               name.startswith('stages.0.') or \
               name.startswith('stages.1.') or \
               name.startswith('stages.2.'):
                param.requires_grad = False
                
        # 3. Attach a Global Average Pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 4. Branch this vector into three parallel MLP heads
        # ConvNeXt V2 base has a 1024-d output feature map at the final stage
        embed_dim = 1024
        
        # normal_abnormal_head (binary -> 1 output)
        self.normal_abnormal_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        )
        
        # pathology_type_head (multi-class -> `num_pathology_classes` outputs)
        self.pathology_type_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_pathology_classes)
        )
        
        # severity_head -> Branched into 5 clinical family sub-vectors (Option A)
        self.macular_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 3) # CNV, DRUSEN, Generic_AMD
        )
        self.diabetic_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 2) # DME, DR
        )
        self.vascular_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 3) # MH, RVO, RAO
        )
        self.fluid_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 1) # CSR
        )
        self.structural_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 2) # ERM, VID
        )

    def forward(self, x: torch.Tensor):
        # 5. Clean forward pass that does not detach the computational graph.
        # forward_features returns the unpooled feature map from Stage 4
        # (e.g., shape: [B, 1024, H, W])
        x = self.backbone.forward_features(x)
        
        # Apply Global Average Pooling (yielding a 1024-d vector per image)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Pass through the main heads
        out_normal = self.normal_abnormal_head(x)
        out_pathology = self.pathology_type_head(x)
        
        # Pass through the 5 clinical sub-vectors for Head 3
        out_macular = self.macular_head(x)
        out_diabetic = self.diabetic_head(x)
        out_vascular = self.vascular_head(x)
        out_fluid = self.fluid_head(x)
        out_structural = self.structural_head(x)
        
        return {
            'normal_abnormal': out_normal,
            'pathology': out_pathology,
            'severity': {
                'macular': out_macular,
                'diabetic': out_diabetic,
                'vascular': out_vascular,
                'fluid': out_fluid,
                'structural': out_structural
            }
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
        """Differential LRs: backbone at backbone_lr, all heads at head_lr."""
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.normal_abnormal_head.parameters()) +
            list(self.pathology_type_head.parameters()) +
            list(self.macular_head.parameters()) +
            list(self.diabetic_head.parameters()) +
            list(self.vascular_head.parameters()) +
            list(self.fluid_head.parameters()) +
            list(self.structural_head.parameters())
        )
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ]


def build_multi_head_model(pretrained=True, warmup=True) -> MultiHeadConvNeXt:
    """
    Factory function for creating the Multi-Head ConvNeXt model.
    """
    model = MultiHeadConvNeXt(num_pathology_classes=5, pretrained=pretrained)
    if warmup:
        model.freeze_backbone()
    return model
