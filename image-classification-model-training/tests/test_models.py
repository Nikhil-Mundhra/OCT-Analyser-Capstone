"""
tests/test_models.py

Comprehensive tests for MultiHeadConvNeXt architecture, feature extraction contracts,
and 4-group weight decay parameter splitting.
"""

import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multi_head_convnext import build_multi_head_model

class TestModels(unittest.TestCase):
    def test_model_forward_pass(self):
        """Verify that the model takes in an image batch and returns the correct dictionary outputs."""
        model = build_multi_head_model(pretrained=False, warmup=True)
        model.eval()
        
        # 4 images, 3 channels, 224x224
        dummy_input = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(dummy_input)
            
        self.assertIn('normal_abnormal', outputs)
        self.assertIn('pathology', outputs)
        
        self.assertEqual(outputs['normal_abnormal'].shape, (4, 1), "H1 should output (B, 1)")
        self.assertEqual(outputs['pathology'].shape, (4, 12), "H2 should output (B, 12)")

    def test_freeze_unfreeze_backbone(self):
        """Verify that the freezing logic correctly targets the stem and stages 0-2."""
        model = build_multi_head_model(pretrained=False, warmup=False)
        
        # Freeze
        model.freeze_backbone()
        for name, param in model.backbone.named_parameters():
            if name.startswith(('stem.', 'stages.0.', 'stages.1.', 'stages.2.')):
                self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen during warmup")
            elif name.startswith('stages.3.'):
                self.assertTrue(param.requires_grad, f"Stage 3 parameter {name} should remain trainable")
                
        # Unfreeze
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            self.assertTrue(param.requires_grad)

    def test_weight_decay_parameter_group_splitting(self):
        """
        Verify weight decay parameter group splitting:
        - Must return 4 parameter groups: [backbone_decay, backbone_no_decay, head_decay, head_no_decay]
        - All parameters with ndim >= 2 (4D Conv kernels, 2D Linear weights) must get weight_decay > 0
        - All 1D biases and LayerNorm/BatchNorm parameters must get weight_decay == 0.0
        """
        model = build_multi_head_model(pretrained=False, warmup=False)
        
        backbone_lr = 1e-5
        head_lr = 1e-4
        target_wd = 1e-4
        
        groups = model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=target_wd)
        
        self.assertEqual(len(groups), 4, "Should return exactly four parameter groups")
        
        # Check group configurations
        self.assertEqual(groups[0]['lr'], backbone_lr)
        self.assertEqual(groups[0]['weight_decay'], target_wd)
        
        self.assertEqual(groups[1]['lr'], backbone_lr)
        self.assertEqual(groups[1]['weight_decay'], 0.0)
        
        self.assertEqual(groups[2]['lr'], head_lr)
        self.assertEqual(groups[2]['weight_decay'], target_wd)
        
        self.assertEqual(groups[3]['lr'], head_lr)
        self.assertEqual(groups[3]['weight_decay'], 0.0)
        
        # Verify complete coverage of all parameters
        num_params_in_groups = sum(len(g['params']) for g in groups)
        num_params_in_model = len([p for p in model.parameters() if p.requires_grad])
        self.assertEqual(num_params_in_groups, num_params_in_model, "All trainable parameters must be covered across groups")

        # Verify parameter classification: 1D biases and norm parameters in no_decay groups
        for g_idx in [1, 3]:  # no_decay groups
            for p in groups[g_idx]['params']:
                self.assertTrue(p.ndim <= 1, "no_decay group should only contain 1D bias or norm tensors")

        for g_idx in [0, 2]:  # decay groups
            for p in groups[g_idx]['params']:
                self.assertTrue(p.ndim >= 2, "decay group should only contain multi-dimensional tensors (ndim >= 2)")

    def test_feature_channels_contract(self):
        """Verify that ConvNeXt V2 Base extracts exact channel dimensions (256, 512, 1024)."""
        model = build_multi_head_model(pretrained=False, warmup=False)
        
        feature_channels = model.backbone.feature_info.channels()
        self.assertEqual(feature_channels, [256, 512, 1024], "Channel dimensions must match baseline contract")
        
        # Concatenated linear input dimension (256 + 512 + 1024 + 1 = 1793)
        expected_concat_dim = sum(feature_channels) + 1
        actual_in_features = model.granular_pathology_head[0].in_features
        self.assertEqual(actual_in_features, expected_concat_dim, "Head input dimension mismatch!")

if __name__ == '__main__':
    unittest.main()
