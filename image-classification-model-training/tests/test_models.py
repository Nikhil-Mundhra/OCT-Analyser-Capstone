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
        
        # Stem and stages.0. shouldn't require grad
        for name, param in model.backbone.named_parameters():
            if name.startswith(('stem.', 'stages.0.', 'stages.1.', 'stages.2.')):
                self.assertFalse(param.requires_grad)
            elif name.startswith('stages.3.'):
                self.assertTrue(param.requires_grad)
                
        # Unfreeze
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            self.assertTrue(param.requires_grad)

    def test_get_param_groups(self):
        """Verify the differential learning rate parameter groups."""
        model = build_multi_head_model(pretrained=False, warmup=False)
        
        backbone_lr = 1e-5
        head_lr = 1e-3
        
        groups = model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr)
        
        self.assertEqual(len(groups), 2, "Should return exactly two param groups")
        
        self.assertEqual(groups[0]['lr'], backbone_lr)
        self.assertEqual(groups[1]['lr'], head_lr)
        
        num_params_in_groups = sum(len(g['params']) for g in groups)
        num_params_in_model = len(list(model.parameters()))
        
        self.assertEqual(num_params_in_groups, num_params_in_model, "All parameters must be in a param group")

    def test_feature_channels_contract(self):
        """Verify that ConvNeXt V2 Base extracts exact channel dimensions (256, 512, 1024)."""
        model = build_multi_head_model(pretrained=False, warmup=False)
        
        # Verify backbone channels
        feature_channels = model.backbone.feature_info.channels()
        self.assertEqual(feature_channels, [256, 512, 1024], "Channel dimensions must match baseline contract")
        
        # Verify concatenated linear input dimension (256 + 512 + 1024 + 1 = 1793)
        expected_concat_dim = sum(feature_channels) + 1
        actual_in_features = model.granular_pathology_head[0].in_features
        self.assertEqual(actual_in_features, expected_concat_dim, "Head input dimension mismatch!")

if __name__ == '__main__':
    unittest.main()
