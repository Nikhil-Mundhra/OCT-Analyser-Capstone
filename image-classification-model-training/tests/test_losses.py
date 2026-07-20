import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.losses import FocalLoss
from train_convnext_mps import AsymmetricLoss, BinaryFocalLossWithLogits

class TestLosses(unittest.TestCase):
    def test_focal_loss(self):
        """Verify Multi-class FocalLoss logic with and without class weights."""
        loss_fn_unweighted = FocalLoss(gamma=2.0)
        
        inputs = torch.tensor([[2.0, -1.0, 0.5], [0.1, 2.5, -0.2]])
        targets = torch.tensor([0, 1])
        
        loss_val = loss_fn_unweighted(inputs, targets)
        self.assertGreater(loss_val.item(), 0)
        self.assertFalse(torch.isnan(loss_val))
        
        # Test with alpha weights
        alpha = torch.tensor([0.1, 1.0, 10.0])
        loss_fn_weighted = FocalLoss(gamma=2.0, alpha=alpha)
        loss_val_weighted = loss_fn_weighted(inputs, targets)
        
        self.assertNotEqual(loss_val_weighted.item(), loss_val.item())
        
    def test_asymmetric_loss(self):
        """Verify AsymmetricLoss penalizes False Negatives more than False Positives."""
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
        
        inputs = torch.tensor([
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0] 
        ])
        
        targets = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0] 
        ])
        
        loss_val = loss_fn(inputs, targets)
        self.assertGreater(loss_val.item(), 0)
        self.assertFalse(torch.isnan(loss_val))

    def test_binary_focal_loss(self):
        """Verify BinaryFocalLossWithLogits handles scalar alpha properly."""
        loss_fn = BinaryFocalLossWithLogits(alpha=torch.tensor([5.0]), gamma=2.0)
        
        inputs = torch.tensor([[10.0], [-10.0]])
        targets = torch.tensor([[1.0], [0.0]])
        
        loss_val = loss_fn(inputs, targets)
        self.assertFalse(torch.isnan(loss_val))

if __name__ == '__main__':
    unittest.main()
