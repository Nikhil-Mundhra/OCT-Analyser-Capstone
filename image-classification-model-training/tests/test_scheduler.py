"""
tests/test_scheduler.py

Unit tests for proportional Cosine Annealing learning rate scheduling.
Verifies that differential learning rate ratios (head vs backbone) remain strictly constant
throughout all epochs without compression or off-by-one errors.
"""

import unittest
import torch
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multi_head_convnext import build_multi_head_model

class TestScheduler(unittest.TestCase):
    def test_proportional_cosine_scheduler_ratio_preservation(self):
        """
        Verify that LambdaLR proportional cosine decay preserves the exact 10:1 ratio
        between head_lr (1e-4) and backbone_lr (1e-5) on every single epoch.
        """
        model = build_multi_head_model(pretrained=False, warmup=False)
        head_lr = 1e-4
        backbone_lr = 1e-5
        finetune_epochs = 20

        groups = model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(groups)

        lr_lambda = lambda ep: 0.01 + (1.0 - 0.01) * 0.5 * (1.0 + math.cos(math.pi * min(ep, finetune_epochs) / max(1, finetune_epochs)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        prev_lrs = None
        for ep in range(finetune_epochs + 1):
            curr_lrs = [group['lr'] for group in optimizer.param_groups]
            
            # Group 0: backbone_decay, Group 2: head_decay
            backbone_curr = curr_lrs[0]
            head_curr = curr_lrs[2]
            
            ratio = head_curr / backbone_curr
            self.assertAlmostEqual(ratio, 10.0, places=4, msg=f"Ratio at epoch {ep} must be exactly 10.0")

            if ep == 0:
                self.assertAlmostEqual(head_curr, 1e-4, places=7)
                self.assertAlmostEqual(backbone_curr, 1e-5, places=8)
            elif ep == finetune_epochs:
                # 1% of initial LRs
                self.assertAlmostEqual(head_curr, 1e-6, places=9)
                self.assertAlmostEqual(backbone_curr, 1e-7, places=10)

            # Monotonicity check
            if prev_lrs is not None:
                self.assertLessEqual(curr_lrs[0], prev_lrs[0], f"LR at epoch {ep} must be <= previous epoch")
                self.assertLessEqual(curr_lrs[2], prev_lrs[2], f"LR at epoch {ep} must be <= previous epoch")

            prev_lrs = curr_lrs
            optimizer.step()
            scheduler.step()

if __name__ == '__main__':
    unittest.main()
