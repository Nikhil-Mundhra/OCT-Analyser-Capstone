import unittest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multi_head_convnext import build_multi_head_model
from training.multi_head_trainer import MultiHeadTrainer

class DummyLoss(nn.Module):
    def forward(self, inputs, targets):
        return torch.tensor(1.0, requires_grad=True)

class TestMultiHeadTrainer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model = build_multi_head_model(pretrained=False, warmup=False)
        criterions = {
            'h1': DummyLoss(),
            'h2': DummyLoss()
        }
        loss_weights = {'h1': 1.0, 'h2': 1.0}
        
        self.trainer = MultiHeadTrainer(
            model=self.model,
            criterions=criterions,
            loss_weights=loss_weights,
            mode="test_mode",
            checkpoint_dir=os.path.join(self.test_dir, "checkpoints"),
            log_dir=os.path.join(self.test_dir, "logs"),
            device=torch.device("cpu"),
            amp_dtype=torch.float32 
        )
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_trainer_batch_size_1_edge_case(self):
        """Verify that validation does not crash with a batch size of exactly 1."""
        images = torch.randn(1, 3, 224, 224)
        labels = {
            'normal_abnormal': torch.tensor([[1.0]]),
            'pathology': torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        }
        
        class MockBatchLoader:
            def __iter__(self):
                yield images, labels
            def __len__(self):
                return 1
                
        try:
            val_loss, h2_f1, h2_recall, h2_acc = self.trainer._val_epoch(MockBatchLoader(), smoke_test=True)
        except ValueError as e:
            self.fail(f"Trainer crashed on batch size 1 with ValueError: {e}")
            
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(h2_f1, float)

    def test_trainer_empty_h2_mask(self):
        """Verify that validation works when NO images in the batch are abnormal."""
        images = torch.randn(4, 3, 224, 224)
        labels = {
            'normal_abnormal': torch.tensor([[0.0], [0.0], [0.0], [0.0]]),
            'pathology': torch.zeros(4, 12)
        }
        
        class MockBatchLoader:
            def __iter__(self):
                yield images, labels
            def __len__(self):
                return 1
                
        try:
            val_loss, h2_f1, h2_recall, h2_acc = self.trainer._val_epoch(MockBatchLoader(), smoke_test=True)
        except Exception as e:
            self.fail(f"Trainer crashed on empty H2 mask with: {e}")
            
        self.assertIsInstance(h2_f1, float)

    def test_metric_extractor_fallback(self):
        """Verify that the default metric_extractor produces multilabel outputs (B, 12) rather than multiclass argmax."""
        # Using the base trainer initialized without metric_extractors
        self.assertIn('h2', self.trainer.metric_extractors)
        
        dummy_logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        extracted = self.trainer.metric_extractors['h2'](dummy_logits)
        
        expected = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        self.assertTrue(torch.all(extracted == expected), "Fallback metric extractor must output multilabel predictions")

if __name__ == '__main__':
    unittest.main()
