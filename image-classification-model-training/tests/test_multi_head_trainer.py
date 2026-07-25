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
        self.criterions = {'h1': DummyLoss(), 'h2': DummyLoss()}
        self.loss_weights = {'h1': 1.0, 'h2': 1.0}

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_multiclass_batch_size_1(self):
        """Verify multi-class mode (default argmax) works with batch size 1."""
        trainer = MultiHeadTrainer(
            model=self.model,
            criterions=self.criterions,
            loss_weights=self.loss_weights,
            mode="test_mode",
            checkpoint_dir=os.path.join(self.test_dir, "checkpoints"),
            log_dir=os.path.join(self.test_dir, "logs"),
            device=torch.device("cpu"),
            amp_dtype=torch.float32 
        )
        
        # Single multiclass index (e.g. class 2)
        images = torch.randn(1, 3, 224, 224)
        labels = {
            'normal_abnormal': torch.tensor([[1.0]]),
            'pathology': torch.tensor([2], dtype=torch.long)
        }
        
        class MockBatchLoader:
            def __iter__(self):
                yield images, labels
            def __len__(self):
                return 1
                
        val_loss, h2_f1, h2_recall, h2_acc = trainer._val_epoch(MockBatchLoader(), smoke_test=True)
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(h2_f1, float)

    def test_multilabel_batch_size_1(self):
        """Verify multi-label mode (injected sigmoid extractor) works with batch size 1."""
        trainer = MultiHeadTrainer(
            model=self.model,
            criterions=self.criterions,
            loss_weights=self.loss_weights,
            mode="test_mode",
            checkpoint_dir=os.path.join(self.test_dir, "checkpoints"),
            log_dir=os.path.join(self.test_dir, "logs"),
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
            metric_extractors={
                'h2': lambda logits: (torch.sigmoid(logits) > 0.5).int()
            }
        )
        
        # 12-class multi-hot target
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
                
        val_loss, h2_f1, h2_recall, h2_acc = trainer._val_epoch(MockBatchLoader(), smoke_test=True)
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(h2_f1, float)

    def test_empty_h2_mask(self):
        """Verify that validation works when NO images in the batch are abnormal."""
        trainer = MultiHeadTrainer(
            model=self.model,
            criterions=self.criterions,
            loss_weights=self.loss_weights,
            mode="test_mode",
            checkpoint_dir=os.path.join(self.test_dir, "checkpoints"),
            log_dir=os.path.join(self.test_dir, "logs"),
            device=torch.device("cpu")
        )
        images = torch.randn(4, 3, 224, 224)
        labels = {
            'normal_abnormal': torch.tensor([[0.0], [0.0], [0.0], [0.0]]),
            'pathology': torch.zeros(4, dtype=torch.long)
        }
        
        class MockBatchLoader:
            def __iter__(self):
                yield images, labels
            def __len__(self):
                return 1
                
        val_loss, h2_f1, h2_recall, h2_acc = trainer._val_epoch(MockBatchLoader(), smoke_test=True)
        self.assertIsInstance(h2_f1, float)

    def test_default_metric_extractor_fallback(self):
        """Verify that default fallback is argmax for multi-class classification."""
        trainer = MultiHeadTrainer(
            model=self.model,
            criterions=self.criterions,
            loss_weights=self.loss_weights,
            mode="test_mode",
            checkpoint_dir=os.path.join(self.test_dir, "checkpoints"),
            device=torch.device("cpu")
        )
        
        dummy_logits = torch.tensor([[10.0, -10.0, 2.0], [-10.0, 10.0, 0.0]])
        extracted = trainer.metric_extractors['h2'](dummy_logits)
        
        expected = torch.tensor([0, 1])
        self.assertTrue(torch.all(extracted == expected), "Default fallback must be argmax for multi-class targets")

if __name__ == '__main__':
    unittest.main()
