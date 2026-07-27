import unittest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil
import types

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

    def test_resume_checkpoint_restores_batch_idx_and_scaler(self):
        """Resume should restore the scaler state and continue from the next batch."""
        source_trainer = MultiHeadTrainer(
            model=self.model,
            criterions=self.criterions,
            loss_weights=self.loss_weights,
            mode="test_mode",
            checkpoint_dir=os.path.join(self.test_dir, "checkpoints"),
            log_dir=os.path.join(self.test_dir, "logs"),
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
        )
        optimizer = torch.optim.SGD(source_trainer.model.parameters(), lr=0.1)
        ckpt_path = os.path.join(source_trainer.ckpt_dir, "resume_checkpoint.pth")
        source_trainer._save_checkpoint(
            "resume_checkpoint.pth",
            optimizer=optimizer,
            scaler=source_trainer.scaler,
            epoch=0,
            phase="warmup",
            batch_idx=3,
            best_val_loss=1.0,
            best_val_macro_f1=0.5,
        )

        resumed_trainer = MultiHeadTrainer(
            model=build_multi_head_model(pretrained=False, warmup=False),
            criterions=self.criterions,
            loss_weights=self.loss_weights,
            mode="test_mode",
            checkpoint_dir=os.path.join(self.test_dir, "checkpoints"),
            log_dir=os.path.join(self.test_dir, "logs"),
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
        )

        recorded = {}

        def fake_train_epoch(self, loader, optimizer, smoke_test=False, accum_steps=1, save_steps=2250, fold_id=0, epoch=0, phase="finetune", best_val_loss=float("inf"), best_val_macro_f1=0.0, hf_repo=None, start_batch=0):
            recorded["start_batch"] = start_batch
            return 0.0

        def fake_val_epoch(self, loader, smoke_test=False):
            return 0.0, 0.0, 0.0, 0.0

        resumed_trainer._train_epoch = types.MethodType(fake_train_epoch, resumed_trainer)
        resumed_trainer._val_epoch = types.MethodType(fake_val_epoch, resumed_trainer)

        class MockLoader:
            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

        resumed_trainer.train(
            MockLoader(),
            MockLoader(),
            warmup_epochs=1,
            finetune_epochs=0,
            smoke_test=True,
            resume_path=ckpt_path,
        )

        self.assertEqual(recorded["start_batch"], 4)
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        self.assertIn("scaler_state_dict", ckpt)


if __name__ == '__main__':
    unittest.main()
