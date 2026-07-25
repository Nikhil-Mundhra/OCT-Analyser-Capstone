import os
import sys
import unittest
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.device import get_device, get_raw_model, ComputeManager

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def freeze_backbone(self):
        return True

class TestDevice(unittest.TestCase):
    def test_get_device(self):
        """Verify device auto-detection returns a valid torch.device instance."""
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_compute_manager_initialization(self):
        """Verify ComputeManager initializes and reports device properties."""
        cm = ComputeManager()
        self.assertIsInstance(cm.device, torch.device)
        self.assertIsInstance(cm.use_data_parallel, bool)

    def test_prepare_model_and_get_raw_model(self):
        """Verify prepare_model and get_raw_model handle model containers properly."""
        cm = ComputeManager(use_data_parallel=False)
        raw = DummyModel()
        prepared = cm.prepare_model(raw)
        
        unwrapped = get_raw_model(prepared)
        self.assertTrue(hasattr(unwrapped, "freeze_backbone"))
        self.assertTrue(unwrapped.freeze_backbone())

    def test_flush_cache(self):
        """Verify flush_cache executes without throwing errors."""
        cm = ComputeManager(cache_flush_interval=5)
        # Flush on interval
        cm.flush_cache(batch_idx=4)
        # Force flush
        cm.flush_cache(force=True)

if __name__ == "__main__":
    unittest.main()
