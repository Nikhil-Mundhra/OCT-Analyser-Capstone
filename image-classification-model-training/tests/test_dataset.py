import unittest
import torch
import sys
import os
import tempfile
import shutil
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MultiHeadOCTDataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(base_dir, "config", "hierarchy.yaml")
        
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock folders expected by hierarchy.yaml class_map
        mock_paths = [
            "Normal (Healthy)/OCT2017",
            "Macular Degeneration Spectrum/Choroidal Neovascularization/CNV",
            "Macular Degeneration Spectrum/DRUSEN",
            "Diabetic Complications/Diabetic Macular Edema (DME)/DME"
        ]
        
        for rel_path in mock_paths:
            full_path = os.path.join(self.test_dir, rel_path)
            os.makedirs(full_path, exist_ok=True)
            img_file = os.path.join(full_path, "sample_0.jpeg")
            Image.new("RGB", (224, 224), color="gray").save(img_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dataset_initialization(self):
        """Verify MultiHeadOCTDataset loads mock dataset correctly."""
        dataset = MultiHeadOCTDataset(
            config_path=self.config_path,
            data_root=self.test_dir,
            transform=None
        )
        self.assertEqual(len(dataset), 4)
        
        # Test item indexing
        image, targets = dataset[0]
        self.assertIn("normal_abnormal", targets)
        self.assertIn("pathology", targets)
        self.assertIsInstance(targets["normal_abnormal"], torch.Tensor)
        self.assertIsInstance(targets["pathology"], torch.Tensor)

    def test_compute_class_weights(self):
        """Verify class weights calculation returns valid non-NaN tensor."""
        dataset = MultiHeadOCTDataset(
            config_path=self.config_path,
            data_root=self.test_dir,
            transform=None
        )
        h2_weights = dataset.compute_class_weights("h2")
        self.assertEqual(h2_weights.shape, (12,))
        self.assertFalse(torch.any(torch.isnan(h2_weights)))

if __name__ == '__main__':
    unittest.main()
