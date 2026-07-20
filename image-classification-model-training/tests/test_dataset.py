import unittest
import torch
import sys
import os
import pandas as pd
from PIL import Image
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_convnext_mps import MultiHeadOCTDataset, compute_loss_weights, PATHOLOGY_CLASSES

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.test_dir, "mock_dataset.csv")
        
        img_dir = os.path.join(self.test_dir, "images")
        os.makedirs(img_dir)
        
        data = []
        for i in range(10):
            img_path = os.path.join(img_dir, f"img_{i}.jpg")
            Image.new("RGB", (224, 224)).save(img_path)
            
            if i < 5:
                # Normal
                data.append({"image_path": img_path, "head1_label": 0, "head3_labels": ""})
            else:
                # Abnormal
                if i % 2 == 0:
                    data.append({"image_path": img_path, "head1_label": 1, "head3_labels": "AMD, CNV"})
                else:
                    data.append({"image_path": img_path, "head1_label": 1, "head3_labels": "DME"})
                    
        df = pd.DataFrame(data)
        df.to_csv(self.csv_file, index=False)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dataset_parsing(self):
        """Verify that the dataset correctly parses Multi-Label strings into tensors."""
        dataset = MultiHeadOCTDataset(self.csv_file, transform=None)
        
        self.assertEqual(len(dataset), 10)
        
        # Check a Normal Image (index 0)
        _, labels_normal = dataset[0]
        self.assertEqual(labels_normal['normal_abnormal'].item(), 0.0)
        self.assertEqual(labels_normal['pathology'].sum().item(), 0.0)
        
        # Check an Abnormal Image with AMD, CNV (index 6, which is i=6 in generation, meaning abnormal even index)
        _, labels_abnormal_1 = dataset[6]
        self.assertEqual(labels_abnormal_1['normal_abnormal'].item(), 1.0)
        
        expected_h2_1 = torch.zeros(12)
        expected_h2_1[PATHOLOGY_CLASSES.index('AMD')] = 1.0
        expected_h2_1[PATHOLOGY_CLASSES.index('CNV')] = 1.0
        self.assertTrue(torch.all(labels_abnormal_1['pathology'] == expected_h2_1))
        
        # Check an Abnormal Image with DME (index 7, which is i=7 in generation, abnormal odd index)
        _, labels_abnormal_2 = dataset[7]
        self.assertEqual(labels_abnormal_2['normal_abnormal'].item(), 1.0)
        
        expected_h2_2 = torch.zeros(12)
        expected_h2_2[PATHOLOGY_CLASSES.index('DME')] = 1.0
        self.assertTrue(torch.all(labels_abnormal_2['pathology'] == expected_h2_2))

    def test_compute_loss_weights(self):
        """Verify that class imbalance weights are computed correctly without NaNs."""
        dataset = MultiHeadOCTDataset(self.csv_file, transform=None)
        device = torch.device("cpu")
        
        h1_w, h2_w = compute_loss_weights(dataset.df, device)
        
        self.assertEqual(h1_w.item(), 5.0)
        
        self.assertEqual(h2_w.shape, (12,))
        self.assertFalse(torch.any(torch.isnan(h2_w)))
        
        self.assertLessEqual(torch.max(h2_w).item(), 10.0)

if __name__ == '__main__':
    unittest.main()
