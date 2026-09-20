"""
tests/test_tissue_cropper.py

Unit tests for Attention U-Net Dynamic Tissue Cropper architecture, unified dataset loader,
loss functions, boundary vector extraction, and pre-flight smoke training.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training" / "segmentation" / "train_tissue_cropper"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "segmentation"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "classification"))
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from models_suite.tissue_cropper.tissue_unet import TissueCroppingUNet
from dataset_unified_tissue import (
    UnifiedTissueDataset,
    get_tissue_augmentations,
    get_train_val_tissue_datasets,
)
from losses import (
    CombinedTissueLoss,
    DiceLoss,
    FocalLoss,
    compute_dice_and_iou,
)
from train import train
from data.preprocessing.unet_cropper import UNetTissueCropper


class TestTissueCropperSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Create mock segmented dataset structure
        self.seg_root = self.tmp_path / "Segmented"
        self.oct5k_imgs = self.seg_root / "OCT5K_Semantic_Segmentation" / "Images"
        self.oct5k_masks = self.seg_root / "OCT5K_Semantic_Segmentation" / "Masks" / "Grading_1"
        self.oct5k_imgs.mkdir(parents=True)
        self.oct5k_masks.mkdir(parents=True)

        # Create mock curated structure
        self.masked_root = self.tmp_path / "Classified-masked"
        self.cur_imgs = self.masked_root / "Images" / "DME"
        self.cur_masks = self.masked_root / "Masks" / "DME"
        self.cur_imgs.mkdir(parents=True)
        self.cur_masks.mkdir(parents=True)

        # Write dummy scans
        mock_img = np.full((100, 100), 50, dtype=np.uint8)
        mock_oct5k_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_oct5k_mask[20:70, :] = 2  # Layer class 2

        mock_cur_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_cur_mask[25:65, :] = 255  # Binary 255

        cv2.imwrite(str(self.oct5k_imgs / "scan01.png"), mock_img)
        cv2.imwrite(str(self.oct5k_masks / "scan01.png"), mock_oct5k_mask)

        cv2.imwrite(str(self.cur_imgs / "scan02.png"), mock_img)
        cv2.imwrite(str(self.cur_masks / "scan02.png"), mock_cur_mask)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_unified_dataset_loading_and_binarization(self):
        dataset = UnifiedTissueDataset(
            segmented_root=self.seg_root,
            masked_root=self.masked_root,
            target_size=(512, 512),
            in_channels=1,
        )
        self.assertEqual(len(dataset), 2)

        # Sample 0 (OCT5K)
        img0, mask0 = dataset[0]
        self.assertEqual(img0.shape, (1, 512, 512))
        self.assertEqual(mask0.shape, (512, 512))
        self.assertEqual(mask0.dtype, torch.int64)
        self.assertEqual(mask0.max().item(), 1)
        self.assertEqual(mask0.min().item(), 0)

        # Sample 1 (Curated)
        img1, mask1 = dataset[1]
        self.assertEqual(img1.shape, (1, 512, 512))
        self.assertEqual(mask1.shape, (512, 512))
        self.assertEqual(mask1.max().item(), 1)

    def test_tissue_augmentations(self):
        aug = get_tissue_augmentations(is_train=True)
        self.assertIsNotNone(aug)
        dummy_img = np.full((512, 512), 80, dtype=np.uint8)
        dummy_mask = np.zeros((512, 512), dtype=np.uint8)
        dummy_mask[100:300, :] = 1
        res = aug(image=dummy_img, mask=dummy_mask)
        self.assertEqual(res["image"].shape, (512, 512))
        self.assertEqual(res["mask"].shape, (512, 512))

    def test_tissue_unet_forward_pass_and_boundary_extraction(self):
        model = TissueCroppingUNet(in_channels=1, num_classes=2, base_c=16)
        x = torch.randn(2, 1, 256, 256)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 2, 256, 256))

        # Test 1D boundary curve extraction
        mock_prob = np.zeros((200, 300), dtype=float)
        mock_prob[40:120, :] = 0.95
        y_top, y_bot = TissueCroppingUNet.extract_boundary_vectors(mock_prob, threshold=0.5, min_thickness=15)
        self.assertEqual(len(y_top), 300)
        self.assertEqual(len(y_bot), 300)
        self.assertTrue(np.all(y_top < y_bot))
        self.assertTrue(np.all(y_bot >= y_top + 15))

    def test_loss_functions_and_metrics(self):
        loss_fn = CombinedTissueLoss(ce_weight=1.0, dice_weight=1.0, focal_weight=0.5)
        logits = torch.randn(2, 2, 64, 64, requires_grad=True)
        targets = torch.randint(0, 2, (2, 64, 64), dtype=torch.long)

        loss = loss_fn(logits, targets)
        self.assertTrue(loss.item() > 0)

        loss.backward()
        self.assertIsNotNone(logits.grad)

        dice, iou = compute_dice_and_iou(logits, targets)
        self.assertTrue(0.0 <= dice <= 1.0)
        self.assertTrue(0.0 <= iou <= 1.0)

    def test_unet_cropper_inference_pipeline(self):
        cropper = UNetTissueCropper(device="cpu", target_size=(256, 256))
        mock_scan = np.full((300, 400, 3), 40, dtype=np.uint8)
        mock_scan[80:180, :, :] = 170

        cropped, y_top, y_bot, crop_box = cropper.crop_and_suppress_background(mock_scan, margin=10)
        self.assertEqual(len(y_top), 400)
        self.assertEqual(len(y_bot), 400)
        self.assertEqual(len(cropped.shape), 3)
        self.assertTrue(cropped.shape[0] <= 300)
        self.assertEqual(cropped.shape[1], 400)

    def test_white_bar_detection(self):
        from data.preprocessing.white_bars import detect_and_process_white_bars
        # 1. Verify scan where tissue touches top edge without a wide banner is NOT cut/notched
        dme5_path = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified/Diabetic Complications/Diabetic Macular Edema (DME)/DME/DME-1695472-5.jpeg")
        if dme5_path.exists():
            im = cv2.imread(str(dme5_path))
            cleaned = detect_and_process_white_bars(im)
            diff = np.sum(cleaned != im)
            self.assertEqual(diff, 0, "Tissue touching top edge was erroneously notched!")

        # 2. Verify synthetic image with real wide banner (> 35 cols) is cleanly removed
        synthetic_banner = np.full((100, 200), 30, dtype=np.uint8)
        synthetic_banner[:10, 20:120] = 250  # 100-col wide white banner
        cleaned_syn = detect_and_process_white_bars(synthetic_banner)
        self.assertTrue(np.all(cleaned_syn[:10, 20:120] == 0), "Synthetic banner was not removed!")

    def test_train_smoke_test(self):
        # Run pre-flight smoke test
        best_dice = train(epochs=2, smoke_test=True)
        self.assertTrue(best_dice >= 0.0)


if __name__ == "__main__":
    unittest.main()
