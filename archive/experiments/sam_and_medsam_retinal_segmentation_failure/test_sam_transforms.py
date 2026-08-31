"""
tests/test_sam_transforms.py
Unit test suite for SAM 2 / MedSAM preprocessing transforms, multi-channel composites,
and automated retinal tissue prompt generation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add repository root and training/classification to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training" / "classification"))

from data.preprocessing.sam_transforms import (
    build_sam_multichannel_inputs,
    generate_retinal_tissue_prompts,
    mask_to_smooth_envelope,
    draw_prompt_visualization,
)


class TestSAMTransforms(unittest.TestCase):
    """Test suite for SAM 2 / MedSAM medical preprocessing and prompt generation."""

    def setUp(self):
        # Create a synthetic 496x512 OCT B-scan
        self.h = 496
        self.w = 512
        self.img = np.zeros((self.h, self.w), dtype=np.uint8)
        # Background noise
        self.img[:150, :] = np.random.randint(0, 15, (150, self.w), dtype=np.uint8)
        # Retinal parenchyma band
        self.img[150:300, :] = np.random.randint(70, 130, (150, self.w), dtype=np.uint8)
        # Hyperreflective RPE band
        self.img[260:280, :] = np.random.randint(180, 240, (20, self.w), dtype=np.uint8)
        # Scanner metadata bar (to test blanking)
        self.img[:5, :] = 250

    def test_multichannel_inputs_shapes_and_types(self):
        """Verify that multi-channel composites and colormaps generate valid shapes and dtypes."""
        variants = build_sam_multichannel_inputs(self.img)
        
        self.assertIn("raw_gray", variants)
        self.assertIn("clahe", variants)
        self.assertIn("sobel_y", variants)
        self.assertIn("composite_3c", variants)
        self.assertIn("viridis_3c", variants)
        self.assertIn("jet_3c", variants)

        self.assertEqual(variants["raw_gray"].shape, (self.h, self.w))
        self.assertEqual(variants["clahe"].shape, (self.h, self.w))
        self.assertEqual(variants["sobel_y"].shape, (self.h, self.w))
        self.assertEqual(variants["composite_3c"].shape, (self.h, self.w, 3))
        self.assertEqual(variants["viridis_3c"].shape, (self.h, self.w, 3))
        self.assertEqual(variants["jet_3c"].shape, (self.h, self.w, 3))

        self.assertEqual(variants["composite_3c"].dtype, np.uint8)
        self.assertEqual(variants["viridis_3c"].dtype, np.uint8)

        # Ensure metadata bar (>=190) is blanked in clean image
        self.assertEqual(variants["raw_gray"][0, 0], 0)

    def test_prompt_generation_accuracy(self):
        """Verify that positive points, negative points, and bounding box align with retinal mass."""
        prompts = generate_retinal_tissue_prompts(self.img, num_pos_points=7)

        pos_coords = prompts["pos_coords"]
        neg_coords = prompts["neg_coords"]
        box = prompts["box"]
        y_center = prompts["y_center"]

        # Retinal core should be located in the tissue region (150 <= y <= 300)
        self.assertTrue(150 <= y_center <= 300, f"Expected y_center in [150, 300], got {y_center}")

        # Bounding box should span full width and enclose y_center
        self.assertEqual(box[0], 0)
        self.assertEqual(box[2], self.w)
        self.assertTrue(box[1] < y_center < box[3])

        # Positive coordinates should all be placed within the retinal tissue band [150, 300]
        self.assertEqual(len(pos_coords), 7)
        for pt in pos_coords:
            self.assertTrue(150 <= pt[1] <= 300, f"Expected positive pt y in [150, 300], got {pt[1]}")
            self.assertTrue(0 <= pt[0] < self.w)

        # Negative points should be placed in vitreous or retrobulbar space
        for pt in neg_coords:
            self.assertTrue(pt[1] < 150 or pt[1] > 260, f"Expected negative pt y outside tissue, got {pt[1]}")

    def test_mask_to_envelope_smoothness(self):
        """Verify that binary masks are converted into smooth continuous bounding envelopes."""
        binary_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        binary_mask[160:290, :] = 255

        envelope_mask, y_top_out, y_bot_out = mask_to_smooth_envelope(
            binary_mask, margin_top=15, margin_bottom=20, gaussian_sigma=5.0
        )

        self.assertEqual(envelope_mask.shape, (self.h, self.w))
        self.assertEqual(len(y_top_out), self.w)
        self.assertEqual(len(y_bot_out), self.w)

        # Top boundary should include margin (160 - 15 = 145)
        # Bottom boundary should include margin (289 + 20 = 309)
        self.assertAlmostEqual(np.mean(y_top_out), 145.0, delta=3.0)
        self.assertAlmostEqual(np.mean(y_bot_out), 309.0, delta=3.0)
        self.assertTrue(np.all(y_top_out < y_bot_out))

    def test_prompt_visualization_rendering(self):
        """Verify prompt overlay drawing function."""
        prompts = generate_retinal_tissue_prompts(self.img)
        vis = draw_prompt_visualization(np.zeros((self.h, self.w, 3), dtype=np.uint8), prompts)
        self.assertEqual(vis.shape, (self.h, self.w, 3))
        # Non-zero pixels should exist due to overlays
        self.assertTrue(np.sum(vis) > 0)


if __name__ == "__main__":
    unittest.main()
