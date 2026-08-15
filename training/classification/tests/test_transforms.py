"""
tests/test_transforms.py

Unit tests for OCT medical image augmentation pipelines in data/transforms.py.
Verifies strict medical constraints:
- NO vertical flips (preserves vitreous to RPE superior-inferior anatomical ordering)
- Bounded max rotation angle (<= 0.09 rad / ~5 degrees)
- Deterministic validation pipeline
"""

import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import get_transforms
from monai.transforms import RandFlip, RandRotate

class TestTransforms(unittest.TestCase):
    def test_train_transforms_medical_constraints(self):
        """
        Verify that training transforms enforce medical OCT constraints:
        1. Vertical flip (spatial_axis=0) must NOT be present.
        2. Horizontal flip (spatial_axis=1) MUST be present.
        3. RandRotate range_x must be <= 0.09 (~5 degrees max).
        """
        train_pipeline = get_transforms("train")
        
        has_horizontal_flip = False
        has_vertical_flip = False
        max_rotation = 0.0

        for t in train_pipeline.transforms:
            if isinstance(t, RandFlip):
                axis = getattr(t, 'spatial_axis', getattr(getattr(t, 'flipper', None), 'spatial_axis', None))
                if axis == 0:
                    has_vertical_flip = True
                elif axis == 1:
                    has_horizontal_flip = True
            elif isinstance(t, RandRotate):
                range_val = t.range_x
                max_rotation = max(abs(x) for x in range_val) if isinstance(range_val, (list, tuple)) else abs(range_val)

        self.assertFalse(has_vertical_flip, "Vertical flip (spatial_axis=0) MUST NOT be present in OCT transforms!")
        self.assertTrue(has_horizontal_flip, "Horizontal flip (spatial_axis=1) must be present in OCT transforms.")
        self.assertLessEqual(max_rotation, 0.09, "Rotation angle range_x must be <= 0.09 rad (~5 degrees max)")

    def test_val_transforms_deterministic(self):
        """Verify that validation transforms pipeline contains no random augmentations."""
        val_pipeline = get_transforms("val")
        for t in val_pipeline.transforms:
            self.assertFalse(
                isinstance(t, (RandFlip, RandRotate)),
                f"Validation transform should not contain random augmentation {type(t)}"
            )

if __name__ == '__main__':
    unittest.main()
