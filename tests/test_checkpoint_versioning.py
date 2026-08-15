import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import tempfile
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.oct_analyzer.checkpoint_versioning import (
    resolve_and_create_version_dir,
    update_version_metadata_metrics,
    get_git_info,
    get_hardware_info
)

class TestCheckpointVersioning(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_auto_version_increment(self):
        base_dir = self.temp_dir / "checkpoints" / "test_model"
        
        # First call should create v1
        v1_dir, v1_tag = resolve_and_create_version_dir(base_dir=base_dir, requested_version="auto")
        self.assertEqual(v1_tag, "v1")
        self.assertTrue(v1_dir.exists())
        self.assertTrue((v1_dir / "version_metadata.md").exists())
        
        # Second call should create v2
        v2_dir, v2_tag = resolve_and_create_version_dir(base_dir=base_dir, requested_version="auto")
        self.assertEqual(v2_tag, "v2")
        self.assertTrue(v2_dir.exists())
        self.assertTrue((v2_dir / "version_metadata.md").exists())

    def test_explicit_version(self):
        base_dir = self.temp_dir / "checkpoints" / "test_model"
        v_custom_dir, v_custom_tag = resolve_and_create_version_dir(
            base_dir=base_dir,
            requested_version="v10",
            args_dict={"epochs": 5, "lr": 1e-4}
        )
        self.assertEqual(v_custom_tag, "v10")
        self.assertTrue((v_custom_dir / "version_metadata.md").exists())
        content = (v_custom_dir / "version_metadata.md").read_text()
        self.assertIn("- **`epochs`**: `5`", content)
        self.assertIn("- **`lr`**: `0.0001`", content)

    def test_update_version_metadata_metrics(self):
        base_dir = self.temp_dir / "checkpoints" / "test_model"
        v1_dir, _ = resolve_and_create_version_dir(base_dir=base_dir, requested_version="v1")
        
        update_version_metadata_metrics(v1_dir, {
            "Validation Loss": "0.1810",
            "Macro F1": "0.8090"
        })
        
        content = (v1_dir / "version_metadata.md").read_text()
        self.assertIn("- **Validation Loss**: `0.1810`", content)
        self.assertIn("- **Macro F1**: `0.8090`", content)

    def test_hardware_and_git_info(self):
        hw = get_hardware_info()
        self.assertIn("device", hw)
        self.assertIn("torch_version", hw)
        self.assertIn("os_arch", hw)

        git = get_git_info()
        self.assertIn("hash", git)
        self.assertIn("branch", git)

if __name__ == "__main__":
    unittest.main()
