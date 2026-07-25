import os
import sys
import unittest
import tempfile
import shutil
import subprocess
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestRefactoringFunctional(unittest.TestCase):
    """
    Functional tests auditing the repository refactoring:
    - Path resolution from scripts/
    - Script importability and CLI flag handling
    - Manifest generation to data/
    """
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scripts_dir = os.path.join(self.base_dir, "scripts")
        self.config_path = os.path.join(self.base_dir, "config", "hierarchy.yaml")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_config_path_resolution(self):
        """Verify hierarchy.yaml config file exists and is accessible from scripts/ context."""
        self.assertTrue(os.path.exists(self.config_path), f"Config file missing at {self.config_path}")

    def test_train_convnext_cli_help(self):
        """Verify train_convnext.py script in scripts/ executes --help clean without import errors."""
        train_script = os.path.join(self.scripts_dir, "train_convnext.py")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU execution
        
        result = subprocess.run(
            [sys.executable, train_script, "--help"],
            capture_output=True,
            text=True,
            env=env
        )
        self.assertEqual(result.returncode, 0, f"train_convnext.py --help failed with stderr: {result.stderr}")
        self.assertIn("Train Multi-Head ConvNeXt", result.stdout)

    def test_evaluate_best_model_imports(self):
        """Verify evaluate_best_model.py in scripts/ can be imported cleanly."""
        sys.path.append(self.scripts_dir)
        try:
            import evaluate_best_model
            self.assertTrue(hasattr(evaluate_best_model, "GradCAM"))
            self.assertTrue(hasattr(evaluate_best_model, "load_model"))
            self.assertTrue(hasattr(evaluate_best_model, "get_data_loader"))
        except ImportError as e:
            self.fail(f"Failed to import evaluate_best_model: {e}")

    def test_generate_manifest_functional(self):
        """Verify generate_manifest.py in scripts/ generates a manifest CSV in target output directory."""
        mock_dataset = os.path.join(self.temp_dir, "mock_data")
        os.makedirs(os.path.join(mock_dataset, "Normal (Healthy)"), exist_ok=True)
        img_path = os.path.join(mock_dataset, "Normal (Healthy)", "sample1.png")
        Image.new("RGB", (100, 100), color="white").save(img_path)

        out_csv = os.path.join(self.temp_dir, "data", "dataset_manifest.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        manifest_script = os.path.join(self.scripts_dir, "generate_manifest.py")
        result = subprocess.run(
            [sys.executable, manifest_script, "--dataset_root", mock_dataset, "--output_path", out_csv],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, f"generate_manifest.py failed: {result.stderr}")
        self.assertTrue(os.path.exists(out_csv), "Manifest CSV was not created at expected output path")

    def test_root_directory_cleanliness(self):
        """Verify no loose execution scripts or heavy csv files remain at subproject root."""
        root_files = os.listdir(self.base_dir)
        forbidden_files = ["train_convnext_mps.py", "evaluate_best_model.py", "generate_manifest.py", "dataset_manifest.csv"]
        for forbidden in forbidden_files:
            self.assertNotIn(forbidden, root_files, f"Refactored artifact '{forbidden}' still present in subproject root!")

if __name__ == "__main__":
    unittest.main()
