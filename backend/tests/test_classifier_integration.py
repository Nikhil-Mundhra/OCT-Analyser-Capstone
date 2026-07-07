import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import tempfile
import os

from backend.oct_analyzer.api import app
import backend.oct_analyzer.api as api_module
from backend.oct_analyzer.classifier_integration import ClassifierWrapper, get_classifier
import backend.oct_analyzer.classifier_integration as ci
import backend.oct_analyzer.mvp_pipeline as mvp_pipeline

client = TestClient(app)

def test_predict_image_invalid_suffix():
    response = client.post("/predict", files={"file": ("test.txt", b"dummy")})
    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported file type"}

def test_predict_image_success(monkeypatch):
    class MockClassifier:
        def predict(self, img_path, gradcam=True):
            return {
                "Level1": {"prediction": "NORMAL", "confidence": 0.99},
                "Final_Diagnosis": "NORMAL",
                "gradcams": {}
            }
    
    monkeypatch.setattr(api_module, "get_classifier", lambda: MockClassifier())
    
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
        
    try:
        with open(tmp_path, "rb") as f:
            response = client.post("/predict", files={"file": ("test.png", f, "image/png")})
        assert response.status_code == 200
        data = response.json()
        assert data["level1_prediction"] == "NORMAL"
        assert data["final_diagnosis"] == "NORMAL"
    finally:
        os.remove(tmp_path)

def test_predict_image_error(monkeypatch):
    class MockClassifier:
        def predict(self, img_path, gradcam=True):
            return {"error": "Mock error"}
            
    monkeypatch.setattr(api_module, "get_classifier", lambda: MockClassifier())
    
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
        
    try:
        with open(tmp_path, "rb") as f:
            response = client.post("/predict", files={"file": ("test.png", f, "image/png")})
        assert response.status_code == 400
        assert response.json()["detail"] == "Mock error"
    finally:
        os.remove(tmp_path)

def test_classifier_wrapper_singleton(monkeypatch):
    ci.ClassifierWrapper._instance = None
    
    class MockPipeline:
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, image_path, gradcam=True):
            return {"mock": True}
            
    monkeypatch.setattr(ci, "OCTInferencePipeline", MockPipeline)
    
    wrapper1 = get_classifier()
    wrapper2 = get_classifier()
    assert wrapper1 is wrapper2
    assert wrapper1.predict("dummy_path") == {"mock": True}

def test_classifier_wrapper_missing_pipeline(monkeypatch):
    ci.ClassifierWrapper._instance = None
    monkeypatch.setattr(ci, "OCTInferencePipeline", None)
    with pytest.raises(RuntimeError, match="OCTInferencePipeline is not available."):
        ClassifierWrapper()

def test_process_scan_classifier_exception(monkeypatch):
    def mock_get_classifier():
        raise ValueError("Simulated classifier error")
        
    monkeypatch.setattr(mvp_pipeline, "get_classifier", mock_get_classifier)
    
    from backend.oct_analyzer.scan_types import NormalizedScan
    scan = NormalizedScan(
        volume=np.zeros((1, 10, 10), dtype=np.float32),
        spacing_mm=(1.0, 1.0, 1.0),
        source_format="vol",
        metadata={},
        warnings=[]
    )
        
    result = mvp_pipeline.process_scan(scan)
    assert result["level1"] == {}
    
def test_process_scan_classifier_success(monkeypatch):
    class MockClassifier:
        def predict(self, img_path, gradcam=True):
            return {
                "Level1": {"prediction": "NORMAL", "confidence": 0.99},
                "Final_Diagnosis": "NORMAL"
            }
            
    monkeypatch.setattr(mvp_pipeline, "get_classifier", lambda: MockClassifier())
    
    from backend.oct_analyzer.scan_types import NormalizedScan
    scan = NormalizedScan(
        volume=np.zeros((1, 10, 10), dtype=np.float32),
        spacing_mm=(1.0, 1.0, 1.0),
        source_format="vol",
        metadata={},
        warnings=[]
    )
        
    result = mvp_pipeline.process_scan(scan)
    assert result["diagnosis"] == "NORMAL"
    assert result["confidence"] == 0.99
    assert result["level1"] == {"prediction": "NORMAL", "confidence": 0.99}
    
def test_classifier_integration_import_error(monkeypatch):
    import builtins
    import importlib
    
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if "scripts.inference_pipeline" in name:
            raise ImportError("Simulated import error")
        return real_import(name, *args, **kwargs)
        
    monkeypatch.setattr(builtins, "__import__", fake_import)
    
    # Reload the module to trigger the import again
    importlib.reload(ci)
    assert ci.OCTInferencePipeline is None
    
    # Restore for other tests
    importlib.reload(ci)

def test_main_cli(monkeypatch):
    from backend.oct_analyzer import main
    import builtins
    
    # Mock load_oct_volume to simulate success
    def mock_load(path):
        return np.zeros((10, 10, 10)), (1.0, 1.0, 1.0)
    monkeypatch.setattr(main, "load_oct_volume", mock_load)
    
    # Mock pipeline
    monkeypatch.setattr(main, "get_preprocessing_pipeline", lambda: lambda x: x)
    
    # Mock flatten
    monkeypatch.setattr(main, "flatten_volume_to_rpe", lambda x: x)
    
    main.main()

def test_main_cli_execution(monkeypatch):
    import runpy
    import sys
    
    # Mock load_oct_volume to prevent file errors
    import backend.oct_analyzer.main as main
    monkeypatch.setattr(main, "load_oct_volume", lambda path: (np.zeros((10,10,10)), (1,1,1)))
    monkeypatch.setattr(main, "get_preprocessing_pipeline", lambda: lambda x: x)
    monkeypatch.setattr(main, "flatten_volume_to_rpe", lambda x: x)
    
    # Run module as main
    runpy.run_module("backend.oct_analyzer.main", run_name="__main__")
