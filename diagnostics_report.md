# Model Health Diagnostics Report
**Target Checkpoint:** `backend/core_ml/classification/weights/multi_head_mps/fold0_best_model.pth`

## Weight Health Check - ❌ FAILED
```text
[STDERR]
Traceback (most recent call last):
  File "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/tests/diagnostics/test_weights.py", line 3, in <module>
    from backend.core_ml.segmentation.models.unet import HierarchicalUNet
ModuleNotFoundError: No module named 'backend'
```

## Tensor Scaling Check - ❌ FAILED
```text
[STDERR]
Traceback (most recent call last):
  File "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/tests/diagnostics/test_div255.py", line 3, in <module>
    from backend.core_ml.segmentation.models.unet import HierarchicalUNet
ModuleNotFoundError: No module named 'backend'
```

## Bias & Features Check - ❌ FAILED
```text
[STDERR]
Traceback (most recent call last):
  File "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/tests/diagnostics/test_features.py", line 3, in <module>
    from backend.core_ml.segmentation.models.unet import HierarchicalUNet
ModuleNotFoundError: No module named 'backend'
```

## Grad-CAM Explainability Check - ✅ PASSED
```text
Shape after cvtColor: (512, 512, 3)
```
