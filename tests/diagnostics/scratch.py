import numpy as np
from backend.core_ml.segmentation.inference.analyzer import SegmentationAnalyzer

# Generate dummy mask and test analyzer
mask = np.zeros((512, 512), dtype=np.uint8)
mask[100:200, :] = 1 # some layer
mask[200:300, 100:200] = 9 # fluid lesion

analyzer = SegmentationAnalyzer()
try:
    analysis = analyzer.analyze(mask)
    print("Success!", analysis.clinical_metrics.total_fluid_area)
except Exception as e:
    import traceback
    traceback.print_exc()
