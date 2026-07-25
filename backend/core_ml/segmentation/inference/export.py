import json
from pathlib import Path
from backend.core_ml.segmentation.inference.data_models import OCTScanAnalysis

class InferenceExporter:
    @staticmethod
    def to_json_file(analysis: OCTScanAnalysis, filepath: str) -> None:
        """
        Exports the OCTScanAnalysis to a JSON file.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w') as f:
            f.write(analysis.to_json())
            
    @staticmethod
    def to_json_string(analysis: OCTScanAnalysis) -> str:
        """
        Exports the OCTScanAnalysis to a JSON string (for direct API responses).
        """
        return analysis.to_json()
