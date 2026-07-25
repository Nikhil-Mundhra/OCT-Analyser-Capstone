import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

@dataclass
class Point:
    x: int
    y: int

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclass
class RetinalLayer:
    """
    Represents a continuous anatomical boundary layer (e.g. ILM, RPE).
    Stored as a sequence of points (x, y) forming a 1D spline across the image width.
    """
    class_id: int
    class_name: str
    boundary_points: List[Point]
    avg_depth: float

@dataclass
class LesionInstance:
    """
    Represents a discrete pathological finding (e.g. Fluid, Drusen).
    Stored as a closed polygon contour.
    """
    class_id: int
    class_name: str
    polygon: List[Point]
    bounding_box: BoundingBox
    area_pixels: float
    # Optional clinical metrics based on geometry
    max_height: Optional[float] = None
    max_width: Optional[float] = None

@dataclass
class ClinicalMetrics:
    """
    Global clinical metrics calculated for the entire scan.
    """
    average_retinal_thickness: float
    total_fluid_area: float
    max_fluid_height: float

@dataclass
class OCTScanAnalysis:
    """
    The root object representing the full analysis of a single OCT B-scan.
    """
    image_width: int
    image_height: int
    layers: List[RetinalLayer]
    lesions: List[LesionInstance]
    clinical_metrics: ClinicalMetrics
    model_version: str = "v1.0"
    
    def to_json(self) -> str:
        """Serializes the analysis object to a JSON string."""
        return json.dumps(asdict(self), indent=2)
