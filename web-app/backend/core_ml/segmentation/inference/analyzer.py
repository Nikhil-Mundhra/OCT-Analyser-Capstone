import cv2
import numpy as np
from typing import List, Dict
from backend.core_ml.segmentation.inference.data_models import Point, BoundingBox, RetinalLayer, LesionInstance, ClinicalMetrics, OCTScanAnalysis

class SegmentationAnalyzer:
    """
    Parses a dense (H, W) granular segmentation mask into an object-oriented
    OCTScanAnalysis containing discrete vector instances and metrics.
    """
    
    # Example mapping (should match your dataset's exact class labels)
    CLASS_MAP = {
        0: "Background",
        1: "ILM",
        2: "NFL-IPL",
        3: "INL",
        4: "OPL",
        5: "ONL-ISM",
        6: "ISE",
        7: "OS-RPE",
        8: "RPE",
        9: "Fluid",
        10: "Hard Drusen",
        11: "Soft Drusen",
        12: "PED",
        13: "Geographic Atrophy",
        14: "Hyper-reflective Foci"
    }

    def __init__(self, layer_classes=list(range(1, 9)), lesion_classes=list(range(9, 15))):
        self.layer_classes = layer_classes
        self.lesion_classes = lesion_classes

    def analyze(self, mask: np.ndarray) -> OCTScanAnalysis:
        height, width = mask.shape
        
        layers = self._extract_layers(mask)
        lesions = self._extract_lesions(mask)
        metrics = self._calculate_metrics(layers, lesions, width)
        
        return OCTScanAnalysis(
            image_width=width,
            image_height=height,
            layers=layers,
            lesions=lesions,
            clinical_metrics=metrics,
            model_version="unet_hierarchical_v1.0"
        )
        
    def _extract_layers(self, mask: np.ndarray) -> List[RetinalLayer]:
        layers = []
        for class_id in self.layer_classes:
            binary_mask = (mask == class_id).astype(np.uint8) * 255
            
            # For continuous layers, we could extract the top boundary. 
            # A simple approach is taking the argmax along the y-axis for each x.
            # But since layers might have gaps, contours are safer.
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # To keep it as a simple "layer" concept for the frontend, we just take the largest contour
            # and format it as a boundary. (Alternatively, return all contours as polygons).
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            # Simplify contour to reduce payload size (Ramer-Douglas-Peucker)
            epsilon = 0.001 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            points = [Point(x=int(pt[0][0]), y=int(pt[0][1])) for pt in approx]
            
            # Compute average depth (mean y)
            if points:
                avg_depth = float(np.mean([pt.y for pt in points]))
            else:
                avg_depth = 0.0
                
            layers.append(RetinalLayer(
                class_id=class_id,
                class_name=self.CLASS_MAP.get(class_id, f"Layer_{class_id}"),
                boundary_points=points,
                avg_depth=avg_depth
            ))
            
        return layers
        
    def _extract_lesions(self, mask: np.ndarray) -> List[LesionInstance]:
        lesions = []
        for class_id in self.lesion_classes:
            binary_mask = (mask == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Filter out microscopic noise
                if area < 5.0:
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                bbox = BoundingBox(xmin=int(x), ymin=int(y), xmax=int(x+w), ymax=int(y+h))
                
                # Simplify polygon
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                polygon = [Point(x=int(pt[0][0]), y=int(pt[0][1])) for pt in approx]
                
                lesions.append(LesionInstance(
                    class_id=class_id,
                    class_name=self.CLASS_MAP.get(class_id, f"Lesion_{class_id}"),
                    polygon=polygon,
                    bounding_box=bbox,
                    area_pixels=float(area),
                    max_width=float(w),
                    max_height=float(h)
                ))
        return lesions
        
    def _calculate_metrics(self, layers: List[RetinalLayer], lesions: List[LesionInstance], width: int) -> ClinicalMetrics:
        # Example Metric: Total fluid area
        fluid_area = sum([L.area_pixels for L in lesions if L.class_name == "Fluid"])
        
        # Example Metric: Max fluid height
        fluid_heights = [L.max_height for L in lesions if L.class_name == "Fluid"]
        max_fluid_h = max(fluid_heights) if fluid_heights else 0.0
        
        # Example Metric: Average Retinal Thickness
        # Estimated as distance between topmost layer (ILM) and bottommost layer (RPE)
        layer_depths = [layer.avg_depth for layer in layers]
        if len(layer_depths) >= 2:
            thickness = float(max(layer_depths) - min(layer_depths))
        else:
            thickness = 0.0
            
        return ClinicalMetrics(
            average_retinal_thickness=thickness,
            total_fluid_area=float(fluid_area),
            max_fluid_height=float(max_fluid_h)
        )
