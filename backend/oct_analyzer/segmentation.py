from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any
import sys
import os
import cv2
import torch
import numpy as np

SEGMENTATION_ATLAS_ENV = "OCT_LAYER_ATLAS"
DEFAULT_LAYER_COUNT = 15

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_ML_SEG_DIR = PROJECT_ROOT / "backend" / "core_ml" / "segmentation"

try:
    from backend.core_ml.segmentation.models.unet import HierarchicalUNet
    from backend.core_ml.classification.utils.gradcam import HierarchicalUNetGradCAM
except ImportError:
    HierarchicalUNet = None
    HierarchicalUNetGradCAM = None

def atlas_path_from_env(env: dict[str, str] = os.environ) -> Path | None:
    path = env.get(SEGMENTATION_ATLAS_ENV)
    if path:
        return Path(path.strip()).expanduser()
    return None

class UnifiedOCTAnalyzer:
    _instance = None

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model = None
        checkpoint_path = CORE_ML_SEG_DIR / "weights" / "unet_hierarchical_best_cls.pth"
            
        if HierarchicalUNet is not None and checkpoint_path.exists():
            try:
                self.model = HierarchicalUNet(n_channels=1, n_coarse_classes=3, n_granular_classes=15)
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print("Successfully loaded Unified HierarchicalUNet.")
            except Exception as e:
                print(f"Failed to load UNet model: {e}")
                self.model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze_volume(self, volume: np.ndarray, gradcam: bool = True) -> tuple[np.ndarray, dict]:
        if self.model is None:
            raise RuntimeError("UNet model not loaded")
        
        z_dim, y_dim, x_dim = volume.shape
        labels_3d = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)
        batch_array = np.zeros((y_dim, 1, 512, 512), dtype=np.float32)
        
        for y in range(y_dim):
            slice_2d = volume[:, y, :]
            slice_min = slice_2d.min()
            slice_max = slice_2d.max()
            if slice_max > slice_min:
                slice_norm = (slice_2d - slice_min) / (slice_max - slice_min)
            else:
                slice_norm = np.zeros_like(slice_2d, dtype=np.float32)
                
            resized = cv2.resize(slice_norm, (512, 512), interpolation=cv2.INTER_LINEAR)
            batch_array[y, 0, :, :] = resized
            
        tensor_batch = torch.from_numpy(batch_array).to(self.device)
        
        # Scale from [0, 1] to [-1, 1] because the classification heads were trained on [-1, 1]
        tensor_batch = (tensor_batch * 2.0) - 1.0
        
        coarse_logits_list = []
        granular_preds_list = []
        cls_logits_dict_list = []
        
        batch_size = 8
        with torch.no_grad():
            for i in range(0, y_dim, batch_size):
                end_i = min(i + batch_size, y_dim)
                sub_batch = tensor_batch[i:end_i]
                coarse, granular, cls = self.model(sub_batch, task="both")
                
                granular_preds_list.append(torch.argmax(granular, dim=1).cpu().numpy().astype(np.uint8))
                coarse_logits_list.append(coarse.cpu())
                cls_logits_dict_list.append({k: v.cpu() if isinstance(v, torch.Tensor) else {k2: v2.cpu() for k2, v2 in v.items()} for k, v in cls.items()})

            granular_preds = np.concatenate(granular_preds_list, axis=0)
            
            cls_logits = {}
            cls_logits["normal_abnormal"] = torch.cat([d["normal_abnormal"] for d in cls_logits_dict_list], dim=0)
            cls_logits["pathology"] = torch.cat([d["pathology"] for d in cls_logits_dict_list], dim=0)
            cls_logits["severity"] = {}
            for head in ["macular", "diabetic", "vascular", "fluid", "structural"]:
                cls_logits["severity"][head] = torch.cat([d["severity"][head] for d in cls_logits_dict_list], dim=0)
            
            best_slice_idx = int(torch.argmax(cls_logits["normal_abnormal"][:, 0]))

            l1_logits_best = cls_logits["normal_abnormal"][best_slice_idx]
            l2_logits_best = cls_logits["pathology"][best_slice_idx]
            
            l1_probs = torch.sigmoid(l1_logits_best)
            l1_pred = "Abnormal" if l1_probs[0] > 0.5 else "Normal"
            level1_res = {"prediction": l1_pred, "confidence": float(l1_probs[0]) if l1_pred == "Abnormal" else 1 - float(l1_probs[0])}

            l2_names = ["Macular Degeneration", "Diabetic Complications", "Vascular Occlusions", "Fluid Accumulation", "Structural Issues"]
            l2_probs = torch.softmax(l2_logits_best, dim=0)
            l2_idx = int(torch.argmax(l2_probs))
            level2_res = {"prediction": l2_names[l2_idx], "confidence": float(l2_probs[l2_idx])}

            level3_res = {}
            if l1_pred == "Abnormal":
                l3_map = {
                    0: ("macular", ["CNV (Wet AMD)", "Drusen (Dry AMD)", "Generic AMD"]),
                    1: ("diabetic", ["DME", "DR"]),
                    2: ("vascular", ["Macular Hole", "RVO", "RAO"]),
                    3: ("fluid", ["CSR"]),
                    4: ("structural", ["ERM", "VID"])
                }
                head_key, head_class_names = l3_map[l2_idx]
                l3_logits_best = cls_logits["severity"][head_key][best_slice_idx]
                l3_probs = torch.sigmoid(l3_logits_best)
                l3_idx = int(torch.argmax(l3_probs))
                level3_res = {
                    "prediction": head_class_names[l3_idx], 
                    "confidence": float(l3_probs[l3_idx]),
                    "probs": {name: float(prob) for name, prob in zip(head_class_names, l3_probs)}
                }

            classification_results = {
                "Level1": level1_res,
                "Level2": level2_res,
                "Level3": level3_res,
                "Final_Diagnosis": level3_res.get("prediction", "Healthy"),
                "confidence": level3_res.get("confidence", level1_res["confidence"]),
                "best_slice_idx": best_slice_idx
            }

        if gradcam and HierarchicalUNetGradCAM is not None:
            best_slice = tensor_batch[best_slice_idx:best_slice_idx+1]
            cam_extractor = HierarchicalUNetGradCAM(self.model, self.model.down4.aspp)
            gradcams_base64 = {}
            
            with torch.enable_grad():
                cam_1 = cam_extractor.generate_cam(best_slice, target_head=1)
                gradcams_base64["L1"] = self._cam_to_base64(best_slice, cam_1)
                
                cam_2 = cam_extractor.generate_cam(best_slice, target_head=2, target_class=l2_idx)
                gradcams_base64["L2"] = self._cam_to_base64(best_slice, cam_2)
                
                if l1_pred == "Abnormal":
                    cam_3 = cam_extractor.generate_cam(best_slice, target_head=3, sub_head=head_key, target_class=l3_idx)
                    gradcams_base64["L3"] = self._cam_to_base64(best_slice, cam_3)
            
            classification_results["gradcams"] = gradcams_base64

        for y in range(y_dim):
            pred_2d = granular_preds[y]
            resized_back = cv2.resize(pred_2d, (x_dim, z_dim), interpolation=cv2.INTER_NEAREST)
            labels_3d[:, y, :] = resized_back
            
        return labels_3d, classification_results

    def _cam_to_base64(self, tensor_slice, cam) -> str:
        import base64
        from io import BytesIO
        from PIL import Image
        img_np = tensor_slice[0, 0].cpu().numpy()
        
        # Min-Max scale the normalized tensor back to [0, 1] before uint8 cast
        img_min = img_np.min()
        img_max = img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
        else:
            img_np = np.zeros_like(img_np)
            
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        superimposed = HierarchicalUNetGradCAM.overlay_cam(img_pil, cam)
        
        buffer = BytesIO()
        superimposed.save(buffer, format="JPEG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    mode: str
    warning: str = ""

LayerSegmenter = Callable[[np.ndarray, tuple[float, float, float]], tuple[SegmentationResult, dict[str, Any]]]

def segment_retinal_layers(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    segmenter: LayerSegmenter | None = None,
) -> tuple[SegmentationResult, dict[str, Any]]:
    if segmenter is not None:
        result, cls_results = segmenter(volume, spacing_mm)
        return _validated_result(result, volume.shape), cls_results

    analyzer = UnifiedOCTAnalyzer.get_instance()
    if analyzer.model is not None:
        try:
            labels, cls_results = analyzer.analyze_volume(volume)
            result = SegmentationResult(labels=labels, mode="ai")
            return _validated_result(result, volume.shape), cls_results
        except Exception as e:
            print(f"AI Analysis Failed: {e}")
            warning = f"UNet segmentation failed: {e}. Falling back to placeholder."
    else:
        warning = "UNet model not loaded. Falling back to deterministic placeholder layer segmentation"

    labels = placeholder_segment_layers(volume.shape, num_layers=DEFAULT_LAYER_COUNT)
    result = SegmentationResult(labels=labels, mode="placeholder", warning=warning)
    
    atlas_path = atlas_path_from_env()
    if atlas_path:
        result = SegmentationResult(labels=labels, mode="placeholder", warning="Atlas asset configured. " + warning)

    return _validated_result(result, volume.shape), {}

def placeholder_segment_layers(shape: tuple[int, int, int], num_layers: int = DEFAULT_LAYER_COUNT) -> np.ndarray:
    z_dim, y_dim, x_dim = shape
    if z_dim < num_layers:
        num_layers = z_dim
    labels = np.zeros(shape, dtype=np.uint8)
    edges = np.linspace(0, z_dim, num_layers + 1, dtype=int)
    for index in range(num_layers):
        labels[edges[index]:edges[index + 1], :, :] = index + 1
    if num_layers < DEFAULT_LAYER_COUNT:
        labels[labels == 0] = num_layers
    return labels

def _validated_result(result: SegmentationResult, expected_shape: tuple[int, int, int]) -> SegmentationResult:
    labels = validate_segmentation_labels(result.labels, expected_shape)
    return SegmentationResult(labels=labels, mode=result.mode, warning=result.warning)

def validate_segmentation_labels(
    labels: np.ndarray,
    expected_shape: tuple[int, int, int],
    max_label: int = DEFAULT_LAYER_COUNT,
) -> np.ndarray:
    array = np.asarray(labels)
    if array.shape != tuple(expected_shape):
        raise ValueError(f"Segmentation labels must match volume shape {expected_shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Segmentation labels must use integer layer IDs")
    if array.size and (int(array.min()) < 0 or int(array.max()) > max_label):
        raise ValueError(f"Segmentation labels must be between 0 and {max_label}")
    return array.astype(np.uint8, copy=False)
