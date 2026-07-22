import torch
import numpy as np

class RetinaCropper:
    """
    Segmentation-Driven Cropper for OCT B-scans.
    Uses the 5-layer segmentation output to zero-out vitreous/background and 
    tightly crop or mask the retinal tissue before feeding to ConvNeXt.
    """
    def __init__(self, background_class_id: int = 0):
        self.bg_id = background_class_id

    def mask_background(self, image_tensor: torch.Tensor, seg_mask: torch.Tensor) -> torch.Tensor:
        """
        Zeros out the background while preserving 100% of the retinal tissue layers.
        
        Args:
            image_tensor: (B, C, H, W) or (C, H, W) input scan
            seg_mask: (B, H, W) or (H, W) segmentation mask (integer class IDs)
        Returns:
            masked_image: Same shape as image_tensor, background set to 0.
        """
        retina_mask = (seg_mask != self.bg_id).unsqueeze(1 if image_tensor.dim() == 4 else 0)
        return image_tensor * retina_mask.to(image_tensor.device)

    def crop_retina_bbox(self, image_tensor: torch.Tensor, seg_mask: torch.Tensor, margin: int = 10):
        """
        Extracts a tight bounding box around the retinal tissue layer.
        
        Args:
            image_tensor: (1, H, W) tensor
            seg_mask: (H, W) tensor
            margin: Extra padding around tissue bounds
        Returns:
            cropped_tensor: Cropped (1, H_crop, W_crop) tensor
            bbox: (ymin, ymax, xmin, xmax)
        """
        non_bg_coords = torch.nonzero(seg_mask != self.bg_id)
        if non_bg_coords.numel() == 0:
            return image_tensor, (0, image_tensor.shape[-2], 0, image_tensor.shape[-1])
        
        ymin, xmin = non_bg_coords.min(dim=0).values
        ymax, xmax = non_bg_coords.max(dim=0).values
        
        h, w = seg_mask.shape[-2:]
        ymin = max(0, int(ymin) - margin)
        ymax = min(h, int(ymax) + margin)
        xmin = max(0, int(xmin) - margin)
        xmax = min(w, int(xmax) + margin)
        
        cropped = image_tensor[..., ymin:ymax, xmin:xmax]
        return cropped, (ymin, ymax, xmin, xmax)
