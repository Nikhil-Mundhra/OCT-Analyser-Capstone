import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class MultiHeadGradCAM:
    """
    Grad-CAM for Multi-Head ConvNeXt V2 models.
    Supports targeting a specific head (1, 2, or 3) and a specific class within that head.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.activations = output.clone().detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_head=2, target_class=None):
        """
        Generates the Class Activation Map (CAM).
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W)
            target_head (int): Which head to explain (1 or 2). Defaults to 2 (Pathology Routing).
            target_class (int, optional): The class within the target head. 
                                          If None, uses the class with the highest score.
        """
        self.model.zero_grad()
        
        # Forward pass returns a dict
        outputs = self.model(input_tensor)
        
        if target_head == 1:
            target_output = outputs['normal_abnormal']
        elif target_head == 2:
            target_output = outputs['pathology']
        else:
            raise ValueError(f"Unknown target_head: {target_head}")
        
        # If binary (Head 1), target_output is [1, 1]
        if target_output.shape[1] == 1:
            score = target_output[0, 0]
        else:
            if target_class is None:
                # for multi-class, find highest activation
                target_class = torch.softmax(target_output, dim=1).argmax(dim=1).item()
            score = target_output[0, target_class]
        
        # Backward pass
        score.backward()
        
        gradients = self.gradients.detach().cpu().numpy()[0]   # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0] # (C, H, W)
        
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        cam_max = np.max(cam)
        cam_min = np.min(cam)
        
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam

    @staticmethod
    def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
        if cam.shape != img_pil.size[::-1]:
            cam = cv2.resize(cam, img_pil.size)
            
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed = cv2.addWeighted(img_cv2, 1 - alpha, heatmap, alpha, 0)
        superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(superimposed)
