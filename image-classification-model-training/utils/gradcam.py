import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    Supports visualizing the regions of the input image that are most important
    for the model's prediction.
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
        import logging
        logger = logging.getLogger("GradCAM")
        logger.info(f"[Hook] Captured Activations - Max: {self.activations.max().item():.4f}, Min: {self.activations.min().item():.4f}")

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; we want the first element
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generates the Class Activation Map (CAM).
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W)
            target_class (int, optional): The class to generate CAM for. 
                                          If None, uses the class with highest score.
        
        Returns:
            np.ndarray: The normalized CAM (H, W) in range [0, 1].
        """
        # Ensure we have gradients enabled for this forward/backward pass
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Extract the score for the target class
        score = output[0, target_class]
        
        # Backward pass
        score.backward()
        
        # Get activations and gradients from the hooks
        gradients = self.gradients.detach().cpu().numpy()[0]   # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0] # (C, H, W)
        
        # Compute the channel weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        import logging
        logger = logging.getLogger("GradCAM")
        logger.info(f"Gradients - Max: {np.max(gradients):.4e}, Min: {np.min(gradients):.4e}, Sum: {np.sum(gradients):.4e}")
        logger.info(f"Activations - Max: {np.max(activations):.4f}, Min: {np.min(activations):.4f}, Sum: {np.sum(activations):.4f}")
        logger.info(f"Weights - Max: {np.max(weights):.4e}, Min: {np.min(weights):.4e}, Sum: {np.sum(weights):.4e}")
        
        # Compute the weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        import logging
        logger = logging.getLogger("GradCAM")
        logger.info(f"Raw CAM - Max: {np.max(cam):.4f}, Min: {np.min(cam):.4f}")
            
        # Apply ReLU to keep only features that have a positive influence on the target class
        cam = np.maximum(cam, 0)
        
        # Normalize the CAM to [0, 1]
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        cam_max = np.max(cam)
        cam_min = np.min(cam)
        logger.info(f"Post-ReLU Resized CAM - Max: {cam_max:.4f}, Min: {cam_min:.4f}")
        
        cam = cam - cam_min
        cam = cam / (cam_max + 1e-8)
        
        return cam

    @staticmethod
    def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """
        Overlays the CAM heatmap onto the original image.
        
        Args:
            img_pil (PIL.Image.Image): Original image (RGB).
            cam (np.ndarray): Normalized CAM (H, W) in [0, 1].
            alpha (float): Blending factor.
            
        Returns:
            PIL.Image.Image: Superimposed image.
        """
        # Ensure cam is same size as image
        if cam.shape != img_pil.size[::-1]:
            cam = cv2.resize(cam, img_pil.size)
            
        # Convert PIL to cv2 (RGB to BGR for colormap)
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Convert CAM to 8-bit heatmap
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed = cv2.addWeighted(img_cv2, 1 - alpha, heatmap, alpha, 0)
        
        # Convert back to PIL
        superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(superimposed)
