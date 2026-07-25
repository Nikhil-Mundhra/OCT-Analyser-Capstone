import numpy as np
from PIL import Image

class GaussianNoiseCornersFiltered(object):
    """
    Blacks out bottom corners to hide scanner UI compasses/logos, using filtered grayscale noise.
    Archived version: Uses a 4-sided boundary sampling strategy, filters out white frames, 
    and applies monochromatic Gaussian noise matching the local background statistics.
    """
    def __init__(self, fraction=0.12, x_offset_frac=0.03, y_offset_frac=0.03):
        self.fraction = fraction
        self.x_offset_frac = x_offset_frac
        self.y_offset_frac = y_offset_frac

    def __call__(self, img):
        w, h = img.size
        base_dim = max(w, h)
        box_size = int(base_dim * self.fraction)
        x_off = int(base_dim * self.x_offset_frac)
        y_off = int(base_dim * self.y_offset_frac)
        
        # Bottom Left
        x1 = x_off
        y1 = h - box_size - y_off
        x2 = x_off + box_size
        y2 = h - y_off
        
        img_np = np.array(img)
        pad = int(base_dim * 0.02) # 2% padding for sampling
        
        # Define the 4 regions (handling boundaries)
        top_region = img_np[max(0, y1-pad):y1, max(0, x1):min(w, x2)]
        bottom_region = img_np[y2:min(h, y2+pad), max(0, x1):min(w, x2)]
        left_region = img_np[max(0, y1):min(h, y2), max(0, x1-pad):x1]
        right_region = img_np[max(0, y1):min(h, y2), x2:min(w, x2+pad)]
        
        # Flatten and concatenate
        pixels = []
        if top_region.size > 0: pixels.append(top_region.reshape(-1, 3))
        if bottom_region.size > 0: pixels.append(bottom_region.reshape(-1, 3))
        if left_region.size > 0: pixels.append(left_region.reshape(-1, 3))
        if right_region.size > 0: pixels.append(right_region.reshape(-1, 3))
        
        if pixels:
            all_pixels = np.concatenate(pixels, axis=0)
            
            # Filter out bright pixels (e.g. white frames) assuming background is dark
            # Only keep pixels where the maximum RGB value is less than 80
            dark_pixels = all_pixels[np.max(all_pixels, axis=1) < 80]
            
            if len(dark_pixels) > 0:
                mean = np.mean(dark_pixels, axis=0)
                std = np.std(dark_pixels, axis=0)
            else:
                mean = np.array([20, 20, 20])
                std = np.array([5, 5, 5])
        else:
            # Fallback if image is too small
            mean = np.array([20, 20, 20])
            std = np.array([5, 5, 5])
            
        # Generate Gaussian noise for the target region
        target_shape = img_np[y1:y2, x1:x2].shape
        if target_shape[0] > 0 and target_shape[1] > 0:
            # OCT is grayscale; generating independent RGB noise creates colorful speckles.
            # Generate 1-channel grayscale noise and repeat it across RGB to stay grayscale.
            gray_mean = np.mean(mean)
            gray_std = np.mean(std)
            noise_1d = np.random.normal(loc=gray_mean, scale=gray_std, size=(target_shape[0], target_shape[1], 1))
            noise_3d = np.repeat(noise_1d, 3, axis=2)
            noise = np.clip(noise_3d, 0, 255).astype(np.uint8)
            img_np[y1:y2, x1:x2] = noise
            return Image.fromarray(img_np)
            
        return img
