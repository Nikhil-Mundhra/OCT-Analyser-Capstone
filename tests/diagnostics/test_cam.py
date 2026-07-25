import numpy as np
from PIL import Image
import cv2

img_np = np.zeros((512, 512), dtype=np.uint8)
img_pil = Image.fromarray(img_np)

try:
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    print("Shape after cvtColor:", img_cv2.shape)
except Exception as e:
    print("Error:", e)
