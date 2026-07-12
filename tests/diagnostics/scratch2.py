from PIL import Image
import numpy as np
import cv2

img_np = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
img_pil = Image.fromarray(img_np)

try:
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    print("Shape after RGB2BGR:", img_cv2.shape)
except Exception as e:
    print("Error 1:", e)

# The code in gradcam.py:
img_cv2 = np.array(img_pil)
if len(img_cv2.shape) == 2:
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2RGB)

print("Correct Shape:", img_cv2.shape)

