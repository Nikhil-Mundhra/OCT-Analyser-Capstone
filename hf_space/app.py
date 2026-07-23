import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import torch
torch.set_num_threads(1)


import torch.nn.functional as F
import numpy as np
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    gr = None
    HAS_GRADIO = False

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add repository root and model subdirectories to path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
m2_path = WORKSPACE_ROOT / "models_suite/model2_choroidalyzer"
if str(m2_path) not in sys.path:
    sys.path.insert(0, str(m2_path))


# Import models from models_suite
from models_suite.model1_oct5k_layers.unet_layers import RetinalLayersUNet
from models_suite.model2_choroidalyzer.choroidalyze.model import UNet as ChoroidalyzerUNet

from models_suite.model3_hrf_dme.hrf_aunet import HRFAttentionUNet as HRF_AttentionUNet

from models_suite.model4_oimhs_hole_cysts.oimhs_unet import OIMHSUNet
from models_suite.model5_oct5k_detection.detector import OCTPathologyDetector, OCT5K_DETECTION_CLASSES

try:
    import spaces
    IS_HF_SPACE = True
except ImportError:
    spaces = None
    IS_HF_SPACE = False

# Device configuration:
# Local runs: use Mac GPU (mps/cuda) if available
# Deployed HF Space: ConvNeXtV2 uses @spaces.GPU for ZeroGPU, Segmentation models use CPU only
if IS_HF_SPACE:
    SEGMENT_DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    SEGMENT_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    SEGMENT_DEVICE = torch.device("mps")
else:
    SEGMENT_DEVICE = torch.device("cpu")

DEVICE = SEGMENT_DEVICE
print(f"Initializing OCT Analyzer Suite | Deployed HF Space: {IS_HF_SPACE} | Segmentation Device: {SEGMENT_DEVICE}")

try:
    import numpy._core.multiarray as np_multiarray
    torch.serialization.add_safe_globals([np_multiarray.scalar])
except Exception:
    pass

def safe_load_ckpt(cp_path, device):
    try:
        return torch.load(cp_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(cp_path, map_location=device, weights_only=False)




# Initialize model instances and load checkpoints
def load_suite():
    print("Loading M1...", flush=True)
    m1 = RetinalLayersUNet(in_channels=1, num_classes=6)
    cp1 = WORKSPACE_ROOT / "models_suite/model1_oct5k_layers/checkpoints/best_model.pth"
    if cp1.exists():
        print("M1 checkpoint file exists. Calling safe_load_ckpt...", flush=True)
        ckpt = safe_load_ckpt(cp1, DEVICE)
        print("M1 safe_load_ckpt returned! Loading state dict...", flush=True)
        m1.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m1.to(DEVICE).eval()
    print("M1 Done.", flush=True)


    print("Loading M2...", flush=True)
    m2 = ChoroidalyzerUNet(in_channels=1, out_channels=3, depth=7, channels='8_doublemax-64', up_type='conv_then_interpolate', extra_out_conv=True)
    cp2 = WORKSPACE_ROOT / "models_suite/model2_choroidalyzer/checkpoints/best_model.pth"
    if cp2.exists():
        ckpt = safe_load_ckpt(cp2, DEVICE)
        m2.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m2.to(DEVICE).eval()
    print("M2 Done.", flush=True)

    print("Loading M3...", flush=True)
    m3 = HRF_AttentionUNet(n_channels=3, n_classes=1)
    cp3 = WORKSPACE_ROOT / "models_suite/model3_hrf_dme/checkpoints/best_model.pth"
    if cp3.exists():
        ckpt = safe_load_ckpt(cp3, DEVICE)
        m3.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m3.to(DEVICE).eval()
    print("M3 Done.", flush=True)

    print("Loading M4...", flush=True)
    m4 = OIMHSUNet(in_channels=1, num_classes=5)
    cp4 = WORKSPACE_ROOT / "models_suite/model4_oimhs_hole_cysts/checkpoints/best_model.pth"
    if cp4.exists():
        ckpt = safe_load_ckpt(cp4, DEVICE)
        m4.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m4.to(DEVICE).eval()
    print("M4 Done.", flush=True)

    print("Loading M5...", flush=True)
    m5 = OCTPathologyDetector(num_classes=10)
    cp5 = WORKSPACE_ROOT / "models_suite/model5_oct5k_detection/checkpoints/best_model.pth"
    if cp5.exists():
        ckpt = safe_load_ckpt(cp5, DEVICE)
        m5.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m5.to(DEVICE).eval()
    print("M5 Done.", flush=True)

    return m1, m2, m3, m4, m5




model1, model2, model3, model4, model5 = load_suite()
print("OCT Analyser 5-Model Suite Loaded Successfully!", flush=True)


# Preprocessing helpers
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    gray = image.convert("L").resize(target_size)
    arr = np.array(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    return gray, tensor

def create_segmentation_overlay(orig_img: Image.Image, mask: np.ndarray, num_classes: int, cmap: list):
    orig_rgb = orig_img.convert("RGB")
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(1, num_classes):
        color = cmap[c % len(cmap)]
        overlay[mask == c] = color

    overlay_img = Image.fromarray(overlay).resize(orig_rgb.size)
    blended = Image.blend(orig_rgb, overlay_img, alpha=0.45)
    return blended

def preprocess_rgb_image(image: Image.Image, target_size=(256, 256)):
    rgb = image.convert("RGB").resize(target_size)
    arr = np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 3, H, W)
    return rgb, tensor

COLOR_MAP = [
    [0, 0, 0],       # 0: BG
    [255, 50, 50],   # 1: Red
    [50, 255, 50],   # 2: Green
    [50, 50, 255],   # 3: Blue
    [255, 255, 50],  # 4: Yellow
    [255, 50, 255],  # 5: Magenta
    [50, 255, 255]   # 6: Cyan
]

# Inference handlers for each model
def predict_model1(image):
    if image is None: return None, "Please upload an OCT scan image."
    gray, tensor = preprocess_image(image, (256, 256))
    with torch.no_grad():
        logits = model1(tensor.to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    overlay = create_segmentation_overlay(image, preds, 6, COLOR_MAP)
    layer_names = ["Background", "ILM -> OPL", "OPL -> IS-OS", "IS-OS -> IBRPE", "IBRPE -> OBRPE", "Choroid & Below"]
    counts = {layer_names[c]: int((preds == c).sum()) for c in range(6)}
    info = f"Retinal Layer Segmentation Complete.\nPixel Breakdown:\n" + "\n".join([f"• {k}: {v} px" for k, v in counts.items()])
    return overlay, info

def predict_model2(image):
    if image is None: return None, "Please upload an OCT scan image."
    gray, tensor = preprocess_image(image, (256, 256))
    with torch.no_grad():
        logits = model2(tensor.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        mask = (probs > 0.5).astype(np.uint8)
    
    overlay = create_segmentation_overlay(image, mask, 2, [[0,0,0], [0, 220, 255]])
    choroid_area = int(mask.sum())
    mean_thickness = float(mask.sum(axis=0).mean())
    info = f"Choroid Region Analysis:\n• Choroid Area: {choroid_area} px\n• Est. Mean Thickness: {mean_thickness:.2f} px"
    return overlay, info

def predict_model3(image):
    if image is None: return None, "Please upload an OCT scan image."
    rgb, tensor = preprocess_rgb_image(image, (256, 256))
    with torch.no_grad():
        logits = model3(tensor.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        mask = (probs > 0.5).astype(np.uint8)
    
    overlay = create_segmentation_overlay(image, mask, 2, [[0,0,0], [255, 0, 120]])
    fluid_px = int(mask.sum())
    info = f"High-Resolution HRF DME Fluid & Lesion Attention Analysis:\n• Pathological Fluid / Lesion Region: {fluid_px} px"
    return overlay, info


def predict_model4(image):
    if image is None: return None, "Please upload an OCT scan image."
    gray, tensor = preprocess_image(image, (256, 256))
    with torch.no_grad():
        logits = model4(tensor.to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    overlay = create_segmentation_overlay(image, preds, 5, COLOR_MAP)
    classes = ["Background", "Macular Hole", "Choroid", "Retina", "Intraretinal Cysts (IRC)"]
    counts = {classes[c]: int((preds == c).sum()) for c in range(5)}
    info = "OIMHS Pathology Analysis:\n" + "\n".join([f"• {k}: {v} px" for k, v in counts.items()])
    return overlay, info

def predict_model5(image, score_threshold=0.5):
    if image is None: return None, "Please upload an OCT scan image."
    orig_rgb = image.convert("RGB")
    gray, tensor = preprocess_image(image, (256, 256))
    # Faster R-CNN expects 3-channel input
    input_tensor = tensor.repeat(1, 3, 1, 1).to(DEVICE)
    
    with torch.no_grad():
        outputs = model5([input_tensor[0]])[0]

    
    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    # Scale boxes back to original image size
    orig_w, orig_h = orig_rgb.size
    scale_x = orig_w / 256.0
    scale_y = orig_h / 256.0

    draw_img = orig_rgb.copy()
    draw = ImageDraw.Draw(draw_img)

    detected_items = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y
            cls_name = OCT5K_DETECTION_CLASSES[label] if label < len(OCT5K_DETECTION_CLASSES) else f"Class {label}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 4, max(0, y1 - 12)), f"{cls_name} {score:.2f}", fill="yellow")
            detected_items.append(f"• {cls_name}: Conf {score:.2%} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    info = f"OCT Pathology Object Detector ({len(detected_items)} objects detected above threshold {score_threshold:.2f}):\n"
    if detected_items:
        info += "\n".join(detected_items)
    else:
        info += "No objects detected above threshold."
        
    return draw_img, info

# Build Gradio UI for HF Space
if HAS_GRADIO:
    with gr.Blocks(title="OCT Analyser 5-Model Microservice Suite") as demo:
        gr.Markdown("# 👁️ OCT Analyser 5-Model Suite (Hugging Face API Services Deployment)")
        gr.Markdown("Comprehensive Optical Coherence Tomography (OCT) Deep Learning Suite providing 5 API services for segmentation, choroid analysis, fluid/lesion quantification, macular hole/cyst detection, and 9-class biomarker object detection.")

        with gr.Tabs():
            with gr.TabItem("Model 1: Retinal Layers U-Net"):
                gr.Markdown("### 6-Class Retinal Layer Segmentation U-Net (OCT5K Benchmark)")
                with gr.Row():
                    with gr.Column():
                        img1 = gr.Image(type="pil", label="Input OCT Scan")
                        btn1 = gr.Button("Segment Retinal Layers", variant="primary")
                    with gr.Column():
                        out_img1 = gr.Image(type="pil", label="6-Layer Segmentation Overlay")
                        txt1 = gr.Textbox(label="Layer Metrics", lines=7)
                btn1.click(predict_model1, inputs=img1, outputs=[out_img1, txt1], api_name="predict_model1")

            with gr.TabItem("Model 2: Choroidalyzer U-Net"):
                gr.Markdown("### Choroid Region & Thickness Quantification U-Net")
                with gr.Row():
                    with gr.Column():
                        img2 = gr.Image(type="pil", label="Input OCT Scan")
                        btn2 = gr.Button("Analyze Choroid Region", variant="primary")
                    with gr.Column():
                        out_img2 = gr.Image(type="pil", label="Choroid Mask Overlay")
                        txt2 = gr.Textbox(label="Choroid Metrics", lines=5)
                btn2.click(predict_model2, inputs=img2, outputs=[out_img2, txt2], api_name="predict_model2")

            with gr.TabItem("Model 3: HRF Attention U-Net"):
                gr.Markdown("### High-Resolution Fluid & Lesion Attention U-Net (HRF DME/AMD)")
                with gr.Row():
                    with gr.Column():
                        img3 = gr.Image(type="pil", label="Input OCT Scan")
                        btn3 = gr.Button("Segment Fluid & Lesions", variant="primary")
                    with gr.Column():
                        out_img3 = gr.Image(type="pil", label="Fluid & Lesion Mask Overlay")
                        txt3 = gr.Textbox(label="Fluid & Lesion Metrics", lines=5)
                btn3.click(predict_model3, inputs=img3, outputs=[out_img3, txt3], api_name="predict_model3")

            with gr.TabItem("Model 4: OIMHS Hole & Cyst U-Net"):
                gr.Markdown("### Macular Hole & Intraretinal Cyst (IRC) U-Net (OIMHS)")
                with gr.Row():
                    with gr.Column():
                        img4 = gr.Image(type="pil", label="Input OCT Scan")
                        btn4 = gr.Button("Detect Hole & Cysts", variant="primary")
                    with gr.Column():
                        out_img4 = gr.Image(type="pil", label="Hole & Cyst Segmentation Overlay")
                        txt4 = gr.Textbox(label="Pathology Metrics", lines=6)
                btn4.click(predict_model4, inputs=img4, outputs=[out_img4, txt4], api_name="predict_model4")

            with gr.TabItem("Model 5: OCT Pathology Detector"):
                gr.Markdown("### Faster R-CNN 9-Class Biomarker Object Detector")
                with gr.Row():
                    with gr.Column():
                        img5 = gr.Image(type="pil", label="Input OCT Scan")
                        thresh = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Confidence Threshold")
                        btn5 = gr.Button("Detect Biomarker Bounding Boxes", variant="primary")
                    with gr.Column():
                        out_img5 = gr.Image(type="pil", label="Biomarker Bounding Box Detections")
                        txt5 = gr.Textbox(label="Detection Results", lines=8)
                btn5.click(predict_model5, inputs=[img5, thresh], outputs=[out_img5, txt5], api_name="predict_model5")

if __name__ == "__main__":
    if HAS_GRADIO:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("Gradio is not installed locally. Run pip install gradio to launch the server web interface.")

