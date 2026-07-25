import os
import spaces
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import torch
torch.set_num_threads(1)
import tempfile

from pathlib import Path
import numpy as np
from PIL import Image

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    gr = None
    HAS_GRADIO = False

IS_HF_SPACE = os.getenv("SPACE_ID") is not None or os.getenv("SPACES_ZERO_GPU") is not None

# Add workspace root / parent paths to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.core_ml.classification.scripts.inference_pipeline import OCTInferencePipeline

print("=====================================================================")
print("Initializing ZeroGPU Multi-Head ConvNeXt V2 Classification Space")
print(f"IS_HF_SPACE: {IS_HF_SPACE}")
print("=====================================================================")

# Cold Start Penalty Mitigation (ZeroGPU Rule #2):
# Model weights are loaded globally onto CPU memory during Space startup phase.
print("1. Loading ConvNeXt V2 Multi-Head Model onto CPU memory...", flush=True)
pipeline = OCTInferencePipeline(device="cpu")
print("✅ ConvNeXt V2 Model loaded into CPU memory!", flush=True)


# Inference function wrapped in ZeroGPU decorator
def _run_classification(image_input, generate_gradcam=True):
    if image_input is None:
        return {"error": "Please upload a valid OCT scan image."}

    img_path = None
    if isinstance(image_input, str):
        img_path = image_input
    elif isinstance(image_input, dict) and "path" in image_input:
        img_path = image_input["path"]
    elif hasattr(image_input, "name") and isinstance(image_input.name, str):
        img_path = image_input.name
    elif hasattr(image_input, "path") and isinstance(image_input.path, str):
        img_path = image_input.path
    elif hasattr(image_input, "save"):
        temp_dir = Path(tempfile.gettempdir())
        img_path = str(temp_dir / "temp_input_scan.png")
        image_input.save(img_path)
    else:
        try:
            temp_dir = Path(tempfile.gettempdir())
            img_path = str(temp_dir / "temp_input_scan.png")
            Image.fromarray(np.array(image_input)).save(img_path)
        except Exception as err:
            return {"error": f"Invalid image format received: {type(image_input)} - {err}"}

    # Dynamic CUDA transfer inside GPU context
    if torch.cuda.is_available():
        pipeline.model.to("cuda")
        pipeline.device = torch.device("cuda")

    # Sub-60s Execution (Forward Pass + Grad-CAM)
    result = pipeline.predict(img_path, gradcam=generate_gradcam)
    return result

HAS_ZEROGPU = os.getenv("SPACES_ZERO_GPU") is not None or os.getenv("ZERO_GPU") is not None

if HAS_ZEROGPU and IS_HF_SPACE and spaces is not None:
    @spaces.GPU
    def predict_multi_head(image, gradcam=True):
        return _run_classification(image, generate_gradcam=gradcam)
else:
    def predict_multi_head(image, gradcam=True):
        return _run_classification(image, generate_gradcam=gradcam)

if HAS_GRADIO:
    with gr.Blocks(title="ConvNeXt V2 Multi-Head OCT Classifier (ZeroGPU)") as demo:
        gr.Markdown("# 👁️ ConvNeXt V2 Multi-Head OCT Pathology Classifier")
        gr.Markdown("ZeroGPU-accelerated hierarchical disease classification (15 pathology classes) with Grad-CAM explainability.")

        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(type="filepath", label="Input OCT Scan")
                chk_gradcam = gr.Checkbox(value=True, label="Generate Grad-CAM Heatmaps")
                btn_run = gr.Button("Classify Scan", variant="primary")
            with gr.Column():
                out_json = gr.JSON(label="Hierarchical Diagnosis Result")
                out_cam = gr.Image(type="pil", label="Grad-CAM Pathology Attention Overlay")

        def gradio_adapter(img, use_cam):
            res = predict_multi_head(img, gradcam=use_cam)
            if isinstance(res, dict) and "error" in res:
                return res, None

            cam_img = None
            if use_cam and isinstance(res, dict) and "gradcams" in res:
                cam_data = res["gradcams"].get("L2") or res["gradcams"].get("L1")
                if cam_data and isinstance(cam_data, str) and cam_data.startswith("data:image"):
                    import base64, io
                    base64_data = cam_data.split(",")[1]
                    cam_bytes = base64.b64decode(base64_data)
                    cam_img = Image.open(io.BytesIO(cam_bytes))

            return res, cam_img

        btn_run.click(gradio_adapter, inputs=[inp_img, chk_gradcam], outputs=[out_json, out_cam], api_name="predict_multi_head")

    demo.queue()

if __name__ == "__main__":
    if HAS_GRADIO:
        demo.launch(show_error=True)
