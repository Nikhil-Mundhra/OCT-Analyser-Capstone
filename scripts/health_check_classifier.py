import urllib.request
import json
import time
import os
from PIL import Image, ImageDraw

def check_space_health():
    url = 'https://huggingface.co/api/spaces/NMundhra/OCT-Image-Classifier-Model'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req)
        data = json.loads(res.read().decode())
        stage = data.get('runtime', {}).get('stage')
        print(f"Hugging Face Space Live Stage: {stage}")

        if stage == 'RUNNING':
            from gradio_client import Client, handle_file
            print("Connecting to live Hugging Face Classifier Space...")
            img_path = 'health_check_scan.png'
            img = Image.new('RGB', (256, 256), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 100, 256, 120], fill=(200, 200, 200))
            img.save(img_path)

            client = Client('NMundhra/OCT-Image-Classifier-Model')
            result = client.predict(handle_file(os.path.abspath(img_path)), True, api_name='/predict_multi_head')
            print("================ LIVE PREDICTION RESULTS ================")
            print("Diagnosis JSON:", json.dumps(result[0], indent=2))
            print("GradCAM Overlay Image Path:", result[1])
            print("STATUS: 100% WORKING & ONLINE 🟢")
            print("=========================================================")
        else:
            print(f"Space is currently in stage: '{stage}'. Please wait for build to finalize.")
    except Exception as e:
        print(f"Health check error: {e}")

if __name__ == "__main__":
    check_space_health()
