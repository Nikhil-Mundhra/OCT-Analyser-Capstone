---
title: OCT Analyser 5 Model Suite API Services
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
short_description: Deep Learning API Suite for OCT Retinal Segmentation and Detection
---

# OCT Analyser 5-Model Microservice Suite

This Hugging Face Space hosts the unified 5-model deep learning inference API suite for Optical Coherence Tomography (OCT) image processing.

## Packaged Models
1. **Model 1 (RetinalLayersUNet)**: 6-Class Retinal Layer Segmentation U-Net (OCT5K Benchmark)
2. **Model 2 (ChoroidalyzerUNet)**: Choroid Region & Thickness Quantification U-Net
3. **Model 3 (HRF_AttentionUNet)**: High-Resolution Fluid & Lesion Attention U-Net (HRF DME/AMD)
4. **Model 4 (OIMHSUNet)**: Macular Hole & Intraretinal Cyst U-Net (OIMHS)
5. **Model 5 (OCTPathologyDetector)**: Faster R-CNN 9-Class Biomarker Object Detector

## API Usage via Gradio Client Python SDK
```python
from gradio_client import Client

client = Client("your-username/oct-analyser-suite")

# Model 1 Retinal Layers
result1 = client.predict(image="path/to/oct.png", api_name="/predict_model1")

# Model 5 Biomarker Detector
result5 = client.predict(image="path/to/oct.png", score_threshold=0.5, api_name="/predict_model5")
```
