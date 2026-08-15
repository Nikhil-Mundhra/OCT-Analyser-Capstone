# Core ML Weights

**Date of Copy**: Sat Jul 11 18:33:20 IST 2026

The `.pth` weight files in `classification/weights` and `segmentation/weights` are **COPIES** of the original weights located in the respective training repositories (`image-classification-model-training` and `OCT-Segmentation-Model`). 

This directory (`core_ml`) exists to isolate the backend API from the training environments. If a model is retrained, the updated `.pth` files must be explicitly copied into this directory for the application to serve the new model.
