# Legacy Architecture Archive

This directory contains the legacy 3-stage cascade architecture (ResNet-50 L1 -> EfficientNet-B2 L2 -> EfficientNet-B0 L3).

**Why was this archived?**
The project was migrated to a single Multi-Head ConvNeXtV2 model to simplify the training and inference pipelines and reduce the total parameter count.

**What is here?**
*   `models/`: The PyTorch model definitions for Gatekeeper, Router, and Specialist.
*   `scripts/`: The 5-fold cross-validation training scripts, calibration scripts, and inference pipeline.
*   `hf_space/`: The legacy inference pipeline for the Hugging Face space.

**Where are the weights?**
To keep the git repository lean, the ~1.2 GB of `.pth` checkpoint files for the legacy models have been deleted locally. The canonical trained weights are preserved on the Hugging Face Hub `dev` branch if they need to be retrieved.
