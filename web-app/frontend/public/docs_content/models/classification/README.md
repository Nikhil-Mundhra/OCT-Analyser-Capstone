# OCT Multi-Head Classification Pipeline

This folder contains the authoritative documentation for the design, architecture, and engineering optimizations behind the PyTorch OCT Image Classification Pipeline. 

> [!NOTE]
> This documentation reflects the modernized unified architecture (post-July 2026), superseding the legacy 3-Level Gatekeeper/Router framework.

## Documentation Suite

### 1. [Architecture & Model Design](/docs/classification-architecture)
Detailed breakdown of the unified **Multi-Head ConvNeXt V2** architecture. Covers the dual-stream feature processing, the CBAM Attention Module, and the Hierarchical Conditioning protocol that safely passes binary probabilities into the multi-class pathology router.

### 2. [Training Pipeline & Optimizations](/docs/classification-training_pipeline)
Documentation of the robust training engine (`MultiHeadTrainer`). Covers the Two-Phase (Warm-up & Fine-Tuning) protocol, Adam Momentum Porting, Automatic Mixed Precision (GradScaler) setups, and the exact asymmetric loss configuration (BCE vs. Focal Loss) used to conquer extreme class imbalance.

### 3. [Data & Augmentation Strategy](/docs/classification-data_augmentation)
Details the medical-specific preprocessing utilizing the MONAI library. Covers Contrast Limited Adaptive Histogram Equalization (CLAHE), safe spatial transforms, and the explicit exclusion of destructive spatial crops (`RandomResizedCrop`) to protect peripheral biological markers.
