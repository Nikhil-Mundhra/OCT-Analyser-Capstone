# OCT Hierarchical Classification Pipeline Documentation

This folder contains the comprehensive documentation for the design, architecture, and engineering optimizations behind the PyTorch OCT Image Classification Pipeline.

## Table of Contents

### 1. [Architecture & Flow](architecture.md)
Detailed breakdown of the 3-Level Hierarchical structure (Gatekeeper → Router → Specialists). Explains the rationale behind model selection (ResNet-50 vs EfficientNet-B0), resolution pipelines (224px vs 384px), and the specific logic for separating and grouping diseases (like the AMD and Vascular splits).

### 2. [Data Pipeline & Imbalance Strategy](data_pipeline.md)
Explains how the pipeline ingests 86,120 images from three disparate datasets (OCTDL, Kermany/UCSD, OCTID). Details the three-tiered defense against extreme long-tail class imbalance using Stratified K-Fold sampling, WeightedRandomSampler, and Focal Loss.

### 3. [Training Engine & Optimisations](training.md)
Breaks down the `HierarchyTrainer` loop, including the Two-Phase training protocol (head warmup vs backbone fine-tuning). Details the hardware-specific optimizations applied for the Apple Silicon M2 Pro (24GB Unified Memory), including the use of MPS `float16` AMP, DataLoader worker tuning, and why `torch.compile` is explicitly disabled.

---

*This documentation reflects the pipeline configuration and decisions as of June 2026.*
