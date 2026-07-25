# OCT Hierarchical Classification Pipeline Documentation

This folder contains the comprehensive documentation for the design, architecture, and engineering decisions behind the PyTorch OCT Image Classification Pipeline.

## Table of Contents

### 1. [Architecture & Flow](architecture.md)
Detailed breakdown of the 3-Level Hierarchical structure (Gatekeeper → Router → Specialists). Explains the model selection decisions, resolution pipelines (224px vs 384px), and the specific logic for separating and grouping diseases (AMD splits, Vascular aggregation/re-separation).

> **Level 1 Backbone:** EfficientNet-B3 (upgraded from ResNet-50, 1st July, 2026)

### 2. [Data Pipeline & Imbalance Strategy](data_pipeline.md)
How the pipeline ingests 86,120 images from three disparate datasets (OCTDL, Kermany/UCSD, OCTID). Details the three-tiered defence against extreme long-tail class imbalance (Stratified K-Fold, WeightedRandomSampler, Focal Loss) and the CLAHE preprocessing step that normalises contrast across scanner manufacturers.

### 3. [Training Engine & Optimisations](training.md)
The `HierarchyTrainer` loop: Two-Phase protocol (head warm-up → backbone fine-tuning), checkpoint selection on `val_macro_f1`, cosine warm restarts, and Apple Silicon M2 Pro hardware optimisations. Also covers the post-training calibration pipeline: ROC-derived threshold, temperature scaling, TTA, and Grad-CAM.

### 4. [Level 1 Improvements (1st July, 2026)](improvements.md)
A detailed record of all improvements made to the Level 1 Gatekeeper model, the rationale for each change, and the expected metric impact. Use this as a reference when evaluating the new trained checkpoint.

### 5. [Project To-Do List](TODO.md)
Prioritised backlog of outstanding work across all three model levels — high, medium, and low priority items, plus a completed work log.

---

*Documentation last updated: 1st July, 2026. Reflects EfficientNet-B3 backbone and full calibration pipeline.*
