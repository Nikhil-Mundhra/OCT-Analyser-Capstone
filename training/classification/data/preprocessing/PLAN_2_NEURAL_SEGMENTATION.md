# Plan 2: Neural Retinal & Choroidal Segmentation Pipeline (U-Net / Foundation Model)

## Overview
Plan 2 transitions the OCT-Analyser-Capstone preprocessing pipeline from classical heuristic/graph methods to an end-to-end, zero-parameter deep learning segmentation architecture. It leverages ground-truth healthy tissue masks from `OCT5K` combined with high-confidence consensus pseudo-labels from the `Classified` dataset.

---

## 1. Objectives
1. **Zero-Parameter Inference**: Eliminate all handcrafted thresholding, gradient weights, and spatial parameters.
2. **Pathology Invariance**: Segment retinal tissue, RPE, and choroid accurately across all 15 disease categories (severe DME domes, PED, choroidal neovascularization, atrophy, and vitreous haze).
3. **High Throughput**: Achieve $< 5\text{ms}$ inference latency per scan on Apple Silicon GPU (`mps`) / NVIDIA CUDA.
4. **Unified Multi-Task Alignment**: Provide pre-trained encoder weights for the downstream hierarchical classification network.

---

## 2. Architecture Specification

### Backbone Options
- **Primary**: Lightweight U-Net with MobileNetV3-Large or ConvNeXt-Femto encoder.
- **Alternative**: SegFormer-B0 or ResNet-18 U-Net with multi-scale feature aggregation.

### Input & Output Channels
- **Input**: $(B, 1, 384, 384)$ normalized grayscale OCT slice.
- **Output**: $(B, 3, 384, 384)$ multi-class probability logits:
  - Channel 0: Neurosensory Retina (ILM to RPE)
  - Channel 1: Choroidal Stroma (RPE to CSI / Scleral Boundary)
  - Channel 2: Background / Vitreous Cavity & Retrobulbar Space

---

## 3. Training & Pseudo-Labeling Active Learning Workflow

```mermaid
flowchart TD
    A["OCT5K Dataset<br/>(Ground-Truth Masks)"] --> B["Stage 1: Supervised Pre-Training<br/>(Lightweight U-Net)"]
    B --> C["Stage 2: High-Confidence Inference<br/>on Classified Dataset (80k scans)"]
    C --> D{"6-Point Anatomical<br/>Invariant Quality Gate"}
    D -->|Pass (>85%)| E["High-Confidence Pseudo-Label Pool"]
    D -->|Fail (<15%)| F["Coupled Multi-Surface Fallback (Plan 1)"]
    F --> E
    E --> G["Stage 3: Semi-Supervised Fine-Tuning<br/>(Hard DME / Pathological Cases)"]
    G --> H["Final Frozen Segmenter<br/>(deploy to preprocessing pipeline)"]
```

### Step-by-Step Execution Plan

1. **Dataset Preparation**:
   - Ingest `OCT5K` dataset layer masks (`ILM`, `IS/OS`, `RPE`, `Choroid`).
   - Format into canonical $384 \times 384$ tensors with letterbox padding.

2. **Stage 1 (Supervised Baseline Training)**:
   - Train MobileNetV3-UNet on OCT5K using combined Dice + Focal Loss:
     $$\mathcal{L} = \mathcal{L}_{\text{Dice}} + \lambda \mathcal{L}_{\text{Focal}}$$
   - Utilize segmentation-preserving medical augmentations (intensity jitter, Gaussian speckle noise, horizontal flips — *no RandomResizedCrop*).

3. **Stage 2 (Dataset-Wide Consensus Pseudo-Labeling)**:
   - Run inference across all 20 subfolders of `Classified` dataset.
   - Filter outputs through the 6-point anatomical invariant gate:
     - Strict layer ordering ($\text{ILM} < \text{RPE} < \text{Choroid}$).
     - Physiological retinal thickness ($35\text{px} \le \bar{T}_{\text{retina}} \le 350\text{px}$).
     - Continuity and minimum tissue presence.
   - Route difficult cases through Plan 1 (Coupled Graph-Cut + B-Spline) for refinement.

4. **Stage 3 (Semi-Supervised Active Learning)**:
   - Retrain U-Net on combined $(D_{\text{OCT5K}} \cup D_{\text{Classified-Pseudo}})$.
   - Freeze model weights to `models/pretrained_oct_segmenter.pt`.

5. **Deployment**:
   - Replace heuristic boundary generation in `boundaries.py` with `NeuralSegmenter.predict_mask(img_tensor)`.
   - Keep CLI and tuning server active for visual QA and inspection.
