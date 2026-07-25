# 👁️ Multi-Head ConvNeXt V2 Hierarchical OCT Classification Pipeline

A state-of-the-art, multi-task deep learning pipeline for hierarchical retinal pathology classification from Optical Coherence Tomography (OCT) B-scans. Built on top of **ConvNeXt V2 Base** with **Multi-Scale Encoder Aggregation**, **CBAM Attention Modules**, **Strict Hierarchical Conditioning**, and **Focal Loss** for extreme class imbalance mitigation.

---

## 📐 Architecture & Key Features

```mermaid
graph TD
    Input["OCT Scan B-Scan (384×384)"] --> Backbone["ConvNeXt V2 Base Backbone"]
    
    Backbone --> |Stage 1: 28x28| CBAM1["CBAM Attention (256 ch)"]
    Backbone --> |Stage 2: 14x14| CBAM2["CBAM Attention (512 ch)"]
    Backbone --> |Stage 3: 7x7| CBAM3["CBAM Attention (1024 ch)"]
    
    CBAM3 --> GAP_S3["Global Avg Pool"] --> H1["Head 1: Gatekeeper (Normal vs Abnormal)"]
    
    CBAM1 --> GAP1["GAP (256)"]
    CBAM2 --> GAP2["GAP (512)"]
    CBAM3 --> GAP3["GAP (1024)"]
    
    GAP1 & GAP2 & GAP3 --> Concat["Multi-Scale Concatenation (1792 ch)"]
    H1 --> |Sigmoid Prob Detached (+1)| Condition["Hierarchical Conditioning"]
    
    Concat & Condition --> H2_Input["Combined Feature Vector (1793 ch)"]
    H2_Input --> H2["Head 2: 12-Class Pathology Classifier"]
    
    H1 & H2 --> SoftmaxCond["Joint Probability Constraint: P(H2) = P(H2|H1) × P(H1)"]
```

### 🌟 Core Capabilities
- **Multi-Scale Encoder Aggregation (Decoder-less Localization)**: Pools spatial feature maps from multiple encoder depths (Stage 1 @ $28 \times 28$, Stage 2 @ $14 \times 14$, Stage 3 @ $7 \times 7$) directly into the classification head, retaining fine-grained spatial details (such as peripheral cysts or drusen deposits) lost at the bottleneck.
- **CBAM Attention Modules**: Channel & Spatial Convolutional Block Attention Modules (`CBAMBlock`) applied independently at each feature scale before pooling.
- **Dual-Head Output**:
  - **Head 1 (H1 Gatekeeper)**: Binary classification (`NORMAL` vs `ABNORMAL`).
  - **Head 2 (H2 Pathology Classifier)**: 12-class granular disease classification (`CNV`, `DRUSEN`, `AMD`, `General_AMD`, `DME`, `DR`, `MH`, `RVO`, `RAO`, `CSR`, `ERM`, `VID`).
- **Strict Hierarchical Conditioning**:
  - **Feature Conditioning**: Appends H1 sigmoid probability (`torch.sigmoid(out_normal).detach()`) directly to the H2 input vector ($1792 + 1 = 1793$ input channels).
  - **Probability Constraint**: Multiplies final H2 softmax probabilities by H1 probability ($P(H2_{\text{final}}) = P(H2 \mid H1) \times P(H1)$) to eliminate multi-head contradictions (e.g. predicting a disease when H1 probability is 0).

---

## ⚖️ Imbalance-Resilient Loss Strategy

- **Focal Loss ($\gamma=2.0$)**: $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ focuses gradient updates on hard minority classes (`RAO`, `CSR`, `VID`) and down-weights dominant classes (`CNV`, `DRUSEN`).
- **Automated Class Imbalance Weighting ($\alpha_t$)**: Calculates exact per-class inverse frequency weights dynamically from the dataset.
- **Asymmetric Hierarchical Loss Masking**: H2 pathology loss is calculated **only** on samples labeled as Abnormal ($H1 = 1$), preventing Normal scans from producing false disease gradient updates.
- **Label Smoothing ($\varepsilon=0.1$)**: Applied to multi-class targets to prevent overconfidence and absorb multi-source annotation noise across datasets.

---

## 📈 Progressive Training & Optimization

- **Two-Phase Training Protocol**:
  - **Phase 1 (Warmup)**: Backbone frozen (`freeze_backbone()`), training only classification heads and CBAM blocks with higher LR (`1e-3`) to stabilize head weights.
  - **Phase 2 (Finetune)**: Backbone unfrozen (`unfreeze_backbone()`), performing full end-to-end fine-tuning.
- **Differential Learning Rates**: Backbone runs at `1e-5` to preserve pretrained representations, while attention & classification heads run at `1e-4`.
- **Adam Momentum Preservation**: Automatically ports optimizer momentum state from Phase 1 to Phase 2 to prevent Adam momentum shock when unfreezing.
- **Cosine Annealing Warm Restarts**: $T_0=10, T_{\text{mult}}=2, \eta_{\text{min}}=1\text{e-}6$ to cycle learning rates and break out of local minima.
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(max_norm=5.0)` to prevent exploding gradients.

---

## ⚡ Hardware Acceleration & Checkpointing

- **Native FP16 Autocast (MPS & CUDA)**: `torch.autocast(device_type, dtype=torch.float16)` accelerates matrix multiplication on Apple Silicon M2 Pro matrix engines and NVIDIA CUDA Cores, keeping loss calculations safely in FP32.
- **Gradient Accumulation (`--accum-steps`)**: Simulates larger effective batch sizes (e.g. `batch_size=16` $\times$ `accum_steps=2` = effective batch 32) without increasing VRAM.
- **Atomic File Saving**: Writes checkpoints to `.pth.tmp` first and renames via atomic `os.replace()`, guaranteeing no corrupted `.pth` files on process interruptions (`Ctrl+C`, SIGINT).
- **Mid-Epoch Checkpointing (`--save-steps 2250`)**: Periodically saves rolling checkpoint (`fold0_last_model.pth`) mid-epoch.
- **Permanent Epoch Snapshots**: Saves `fold0_epoch_001.pth`, `fold0_epoch_002.pth`, etc. at every completed epoch alongside `fold0_best_model.pth`.
- **Real-Time Cloud Backup (`--hf-repo`)**: Automatically pushes best & last checkpoints to HuggingFace Hub in real-time.

---

## 📂 Directory Layout

```text
image-classification-model-training/
├── config/
│   └── hierarchy.yaml               # Single source of truth for labels & paths
├── data/
│   ├── dataset.py                   # MultiHeadOCTDataset & K-Fold builders
│   ├── transforms.py                # MONAI data augmentation pipelines
│   ├── dataset_manifest.csv         # Full dataset manifest
│   └── micro_dataset/               # Sanity testing micro-dataset
├── models/
│   └── multi_head_convnext.py       # MultiHeadConvNeXt architecture
├── training/
│   ├── multi_head_trainer.py        # MultiHeadTrainer with MPS FP16 & atomic saves
│   ├── losses.py                    # FocalLoss & LabelSmoothingCrossEntropy
│   └── trainer.py                   # Single-head legacy trainer base
├── scripts/
│   ├── train_convnext.py            # Main training execution CLI
│   ├── evaluate_best_model.py       # Checkpoint evaluation script
│   └── generate_manifest.py         # Manifest builder
├── tests/                           # PyTorch unit test suite
└── requirements.txt                 # Dependencies
```

---

## 🚀 Execution & Usage Commands

### 1. Local Training (Apple Silicon Mac - MPS)
```bash
OCT_DATA_ROOT="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified" \
python3 image-classification-model-training/scripts/train_convnext.py \
    --config "image-classification-model-training/config/hierarchy.yaml" \
    --batch-size 16 \
    --accum-steps 2 \
    --num-workers 0 \
    --save-steps 2250 \
    --epochs-warmup 3 \
    --epochs-finetune 20
```

### 2. Kaggle Dual-GPU Training (NVIDIA Dual T4)
Set environment variables in Python:
```python
import os
os.environ["OCT_DATA_ROOT"] = "/kaggle/input/classified-oct-v2/Classified"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = "your_hf_token_here"
```
Run training CLI:
```bash
!python3 image-classification-model-training/scripts/train_convnext.py \
    --config "image-classification-model-training/config/hierarchy.yaml" \
    --batch-size 32 \
    --accum-steps 1 \
    --num-workers 2 \
    --save-steps 2250 \
    --epochs-warmup 3 \
    --epochs-finetune 20 \
    --hf-repo "NMundhra/OCT-Classification-Model"
```

### 3. Resuming Training
To resume cleanly from any saved checkpoint:
```bash
OCT_DATA_ROOT="/path/to/Classified" \
python3 image-classification-model-training/scripts/train_convnext.py \
    --config "image-classification-model-training/config/hierarchy.yaml" \
    --batch-size 16 \
    --resume "image-classification-model-training/checkpoints/multi_head/fold0_last_model.pth"
```

### 4. Running Unit Tests
```bash
KMP_DUPLICATE_LIB_OK=TRUE \
PYTHONPATH=$(pwd)/image-classification-model-training \
python3 -m unittest discover -s image-classification-model-training/tests
```
