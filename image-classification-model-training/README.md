# Multi-Head ConvNeXt V2 Hierarchical OCT Classification Pipeline

A state-of-the-art, multi-task deep learning pipeline for hierarchical retinal pathology classification from Optical Coherence Tomography (OCT) B-scans. Built on top of **ConvNeXt V2 Base** with **Multi-Scale Encoder Aggregation**, **CBAM Attention Modules**, **Strict Hierarchical Conditioning**, **Source-Namespaced Patient Stratified Group K-Fold**, and **Unique-Patient Bounded Focal Loss**.

---

## Architecture & Key Features

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

### Core Capabilities
- **Multi-Scale Encoder Aggregation (Decoder-less Localization)**: Pools spatial feature maps from multiple encoder depths (Stage 1 @ $28 \times 28$, Stage 2 @ $14 \times 14$, Stage 3 @ $7 \times 7$) directly into the classification head, retaining fine-grained spatial details (such as peripheral cysts or drusen deposits) lost at the bottleneck.
- **CBAM Attention Modules**: Channel & Spatial Convolutional Block Attention Modules (`CBAMBlock`) applied independently at each feature scale before pooling.
- **Dual-Head Output**:
  - **Head 1 (H1 Gatekeeper)**: Binary classification (`NORMAL` vs `ABNORMAL`).
  - **Head 2 (H2 Pathology Classifier)**: 12-class granular disease classification (`CNV`, `DRUSEN`, `AMD`, `General_AMD`, `DME`, `DR`, `MH`, `RVO`, `RAO`, `CSR`, `ERM`, `VID`).
- **Strict Hierarchical Conditioning**:
  - **Feature Conditioning**: Appends H1 sigmoid probability (`torch.sigmoid(out_normal).detach()`) directly to the H2 input vector ($1792 + 1 = 1793$ input channels).
  - **Probability Constraint**: Multiplies final H2 softmax probabilities by H1 probability ($P(H2_{\text{final}}) = P(H2 \mid H1) \times P(H1)$) to eliminate multi-head contradictions.

---

## Patient Grouping & Data Integrity

- **Source & Pathology Class Namespacing**: Global patient IDs are constructed as `dataset_key::pathology_class::local_patient_id` using explicit `dataset_key` attributes in `hierarchy.yaml`. This prevents false cross-dataset or cross-class patient ID collisions during cross-validation.
- **Zero Patient Leakage (`StratifiedGroupKFold`)**: Cross-validation uses `StratifiedGroupKFold` on namespaced patient IDs across 7,536 distinct patient groups, guaranteeing that all slices from a patient volume remain strictly in a single fold.
- **Mutual Exclusivity Validation**: Pre-namespaced field audits verify that patient groups map to single pathology classes without cross-dataset ambiguity.

---

## Imbalance-Resilient Loss Strategy

- **Fold-Specific Unique-Patient Focal Loss ($\gamma=2.0$)**: $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ calculated dynamically per fold from unique training patient counts in that specific fold's training set:
  $$\alpha_t = \text{clamp}\left(\frac{1/\sqrt{\text{N\_patients}_t}}{\text{mean}(1/\sqrt{\text{N\_patients}})}, \, 0.4, \, 4.0\right)$$
- **Patient-Level Prevalence Calibration**: Prevents slice-rich multi-volume classes (`CNV` with 47 slices/patient) from inflating weights relative to single-scan classes with similar human patient numbers (`DRUSEN`, `DME`).
- **Asymmetric Loss Masking**: H2 pathology loss is calculated only on Abnormal samples ($H1 = 1$), preventing Normal scans from producing false disease gradient updates.
- **Label Smoothing ($\varepsilon=0.1$)**: Applied to multi-class targets to prevent overconfidence and absorb multi-source annotation noise across datasets.

---

## Progressive Training & Optimization

- **Two-Phase Training Protocol**:
  - **Phase 1 (Warmup)**: Backbone frozen (`freeze_backbone()`), training only classification heads and CBAM blocks with higher LR (`1e-3`) to stabilize head weights.
  - **Phase 2 (Finetune)**: Backbone unfrozen (`unfreeze_backbone()`), performing full end-to-end fine-tuning.
- **Differential Learning Rates**: Backbone runs at `1e-5` to preserve pretrained representations, while attention & classification heads run at `1e-4`.
- **Adam Momentum Preservation**: Automatically ports optimizer momentum state from Phase 1 to Phase 2 to prevent Adam momentum shock when unfreezing.
- **Macro-F1 Early Stopping**: Monitors `h2_macro_f1` with `patience=3`, `mode="max"`, `min_delta=1e-4` after warmup.
- **Dual Best Checkpointing**: Saves `fold0_best_macro_f1.pth` (main model) and `fold0_best_val_loss.pth` independently.
- **DDP Bucket View Optimization**: `DistributedDataParallel(gradient_as_bucket_view=True)` eliminates conv stride warnings and memory copy overhead.

---

## Directory Layout

```text
image-classification-model-training/
├── config/
│   └── hierarchy.yaml               # Single source of truth for labels, paths, & dataset_keys
├── data/
│   ├── dataset.py                   # MultiHeadOCTDataset & StratifiedGroupKFold loaders
│   ├── transforms.py                # MONAI data augmentation pipelines
│   └── dataset_manifest.csv         # Full dataset manifest
├── models/
│   └── multi_head_convnext.py       # MultiHeadConvNeXt architecture & param groups
├── training/
│   ├── multi_head_trainer.py        # MultiHeadTrainer with DDP, AMP, early stopping, & HF upload
│   └── losses.py                    # FocalLoss & LabelSmoothingCrossEntropy
├── scripts/
│   ├── train_convnext.py            # Main training execution CLI
│   ├── audit_h2_targets.py          # Target semantics & patient grouping audit tool
│   ├── test_drusen_memorization.py  # Micro-dataset memorization test script
│   └── analyze_checkpoint_confusion.py # Checkpoint confusion matrix generator
├── tests/                           # PyTorch unit test suite (27 tests)
└── requirements.txt                 # Dependencies
```

---

## Execution & Usage Commands

### 1. Target Semantics & Patient Grouping Audit
```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 image-classification-model-training/scripts/audit_h2_targets.py \
    --config image-classification-model-training/config/hierarchy.yaml
```

### 2. Micro-Dataset Memorization Test
```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 image-classification-model-training/scripts/test_drusen_memorization.py \
    --config image-classification-model-training/config/hierarchy.yaml \
    --epochs 25 \
    --lr 1e-3
```

### 3. Local Training (Apple Silicon Mac - MPS)
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

### 4. Kaggle Dual-GPU DDP Training (NVIDIA Dual T4)
Set environment variables in Python:
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OCT_DATA_ROOT"] = "/kaggle/input/datasets/nikhilmundhra/classified-oct-v2-preprocessed/Classified-preprocessed"
os.environ["HF_TOKEN"] = "your_hf_token_here"
os.environ["HF_REPO_ID"] = "NMundhra/OCT-Classification-Model"
```

Run DDP Multi-GPU training CLI (~1.9x speedup, Global Batch Size 32):
```bash
!torchrun --nproc_per_node=2 image-classification-model-training/scripts/train_convnext.py \
    --config "image-classification-model-training/config/hierarchy.yaml" \
    --checkpoint-dir "/kaggle/working" \
    --batch-size 16 \
    --accum-steps 1 \
    --num-workers 2 \
    --save-steps 2250 \
    --epochs-warmup 3 \
    --epochs-finetune 20 \
    --use-ddp \
    --hf-repo "NMundhra/OCT-Classification-Model"
```

### 5. Running Unit Test Suite
```bash
KMP_DUPLICATE_LIB_OK=TRUE \
python3 -m unittest discover -s image-classification-model-training/tests -p "test_*.py"
```
