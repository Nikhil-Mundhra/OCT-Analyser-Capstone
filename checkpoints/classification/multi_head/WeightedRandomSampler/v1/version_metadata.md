# Model Weight Version Metadata — `v1`

## Run Identification
- **Version Tag**: `v1`
- **Creation Date & Time**: `2026-08-08 19:46:00 (Local Time)`
- **Git Branch**: `dev`
- **Git Commit Hash**: `ba9a066`
- **Commit Description**: `feat(eval): Add author attribution and full telemetry PDF dashboard`

## Hardware & System Specs
- **Compute Accelerator**: `Apple Silicon MPS (Metal Performance Shaders)`
- **PyTorch Version**: `2.3.1`
- **Host OS**: `macOS 15.x (darwin-arm64)`
- **Max MPS Memory Ratio**: `High Watermark = 0.7 | Low Watermark = 0.5`
- **Batch Processing Throughput**: `3.51s per batch of 64 images (~80.5 mins per epoch)`

## Execution Command Line
```bash
OCT_DATA_ROOT="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed" \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \
KMP_DUPLICATE_LIB_OK=TRUE \
python3 image-classification-model-training/scripts/train_convnext.py \
  --config "image-classification-model-training/config/hierarchy.yaml" \
  --epochs-warmup 5 \
  --epochs-finetune 10 \
  --batch-size 64 \
  --num-workers 4 \
  --use-weighted-sampler
```

## Architectural & Training Innovations
- **Pre-Normalization Masked GAP**: Downsampled `valid_mask` (`mode='area'`) across multi-scale feature maps ($S2, S3, S4$).
- **Uniform Initial FocalLoss Alpha ($\alpha = 1.0$)**: Decoupled DataLoader sampling from loss scaling when `WeightedRandomSampler` is active.
- **Dynamic Adaptive Class-Weight Controller**: Real-time validation F1 feedback loop dynamically scaling $\alpha_c$ ($1.0\times \to 5.0\times$) and decaying upon recovery.
- **Un-Frozen Stage 2 & 3 Bottleneck Fine-Tuning**: Provided gradient updates ($\frac{\partial L}{\partial f_{\text{invalid}}} = 0.0$) across bottleneck layers.
- **Joint Spatial Translation Jitter**: Applied $\pm 10\text{ px}$ `RandAffine` translation during fine-tuning.

## Verified Validation Metrics
- **Peak Checkpoint File**: `fold0_best_val_loss.pth`
- **Validation Loss**: `0.1810`
- **H1 (Gatekeeper) Accuracy**: `96.60%`
- **H1 Macro-F1**: `0.9609`
- **H2 (Pathology Multi-Class) Accuracy**: `93.30%`
- **H2 Macro-F1**: `0.9489`
- **Active Pathology Classes**: `12 / 12 (100% active, 0 dead)`
- **Lowest Per-Class F1**: `0.8300 (DRUSEN)`
- **Rare Class Performance**: `RAO = 1.0000 F1 | General_AMD = 1.0000 F1`
