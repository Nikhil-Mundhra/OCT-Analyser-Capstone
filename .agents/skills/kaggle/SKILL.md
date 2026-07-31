---
name: kaggle
description: MANDATORY skill for deploying, writing, running, or debugging model training scripts on Kaggle. Enforces critical rules for preventing DataLoader deadlocks, tail-batch crashes (batch-size 1 squeeze traps), shared memory OOMs, multi-GPU setup, medical spatial augmentation rules, and mandatory local pre-flight smoke testing.
---

# Kaggle Training Guidelines

When deploying PyTorch training scripts to Kaggle (especially those using Multi-processing DataLoaders, OpenCV, and MONAI), you MUST enforce the following rules to prevent silent deadlocks, shared memory crashes, and invalid medical data augmentations:

## 1. Prevent Multiprocessing Deadlocks
Kaggle Notebooks often hang indefinitely at the start of a PyTorch DataLoader epoch due to thread contention between Python's multiprocessing workers and underlying C++ threading libraries (like OpenMP or OpenCV).
Always ensure the following are executed at the **very top** of the main training script, *before* importing `torch`:
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
cv2.setNumThreads(0)
```

## 2. Shared Memory Limits (`/dev/shm`) & DataLoader Configuration
Kaggle kernels and Docker containers have strict limits on shared memory. PyTorch uses shared memory to transfer tensors from DataLoader workers to the main process.
- **Maximum Workers**: Never set `num_workers` higher than `2` for high-resolution images (e.g., 384x384 or 512x512) with large batch sizes (e.g., 64).
- **Fallback (`num_workers=0`)**: If training still hangs, fallback to `num_workers=0` to force synchronous data loading and completely bypass `/dev/shm`.
- **Drop Last Batch (`drop_last=True`)**: Always set `drop_last=True` for training DataLoaders to prevent single-sample micro-batches at epoch ends that break BatchNorm statistics or target tensor shapes.

## 3. Medical Data Augmentation Constraints
Medical OCT scans contain critical disease markers along peripheral edges (e.g., peripheral cysts, drusen).
- **STRICT PROHIBITION**: NEVER use `RandomResizedCrop` or destructive spatial augmentations that slice off image edges (acts as spatial dropout and erases edge-located markers).
- **Segmentation-Driven Cropping**: Use U-Net or morphological thresholding (`TissueMaskCrop`) to dynamically preserve 100% of retinal tissue while zeroing out background scanner text or compass artifacts.

## 4. Multi-GPU Saturation & Distributed Data Parallel (DDP)
When running on Kaggle kernels equipped with dual GPUs (e.g., Dual T4s):
- **DistributedDataParallel (DDP)**: Prefer launching multi-GPU training via `torchrun --nproc_per_node=2` over legacy `nn.DataParallel`. DDP runs isolated per-GPU processes, eliminating Global Interpreter Lock (GIL) contention and yielding ~1.9x speedup.
- **Global Effective Batch Size**: $(\text{Batch Size per GPU}) \times (\text{Number of GPUs}) \times (\text{Accumulation Steps})$. Setting `--batch-size 16` per GPU across 2 GPUs achieves a target Global Batch Size of 32 while reducing per-GPU VRAM overhead.
- **Memory View Optimization**: Pass `gradient_as_bucket_view=True` in `DistributedDataParallel` constructor to assign gradients directly as reduction bucket views, eliminating 1x1 conv stride copy warnings and saving VRAM.

## 5. Model Training Execution Pipelines on Kaggle

### Pipeline A: Multi-Head ConvNeXt Model (`train_convnext.py`)

**Cell 1: Clone Repository and Install Dependencies**
```bash
!rm -rf OCT-Analyser-Capstone && git clone -b dev --depth 1 https://github.com/Nikhil-Mundhra/OCT-Analyser-Capstone.git
%cd OCT-Analyser-Capstone
!pip install -q uv
!UV_NO_PROGRESS=1 uv pip install --system -q -r image-classification-model-training/requirements.txt
```

**Cell 2: Execute DDP Training (Dual T4 GPUs, Global Batch Size 32)**
```bash
!PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
OCT_DATA_ROOT="/kaggle/input/datasets/nikhilmundhra/classified-oct-v2-preprocessed/Classified-preprocessed" \
HF_TOKEN="your_hf_token_here" \
torchrun --nproc_per_node=2 image-classification-model-training/scripts/train_convnext.py \
    --config "image-classification-model-training/config/hierarchy.yaml" \
    --batch-size 16 \
    --accum-steps 1 \
    --num-workers 2 \
    --save-steps 2250 \
    --epochs-warmup 3 \
    --epochs-finetune 20 \
    --hf-repo "NMundhra/OCT-Classification-Model"
```

---

### Pipeline B: Hierarchical U-Net Classifier Fine-Tuning (`train_cls_frozen.py`)

**Cell 1: Install Dependencies (Quietly)**
```bash
!uv pip install --system -r requirements.txt -q
```

**Cell 2: Execute Training**
```bash
!python3 train_cls_frozen.py \
    --cls-data "/kaggle/input/datasets/nikhilmundhra/classified-oct-v2/Classified" \
    --cls-config "image-classification-model-training/config/hierarchy.yaml" \
    --checkpoint "models_suite/model1_oct5k_layers/checkpoints/best_model.pth" \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4
```

## 6. Mandatory Local Pre-Flight Checks (Smoke Testing)
Before deploying **any** changes to a Kaggle training pipeline, it is **MANDATORY** to run a full local smoke test. Kaggle "Save & Run All" commits take hours to queue and run, so catching typos, shape mismatches, or missing imports locally is strictly required.

Always execute the pipeline locally against the `micro_dataset` to verify model construction, tensor routing, metric calculation, and `.pth` checkpoint generation.

**Command for Multi-Head ConvNeXt Smoke Test:**
```bash
OCT_DATA_ROOT="image-classification-model-training/data/micro_dataset" \
python3 image-classification-model-training/scripts/train_convnext.py \
    --config "image-classification-model-training/config/micro_hierarchy.yaml" \
    --batch-size 4 \
    --num-workers 0 \
    --epochs-warmup 1 \
    --epochs-finetune 1 \
    --smoke-test
```

## 7. The Batch-Size 1 Tail Batch `.squeeze()` Trap
When indexing multi-head targets or masks (e.g. `valid_h2_mask`), **NEVER** use unparameterized `.squeeze()` on label tensors.

- **The Bug**: On large datasets (like 88,697 Kaggle samples), the final tail batch of a validation split can have **batch size 1** (shape `[1, 1]`). Calling `.squeeze()` without a dimension parameter on a `[1, 1]` tensor strips **both** dimensions, turning it into a **0D scalar tensor (`[]`)**.
- **The Symptom**: Indexing a 2D logit matrix `[1, 12]` with a 0D boolean scalar tensor in PyTorch silently projects it into a **3D tensor `[1, 1, 12]`**. Downstream loss calculations reading `inputs.size(1)` evaluate the dimension as `1` instead of `12`, causing a `ZeroDivisionError: float division by zero` in label smoothing or loss functions (`eps / (num_classes - 1)`).
- **The Rule**: Always use `.view(-1)` instead of `.squeeze()` to flatten target mask tensors:
  ```python
  # BAD (Collapses [1, 1] into 0D scalar [], mutating tensor rank to 3D on indexing):
  valid_h2_mask = (labels['normal_abnormal'] == 1).squeeze()

  # GOOD (Guarantees 1D tensor [B] regardless of batch size):
  valid_h2_mask = (labels['normal_abnormal'] == 1).view(-1)
  ```

## 8. Checkpoint Saving, Safe Resuming, and Path Validation Across Session Timeouts
Kaggle sessions have a strict 9-hour maximum execution limit.
- **Automatic Checkpoint Persistence**: Ensure trainers save `fold0_best_model.pth` and `fold0_last_model.pth` to `/kaggle/working/checkpoints/` after every epoch.
- **Safe Checkpoint Path Validation**: Always verify `if resume_path and os.path.exists(resume_path):` before calling `torch.load()`. If the file path is invalid or missing, log a warning and fall back to fresh model initialization rather than crashing mid-run with an unhandled `FileNotFoundError`.
- **Seamless Resuming**: When `--resume` points to an existing `.pth` file, restore model parameters, optimizer momentum, scheduler states, and epoch indices cleanly.
- **Fail-Safe Real-Time Cloud Backup (HF Hub)**: Pass `--hf-repo username/repo_name` and set `HF_TOKEN` in the environment to stream peak model checkpoints directly to Hugging Face Hub in real-time.

## 9. Automatic Mixed Precision (AMP) & VRAM Optimization
Heavy architectures combining ConvNeXt V2 Base with multi-scale CBAM attention blocks require careful memory management:
- **`PYTORCH_CUDA_ALLOC_CONF`**: Always set `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` at script startup to prevent VRAM fragmentation crashes.
- **Autocast Dtype**: Use `torch.autocast(device_type="cuda", dtype=torch.float16)` (or `torch.bfloat16` on Ampere/A100 GPUs) combined with `torch.amp.GradScaler('cuda')` to accelerate matrix multiplications while computing loss safely in FP32.

## 10. Seed & Determinism Protocol
To guarantee reproducible K-Fold cross-validation splits and model initializations across Kaggle runs, enforce seed setting across all RNG backends:
```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```
