---
name: kaggle
description: MANDATORY skill for deploying, writing, running, or debugging model training scripts on Kaggle. Enforces critical rules for preventing DataLoader deadlocks, tail-batch crashes (batch-size 1 squeeze traps), shared memory OOMs, multi-GPU setup, and mandatory local pre-flight smoke testing.
---

# Kaggle Training Guidelines

When deploying PyTorch training scripts to Kaggle (especially those using Multi-processing DataLoaders and OpenCV/MONAI), you MUST enforce the following rules to prevent silent deadlocks and shared memory crashes:

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

## 2. Shared Memory Limits (`/dev/shm`)
Kaggle kernels and Docker containers have strict limits on shared memory. PyTorch uses shared memory to transfer tensors from DataLoader workers to the main process.
- **Maximum Workers**: Never set `num_workers` higher than `2` for high-resolution images (e.g., 384x384) with large batch sizes (e.g., 64). 
- If training still hangs, fallback to `num_workers=0` to force synchronous data loading and completely bypass `/dev/shm`.

## 3. Progress Tracking
- Avoid printing high-frequency updates directly to `stdout`/`stderr` inside loops, as it can flood and crash the Kaggle web interface.
- If using `tqdm`, the user may request it to be removed to keep logs clean. Respect the user's preference for clean, epoch-level logging over batch-level progress bars.

## 4. Multi-GPU Saturation
When running on Kaggle kernels equipped with dual GPUs (like Dual T4s):
- Use `nn.DataParallel` to automatically split the batch across both GPUs.
- Override default batch sizes via command line (e.g., `--batch-size 64`) so that each GPU receives an optimal chunk (e.g., 32 per GPU) to keep VRAM fully utilized without overflowing.

## 5. Executing the Multi-Head ConvNeXt Model
When the user wants to kick off the final Multi-Head ConvNeXt training on Kaggle, they will typically run this in a Kaggle Notebook using the "Save & Run All (Commit)" feature.

Ensure they use the following blocks in their notebook:

**Cell 1: Install Dependencies (Quietly)**
```bash
!uv pip install --system -r image-classification-model-training/requirements.txt -q
```
*(Note: `UV_NO_PROGRESS=1` is also helpful to reduce console bloat if not using `-q`)*

**Cell 2: Execute Training**
```bash
!PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
OCT_DATA_ROOT="/kaggle/input/datasets/nikhilmundhra/classified-oct-v2/Classified" \
python image-classification-model-training/scripts/train_convnext.py \
    --config "image-classification-model-training/config/hierarchy.yaml" \
    --batch-size 16 \
    --num-workers 0
```
- `OCT_DATA_ROOT`: Crucial for overriding the local data root defined in `hierarchy.yaml` to point to Kaggle's `/kaggle/input/...` directory structure.
- `--batch-size 16`: `convnextv2_base` with multi-scale CBAM attention blocks is memory intensive. A batch size of 16 (8 per T4 GPU) prevents CUDA OutOfMemory errors on 15GB T4 GPUs.
- `--num-workers 0`: Crucial to prevent DataLoader deadlocks (see section 1 and 2).
- `--resume checkpoints/multi_head/fold0_last_model.pth`: (Optional) Seamlessly resumes training from a previous checkpoint if a Kaggle session times out.
- Default epochs (5 warmup + 45 finetune) will be used unless overridden.

## 6. Mandatory Local Pre-Flight Checks (Smoke Testing)
Before deploying **any** changes to the Kaggle training pipeline (even seemingly trivial changes), it is **MANDATORY** to run a full local smoke test. Kaggle "Save & Run All" commits take hours to queue and run, so catching typos, shape mismatches, or missing imports locally is strictly required.

Always execute the pipeline locally against the `micro_dataset` to ensure it successfully builds the models, routes tensors, evaluates metrics, and saves the `.pth` checkpoints.

**Command to run the local smoke test:**
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
*(Note: Ensure you are using `micro_hierarchy.yaml` locally, as the structure of `micro_dataset` does not match the full Kaggle dataset mapped in `hierarchy.yaml`).*

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
- **Why Smoke Tests Miss It**: Local smoke tests using small sample sizes and batch size 4 process batches of shape `[4, 1]`. `.squeeze()` on `[4, 1]` only squeezes dim 1, hiding the 0D collapse that occurs when batch size is 1.

## 8. Checkpoint Saving & Resuming Across Session Timeouts
Kaggle sessions have a strict 9-hour maximum execution limit. Long-running training pipelines (e.g. 50 epochs over 88k images) may get cut off before completing all epochs.

- **Automatic Checkpoint Persistence**: Ensure the trainer saves `fold0_best_model.pth` and `fold0_last_model.pth` to `/kaggle/working/checkpoints/` after every single epoch. When a Kaggle session reaches its time limit or is stopped, Kaggle automatically saves all files in `/kaggle/working`.
- **Seamless Resuming**: Always support a `--resume` argument pointing to `checkpoints/multi_head/fold0_last_model.pth`. When resuming:
  1. Load the model state dictionary.
  2. Restore the phase (`warmup` vs `finetune`) and absolute epoch index.
  3. Restore optimizer and learning rate scheduler states (`optimizer_state_dict`, `scheduler_state_dict`).
  4. Pass `--resume` in the Kaggle command line for subsequent runs to continue seamlessly without wasting completed epochs.
- **Fail-Safe Real-Time Cloud Backup (HF Hub)**: To guarantee zero data loss if Kaggle hard-crashes or times out without committing output:
  - Pass `--hf-repo username/repo_name` and set `HF_TOKEN` in the environment.
  - The trainer automatically streams `fold0_best_model.pth` directly to Hugging Face Hub the exact second a new peak score is achieved. Even if Kaggle times out 4 hours later, the checkpoint is already safe in the cloud.
