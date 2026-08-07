---
name: macbook-cnn-training
description: Execution rules, DataLoader worker limits, Unified Memory management, and backbone preservation protocols for PyTorch CNN training on Apple Silicon Macbooks (MPS).
---

# Apple Silicon Macbook CNN Training Skill

Comprehensive guide for training PyTorch Convolutional Neural Networks (CNNs) and vision models on Apple Silicon Macs (M1/M2/M3/M4) using Metal Performance Shaders (`mps`).

## 1. Environment & Device Initialization

Always configure OpenMP and PyTorch MPS device settings before initializing model training:

```python
import os
import torch

# Prevent macOS OpenMP duplicate library runtime crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# MPS FP16/BF16 Autocast Context
autocast_context = torch.autocast(device_type="mps", dtype=torch.bfloat16)
```

---

## 2. DataLoader & Multiprocessing Rules (`num_workers`)

- **Optimal `num_workers`**: Strictly set `num_workers = 2` (or `0` for small datasets).
- **macOS `spawn` Overhead**: macOS uses Python `spawn` multiprocessing. Setting `num_workers > 2` degrades performance due to inter-process shared memory (`shm`) lock contention and Unified Memory bus congestion.
- **Pin Memory**: Set `pin_memory = False` on MPS (MPS does not support pinned CUDA host memory).

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,        # Sweet spot for macOS multiprocessing
    pin_memory=False,     # Disable pinned memory for MPS
)
```

---

## 3. Unified Memory & Batch Size Constraints

- Size batch size (e.g. `batch_size = 32` for 384x384 resolution) so peak memory remains within physical RAM.
- Exceeding physical RAM forces macOS Unified Memory to page onto SSD swap, which drops training speed by over 10x.

---

## 5. Gradient Checkpointing & MPS Memory Management

- **The Autograd Activation Bottleneck**: When unfreezing deep backbones (e.g. ConvNeXt-V2 with 36 blocks) during full fine-tuning, PyTorch autograd retains intermediate feature activations in RAM for every block across all batch images. At batch size 32-64, activation memory balloons to > 28 GB, triggering macOS NVMe swap and severe GPU memory bandwidth throttling.
- **Gradient Checkpointing (`set_grad_checkpointing(True)`)**:
  - Re-evaluates activation layers on-the-fly during the backward pass instead of caching them in memory.
  - **Memory Impact**: Cuts autograd activation memory by over **80% (from 28.5 GB down to ~3.5 GB)**.
  - **Speed Impact**: Although recomputation adds a minor ~15% FLOP overhead, staying 100% inside physical Unified RAM without touching disk swap makes overall training **up to 10x FASTER** on Apple Silicon hardware.
- **Periodic MPS Cache Clearing**:
  - Always configure `cache_flush_interval = 50` and call `gc.collect()` + `torch.mps.empty_cache()` every 50 batches to prevent PyTorch MPSAllocator from holding un-freed tensor buffers.

```python
# Enable Gradient Checkpointing on timm backbone
if hasattr(model.backbone, "set_grad_checkpointing"):
    model.backbone.set_grad_checkpointing(True)

# Micro-batch accumulation with Gradient Checkpointing
# Batch size 16 + accum 4 = Effective batch size 64 with < 8GB RAM footprint
```
