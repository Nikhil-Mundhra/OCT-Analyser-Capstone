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

## 4. Pretrained Backbone Preservation Protocol

When fine-tuning pretrained vision models (ConvNeXt, ResNet, EfficientNet) on medical/OCT images:

1. **Phase 1 (Warmup - 10 Epochs)**: Keep backbone completely frozen (`freeze_full_backbone()`). Train only classification heads and attention blocks (CBAM) at `lr_head = 1e-4`.
2. **Phase 2 (Gradual Unfreezing)**: Unfreeze only the deepest bottleneck stage at `backbone_lr = 1e-6`, keeping stem & early stages frozen or at `0.1x` learning rate (`2e-7`).
