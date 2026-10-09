# Technical Postmortem: V100 Volta FP16 Arithmetic Overflow & NaN Divergence

**Document ID**: POSTMORTEM-2026-10-09-V100-FP16-OVERFLOW  
**Authors**: Antigravity Autonomous Agent & Nikhil Mundhra  
**Date**: October 9, 2026  
**Status**: Resolved (`train-cnn-models` commit `7cc1550`)  
**Impacted Environments**: NYUAD Jubail HPC (`#SBATCH -p nvidia --gres=gpu:v100:4`)  
**Affected Jobs**: SLURM Job `18724259` (`rnfl_bp_61s_4v100`)

---

## 1. Executive Incident Summary

On October 9, 2026, the zero-leakage 61-subject Retraining Pipeline was launched on **4x NVIDIA Tesla V100-SXM2-32GB GPUs** on node `dn001` via PyTorch DDP (`torchrun --nproc_per_node=4`).

While training initially progressed with rapid convergence across Epochs 1–7 (loss declining from $296.89 \to 73.84$ at $\approx 368$ slices/s), **training diverged to `NaN` at Epoch 8 Step 725**:

```text
Epoch [8/20] Step [1200/1220] Loss: 188.75 (Overlap: 1.000, Thick: 54.0 um, Bnd: 420.5 um)
Epoch [8/20] Total Time: 294.8s | Train Loss: nan | Val Loss: nan | Val Dice: 0.0000 | Val Peri NFL MABE: nan um
Epoch [9/20] Step [725/1220] Loss: nan (Overlap: nan, Thick: nan um, Bnd: nan um)
```

The model finished all 20 epochs with collapsed weights. Subsequent evaluation on the 40 held-out validation scans yielded corrupted metrics ($\text{MABE} \approx 124.41\,\mu\text{m}$, $\text{Dice} \approx 0.2239$) masquerading as a catastrophic failure of the 61-subject training cohort.

---

## 2. Root Cause Analysis: Hardware Precision Mismatch

### 2.1 The Architectural Difference Between A100 and V100

| GPU Architecture | Compute Capability | Native `bfloat16` Support? | FP16 Dynamic Range | Max Representable Value |
| :--- | :---: | :---: | :---: | :---: |
| **NVIDIA A100 (Ampere)** | `SM 8.0` | **YES** | $\approx 10^{-38}$ to $10^{38}$ | **$3.3895 \times 10^{38}$** |
| **NVIDIA V100 (Volta)** | `SM 7.0` | **NO** | $\approx 6.1 \times 10^{-5}$ to $65,504$ | **$65,504$** |

- Prior baseline models (`18045386`, `18223981`, `18563914`) were trained on NYUAD Jubail's **A100 nodes** using `--precision "bf16"`.
- `bfloat16` allocates **8 exponent bits** (identical to single-precision FP32) and 7 mantissa bits. It possesses the exact same dynamic range as FP32 ($10^{38}$), completely eliminating arithmetic overflow during tensor reductions.
- When transitioning to **4x V100 nodes** to accelerate queue turnaround, the job script requested `--precision "fp16"`.
- Standard IEEE 754 `float16` allocates only **5 exponent bits** and 10 mantissa bits. The maximum representable number in FP16 is strictly **$65,504$**. Any summation that exceeds $65,504$ immediately saturates to $+ \infty$.

---

## 3. The Mathematical Failure Mechanism

In 2.5D volumetric retinal OCT segmentation, inputs possess large spatial dimensions:
- Slice spatial resolution: $H = 768$ (axial depth), $W = 320$ (lateral width) $= 245,760$ voxels per slice.
- Per-GPU batch size: $B = 16$ slices.
- Aggregate voxels per forward pass per GPU:
  $$\text{Total Elements} = 16 \times 768 \times 320 = 3,932,160 \text{ voxels}$$

### 3.1 Tversky / Dice Loss Overflow
The dense segmentation overlap loss is formulated as:
$$\mathcal{L}_{\text{Tversky}} = 1 - \frac{\sum_{i=1}^{N} p_i g_i + \epsilon}{\sum_{i=1}^{N} p_i g_i + \alpha \sum_{i=1}^{N} p_i (1 - g_i) + \beta \sum_{i=1}^{N} (1 - p_i) g_i + \epsilon}$$

In PyTorch, `train.py` executed both the network forward pass and the loss function inside the same autocast block:
```python
with get_autocast_context():  # torch.autocast('cuda', dtype=torch.float16)
    preds = model(batch['image'])
    loss_dict = criterion(preds, batch)  # EXECUTED ENTIRELY IN FP16!
```

When evaluating the denominator:
1. If just **1.7%** of the voxels in the batch are foreground or transition boundaries:
   $$\text{Sum} = 3,932,160 \times 0.017 \approx 66,846 > 65,504$$
2. The FP16 accumulator overflows to `inf`.
3. The fraction evaluates to:
   $$\frac{\infty}{\infty} = \text{NaN}$$
4. The PyTorch `GradScaler` detects `NaN`, skips optimizer steps repeatedly, and once unscaled gradients contain `NaN`, model weights irrevocably degrade to all-zeros or random noise.

### 3.2 Unconstrained Optical Edge Grid Sampling
In `losses.py`, the optical gradient alignment loss computes:
$$\text{norm\_y} = \frac{2 \cdot \hat{y}_{\text{NFL}}}{H - 2} - 1.0$$
Before clipping was introduced, if early gradient spikes caused $\hat{y}_{\text{NFL}}$ to overshoot the image boundaries, `grid_sample` in FP16 sampled extrapolated coordinates with unbounded half-precision gradients.

---

## 4. Architectural Resolution (`7cc1550`)

We instituted a **Dual-Tier Numerical Precision Architecture**:

1. **FP32 Loss Calculation Outside Autocast**:
   Model forward passes run in FP16 to utilize Volta Tensor Cores at full speed, but **all prediction tensors are upcast to IEEE 754 `float32` before computing loss functions**:
   ```python
   with get_autocast_context():
       preds = model(batch['image'])

   # Upcast predictions to FP32 outside autocast
   preds_fp32 = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}
   loss_dict = criterion(preds_fp32, batch)
   ```

2. **FP32 Validation Reduction**:
   All validation metrics (`validate()` in `train.py`) now cast predictions to FP32 before accumulating batch statistics, preventing validation MABE and Dice from collapsing.

3. **Coordinate Clamping**:
   In `losses.py`, coordinates for optical edge bilinear sampling are strictly bounded:
   ```python
   norm_y = (2.0 * preds['nfl_pred'] / (H_img - 2) - 1.0).clamp(-1.0, 1.0)
   ```

---

## 5. Standard Operating Procedure for Future HPC Training

1. **When training on NVIDIA A100 / H100**:
   - Always specify `--precision "bf16"`. Native hardware `bfloat16` provides maximum speed with zero overflow risk.
2. **When training on NVIDIA V100 / T4**:
   - Specify `--precision "fp16"`. The updated pipeline ensures all loss reductions occur in FP32 with `GradScaler`.
   - If batch sizes exceed 16 per device, `--precision "fp32"` is recommended as a zero-risk alternative (fitting comfortably in 32GB VRAM).
3. **Never benchmark a model if execution logs contain `Loss: nan`**:
   - Always verify that training logs exhibit smooth loss decay and non-zero validation Dice before ingesting checkpoints into multi-model comparison benchmarks.
