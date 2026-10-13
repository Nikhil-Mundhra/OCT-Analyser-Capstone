# Architectural Decision Record: Balanced Hybrid 20M vs. ViT-Focused 20M for TransUNet Iso-Capacity Standardization

**Document ID:** `ADR-2026-TRANSUNET-20M-BALANCED-HYBRID`  
**Date:** 13 October 2026  
**Status:** Approved & Adopted  
**Scope:** `train-cnn-models/model_training/train_rnfl_volumetric/transunet.py`, `OCT-Analyser-Capstone/docs/technical_specs/`  
**Target Iso-Capacity:** $\sim 20.3\,\text{M}$ Parameters (Matching Dense 3D U-Net Job `18770720` / `18574378`)  

---

## 1. Context & Problem Statement

In the Capstone ophthalmic OCT layer segmentation research program, three primary architectural paradigms are evaluated across the 20-subject held-out cohort (`stratified_held_out_v2.json`):
1. **Bi-Planar Orthogonal 2.5D ResNet** ($\sim 6.58\,\text{M}$ parameters)
2. **Dense Anisotropic 3D U-Net** ($\sim 20.30\,\text{M}$ parameters)
3. **Hybrid Anisotropic TransUNet** ($\sim 5.10\,\text{M}$ parameters in Job `18724310`)

To investigate whether TransUNet's observed boundary dispersion (median MABE $9.34\,\mu\text{m}$ vs. $4.71\,\mu\text{m}$ for Bi-Planar) stems from an unfair parameter deficit ($5.1\,\text{M}$ vs. $20.3\,\text{M}$) or fundamental inductive bias limitations, we are scaling TransUNet to an **iso-capacity $\sim 20.3\,\text{M}$ parameter budget**.

Two primary scaling strategies exist:
- **Strategy A (ViT-Focused 20M)**: Keep the CNN stem/decoder thin ($32 \to 64 \to 128$) and scale the transformer bottleneck exclusively ($L=10$ layers, $d_{\text{model}}=384$, $\text{mlp}=1536$), allocating **$92.8\%$ of all weights** to self-attention.
- **Strategy B (Balanced Hybrid 20M)**: Scale both the convolutional feature pyramids ($64 \to 128 \to 256$, $2\times$ wider stem and decoder) and the transformer bottleneck ($L=8$ layers, $d_{\text{model}}=384$, $\text{mlp}=1536$), allocating **$25\%$ to convolutions and $75\%$ to self-attention**.

---

## 2. Decision: Adopt Strategy B (Balanced Hybrid 20M)

We explicitly reject Strategy A (ViT-Focused 20M) despite its trivial zero-code implementation, and **adopt Strategy B (Balanced Hybrid 20M)** as the scientifically grounded architecture for iso-capacity scaling.

---

## 3. Academic & Theoretical Rationales

### 3.1 Resolving Stem Starvation in Sub-Pixel Boundary Heads
Our standardized architecture employs `MultiScaleBoundaryHead` to regress continuous 1D physical surfaces in micrometers:
$$\hat{y}(x) = \text{Conv1D}\Big(\text{VerticalPool}\big(F_{\text{decoder}}\big) \;\Vert\; \text{VerticalPool}\big(F_{\text{stem}}\big)\Big)$$
- Under **Strategy A**, the native-resolution CNN stem ($F_{\text{stem}}$) has only **32 channels**. Even though the model has $20.2\,\text{M}$ parameters, $93\%$ of those weights reside in the downsampled token grid $(96 \times 20)$ where spatial resolution is $25\,\mu\text{m} \times 300\,\mu\text{m}$ per token. The 1D boundary regression head is starved of high-frequency spatial filter diversity at the native $3.12\,\mu\text{m}$ axial scale.
- Under **Strategy B**, Stage 1 provides **64 channels** of native-resolution stem features. The boundary regression head receives $2\times$ richer optical gradient and reflectivity descriptors directly from the high-frequency optical signal, enabling sharper sub-pixel coordinate regression.

### 3.2 Inductive Bias Allocation in Medical OCT
In ophthalmic SD-OCT, retinal layer interfaces (ILM and NFL) are thin, continuous, micron-scale physical boundaries (often $< 30\,\mu\text{m}$ total thickness). 
- Vision Transformers possess weak intrinsic spatial inductive bias (permutation-invariant self-attention over coarse patch tokens). Pushing $93\%$ of weights into self-attention forces the network to learn spatial locality from scratch with limited medical OCT data, increasing vulnerability to speckle noise and vitreous shadow artifacts.
- Convolutions provide hard **translation equivariance** and **local spatial smoothness priors** ($\mathcal{O}(1)$ spatial inductive bias). Dedicating $\sim 5\,\text{M}$ parameters to wide convolutional feature pyramids ($64 \to 128 \to 256$) ensures that oriented edge filters, speckle-invariant texture descriptors, and high-frequency boundary gradients are preserved across all spatial scales.

### 3.3 Symmetric Skip-Connection Information Flow
In U-Net style architectures, skip connections bridge the semantic gap between abstract bottleneck representations and fine spatial reconstruction:
- **Strategy A creates severe dimensional asymmetry**: the bottleneck features are wide ($d_{\text{model}} = 384$), but the skip connections are narrow ($128, 64, 32, 32$). The Cascaded Upsampler (CUP) decoder is dominated by coarse bottleneck tokens, drowning out fine axial contours in the skip pathways.
- **Strategy B restores symmetric multi-scale flow**: the skip connections have channels matched to the bottleneck ($256, 128, 64, 64$). The skip pathways possess the capacity ($4.9\,\text{M}$ parameters) to transmit high-frequency axial gradient details directly into the decoder.

### 3.4 Sample Complexity & Overfitting Resistance
In medical imaging cohorts ($N \sim 40\text{--}60$ subjects, $\sim 12,000$ B-scans), deep self-attention stacks ($L \ge 10$) with long sequence lengths ($N = 1,920$ tokens) are prone to attention entropy collapse and overfitting. Anchoring $25\%$ of the network's capacity in rigid convolutional weight sharing acts as an essential structural regularizer that enhances held-out generalization across unseen patient anatomies.

### 3.5 Arithmetic Intensity & GPU Optimization
- Computing self-attention over $N = 1,920$ tokens per slice requires an attention matrix of $1,920 \times 1,920 \approx 3.69 \times 10^6$ entries per head, per layer.
- Strategy B reduces transformer depth from $L=10 \to 8$, cutting quadratic attention matrix operations by **$20\%$**.
- The reallocated compute goes into spatial 2D convolutions, which achieve near-peak Tensor Core utilization with linear $\mathcal{O}(HW)$ memory scaling.

---

## 4. Parameter & Channel Specifications

| Component | Standard TransUNet 5.1M (`18724310`) | Scaled Balanced Hybrid 20M (`Proposed`) |
| :--- | :---: | :---: |
| **CNN Stage 1 Channels** | $32$ | **$64$** |
| **CNN Stage 2 Channels** | $64$ | **$128$** |
| **CNN Stage 3 Channels** | $128$ | **$256$** |
| **CNN Stage 4 Projection** | $128 \to 256$ | **$256 \to 384$** |
| **Transformer Embedding ($d_{\text{model}}$)** | $256$ | **$384$** |
| **Transformer Layers ($L$)** | $6$ | **$8$** |
| **Attention Heads ($h$)** | $8$ ($32\,\text{dim/head}$) | **$12$ ($32\,\text{dim/head}$)** |
| **MLP Dimension ($d_{\text{ff}}$)** | $512$ ($2\times$) | **$1536$ ($4\times$)** |
| **Decoder CUP Channels** | $128 \to 64 \to 32 \to 16$ | **$256 \to 128 \to 64 \to 32$** |
| **Boundary Stem Input** | $32$ | **$64$** |
| **CNN Parameters** | $1.37\,\text{M}$ ($26.9\%$) | **$4.96\,\text{M}$ ($24.9\%$)** |
| **Transformer Parameters** | $3.65\,\text{M}$ ($71.6\%$) | **$14.93\,\text{M}$ ($74.7\%$)** |
| **Multi-Scale Heads** | $0.08\,\text{M}$ ($1.5\%$) | **$0.09\,\text{M}$ ($0.4\%$)** |
| **Total Parameter Count** | **$5,099,988$ ($\sim 5.10\,\text{M}$)** | **$19,980,188$ ($\sim 19.98\,\text{M}$)** |

*Parameter Delta against Dense 3D ($20,296,897$)*: **$-1.56\%$** (Strict iso-capacity match).

---

## 5. Implementation & Migration Plan

1. **Source Refactoring (`transunet.py`)**:
   - Parameterize `AnisotropicTransUNetEncoder` with `stage_channels=(32, 64, 128)` as default (preserving exact backwards compatibility for the existing 5.1M checkpoints).
   - In `TransUNetRNFLNet`, expose `stage_channels` and wire `base_channels` dynamically into `cup1..cup4`.
2. **CLI Parameterization (`train.py`)**:
   - Add `--transunet_stage_channels "64,128,256"` and `--base_channels 32` for Balanced Hybrid 20M.
3. **Unit Tests**:
   - Validate backwards compatibility with checkpoint loading of `18724310`.
   - Validate forward/backward passes and parameter count of the 19.98M Balanced Hybrid configuration.
4. **HPC SLURM Script**:
   - Generate `train_rnfl_transunet_20m_jubail.slurm` with optimized batch size and bfloat16 precision.
