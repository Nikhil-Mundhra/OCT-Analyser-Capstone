# Hierarchical OCT/OCTA Image Classification Architecture

> [!NOTE]
> **What exactly did we build?**
> We built a **Hierarchical Convolutional Neural Network (CNN) Ensemble**. 
> It is **NOT** a Recurrent Neural Network (RNN) or a Transformer (like ViT). We utilized **Transfer Learning**, taking state-of-the-art image recognition models (ResNet and EfficientNet) that were pre-trained on millions of real-world images, and specialized them specifically for medical OCT scan analysis.

Instead of training a single massive model to classify all possible diseases at once (which struggles with rare diseases and structural similarities), we built a **triage system** (a cascade/hierarchy of models). It mimics how a medical system works: a general practitioner (Gatekeeper) routes you to a department (Router), which then assigns you to a specialized expert (Specialist).

---

## 1. The Global Data Flow

Here is the visual representation of how a single input scan flows through our hierarchical pipeline during inference:

```mermaid
graph TD
    Input(Input Image Scan) --> L1

    subgraph Level 1: The Gatekeeper
        L1{ResNet-50<br/>Normal vs Abnormal?}
    end

    L1 -->|NORMAL| OutNormal([Diagnosis: NORMAL])
    L1 -->|ABNORMAL| L2

    subgraph Level 2: The Disease Router
        L2{EfficientNet-B2<br/>Which Category?}
    end

    L2 -->|Macular| L3_Mac
    L2 -->|Diabetic| L3_Dia
    L2 -->|Vascular| L3_Vas
    L2 -->|Structural| L3_Str
    L2 -->|Fluid| L3_Flu

    subgraph Level 3: The Specialists (EfficientNet-B0)
        L3_Mac[Macular Specialist]
        L3_Dia[Diabetic Specialist]
        L3_Vas[Vascular Specialist]
        L3_Str[Structural Specialist]
        L3_Flu[Fluid Specialist]
    end

    L3_Mac --> OutMac([CNV / DRUSEN / Generic_AMD])
    L3_Dia --> OutDia([DME / PDR / NPDR])
    L3_Vas --> OutVas([MH / RVO / RAO])
    L3_Str --> OutStr([ERM / CSR])
    L3_Flu --> OutFlu([IRF / SRF])
```

---

## 2. Layer-by-Layer Breakdown

### Level 1: The Gatekeeper (Binary Triage)
- **Model Backbone:** `ResNet-50`
- **Role:** Safely screen out healthy patients. This model is optimized for maximum **Recall**, ensuring we never false-negative a sick patient.
- **Input Resolution:** `224 x 224` pixels (Fast throughput).
- **Output Classes (2):** `NORMAL` vs `ABNORMAL`.

| Flow | Layer | Role in Architecture |
|:---:|---|---|
| 🟢 | **Input** | Raw patient scan (224x224x3) |
| ↓ | **Conv1** | Extracts low-level edge features (7x7 Conv, stride 2) |
| ↓ | **MaxPool** | Reduces spatial dimensions |
| ↓ | **ResBlock 1** | Learns basic textures (3 bottleneck layers) |
| ↓ | **ResBlock 2** | Learns complex patterns (4 bottleneck layers) |
| ↓ | **ResBlock 3** | Learns disease-specific structures (6 bottleneck layers) |
| ↓ | **ResBlock 4** | High-level semantic features (3 bottleneck layers) |
| ↓ | **GAP** | Global Average Pooling: Flattens spatial maps into vector |
| ↓ | **FC Head** | Linear Classification Head (out_features=2) |
| 🏁 | **Output** | Final Binary Triage: NORMAL vs ABNORMAL |

### Level 2: The Disease Router
- **Model Backbone:** `EfficientNet-B2`
- **Role:** Broad categorization. It looks at an abnormal scan and determines the general nature of the pathology. EfficientNet-B2 provides an excellent balance of depth and width to recognize broad structural differences.
- **Input Resolution:** `224 x 224` pixels.
- **Output Classes (5):** `Macular_Degeneration`, `Diabetic_Complications`, `Vascular_Occlusions`, `Structural_Issues`, `Fluid_Accumulation`.

| Flow | Layer | Role in Architecture |
|:---:|---|---|
| 🟢 | **Input** | Abnormal patient scan (224x224x3) |
| ↓ | **Stem** | Initial feature extraction (Conv3x3) |
| ↓ | **MBConv 1** | Mobile Inverted Bottleneck: Efficient spatial filtering |
| ↓ | **MBConv 2** | Includes Squeeze & Excitation: Attends to important feature channels |
| ↓ | **MBConv 3** | Scaled Width & Depth (EfficientNet-B2 scaling) for broader patterns |
| ↓ | **Head** | Conv1x1 & Pooling: Aggregates global features |
| ↓ | **FC Head** | Linear Classification Head (out_features=5) |
| 🏁 | **Output** | Routes to correct specialist (5 Broad Categories) |

### Level 3: The Specialists (5 Distinct Models)
- **Model Backbone:** `EfficientNet-B0` (Fast, highly specialized lightweight models).
- **Role:** Fine-grained sub-typing. Because they only see diseases within their specific category, they can dedicate 100% of their parameters to learning the subtle differences between similar diseases (e.g., distinguishing between RAO and RVO).
- **Input Resolution:** `384 x 384` pixels.
  > [!TIP]
  > **Why the higher resolution?** We bump the resolution significantly at Level 3 because the specialists need to look for minute, fine-grained structural details (like micro-aneurysms or small fluid pockets) that might be blurred out at 224px.
- **Output Classes:** Variable (e.g., 3 classes for Vascular, 2 classes for Fluid).

| Flow | Layer | Role in Architecture |
|:---:|---|---|
| 🟢 | **High-Res Input** | Higher resolution image (384x384x3) to maximize structural detail |
| ↓ | **Stem** | Initial feature extraction (Conv3x3) |
| ↓ | **MBConv 1** | Base Width & Depth (EfficientNet-B0): Fast lightweight filtering |
| ↓ | **MBConv 2** | Includes Squeeze & Excitation: Highly specialized attention |
| ↓ | **Head** | Conv1x1 & Pooling: Aggregates global features |
| ↓ | **FC Head** | Linear Classification Head (out_features=N Subtypes) |
| 🏁 | **Output** | Final Medical Diagnosis (Fine-grained Subtypes) |

---

## 3. The Imbalance Mitigation Engine

Medical datasets are notoriously imbalanced. We have over 26,000 NORMAL images, but only 22 images of RAO. Standard CNNs fail entirely on this. 

We implemented a **three-layered defense mechanism** to force the network to care about rare diseases:

### A. Focal Loss ($\gamma = 2.0$)
Standard Cross-Entropy loss overwhelms the model with easy examples (like the 26,000 normal images). We replaced it with **Focal Loss**, governed by the mathematical formula:

$$ FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$

- **Focusing Parameter ($\gamma=2.0$):** This dynamically scales the loss based on confidence. If the model is 90% confident it's seeing a NORMAL image, the loss drops to near zero. If it struggles with a rare RAO image, the penalty is amplified exponentially.
- **Alpha Weighting ($\alpha$):** We manually inject the inverse frequency weights into the loss function. The network is penalized **52x to 117x harder** for misclassifying rare vascular diseases compared to common ones.

### B. WeightedRandomSampler
During training, we intercept the PyTorch DataLoader. Instead of showing the model random images (which would mean it almost never sees RAO), we sample images with probabilities inversely proportional to their class counts. Every epoch sees an artificially balanced distribution.

### C. Aggressive Augmentation Pipeline
Because we oversample the 22 RAO images heavily, the model would normally memorize them and overfit. To prevent this, we apply severe visual distortions dynamically on the fly:
- **RandomAffine:** Shifts, scales, and shears.
- **RandomErasing:** Literally blacks out random rectangles of the image, forcing the CNN to find multiple distinct features for a disease rather than relying on a single artifact.

---

## 4. Training Strategy: Two-Phase Fine-Tuning

Transfer learning can be volatile if done incorrectly. We use a **Phase-based** training loop for every model in the hierarchy:

1. **Phase 1: Warm-up (Frozen Backbone).** We freeze the millions of pre-trained parameters in the CNN backbone and only allow gradients to update our newly attached classification head. We do this for 5 epochs using a high learning rate ($10^{-3}$).
2. **Phase 2: Fine-tuning (Unfrozen Backbone).** We unfreeze the entire network and drop the learning rate drastically ($10^{-4}$ to $10^{-5}$). We allow the CNN to gently adapt its edge-and-texture filters specifically to OCT scans, without destroying the powerful generalized features it learned from ImageNet.
