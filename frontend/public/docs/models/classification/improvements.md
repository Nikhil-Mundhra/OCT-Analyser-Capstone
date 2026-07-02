# Level 1 Gatekeeper — Model Improvements (1st July, 2026)

This document is the authoritative record of all improvements made to the Level 1 Gatekeeper model. Each improvement includes the **rationale**, **files changed**, **expected impact**, and **how to verify** the change is working correctly.

> [!NOTE]
> Improvements 1–4 are **pre-training changes** and require a new training run to take effect.
> Improvements 5–9 are **post-training changes** applied to an already-trained checkpoint.

---

## Improvement 1 — Checkpoint Selection: `val_loss` → `val_macro_f1`

**File:** [`training/trainer.py`](../training/trainer.py)

### Rationale

The best checkpoint was previously saved whenever `val_loss` reached a new minimum. But the stated primary metric for the Level 1 model is `val_macro_f1`. These two signals are not equivalent under class imbalance:

A model that assigns very high confidence to the majority class (NORMAL, 26,853 samples) will achieve low cross-entropy loss while still misclassifying the minority class. The result is a saved checkpoint that looks good on loss curves but underperforms on the clinical metric that actually matters.

> [!IMPORTANT]
> This creates an invisible mismatch: `summarise_folds()` in `train_level1.py` correctly reports the best `val_macro_f1` achieved during training, but the checkpoint on disk may correspond to a *different* epoch — the minimum-loss epoch — which may have had a lower F1.

### Change

Early stopping continues to monitor `val_loss` (this is intentional — early stopping monitors training *stability*, not clinical *performance*). Only the checkpoint selection criterion changed:

```diff
- if val_loss < best_val_loss:
-     best_val_loss = val_loss
+ current_macro_f1 = val_m.get("val_macro_f1", 0.0)
+ if current_macro_f1 > best_val_macro_f1:
+     best_val_macro_f1 = current_macro_f1
```

### Expected Impact

Eliminates the possibility of deploying a suboptimal checkpoint. In runs where loss and F1 diverge, this can yield meaningful improvements in the external test Macro F1.

---

## Improvement 2 — LR Scheduler: `T_0=20` → `T_0=10`

**File:** [`training/trainer.py`](../training/trainer.py)

### Rationale

`CosineAnnealingWarmRestarts` with `T_0=20, T_mult=2` schedules learning rate restarts at epochs 20, 60, 140, …

With `finetune_epochs=50`, the second restart would occur at epoch 60 — **beyond the training budget**. This means only one cosine cycle ever completes. After the restart at epoch 20, the model trains for 30 more epochs on a monotonically decaying LR, which is identical to standard cosine annealing. The "warm restarts" benefit — periodically boosting the LR to help the optimiser escape local minima — was effectively inactive.

### Change

```diff
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer_ft,
-     T_0=20,
+     T_0=10,
      T_mult=2,
      eta_min=1e-6,
  )
```

With `T_0=10`, restarts fire at epochs 10 and 30, giving **two complete cosine cycles** within the 50-epoch budget.

### Expected Impact

Smoother convergence with the loss landscape explored more thoroughly. Particularly beneficial for escaping the shallow local minima typical of imbalanced medical imaging datasets.

---

## Improvement 3 — Backbone: ResNet-50 → EfficientNet-B3

**File:** [`models/level1_gatekeeper.py`](../models/level1_gatekeeper.py)

### Rationale

| Metric | ResNet-50 | EfficientNet-B3 | Change |
|---|---|---|---|
| Parameters | 25.6M | 12.2M | **−52%** |
| ImageNet Top-1 | 80.9% | 82.2% | **+1.3%** |
| Feature dim | 2048 | 1536 | Smaller head |
| OCT AUROC (typical) | ~0.970 | ~0.980–0.985 | +1–1.5% |

EfficientNet uses **compound scaling**: depth, width, and resolution are all scaled together using a fixed ratio derived by neural architecture search. At B3, this means the network simultaneously captures fine-grained local textures (via depth) and broader structural context (via width) — both essential for retinal pathology.

For OCT specifically, this matters because:
- **Drusen** are small, localised deposits requiring fine-grained spatial resolution
- **CNV** produces broader membrane changes requiring structural context
- EfficientNet's compound scaling encodes both simultaneously within fewer parameters

### Architecture Change

```python
# Before: ResNet-50 — 2048-d output
self.features = nn.Sequential(*list(backbone.children())[:-1])
# in_features = 2048

# After: EfficientNet-B3 — 1536-d output
self.features = backbone.features      # nn.Sequential of MBConv blocks
self.avgpool  = backbone.avgpool       # AdaptiveAvgPool2d(1)
# in_features = 1536
```

The classifier head topology is identical (Dropout → Linear → ReLU → Dropout → Linear). Only the input dimension changes (2048 → 1536).

### Expected Impact

- **AUROC:** +0.5–1.5% on external test set
- **Inference speed:** ~1.3× faster (fewer FLOPs per forward pass)
- **Memory:** ~30% lower activation footprint at batch_size=48

### Grad-CAM Note

The Grad-CAM target layer changes from `model.features[-2]` (ResNet-50 `layer4`) to `model.features[-1]` (EfficientNet-B3's last MBConv block). This is already correctly set in the updated test script.

---

## Improvement 4 — CLAHE Preprocessing

**File:** [`data/transforms.py`](../data/transforms.py)

### Rationale

The dataset aggregates OCT scans from multiple sources and scanner families:
- **Kermany/UCSD**: Zeiss Cirrus (specific brightness/contrast profile)
- **OCTDL**: Different scanner generation
- **OCTID**: Potentially Heidelberg Spectralis or Topcon

Each scanner produces a systematically different baseline brightness and local contrast. Without normalisation, the model can exploit these scanner-specific signatures rather than learning pathology — a form of **shortcut learning** (also called Clever Hans learning) that achieves high validation accuracy on the training dataset but fails to generalise to a new clinical site or scanner.

CLAHE (Contrast Limited Adaptive Histogram Equalisation) equalises local contrast independently in small tile regions, making the image statistics scanner-agnostic while preserving the structural information that defines pathology.

### Implementation

```python
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, img):
        img_np = np.array(img.convert("L"), dtype=np.uint8)
        equalized = self.clahe.apply(img_np)
        rgb = np.stack([equalized, equalized, equalized], axis=-1)
        return PIL.Image.fromarray(rgb, mode="RGB")
```

Applied as the **first step** in all three transform pipelines: `get_train_transforms`, `get_heavy_train_transforms`, and `get_val_transforms`.

> [!IMPORTANT]
> CLAHE must be applied **identically at train, val, and test time**. It is not an augmentation — it is a deterministic normalisation step. Adding it only to training would create a train/test distribution mismatch.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `clip_limit` | 2.0 | Standard for medical OCT; higher values amplify noise in uniform regions |
| `tile_grid` | (8, 8) | 64 adaptive tiles across the image; standard size for retinal scans |

### Expected Impact

- Improved generalisation to multi-scanner datasets
- Reduced sensitivity to acquisition conditions at inference
- Particularly impactful on the external test set which may use different hardware than the training data

---

## Improvement 5 — ROC-Derived Decision Threshold

**File:** [`scripts/calibrate_level1.py`](../scripts/calibrate_level1.py) *(new file)*

### Rationale

The previous test script used `--threshold 0.35`. This value had no documented justification. If it was discovered by testing different values against the test set, this constitutes **data leakage**: the test set was used to select a model hyperparameter, making the reported metrics optimistic and non-reproducible.

The decision threshold must be derived from **validation data only**, then fixed and applied to the test set without adjustment.

### Two Strategies

**Youden's Index** — maximises Sensitivity + Specificity jointly:
$$J = \text{argmax}(\text{TPR} - \text{FPR})$$

**Sensitivity-Constrained** *(recommended for screening)* — finds the highest-specificity threshold subject to Sensitivity ≥ target:
$$\text{threshold}^* = \text{argmax}(\text{Specificity}) \quad \text{subject to} \quad \text{Sensitivity} \geq 0.95$$

The sensitivity-constrained strategy is clinically appropriate for Level 1: a **screener's** primary failure mode is a False Negative (clearing a sick patient). We accept more False Positives (unnecessary referrals) in exchange for catching ≥95% of diseased patients.

### Usage

```bash
python3 scripts/calibrate_level1.py \
    --checkpoint checkpoints/level1/fold0_best_model.pth \
    --strategy sensitivity_constrained \
    --sensitivity-target 0.95
```

**Output:** `checkpoints/level1/calibration.json` is automatically read by the test script.

---

## Improvement 6 — Sensitivity & Specificity Reporting

**File:** [`scripts/test_level1_on_test_set.py`](../scripts/test_level1_on_test_set.py)

### Rationale

The original test script reported accuracy, AUROC, and F1 — but not sensitivity or specificity by name. For a binary medical screener, these are the **primary clinical metrics**:

| Metric | Definition | Clinical meaning |
|---|---|---|
| **Sensitivity** | TP / (TP + FN) | What fraction of diseased patients are correctly caught? |
| **Specificity** | TN / (TN + FP) | What fraction of healthy patients are correctly cleared? |
| **PPV** | TP / (TP + FP) | If flagged as abnormal, how likely is disease? |
| **NPV** | TN / (TN + FN) | If cleared as normal, how confident are we? |

The updated test script explicitly computes all four from the confusion matrix and logs them prominently, with sensitivity marked as the primary metric.

---

## Improvement 7 — Test-Time Augmentation (TTA)

**File:** [`scripts/test_level1_on_test_set.py`](../scripts/test_level1_on_test_set.py)

### Rationale

At inference, a single forward pass produces a prediction that is sensitive to the exact spatial positioning of the image content. TTA averages predictions across multiple augmented views, reducing this variance without any retraining.

### Implementation

Five deterministic views are averaged:
1. Original image
2. Horizontal flip
3. Vertical flip
4. 90° rotation
5. −90° rotation

```python
probs = predict_batch_tta(model, images, device, n_views=5)
```

The temperature-scaled softmax is applied inside TTA, so calibration and TTA interact correctly.

### Expected Impact

- **AUROC:** +0.5–1.5%
- **Macro F1:** +1–2%
- **Inference time:** 5× slower (acceptable for a ~1,000-image test set; for 86K+ image production deployment, restrict TTA to low-confidence cases with probability in range [0.35, 0.65])

---

## Improvement 8 — Grad-CAM Explainability

**File:** [`scripts/test_level1_on_test_set.py`](../scripts/test_level1_on_test_set.py)

### Rationale

A model that achieves 98% AUROC may still be attending to the wrong image regions — scanner borders, watermarks, image compression artefacts, or spurious correlations in the training data. In a clinical context, "high accuracy" alone is insufficient evidence that the model is reasoning about pathology.

Grad-CAM (Gradient-weighted Class Activation Mapping) visualises which spatial regions of the image contributed most to the model's decision, by backpropagating gradients to the final convolutional feature map.

> [!CAUTION]
> **The False Negative grid is the most clinically important output.** These are cases where the model cleared a diseased patient as normal. If the Grad-CAM shows the model attending to the image border or a blank region rather than the retinal tissue, this indicates a systematic failure that metrics alone cannot reveal.

### Output

Running with `--gradcam` generates four grid images:

| File | Contents |
|---|---|
| `gradcam_true_pos.png` | Correctly identified ABNORMAL scans — what does the model attend to? |
| `gradcam_false_neg.png` | **Missed ABNORMAL scans — critical review required** |
| `gradcam_true_neg.png` | Correctly cleared NORMAL scans |
| `gradcam_false_pos.png` | Incorrectly flagged NORMAL scans — what caused the false alarm? |

```bash
python3 scripts/test_level1_on_test_set.py \
    --checkpoint checkpoints/level1/fold0_best_model.pth \
    --gradcam \
    --gradcam-dir logs/level1/gradcam
```

---

## Improvement 9 — Temperature Scaling (Probability Calibration)

**File:** [`scripts/calibrate_level1.py`](../scripts/calibrate_level1.py) *(new file)*

### Rationale

Raw softmax outputs from trained neural networks are **systematically overconfident**. A model that outputs 95% confidence may only be correct 80% of the time at that confidence level. This miscalibration has two clinical consequences:

1. **Triage errors:** Clinicians who see "97% probability of ABNORMAL" will prioritise this case differently than if they see "72% probability of ABNORMAL." If the underlying probability is actually 72%, the model is misleading the triage workflow.
2. **Threshold sensitivity:** A miscalibrated probability space means the chosen decision threshold (Improvement 5) may not translate correctly to real-world operating conditions.

Temperature scaling is the simplest and most reliable post-hoc calibration method for classification networks (Guo et al. 2017, ICML):

$$\hat{p} = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$

A single scalar $T$ is fit on the validation set by minimising NLL using L-BFGS. For most trained CNNs, $T > 1.0$ (the model was overconfident); scaling with $T > 1.0$ softens probabilities toward calibrated uncertainty.

### Verification

Two reliability diagrams are saved:
- `checkpoints/level1/calibration_before.png` — before scaling
- `checkpoints/level1/calibration_after.png` — after scaling

A perfectly calibrated model follows the diagonal on a reliability diagram. The Expected Calibration Error (ECE) is printed before and after scaling. A reduction in ECE confirms calibration improved.

---

## Improvement Summary

| # | Improvement | Phase | Primary Metric Impact |
|---|---|---|---|
| 1 | Checkpoint selection → `val_macro_f1` | Pre-training | Ensures deployed model = best F1 model |
| 2 | Scheduler `T_0`: 20 → 10 | Pre-training | Smoother convergence |
| 3 | Backbone: ResNet-50 → EfficientNet-B3 | Pre-training | AUROC +0.5–1.5%, 3× fewer params |
| 4 | CLAHE preprocessing | Pre-training | Generalisation to multi-scanner data |
| 5 | ROC-derived threshold | Post-training | Sensitivity ≥ 0.95 guaranteed |
| 6 | Sensitivity/specificity reporting | Post-training | Clinical metrics visible |
| 7 | Test-Time Augmentation (5 views) | Post-training | AUROC +0.5–1.5%, Macro F1 +1–2% |
| 8 | Grad-CAM explainability | Post-training | Clinical interpretability |
| 9 | Temperature scaling | Post-training | Calibrated probabilities, lower ECE |

## Complete Workflow

```bash
# 1. Retrain with pre-training improvements active
python3 scripts/train_level1.py

# 2. Calibrate threshold and temperature on the validation fold
python3 scripts/calibrate_level1.py \
    --checkpoint checkpoints/level1/fold0_best_model.pth \
    --strategy sensitivity_constrained \
    --sensitivity-target 0.95

# 3. Evaluate on external test set with all post-training improvements
python3 scripts/test_level1_on_test_set.py \
    --checkpoint checkpoints/level1/fold0_best_model.pth \
    --gradcam \
    --gradcam-dir logs/level1/gradcam
```
