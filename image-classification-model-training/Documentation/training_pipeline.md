# Training Pipeline & Optimizations

The `MultiHeadTrainer` orchestrates a highly robust training loop designed to conquer extreme class imbalances while maintaining numerical stability during transfer learning.

## 1. Two-Phase Training Protocol
Transfer learning from natural images (ImageNet) to grayscale medical OCT scans requires a delicate transition to prevent the destruction of the pre-trained weights.

- **Phase 1 (Warm-Up)**: 
  - Backbone (Stem & Stages 0-2) is fully **frozen**.
  - Only the classification heads and CBAM attention module are trained.
  - Runs at a higher learning rate (`1e-3`) for 3-5 epochs to orient the new MLPs to the OCT feature space.
- **Phase 2 (Fine-Tuning)**:
  - The entire backbone is **unfrozen**.
  - Employs **differential learning rates**: the fragile backbone trains slowly (`1e-4` to `5e-6`), while the heads continue converging at `1e-3`.

### Adam Momentum Porting (Crucial Optimization)
When transitioning from Phase 1 to Phase 2, a new `AdamW` optimizer is instantiated to include the newly unfrozen backbone parameters. 
Normally, this would erase the accumulated momentum and variance buffers for the classification heads, causing a violent "momentum shock" that shoots massive error gradients into the fragile backbone, resulting in `NaN` losses. 
Our trainer explicitly iterates through the old Phase 1 optimizer and selectively **ports the Adam state dictionary** into the Phase 2 optimizer, ensuring a completely seamless and mathematically stable transition.

## 2. Numerical Stability
- **Automatic Mixed Precision (AMP)**: Operations are cast to `float16` for memory efficiency.
- **GradScaler**: To combat FP16's narrow numerical ceiling, the loss is scaled before `.backward()` and unscaled before the optimizer steps. This entirely eliminates the classic `NaN` gradient explosion when unfreezing large backbones like ConvNeXt.
- **Gradient Clipping**: All gradients are strictly clamped to a `max_norm=5.0`.

## 3. Asymmetric Loss & Hierarchical Masking
Because the dataset is heavily skewed toward "Normal" images, we employ specific loss strategies:

- **Head 1 Loss**: Standard `BCEWithLogitsLoss`.
- **Head 2 Loss**: MONAI's `FocalLoss` (`gamma=2.0`, `label_smoothing=0.1`). We dynamically calculate the `alpha` weights directly from the dataset class frequencies to force the network to pay attention to severe minority classes.
- **Hierarchical Masking**: We physically zero-out the H2 loss for any image that is actually "Normal". This prevents the Router head from being overwhelmed and confused by trying to route healthy tissue into a disease category.

## 4. Evaluation Metrics
The trainer tracks validation loss for Early Stopping (patience=10), but exclusively saves the `_best_model.pth` based on the **H2 Macro F1 Score**. 
> [!IMPORTANT]
> The evaluation loop correctly utilizes `torch.argmax` on the H2 predictions. This prevents shape mismatches (`ValueError: Classification metrics can't handle a mix of multiclass and multilabel-indicator targets`) when `sklearn.metrics.f1_score` evaluates the multi-class Router predictions.
