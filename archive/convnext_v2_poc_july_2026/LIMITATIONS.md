# ConvNeXt V2 Standalone Classifier (PoC)

**Date Archived:** July 18, 2026
**Architecture:** ConvNeXt V2 (Base) with Hierarchical Multi-Head Output (H1, H2, H3).
**Training Time:** ~4 Hours on Kaggle (2x T4 GPUs). Halted at Epoch 6.

## Clinical Performance Snapshot
* **Head 1 (Triage / Normal vs Abnormal):** 98% Accuracy / 98% F1
* **Head 2 (Disease Router):** 93% Accuracy / 75% Macro F1
* **Head 3 (Granular Subtypes):**
  * CNV: 95% F1
  * DME: 100% F1
  * ERM: 93% F1

## Discovered Limitations

### 1. The Fluid Accumulation (CSR) Zero-Weight Bug
* **Issue:** The model achieved only a 10% F1 score (5% Recall) on Central Serous Retinopathy (CSR).
* **Cause:** A mathematical bug in `compute_loss_weights`. Because CSR was the *only* label mapped under the "Fluid Accumulation" family, the dataloader calculated its intra-family negative samples as `0`. This resulted in `pos_weight = 0.0`. The model received no reward for correctly predicting CSR, causing it to default to predicting `0` (negative) to minimize false positive penalties.
* **Resolution for Next Phase:** Introduce a lower bound `weights[idx] = max(1.0, (neg_count / max(1, count)))` to prevent zero gradients.

### 2. High False Negative Rate in Triage (H1)
* **Issue:** Head 1 exhibited a 2.15% False Negative Rate (missing 264 actually diseased scans out of 12,286). In a clinical triage setting, False Negatives are drastically more dangerous than False Positives.
* **Cause:** The evaluation used a standard `0.5` sigmoid threshold.
* **Resolution for Next Phase:** Shift the Head 1 operating point threshold from `0.5` down to `0.2`. This trades a minor increase in False Positives for a near-zero False Negative Rate.

### 3. Extreme Data Scarcity (VID, DR, MH)
* **Issue:** Classes like Vitreo-macular Interface Disease (VID) only had ~60 training samples, resulting in a fragile 67% F1 score.
* **Cause:** Standard Random Sampling feeds rare diseases at their natural, microscopic frequencies. Loss scaling helped, but cannot synthesize missing visual features.
* **Resolution for Next Phase:** Implement a `WeightedRandomSampler` in the PyTorch dataloader to oversample rare classes, or use targeted augmentation strictly on minority classes.

### 4. Overreliance on Blackout/Letterbox Padding
* **Issue:** The model relies on raw pixel destruction (`BlackoutCorners`) to avoid learning UI compass artifacts from the training data.
* **Resolution for Next Phase (The Capstone Goal):** Pivot to a modular Two-Stage pipeline. A specialized U-Net will identify and crop out 100% of the retinal tissue dynamically, completely eliminating the need for rigid pixel blackouts.
