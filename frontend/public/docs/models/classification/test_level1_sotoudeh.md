# Level 1 Model Evaluation: Sotoudeh-Paima OCT Dataset

## Overview
This document outlines the evaluation of the **Level 1 Gatekeeper Model** (Normal vs. Abnormal) on the labeled retinal Optical Coherence Tomography (OCT) dataset for the classification of Normal, Drusen, and CNV cases.

> [!NOTE]
> The test is currently executing on a purely **CPU-bound** environment. This document will be updated with the final evaluation metrics once the background task completes.

## Dataset Structure & Labeling
The dataset consists of more than 16,000 retinal OCT B-scans from 441 cases (Normal: 120, Drusen: 160, CNV: 161), acquired at Noor Eye Hospital, Tehran, Iran.

The folder structure is highly optimized for slice-level supervision:
1. **Diagnosis (Top-Level):** Folders like `CNV`, `DRUSEN`, and `NORMAL` represent the overarching clinical diagnosis for a patient's eye.
2. **Patient ID (Sub-Level):** Inside each diagnosis folder, patients are separated into numbered folders (e.g., `CNV/1/`, `CNV/2/`). This represents a single 3D OCT volume.
3. **B-Scans (File-Level):** Inside the patient folder, individual 2D slices (B-scans) are stored as `.jpg` images. Crucially, the images are labeled based on **what is actually visible in that specific slice** (e.g., `000_Drusen.jpg`, `011_CNV.jpg`, `003_Normal.jpg`), rather than just inheriting the overall patient diagnosis.

## Test Configuration
- **Model Target:** Level 1 Gatekeeper (Binary Classification)
- **Label Mapping:** 
  - `Normal` → `0`
  - `Drusen` / `CNV` → `1` (Abnormal)
- **Evaluation Strategy:** Option 1 (All Images) 
  - The evaluation evaluates all 16,822 individual images, prioritizing full coverage across all slices rather than solely evaluating the worst-case condition.
- **Hardware:** CPU Only
- **Decision Threshold:** `0.35` (lowered from default `0.50` to reduce False Negatives, prioritizing safety for abnormal case screening)

## Acknowledgments and Citation
If this dataset is utilized in further research or deployment, it must be acknowledged and cited as follows:

> **Sotoudeh-Paima, S., Jodeiri, A., Hajizadeh, F., & Soltanian-Zadeh, H. (2022). Multi-scale convolutional neural network for automated AMD classification using retinal OCT images. *Computers in biology and medicine, 144*, 105368.**

The original implementation of the above-mentioned publication is available at the following repository:
[SamanSotoudeh/Multi-scale-convolutional-neural-network-for-automated-AMD-classification-using-retinal-OCT-images](https://github.com/SamanSotoudeh/Multi-scale-convolutional-neural-network-for-automated-AMD-classification-using-retinal-OCT-images)

---

## Results (Threshold: 0.35)
*Evaluation completed on 16,822 test images using CPU.*

- **Accuracy:** 0.8148 (81.48%)
- **AUROC:** 0.9349
- **Macro F1:** 0.8136
- **Weighted F1:** 0.8133

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.90 | 0.72 | 0.80 | 8584 |
| **Abnormal** | 0.76 | 0.92 | 0.83 | 8238 |
