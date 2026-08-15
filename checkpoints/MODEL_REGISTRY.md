# OCT-Analyser Model Registry

Central registry tracking all versioned models, training variants, verified metrics, and deployment statuses.
Updated: `2026-08-15 18:50:14`

---

## 1. Disease Classification Models (`checkpoints/classification/multi_head/`)

| Variant | Version | Release Date | Architecture | Best Val Loss | Macro F1 | Deployment Status | Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **WeightedRandomSampler** | `v1` | 2026-08-08 | Hierarchical ConvNeXt + Masked GAP | **0.1810** | **0.8090** | **Active Production SOTA** | `checkpoints/classification/multi_head/WeightedRandomSampler/v1/` |
| **LossOptimized** | `v1` | 2026-08-06 | Hierarchical ConvNeXt | 0.2084 | 0.5556 | Archived | `checkpoints/classification/multi_head/LossOptimized/v1/` |
| **Baseline** | `v1` | 2026-08-05 | Hierarchical ConvNeXt | 0.3609 | 0.5635 | Archived | `checkpoints/classification/multi_head/Baseline/v1/` |

---

## 2. Retinal Tissue Segmentation Models (`checkpoints/segmentation/`)

| Model | Version | Release Date | Architecture | Primary Metric | Validation Loss | Deployment Status | Canonical Suite Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: OCT5K Layers** | `v1` | 2026-07-23 | 6-Class Retinal Layer U-Net | mDice: **0.9452** | 0.0553 | **Active Production** | `models_suite/model1_oct5k_layers/checkpoints/best_model.pth` |
| **Model 2: Choroidalyzer** | `v1` | 2026-07-23 | Choroid Region & Thickness U-Net | Choroid Dice: **0.9610** | 0.0312 | **Active Production** | `models_suite/model2_choroidalyzer/checkpoints/best_model.pth` |
| **Model 3: HRF DME** | `v1` | 2026-07-23 | Fluid & Lesion Attention U-Net | Fluid Dice: **0.9380** | 0.0420 | **Active Production** | `models_suite/model3_hrf_dme/checkpoints/best_model.pth` |
| **Model 4: OIMHS** | `v1` | 2026-07-23 | Macular Hole & Cysts U-Net | Hole & Cyst Dice: **0.9701** | 0.0299 | **Active Production** | `models_suite/model4_oimhs_hole_cysts/checkpoints/best_model.pth` |

---

## 3. Pathology Detection Models (`checkpoints/detection/`)

| Model | Version | Release Date | Architecture | Primary Metric | Training Loss | Deployment Status | Canonical Suite Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 5: OCT5K Detector** | `v1` | 2026-07-23 | Faster R-CNN ResNet-50-FPN | mAP@0.5: **0.8650** | 0.1990 | **Active Production** | `models_suite/model5_oct5k_detection/checkpoints/best_model.pth` |

---

## Standardized Checkpoint Layout

All future model training runs must strictly follow the hierarchical folder structure:
```
checkpoints/
├── classification/<model_type>/<experiment_variant>/<version_tag>/
├── segmentation/<model_name>/<version_tag>/
└── detection/<model_name>/<version_tag>/
```
Each version folder must contain:
1. Model weights (`.pth`)
2. `version_metadata.md` (System specs, Git commit SHA, hyperparameters, and verified metrics)
3. Any evaluation artifacts (`eval_cache.pth`, `telemetry_summary.json`, HTML/PDF reports)
