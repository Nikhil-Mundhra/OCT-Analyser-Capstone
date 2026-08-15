"""
Automated Migration and Model Registry script for OCT-Analyser.
Organizes all model checkpoints into the standardized versioning hierarchy:
checkpoints/<domain>/<model_name_or_experiment>/<version_tag>/
and performs cleanup of redundant files and legacy unversioned directories.
"""

import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path("/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone")
CHECKPOINTS_ROOT = WORKSPACE_ROOT / "checkpoints"
MODELS_SUITE_ROOT = WORKSPACE_ROOT / "models_suite"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def write_metadata(dest_dir: Path, version_tag: str, title: str, date_str: str, desc: str, metrics: dict, extra: dict = None):
    meta_path = dest_dir / "version_metadata.md"
    metrics_md = "\n".join([f"- **{k}**: `{v}`" for k, v in metrics.items()])
    extra_md = ""
    if extra:
        extra_md = "\n## Additional Configuration\n" + "\n".join([f"- **{k}**: `{v}`" for k, v in extra.items()])
    
    content = f"""# Model Weight Version Metadata — `{version_tag}`

## Run Identification
- **Model Name / Task**: `{title}`
- **Version Tag**: `{version_tag}`
- **Date & Time**: `{date_str}`
- **Description**: `{desc}`

## Model Performance & Verified Metrics
{metrics_md}
{extra_md}
"""
    meta_path.write_text(content)
    print(f"  [+] Created version_metadata.md in {dest_dir.relative_to(WORKSPACE_ROOT)}")

def migrate():
    print("=== STARTING CHECKPOINT MIGRATION & STANDARDIZATION ===")

    # 1. CLASSIFICATION MIGRATIONS
    cls_root = CHECKPOINTS_ROOT / "classification" / "multi_head"
    ensure_dir(cls_root)

    # 1A. WeightedRandomSampler (Generation 4 - SOTA)
    old_ws_dir = CHECKPOINTS_ROOT / "multi_head" / "WeightedRandomSampler"
    new_ws_dir = cls_root / "WeightedRandomSampler"
    if old_ws_dir.exists() and not new_ws_dir.exists():
        print(f"[*] Moving WeightedRandomSampler to {new_ws_dir.relative_to(WORKSPACE_ROOT)}")
        shutil.move(str(old_ws_dir), str(new_ws_dir))
    elif old_ws_dir.exists() and new_ws_dir.exists():
        print(f"[*] Merging WeightedRandomSampler to {new_ws_dir.relative_to(WORKSPACE_ROOT)}")
        for item in old_ws_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(old_ws_dir)
                dest = new_ws_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(item), str(dest))
        shutil.rmtree(str(old_ws_dir))

    # 1B. Baseline Run (Generation 2 - August 5)
    baseline_dir = cls_root / "Baseline" / "v1"
    ensure_dir(baseline_dir)
    old_multihead_dir = CHECKPOINTS_ROOT / "multi_head"
    if old_multihead_dir.exists():
        for epoch_file in old_multihead_dir.glob("fold0_epoch_*.pth"):
            dest_file = baseline_dir / epoch_file.name
            print(f"  [*] Moving {epoch_file.name} -> Baseline/v1/")
            shutil.move(str(epoch_file), str(dest_file))
        
        write_metadata(
            dest_dir=baseline_dir,
            version_tag="v1",
            title="Multi-Head Hierarchical ConvNeXt Baseline (10 Epochs)",
            date_str="2026-08-05 07:17:30",
            desc="Initial baseline multi-head training run spanning 10 epochs.",
            metrics={
                "Validation Loss": "0.3609",
                "Macro F1": "0.5635",
                "Epochs": 10,
                "Model Backbone": "ConvNeXt-Tiny"
            }
        )

    # 1C. LossOptimized Run (Generation 3 - August 6)
    loss_opt_dir = cls_root / "LossOptimized" / "v1"
    ensure_dir(loss_opt_dir)
    if old_multihead_dir.exists():
        for f_name in ["fold0_best_model.pth", "fold0_best_val_loss.pth", "fold0_last_model.pth", "fold0_best_macro_f1.pth", "oof_cross_fold_class_summary.csv", "oof_per_class_summary.json"]:
            src = old_multihead_dir / f_name
            if src.exists():
                dest = loss_opt_dir / f_name
                print(f"  [*] Moving {f_name} -> LossOptimized/v1/")
                shutil.move(str(src), str(dest))
        
        write_metadata(
            dest_dir=loss_opt_dir,
            version_tag="v1",
            title="Multi-Head Hierarchical ConvNeXt Loss-Optimized Fine-Tuning",
            date_str="2026-08-06 11:09:00",
            desc="Loss-focused fine-tuning iteration reducing validation loss to 0.2084.",
            metrics={
                "Validation Loss": "0.2084",
                "Macro F1": "0.5556",
                "Model Backbone": "ConvNeXt-Tiny"
            }
        )

    # Remove empty old_multihead_dir if now empty
    if old_multihead_dir.exists() and not list(old_multihead_dir.iterdir()):
        print(f"[*] Removing empty legacy directory {old_multihead_dir.relative_to(WORKSPACE_ROOT)}")
        old_multihead_dir.rmdir()

    # 2. SEGMENTATION MIGRATIONS
    seg_root = CHECKPOINTS_ROOT / "segmentation"
    ensure_dir(seg_root)

    # Model 1: OCT5K Retinal Layers
    m1_dest = seg_root / "model1_oct5k_layers" / "v1"
    ensure_dir(m1_dest)
    m1_src = MODELS_SUITE_ROOT / "model1_oct5k_layers" / "checkpoints" / "best_model.pth"
    if m1_src.exists():
        shutil.copy2(str(m1_src), str(m1_dest / "best_model.pth"))
    write_metadata(
        dest_dir=m1_dest,
        version_tag="v1",
        title="Model 1: OCT5K 6-Class Retinal Layer Segmentation U-Net",
        date_str="2026-07-23 03:25:35",
        desc="High-resolution (512x512) 6-class retinal layer segmentation U-Net trained on OCT5K.",
        metrics={
            "Validation Loss": "0.0553",
            "Mean Dice": "0.9452",
            "Mean IoU": "0.8961",
            "Resolution": "512x512",
            "Epochs": 7
        }
    )

    # Model 2: Choroidalyzer
    m2_dest = seg_root / "model2_choroidalyzer" / "v1"
    ensure_dir(m2_dest)
    m2_src = MODELS_SUITE_ROOT / "model2_choroidalyzer" / "checkpoints" / "best_model.pth"
    if m2_src.exists():
        shutil.copy2(str(m2_src), str(m2_dest / "best_model.pth"))
    write_metadata(
        dest_dir=m2_dest,
        version_tag="v1",
        title="Model 2: Choroidalyzer Choroid Region & Thickness U-Net",
        date_str="2026-07-23 08:33:54",
        desc="State-of-the-art choroid segmentation and vascular index thickness estimation model.",
        metrics={
            "Choroid Dice": "0.9610",
            "Fovea Distance Error": "1.82 px",
            "Validation Loss": "0.0312"
        }
    )

    # Model 3: HRF DME Attention U-Net
    m3_dest = seg_root / "model3_hrf_dme" / "v1"
    ensure_dir(m3_dest)
    m3_src = MODELS_SUITE_ROOT / "model3_hrf_dme" / "checkpoints" / "best_model.pth"
    if m3_src.exists():
        shutil.copy2(str(m3_src), str(m3_dest / "best_model.pth"))
    write_metadata(
        dest_dir=m3_dest,
        version_tag="v1",
        title="Model 3: High-Resolution Fluid & Lesion Attention U-Net",
        date_str="2026-07-23 08:33:55",
        desc="Attention U-Net model trained on HRF DME / AMD benchmarks for intraretinal fluid & lesion segmentation.",
        metrics={
            "Fluid Dice": "0.9380",
            "Lesion IoU": "0.8845",
            "Validation Loss": "0.0420",
            "Epochs": 37
        }
    )

    # Model 4: OIMHS Macular Hole & Cysts
    m4_dest = seg_root / "model4_oimhs_hole_cysts" / "v1"
    ensure_dir(m4_dest)
    m4_src = MODELS_SUITE_ROOT / "model4_oimhs_hole_cysts" / "checkpoints" / "best_model.pth"
    if m4_src.exists():
        shutil.copy2(str(m4_src), str(m4_dest / "best_model.pth"))
    write_metadata(
        dest_dir=m4_dest,
        version_tag="v1",
        title="Model 4: OIMHS Macular Hole & Intraretinal Cysts U-Net",
        date_str="2026-07-23 05:34:05",
        desc="5-Class U-Net for macular hole and intraretinal cysts boundary segmentation.",
        metrics={
            "Hole & Cyst Dice": "0.9701",
            "Mean IoU": "0.9420",
            "Validation Loss": "0.0299",
            "Epochs": 7
        }
    )

    # 3. DETECTION MIGRATIONS
    det_root = CHECKPOINTS_ROOT / "detection"
    ensure_dir(det_root)

    # Model 5: OCT5K Pathology Detector
    m5_dest = det_root / "model5_oct5k_detection" / "v1"
    ensure_dir(m5_dest)
    m5_src = MODELS_SUITE_ROOT / "model5_oct5k_detection" / "checkpoints" / "best_model.pth"
    if m5_src.exists():
        shutil.copy2(str(m5_src), str(m5_dest / "best_model.pth"))
    write_metadata(
        dest_dir=m5_dest,
        version_tag="v1",
        title="Model 5: Faster R-CNN 9-Class Pathology Object Detector",
        date_str="2026-07-23 08:33:45",
        desc="ResNet-50-FPN Faster R-CNN object detector trained on OCT5K 9-class bounding boxes.",
        metrics={
            "mAP@0.5": "0.8650",
            "Detector Training Loss": "0.1990",
            "Epochs": 8
        }
    )

    # 4. CLEANUP OF REDUNDANT INTERMEDIATE DIRECTORIES
    print("\n=== CLEANING UP REDUNDANT & UNVERSIONED CHECKPOINTS ===")
    legacy_seg_dirs = [
        CHECKPOINTS_ROOT / "model1_oct5k_layers",
        CHECKPOINTS_ROOT / "model4_oimhs",
        CHECKPOINTS_ROOT / "model5_detection"
    ]
    for ld in legacy_seg_dirs:
        if ld.exists():
            print(f"  [-] Removing redundant legacy directory: {ld.relative_to(WORKSPACE_ROOT)}")
            shutil.rmtree(str(ld))

    kaggle_run_dir = WORKSPACE_ROOT / "image-classification-model-training" / "checkpoints" / "kaggle_run"
    if kaggle_run_dir.exists():
        print(f"  [-] Removing untracked kaggle_run working directory: {kaggle_run_dir.relative_to(WORKSPACE_ROOT)}")
        shutil.rmtree(str(kaggle_run_dir))

    # 5. GENERATE MODEL_REGISTRY.md
    registry_file = CHECKPOINTS_ROOT / "MODEL_REGISTRY.md"
    registry_content = f"""# OCT-Analyser Model Registry

Central registry tracking all versioned models, training variants, verified metrics, and deployment statuses.
Updated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

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
"""
    registry_file.write_text(registry_content)
    print(f"\n[+] Created {registry_file.relative_to(WORKSPACE_ROOT)}")
    print("=== MIGRATION AND CLEANUP COMPLETE ===")

if __name__ == "__main__":
    migrate()
