# Project To-Do List

*Last updated: 1st July, 2026*

---

## 🔴 High Priority

- [x] **Review Level 2 Router** — audit architecture, training protocol, and test script against industry standards (same methodology as the Level 1 review)
- [x] **Review Level 3 Specialists** — audit all 5 specialist models (Macular, Diabetic, Vascular, Fluid, Structural) individually
- [x] **Build end-to-end inference pipeline** — connect L1 → L2 → L3 into a single callable function that takes a raw OCT scan and returns a final diagnosis with confidence scores

---

## 🟡 Medium Priority

- [x] **Apply Level 1 improvements to Level 2 & Level 3** — once the L1 improvements are validated on the new checkpoint, port the same changes (CLAHE, checkpoint selection, TTA, Grad-CAM, calibration) to the router and specialist models
- [ ] **Run calibrate_level1.py** — after Level 1 retraining completes, derive the ROC threshold and temperature scalar from the validation fold
- [ ] **Review Grad-CAM outputs** — after running `test_level1_on_test_set.py --gradcam`, manually inspect `gradcam_false_neg.png` to confirm the model attends to retinal pathology and not scanner artifacts
- [ ] **Update training results in `training.md`** — replace the current 5-fold CV metrics with the new EfficientNet-B3 results once retraining is complete

---

## 🟢 Low Priority

- [ ] **Retrain Level 1** — run `train_level1.py` with all pre-training improvements active (EfficientNet-B3, CLAHE, checkpoint selection on `val_macro_f1`, scheduler `T_0=10`)
- [ ] **Increase batch size for Level 1** — EfficientNet-B3 uses ~30% less activation memory than ResNet-50 at the same batch size; consider increasing from 48 → 64 and update `config/hierarchy.yaml`
- [ ] **Write a project-level README** — a root-level `README.md` (outside of `Documentation/`) with setup instructions, environment requirements, and a quickstart guide for someone cloning the repo fresh
- [ ] **Add unit tests for transforms pipeline** — verify CLAHE output properties and that all transform chains produce the correct tensor shapes for each mode/resolution combination
- [ ] **Explore Vision Transformer (ViT-B/16)** — benchmark against EfficientNet-B3 on the Level 1 task; ViT models have shown strong performance on OCT in 2024–2025 literature, particularly for AMD detection

---

## ✅ Completed

- [x] Review Level 1 Gatekeeper architecture, training, and tests against 2024–2025 industry standards
- [x] Fix checkpoint selection: `val_loss` → `val_macro_f1`
- [x] Fix LR scheduler cycle: `T_0=20` → `T_0=10`
- [x] Swap Level 1 backbone: ResNet-50 → EfficientNet-B3
- [x] Add CLAHE preprocessing to all transform pipelines (train, heavy train, val)
- [x] Create `scripts/calibrate_level1.py` — ROC-derived threshold + temperature scaling
- [x] Update `scripts/test_level1_on_test_set.py` — sensitivity/specificity, TTA (5 views), Grad-CAM
- [x] Update `Documentation/architecture.md`, `training.md`, `data_pipeline.md`, `README.md`
- [x] Create `Documentation/improvements.md` — full rationale and changelog for all Level 1 improvements
- [x] Install `opencv-python-headless` and `torchcam` dependencies
