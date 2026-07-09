# OCT Analyzer Capstone - Project To-Do List

*Last updated: 8th July, 2026*

---

## 🔴 High Priority (Current Focus)

- [ ] **Finalize Multi-Head ConvNeXt V2 Model**
  - [x] Complete the dataset mapping, loss functions (BCE + CrossEntropy), and training loop in `ConvNeXtV2.ipynb`.
  - [ ] Generate dataset manifest (`dataset_manifest.csv`) from local Classified dataset.
  - [ ] Execute Apple Silicon (MPS) optimized Python training script (`train_convnext_mps.py`).
  - [ ] Validate performance across all heads (Normal/Abnormal, Pathology Category, Specific Sub-Type).
  - [ ] Export the final trained checkpoint (`multi_head.pth`) for inference.

- [x] **Integrate 15-Layer U-Net Segmentation**
  - Replace the deterministic 12-layer placeholder in the FastAPI backend (`backend/oct_analyzer/mvp_pipeline.py`).
  - Wire up the actual PyTorch inference using the newly developed 15-layer Hierarchical U-Net model from `OCT-Segmentation-Model/main.py`.

- [x] **Implement Live Classification Inference**
  - Replace the demo classification placeholders in the backend API.
  - Connect the backend to the live inference model (NOTE: currently falling back to random initialization due to missing weights).

---

## 🟡 Medium Priority (Infrastructure & Pipeline)

- [x] **Background Job Queue & Persistence**
  - Processing heavy 3D volumetric scans synchronously in FastAPI is prone to timeouts.
  - Implement a background job queue (e.g., Celery + Redis).
  - Update the upload endpoint to return a `task_id` and have the Next.js frontend poll for status.
  - Add database persistence to store scan states rather than relying on local filesystem/process memory.

- [x] **Legacy Ensemble Support (Optional)**
  - *If maintaining the old Hierarchical Ensemble instead of ConvNeXt:*
    - Review Level 3 Specialists (audit all 5 specialist models).
    - Port Level 1 improvements (CLAHE, TTA, calibration) to Level 2 and Level 3 models.
    - Run `calibrate_level1.py` to derive ROC threshold and temperature scalar.

---

## 🟢 Low Priority (Future Roadmap & Enterprise)

- [x] **Proprietary Archive Support**
  - Expand ingestion beyond standard `.dcm` and `.vol` to include proprietary formats (e.g., Solix `.fds` archives).
- [x] **Enterprise Capabilities**
  - Add PDF report generation for clinical exports.
  - Implement user authentication and HIPAA-compliant access controls.
  - Add audit-grade clinical logging.
- [x] **Clinical Validation**
  - Conduct formal clinical trials and robust accuracy evaluations (the app cannot be used for real patient diagnosis until this is complete).
