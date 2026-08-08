# OCT Analyzer Capstone - Project To-Do List

*Last updated: 8th July, 2026*

---

## 🔴 High Priority (Current Focus)

- [ ] **Finalize Multi-Head ConvNeXt V2 Model**
  - [x] Complete the dataset mapping, loss functions (BCE + CrossEntropy), and training loop in `ConvNeXtV2.ipynb`.
  - [x] Generate dataset manifest (`dataset_manifest.csv`) from local Classified dataset.
  - [x] Execute Apple Silicon (MPS) optimized Python training script (`train_convnext_mps.py`) [Smoke tests passed locally, full training offloaded to Kaggle].
  - [ ] Validate performance across all heads (Normal/Abnormal, Pathology Category, Specific Sub-Type) [Pending Kaggle execution].
  - [ ] Export the final trained checkpoint (`multi_head.pth`) for inference [Pending Kaggle execution].

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

- [x] **Column-Wise Vitreous-Moat White Bar Removal & Dataset Preprocessing**
  - Implemented Column-Wise Raycasting in `scripts/preprocess_dataset.py` to dynamically remove irregular, thick (35%+), or slanted/diagonal white header/footer bars without bleeding into retinal tissue.
  - Added bottom corner compass/logo zeroing, Otsu tissue boundary masking, and red evaluation dataset mode (`--highlight-red`).
  - Generated and validated full 88,804 image preprocessed dataset `Classified-preprocessed-R3`.

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

- [ ] **Future Deep Learning Architecture Enhancements**
  - **Graph-Search Correction (Anatomical Constraints):** Implement topological constraints (Viterbi/Dijkstra) to prevent impossible layer crossings, but MUST design soft-constraints to handle broken topology (Macular Holes), obliterated boundaries (CNV scars), extreme elevations (PEDs), shadow artifacts, and massive 3D volume latency.
  - **Uncertainty Head (Monte Carlo Dropout):** Add pixel-level confidence maps to prevent automation bias, but MUST engineer solutions for the N-pass latency penalty (30x slower inference), alert fatigue from aleatoric noise (subretinal fluid debris), calibration drift across different OCT scanners, and false confidence in OOD scans.
  - **Regression Head (Functional Outputs):** Add continuous metric predictions (retinal thickness, contrast sensitivity), but MUST resolve the structure-function biological disconnect, noisy subjective clinical ground truth labels, non-linear disease inflection points (thinning -> thickening -> thinning), and MSE loss function dominance destroying the shared classification encoder.
  - **Segmentation-Driven Cropping (UI/Artifact Defense):** Use the 15-layer Hierarchical U-Net segmentation model to explicitly isolate retinal tissue and black out UI artifacts (like scanner compasses or logos) before passing the image to the classifier. This prevents the classifier from learning spurious shortcuts based on scanner-specific UI overlays.
  - **Segmentation Bounding Box (Smart Padding):** Extend the segmentation pipeline to dynamically calculate the tightest bounding box around the retinal tissue, crop to that box, and letterbox it to a perfect square. This maximizes the biological resolution inside the 224x224 input tensor by eliminating empty background space, significantly boosting the classifier's spatial awareness.
  - **GradCAM Support:** Generating traditional classification GradCAMs requires rewriting the visualization logic for U-Net bottlenecks, as the unified model uses a single-channel segmentation network.
