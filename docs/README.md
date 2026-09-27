# OCT-Analyser-Capstone Documentation

This directory contains the clinical documentation, cohort evaluation reports, architectural specifications, and implementation guides for the **OCT-Analyser-Capstone** clinical suite.

---

## 🗂️ Documentation Sections

### 1. [Clinical Cohort Benchmark Reports (`cohort_reports/`)](./cohort_reports/)
Executive-level evaluation reports and PDF benchmarks comparing deep-learning segmentation models against clinician reference standards and commercial Solix baselines across the 23-subject cohort (46 paired volumes).

- [**`2026-09-26_biplanar_orthogonal_18223981/`**](./cohort_reports/2026-09-26_biplanar_orthogonal_18223981/): **Bi-Planar Orthogonal Heavy Model (Current State-of-the-Art)**
  - Dual-view horizontal/vertical fusion, Focal Tversky ($\alpha=0.3, \beta=0.7$) + ReLayNet boundary BCE + Column thickness integral loss.
  - Achieved **$3.02\,\mu\text{m}$ MABE on OD acquisitions** and **$4.62\,\mu\text{m}$ MABE on held-out validation**.
  - Formats: [Markdown](./cohort_reports/2026-09-26_biplanar_orthogonal_18223981/executive_cohort_rnfl_report_biplanar.md) &bull; [Executive PDF](./cohort_reports/2026-09-26_biplanar_orthogonal_18223981/executive_cohort_rnfl_report_biplanar.pdf)
- [**`2026-09-21_heavy_volumetric_18045386/`**](./cohort_reports/2026-09-21_heavy_volumetric_18045386/): Volumetric 2.5D ResNet Heavy (`base_channels=32`, ~6.58M params; MABE $61.04\,\mu\text{m}$).
- [**`2026-09-21_light_volumetric_18043443/`**](./cohort_reports/2026-09-21_light_volumetric_18043443/): Volumetric 2.5D ResNet Light (`base_channels=16`, ~1.65M params; MABE $62.47\,\mu\text{m}$).
- [**`2026-09-21_expanded_cohort_23subj/`**](./cohort_reports/2026-09-21_expanded_cohort_23subj/): Expanded 23-subject Solix baseline benchmark.
- [**`2026-09-14_pilot_cohort_11subj/`**](./cohort_reports/2026-09-14_pilot_cohort_11subj/): Initial 11-subject pilot cohort benchmark.
- [**`baseline_unet_vs_heuristic_report/`**](./cohort_reports/baseline_unet_vs_heuristic_report/): Commercial Solix algorithm vs 2D U-Net benchmark.
- [**`consolidated_executive_archive/`**](./cohort_reports/consolidated_executive_archive/): Archived milestone executive reports and assets.

---

### 2. [Clinical Reference Guides (`clinical_guides/`)](./clinical_guides/)
Domain-specific ophthalmic literature and anatomical references:
- [**`retinal_layers_optic_neuritis_oct_guide.md`**](./clinical_guides/retinal_layers_optic_neuritis_oct_guide.md): Anatomical reference covering retinal layer boundaries (ILM, RNFL, GCL, IPL, INL, OPL, ONL, ELM, IS/OS, RPE, BM), thickness profiles, and specific biomarkers for Optic Neuritis.
- [**`biomarker_mapping/`**](./clinical_guides/biomarker_mapping/): Algorithmic definitions and coordinate mapping specs for neuro-ophthalmic biomarkers.

---

### 3. [Technical & Architectural Specifications (`technical_specs/`)](./technical_specs/)
Engineering design documents, preprocessing guidelines, and incident reviews:
- [**`hierarchical_classification_architecture.md`**](./technical_specs/hierarchical_classification_architecture.md): Hierarchical multi-stage classification pipeline for ONH and retinal abnormalities.
- [**`preprocessing_tuning_guide.md`**](./technical_specs/preprocessing_tuning_guide.md): Spatial geometry normalization, white-bar metadata masking, tissue cropping, and aspect ratio alignment.
- [**`oct_segmentation_dataset_search.md`**](./technical_specs/oct_segmentation_dataset_search.md): Systematic survey of public and clinical OCT segmentation datasets (DUKE, RETOUCH, AROI, Solix).
- [**`oct_layer_segmentation_validation_correction_release.md`**](./technical_specs/oct_layer_segmentation_validation_correction_release.md): Quality control validation protocols and manual clinician correction procedures.
- [**`implementation_info.md`**](./technical_specs/implementation_info.md): Core platform architectural specifications and dependencies.
- [**`h1_sensitivity_diagnostic_report.json`**](./technical_specs/h1_sensitivity_diagnostic_report.json): Diagnostic sensitivity analysis for hypothesis testing.
- [**`postmortems/`**](./technical_specs/postmortems/): Engineering incident reports, including [SAM & MedSAM Retinal Segmentation Failure Analysis](./technical_specs/postmortems/SAM_MEDSAM_RETINAL_SEGMENTATION_FAILURE.md).
