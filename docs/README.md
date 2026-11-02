# OCT-Analyser-Capstone Documentation

This directory contains the clinical documentation, multi-architecture cohort evaluation benchmarks, technical specifications, and clinical reference guides for the **OCT-Analyser-Capstone** suite.

---

## 🗂️ Documentation Sections

### 1. [Clinical Cohort Benchmark Reports (`cohort_reports/`)](./cohort_reports/)
Executive-level evaluation reports, multi-model research benchmarks, and publication-ready PDFs comparing frontier deep-learning segmentation architectures against clinician-audited ground truth and commercial Optovue Solix baselines on the zero-leakage held-out validation cohort (`stratified_held_out_v2.json`, $N = 20$ subjects / $40$ paired OD/OS 3D volumes).

#### 🏆 Cross-Architecture Comparative Benchmarks
- [**`orthogonal_fractional_context_benchmark/`**](./cohort_reports/orthogonal_fractional_context_benchmark/): **Orthogonal Fractional Volumetric Context (`2.3D` / `2.5D` / `2.7D` / `2.9D`) & Multi-Horizon Spatially-Adaptive Router (Current Overall SOTA)**
  - Evaluates `OrthogonalContextRNFLNet` with a 3D depth-collapsing stem (`SliceContextFusion`) across $K \in \{3, 5, 7, 9\}$ slice contexts and fuses all 4 horizons via `MultiHorizonEnsemblePredictor` (`18981209`).
  - Achieved **$4.60\,\mu\text{m}$ median MABE** ($6.06\,\mu\text{m}$ mean), **$0.9533$ median Volumetric Dice** ($0.9457$ mean), **$0.9448$ mean Optic Cup IoU**, and **$19.88\,\mu\text{m}$ mean $P_{95}$ error** on the human-audited tier.
  - Formats: [Markdown](./cohort_reports/orthogonal_fractional_context_benchmark/research_report_orthogonal_fractional_context_benchmark.md) &bull; [Research PDF](./cohort_reports/orthogonal_fractional_context_benchmark/research_report_orthogonal_fractional_context_benchmark.pdf)
- [**`standardized_iso_capacity_comparison/`**](./cohort_reports/standardized_iso_capacity_comparison/): **Standardized Iso-Capacity Multi-Architecture Benchmark (Bi-Planar 2.5D vs. Dense 3D U-Net vs. TransUNet 20M)**
  - Benchmarks **Bi-Planar 2.5D (`18563914`, 6.58M)** against **Standardized Dense 3D U-Net (`18770720`, 20.30M)**, **Dense 3D Pure-Voxel Baseline (`18574378`, 20.30M)**, **TransUNet 5.1M (`18724310`)**, and **TransUNet Balanced Hybrid 20M (`18777225_ep12`, 19.97M)** under the standardized 5-head multi-task contract.
  - Establishes Bi-Planar 2.5D Pareto dominance ($4.71\,\mu\text{m}$ median MABE, $0.9534$ Surface Dice / $0.9518$ Raw Voxel Dice) and documents monotonic TransUNet scaling ($9.34 \to 8.27 \to 7.48\,\mu\text{m}$ median MABE; $15.15\,\mu\text{m}$ max outlier bound).
  - Formats: [Markdown](./cohort_reports/standardized_iso_capacity_comparison/research_report_standardized_iso_capacity_comparison.md) &bull; [Research PDF](./cohort_reports/standardized_iso_capacity_comparison/research_report_standardized_iso_capacity_comparison.pdf)
- [**`three_way_architecture_comparison/`**](./cohort_reports/three_way_architecture_comparison/) & [**`three_way_standardized_architecture_comparison/`**](./cohort_reports/three_way_standardized_architecture_comparison/): **Three-Way Paradigm Synthesis (Bi-Planar 2.5D, Dense 3D, TransUNet)**
  - Concise tri-model benchmark reports and B-scan contour comparisons (`BEH0314 OD`, B-scan 160).
  - Formats: [Iso-Capacity Markdown](./cohort_reports/three_way_architecture_comparison/research_report_tri_model_comparison.md) &bull; [Iso-Capacity PDF](./cohort_reports/three_way_architecture_comparison/research_report_tri_model_comparison.pdf) &bull; [Standardized 5M Markdown](./cohort_reports/three_way_standardized_architecture_comparison/research_report_tri_model_standardized_comparison.md) &bull; [Standardized 5M PDF](./cohort_reports/three_way_standardized_architecture_comparison/research_report_tri_model_standardized_comparison.pdf)
- [**`multi_model_validation_benchmark/`**](./cohort_reports/multi_model_validation_benchmark/): **5-Model Progression Benchmark on Stratified Held-Out V2 (`stratified_held_out_v2.json`)**
  - Compares Unilateral 2.5D Light (`18043443`), Unilateral 2.5D Heavy (`18045386`), Bi-Planar 23-Subj (`18223981`), Bi-Planar 61-Subj Zero-Leakage (`18732241`), and Bi-Planar V2 Held-Out (`18563914`).
  - Formats: [Markdown](./cohort_reports/multi_model_validation_benchmark/multi_model_validation_benchmark_report.md) &bull; [PDF](./cohort_reports/multi_model_validation_benchmark/multi_model_validation_benchmark_report.pdf)
- [**`3d_vs_biplanar_comparison/`**](./cohort_reports/3d_vs_biplanar_comparison/): **Dense Anisotropic 3D U-Net vs. Bi-Planar 2.5D Direct Comparison**
  - Formats: [Markdown](./cohort_reports/3d_vs_biplanar_comparison/research_report_3d_vs_biplanar_comparison.md) &bull; [PDF](./cohort_reports/3d_vs_biplanar_comparison/research_report_3d_vs_biplanar_comparison.pdf)

#### 📊 Single-Model Executive Cohort Reports
- [**`2026-10-10_biplanar_61subj_18732241/`**](./cohort_reports/2026-10-10_biplanar_61subj_18732241/): Retrained Zero-Leakage 61-Subject Bi-Planar Orthogonal Heavy (4x V100 DDP FP32-Loss Fix; $5.34\,\mu\text{m}$ median MABE, $0.9390$ median Dice, $+61.3\%$ audit correction gain).
- [**`2026-10-10_transunet_hybrid_18724310/`**](./cohort_reports/2026-10-10_transunet_hybrid_18724310/): Anisotropic TransUNet Hybrid Baseline (5.13M params, 4x V100 DDP; $9.34\,\mu\text{m}$ median MABE, $0.9138$ median Dice, $+44.3\%$ audit correction gain).
- [**`2026-10-09_biplanar_61subj_18724259/`**](./cohort_reports/2026-10-09_biplanar_61subj_18724259/): Pre-fix 4x V100 FP16 overflow diagnostic run ($124.41\,\mu\text{m}$ MABE; documented in V100 FP16 Postmortem).
- [**`2026-10-08_transunet_hybrid_18710145/`**](./cohort_reports/2026-10-08_transunet_hybrid_18710145/): Initial 2x V100 resumed TransUNet Hybrid exploratory run ($31.90\,\mu\text{m}$ median MABE).
- [**`2026-10-07_canonical_robust_18697275/`**](./cohort_reports/2026-10-07_canonical_robust_18697275/): Canonical Robust Bi-Planar with Constrained STN Canonicalizer & Physics-Informed Optical Noise ($9.77\,\mu\text{m}$ median MABE, $0.8591$ median Dice).
- [**`2026-10-04_rnfl_3d_expanded_18574378/`**](./cohort_reports/2026-10-04_rnfl_3d_expanded_18574378/): Dense Anisotropic 3D U-Net Baseline (`18574378`, ~20.30M params; **$5.50\,\mu\text{m}$ median MABE**, **$0.9471$ median Dice**).
- [**`2026-10-03_biplanar_expanded_18563914/`**](./cohort_reports/2026-10-03_biplanar_expanded_18563914/): **Lead Single-Model Production Checkpoint — Bi-Planar Orthogonal 2.5D Heavy (`18563914`, 6.58M params)** (**$4.71\,\mu\text{m}$ median MABE**, **$0.9518$ median Dice** on 40 held-out volumes).
- [**`2026-09-28_biplanar_expanded_18266075/`**](./cohort_reports/2026-09-28_biplanar_expanded_18266075/): Expanded 61-Subject Bi-Planar Orthogonal Heavy (OD $2.59\,\mu\text{m}$ / OS $2.70\,\mu\text{m}$ pooled median MABE; $5.12\,\mu\text{m}$ validation MABE).
- [**`2026-09-26_biplanar_orthogonal_18223981/`**](./cohort_reports/2026-09-26_biplanar_orthogonal_18223981/): Initial 23-Subject Bi-Planar Orthogonal Heavy ($3.02\,\mu\text{m}$ OD MABE, $4.62\,\mu\text{m}$ held-out validation MABE).
- [**`2026-09-21_heavy_volumetric_18045386/`**](./cohort_reports/2026-09-21_heavy_volumetric_18045386/) & [**`2026-09-21_light_volumetric_18043443/`**](./cohort_reports/2026-09-21_light_volumetric_18043443/): Historical unilateral 2.5D ResNet Heavy (~6.58M params) and Light (~1.65M params) baselines.
- [**`2026-09-21_expanded_cohort_23subj/`**](./cohort_reports/2026-09-21_expanded_cohort_23subj/), [**`2026-09-14_pilot_cohort_11subj/`**](./cohort_reports/2026-09-14_pilot_cohort_11subj/), [**`baseline_unet_vs_heuristic_report/`**](./cohort_reports/baseline_unet_vs_heuristic_report/), & [**`consolidated_executive_archive/`**](./cohort_reports/consolidated_executive_archive/): Early pilot and baseline cohort archives.

---

### 2. [Clinical Reference Guides (`clinical_guides/`)](./clinical_guides/)
Domain-specific ophthalmic literature and anatomical references:
- [**`retinal_layers_optic_neuritis_oct_guide.md`**](./clinical_guides/retinal_layers_optic_neuritis_oct_guide.md): Anatomical reference covering retinal layer boundaries (ILM, RNFL, GCL, IPL, INL, OPL, ONL, ELM, IS/OS, RPE, BM), thickness profiles, and specific biomarkers for Optic Neuritis.
- [**`biomarker_mapping/`**](./clinical_guides/biomarker_mapping/): Algorithmic definitions and coordinate mapping specs for neuro-ophthalmic biomarkers.

---

### 3. [Technical & Architectural Specifications (`technical_specs/`)](./technical_specs/)
Engineering design documents, architectural decision records (ADRs), preprocessing guidelines, and incident postmortems:
- [**`transunet_20m_balanced_hybrid_architecture_decision.md`**](./technical_specs/transunet_20m_balanced_hybrid_architecture_decision.md): ADR justifying the Balanced Hybrid 20M TransUNet design (25% CNN pyramid `64-128-256` / 75% ViT-8 bottleneck `d=384, mlp=1536`, 19.98M params) over ViT-only scaling to prevent native-resolution stem starvation.
- [**`dense_3d_multitask_standardization_finding.md`**](./technical_specs/dense_3d_multitask_standardization_finding.md): Mechanistic analysis of 5-head multi-task standardization in `AnisotropicRNFLUNet3D` (`18770720` vs. `18574378`), documenting cross-objective gradient interference between physical-micron surface regression and 3D voxel logits.
- [**`hierarchical_classification_architecture.md`**](./technical_specs/hierarchical_classification_architecture.md): Mathematical formulation and calibration on $N = 17,761$ validation scans for the Multi-Head ConvNeXt V2 classifier with a Calibrated Tri-State Clinical Triage Layer and Free-Energy Open-Set OOD gate.
- [**`preprocessing_tuning_guide.md`**](./technical_specs/preprocessing_tuning_guide.md): Coupled multi-surface graph optimization, continuous B-spline regularization, vitreous-moat banner blanking, and interactive tuning server guide.
- [**`oct_segmentation_dataset_search.md`**](./technical_specs/oct_segmentation_dataset_search.md): Survey of public and clinical OCT segmentation datasets (DUKE, RETOUCH, AROI, OCT5K, Solix).
- [**`oct_layer_segmentation_validation_correction_release.md`**](./technical_specs/oct_layer_segmentation_validation_correction_release.md): Quality control validation protocols and manual clinician correction procedures.
- [**`implementation_info.md`**](./technical_specs/implementation_info.md): Full-stack platform ingestion and architecture blueprint.
- [**`h1_sensitivity_diagnostic_report.json`**](./technical_specs/h1_sensitivity_diagnostic_report.json): Diagnostic sensitivity analysis for $H_1$ conditioning in hierarchical classification.
- [**`postmortems/`**](./technical_specs/postmortems/): Engineering incident and architectural postmortems:
  - [**`SAM_MEDSAM_RETINAL_SEGMENTATION_FAILURE.md`**](./technical_specs/postmortems/SAM_MEDSAM_RETINAL_SEGMENTATION_FAILURE.md): Failure analysis and permanent deprecation of zero-shot promptable foundation models (Meta SAM ViT-B & MedSAM 2 Hiera-T) on stratified retinal OCT B-scans.
  - [**`V100_VOLTA_FP16_OVERFLOW_POSTMORTEM.md`**](./technical_specs/postmortems/V100_VOLTA_FP16_OVERFLOW_POSTMORTEM.md): Root-cause analysis of IEEE 754 `float16` accumulator overflow ($>65,504$) in volumetric Tversky/Dice loss on NVIDIA V100 GPUs and the Dual-Tier FP32 loss upcasting fix.

