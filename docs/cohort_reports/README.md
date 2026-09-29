# Clinical Cohort Evaluation Benchmark Reports

This registry catalogs the clinical evaluation benchmark reports, high-throughput cohort runs, and executive PDFs produced across the Optovue Solix 23-subject dataset (46 paired 3D OCT volumes).

---

## 📊 Benchmark Evolution & Model Milestones

| Date | Run Directory / Milestone | Model Architecture & Formulation | Scope | Peripapillary MABE | Dice Overlap | Optic Cup IoU | Formats |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **2026-09-28** | [**`2026-09-28_biplanar_expanded_18266075`**](./2026-09-28_biplanar_expanded_18266075/) | **Expanded 61-Subject Bi-Planar Orthogonal Heavy**<br>Dual-view fusion (horizontal + vertical), continuous 1D surface regression, base_channels=32 (~6.58M params), Corrected OS laterality, Surface-guided RPE clamping & Cup-reach boundary | 61 Subj / 122 Vol<br>*(8 Held-Out / 16 Vol)* | **OD: $2.59\,\mu\text{m}$ (Med)** / $3.01\,\mu\text{m}$ (Mean)<br>**OS: $2.70\,\mu\text{m}$ (Med)** / $3.20\,\mu\text{m}$ (Mean)<br>*(Val Med: $5.12\,\mu\text{m}$)* | **$0.9037$ (Med)**<br>$0.8931$ (Mean)<br>*(Val Med: $\mathbf{0.9481}$)* | **$0.9503$ (Med)**<br>$0.9463$ (Mean) | [Markdown](./2026-09-28_biplanar_expanded_18266075/executive_cohort_rnfl_report_biplanar.md) &bull; [PDF](./2026-09-28_biplanar_expanded_18266075/executive_cohort_rnfl_report_biplanar.pdf) |
| **2026-09-26** | [**`2026-09-26_biplanar_orthogonal_18223981`**](./2026-09-26_biplanar_orthogonal_18223981/) | **Bi-Planar Orthogonal Heavy**<br>Dual-view fusion (horizontal + vertical), Focal Tversky ($\alpha=0.3, \beta=0.7$) + ReLayNet Boundary BCE + Column Thickness Integral | 23 Subj / 46 Vol<br>*(3 Held-Out)* | **OD: $3.02 \pm 0.18\,\mu\text{m}$**<br>*(Validation OD: $4.62\,\mu\text{m}$)*<br>All: $54.47\,\mu\text{m}$ | **$0.8983$ (OD)**<br>$0.7872$ (All) | **$0.9179$ (Median)**<br>$0.8763$ (Mean) | [Markdown](./2026-09-26_biplanar_orthogonal_18223981/executive_cohort_rnfl_report_biplanar.md) &bull; [PDF](./2026-09-26_biplanar_orthogonal_18223981/executive_cohort_rnfl_report_biplanar.pdf) |
| **2026-09-21** | [**`2026-09-21_heavy_volumetric_18045386`**](./2026-09-21_heavy_volumetric_18045386/) | **2.5D ResNet Heavy** (`base_channels=32`, ~6.58M params)<br>Symmetric Dice + Boundary Loss + Peripapillary Weighting | 23 Subj / 46 Vol<br>*(3 Held-Out)* | $61.04 \pm 76.51\,\mu\text{m}$ | $0.7818 \pm 0.1264$ | $0.8521 \pm 0.1130$ | [Markdown](./2026-09-21_heavy_volumetric_18045386/executive_cohort_rnfl_report_heavy_18045386.md) &bull; [PDF](./2026-09-21_heavy_volumetric_18045386/executive_cohort_rnfl_report_heavy_18045386.pdf) |
| **2026-09-21** | [**`2026-09-21_light_volumetric_18043443`**](./2026-09-21_light_volumetric_18043443/) | **2.5D ResNet Light** (`base_channels=16`, ~1.65M params)<br>Symmetric Dice + Boundary Loss + Peripapillary Weighting | 23 Subj / 46 Vol<br>*(3 Held-Out)* | $62.47 \pm 78.92\,\mu\text{m}$ | $0.7801 \pm 0.1248$ | $0.8493 \pm 0.1154$ | [Markdown](./2026-09-21_light_volumetric_18043443/executive_cohort_rnfl_report_light_18043443.md) &bull; [PDF](./2026-09-21_light_volumetric_18043443/executive_cohort_rnfl_report_light_18043443.pdf) |
| **2026-09-21** | [**`2026-09-21_expanded_cohort_23subj`**](./2026-09-21_expanded_cohort_23subj/) | **Initial Expanded Solix Cohort Evaluation**<br>Baseline Volumetric 2.5D architecture across full 23 subjects | 23 Subj / 46 Vol | $61.04\,\mu\text{m}$ | $0.7818$ | $0.8521$ | [Markdown](./2026-09-21_expanded_cohort_23subj/executive_cohort_rnfl_report_23subj_20260921.md) &bull; [PDF](./2026-09-21_expanded_cohort_23subj/executive_cohort_rnfl_report_23subj_20260921.pdf) |
| **2026-09-14** | [**`2026-09-14_pilot_cohort_11subj`**](./2026-09-14_pilot_cohort_11subj/) | **Initial Pilot Cohort Evaluation**<br>Early 2.5D Volumetric Network on 11 subjects | 11 Subj / 22 Vol | $63.12\,\mu\text{m}$ | $0.7745$ | $0.8410$ | [Markdown](./2026-09-14_pilot_cohort_11subj/executive_cohort_rnfl_report_11subj_20260914.md) &bull; [PDF](./2026-09-14_pilot_cohort_11subj/executive_cohort_rnfl_report_11subj_20260914.pdf) |
| **2026-09-13** | [**`baseline_unet_vs_heuristic_report`**](./baseline_unet_vs_heuristic_report/) | **Heuristic Baseline vs 2D U-Net**<br>Commercial Solix algorithm vs early slice-level network | Representative B-scans | Variable | $0.7120$ | N/A | [Markdown](./baseline_unet_vs_heuristic_report/unet_vs_algorithm_segmentation_report.md) &bull; [PDF](./baseline_unet_vs_heuristic_report/unet_vs_algorithm_segmentation_report.pdf) |
| **Archive** | [**`consolidated_executive_archive`**](./consolidated_executive_archive/) | Consolidated milestone summaries & historic executive reports | Historical | $61.04\,\mu\text{m}$ | $0.7818$ | $0.8521$ | [Markdown](./consolidated_executive_archive/executive_cohort_rnfl_report_final.md) &bull; [PDF](./consolidated_executive_archive/executive_cohort_rnfl_report_final.pdf) |

---

## 📁 Directory Structure of Each Report

Each benchmark run is completely self-contained:
```
<run_directory>/
├── executive_cohort_rnfl_report_*.md    # Complete clinical markdown document
├── executive_cohort_rnfl_report_*.pdf   # Publication-ready executive PDF
└── assets/                              # High-resolution visual artifacts
    └── executive_cohort_report/
        ├── cohort_evaluation_metrics.json # Full per-scan metrics manifest
        ├── cohort_summary_chart.png     # Multi-panel statistical distribution
        ├── deep_dive_<subject>_OD.png   # 4-panel quadrant and profile deep dives
        └── gallery_bscan_<subject>_OD.png# Peripapillary B-scan overlays
```
