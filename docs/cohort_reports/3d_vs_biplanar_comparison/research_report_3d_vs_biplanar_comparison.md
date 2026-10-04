<style>
@import url('https://fonts.googleapis.com/css2?family=Carlito:ital,wght@0,400;0,700;1,400;1,700&display=swap');

@page {
    margin: 0.38in 0.44in 0.38in 0.44in !important;
}
body {
    font-family: 'Calibri', 'Carlito', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 10.8px !important;
    line-height: 1.28 !important;
    color: #1e293b !important;
}
h1 {
    font-family: 'Calibri', 'Carlito', sans-serif !important;
    font-size: 16.5px !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
    margin-bottom: 3px !important;
    padding-bottom: 2px !important;
    color: #0969da !important;
    border-bottom: 1.5px solid #0969da !important;
}
h2 {
    font-family: 'Calibri', 'Carlito', sans-serif !important;
    font-size: 12.5px !important;
    font-weight: 700 !important;
    margin-top: 6px !important;
    margin-bottom: 3px !important;
    padding-bottom: 2px !important;
    color: #166534 !important;
    border-bottom: 1px solid #dcfce7 !important;
}
h3 {
    font-family: 'Calibri', 'Carlito', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    margin-top: 5px !important;
    margin-bottom: 2px !important;
    color: #0f172a !important;
}
p, ul, ol {
    margin-top: 2px !important;
    margin-bottom: 3px !important;
}
li {
    margin-bottom: 2px !important;
}
table {
    margin: 4px 0 !important;
    font-size: 9.5px !important;
    line-height: 1.22 !important;
}
th, td {
    padding: 3px 5px !important;
}
th {
    background-color: #f1f5f9 !important;
}
hr {
    margin: 5px 0 !important;
    height: 1px !important;
    background-color: #e2e8f0 !important;
}
.markdown-alert {
    padding: 5px 8px !important;
    margin: 4px 0 !important;
    border-radius: 4px !important;
}
.markdown-alert-title {
    font-size: 11px !important;
    margin-bottom: 2px !important;
}
/* Visual Hero KPI Grid */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 7px;
    margin: 6px 0 8px 0;
}
.kpi-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 6px 8px;
    border-left: 3.5px solid #0969da;
}
.kpi-box.purple { border-left-color: #7c3aed; }
.kpi-box.emerald { border-left-color: #059669; }
.kpi-box.rose { border-left-color: #e11d48; }
.kpi-box.amber { border-left-color: #d97706; }
.kpi-label {
    font-size: 8.5px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #64748b;
    margin-bottom: 1px;
}
.kpi-value {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 1px;
}
.kpi-desc {
    font-size: 8px;
    color: #475569;
    line-height: 1.15;
}
/* Figures */
.fig-row {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 8px;
    align-items: center;
    margin: 4px 0;
}
.report-fig-half {
    max-height: 220px !important;
    max-width: 100% !important;
    height: auto !important;
    margin: 2px auto !important;
    display: block !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
}
.report-fig-wide-banner {
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    margin: 2px 0 3px 0 !important;
    display: block !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
}
.report-fig-stacked {
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    margin: 2px 0 !important;
    display: block !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
}
.two-col-radar {
    display: grid;
    grid-template-columns: 1.05fr 0.95fr;
    gap: 12px;
    align-items: start;
    margin-top: 3px;
}
.radar-left {
    display: flex;
    flex-direction: column;
}
.report-fig-radar {
    max-height: 250px !important;
    width: 100% !important;
    object-fit: contain !important;
    margin: 0 auto !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
}
.radar-right {
    display: flex;
    flex-direction: column;
    gap: 5px;
}
.radar-callout {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 5px 7px;
}
.radar-callout.emerald {
    border-left: 3.5px solid #059669;
}
.radar-callout.purple {
    border-left: 3.5px solid #7c3aed;
}
.radar-callout h4 {
    font-family: 'Calibri', 'Carlito', sans-serif !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
    margin-bottom: 2px !important;
}
.radar-callout.emerald h4 { color: #065f46 !important; }
.radar-callout.purple h4 { color: #5b21b6 !important; }
.radar-callout p {
    margin: 0 !important;
    font-size: 8.5px !important;
    line-height: 1.22 !important;
    color: #334155 !important;
}
.report-fig-wide {
    max-height: 195px !important;
    max-width: 98% !important;
    height: auto !important;
    margin: 3px auto !important;
    display: block !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
}
.fig-caption {
    font-size: 8px !important;
    line-height: 1.2 !important;
    padding: 3px 6px !important;
    margin: 2px 0 4px 0 !important;
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
}
.meta-banner {
    font-size: 9px !important;
    color: #475569 !important;
    margin-bottom: 3px !important;
    line-height: 1.2 !important;
}
.two-col-grid {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
    gap: 14px !important;
    align-items: start !important;
    margin: 5px 0 7px 0 !important;
}
.column-panel {
    min-width: 0 !important;
    padding: 7px 9px !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 5px !important;
    background: #fbfdff !important;
    break-inside: avoid !important;
}
.column-panel h2 {
    margin-top: 0 !important;
}
.column-panel h3 {
    margin-top: 4px !important;
}
.column-panel table {
    width: 100% !important;
    table-layout: fixed !important;
    font-size: 8.8px !important;
}
.column-panel th,
.column-panel td {
    overflow-wrap: anywhere !important;
    vertical-align: top !important;
}
.column-panel ul,
.column-panel ol {
    padding-left: 18px !important;
}
.decision-box {
    border-left: 4px solid #059669 !important;
    background: #ecfdf5 !important;
    padding: 6px 8px !important;
    border-radius: 4px !important;
    margin: 4px 0 !important;
}
.caution-box {
    border-left: 4px solid #d97706 !important;
    background: #fffbeb !important;
    padding: 6px 8px !important;
    border-radius: 4px !important;
    margin: 4px 0 !important;
}
.source-strip {
    margin-top: 6px !important;
    padding: 6px 8px !important;
    border-top: 1px solid #cbd5e1 !important;
    background: #f8fafc !important;
    font-size: 8.5px !important;
}
</style>

# Paired Validation Study: Dense 3D vs. Bi-Planar RNFL Segmentation

<div class="meta-banner">
<strong>Protocol:</strong> Optovue Solix Peripapillary Retinal Nerve Fiber Layer (RNFL) Volumetric Segmentation<br/>
<strong>Validation Cohort:</strong> <code>deidentified-new</code>, 20 subject-disjoint participants and 40 paired OD/OS volumes | <strong>Evaluation:</strong> Native OS restoration, dynamic BMO cup tracking, identical mutually held-out subjects<br/>
<strong>Models:</strong> Dense Anisotropic 3D U-Net (Job <code>18574378</code>) and Bi-Planar 2.5D Orthogonal Fusion (Job <code>18563914</code>) | <strong>Report date:</strong> 4 October 2026
</div>

<div class="kpi-container">
  <div class="kpi-box emerald">
    <div class="kpi-label">Primary Research Candidate</div>
    <div class="kpi-value">Bi-Planar</div>
    <div class="kpi-desc">Lower MABE on 34/40 volumes and stronger performance on the human-corrected tier.</div>
  </div>
  <div class="kpi-box purple">
    <div class="kpi-label">Corrected-Eye MABE</div>
    <div class="kpi-value">5.13 µm</div>
    <div class="kpi-desc">Bi-Planar versus 6.12 µm for 3D; Bi-Planar was lower on all 10 corrected eyes.</div>
  </div>
  <div class="kpi-box rose">
    <div class="kpi-label">Observed Maximum MABE</div>
    <div class="kpi-value">17.02 µm</div>
    <div class="kpi-desc">3D versus 39.64 µm for Bi-Planar; this advantage arose from one severe Bi-Planar failure.</div>
  </div>
  <div class="kpi-box amber">
    <div class="kpi-label">Evidence Status</div>
    <div class="kpi-value">Validation</div>
    <div class="kpi-desc">Both checkpoints were selected using this cohort; an untouched test cohort is still required.</div>
  </div>
</div>

> [!IMPORTANT]
> **Evidence-based conclusion**<br/>
> The current validation evidence favors **Bi-Planar as the primary accuracy model**. It produced lower typical boundary error, better BMO cup overlap, and lower MABE on every eye with an expert-corrected reference. The 3D model avoided one severe Bi-Planar failure and therefore had a lower observed maximum error, but this single event does not establish general outlier immunity. Neither model has been validated for autonomous diagnosis or clinical deployment.

---

<div class="two-col-grid" markdown="1">
<div class="column-panel" markdown="1">

## 1. Study Design and Evidence Hierarchy

Both architectures were trained on the same 119-subject Phase 1 cohort and evaluated on the same frozen 20-subject validation cohort. The split was performed at subject level: OD, OS, and repeat acquisitions from one participant remained in a single partition. No validation subject appeared in the Phase 1 training manifest.

The validation cohort comprised two reference tiers:

1. **Human-audited subjects:** 6 subjects, including 10 eyes with paired pre-correction and expert-corrected boundaries. These corrected eyes provide the strongest available evidence for agreement with expert review.
2. **Accepted-as-segmented subjects:** 14 subjects whose commercial segmentation was accepted without a recorded manual edit. These references are useful for broad morphology assessment but may preserve commercial algorithm bias.

All 40 volumes were evaluated with native coordinate restoration for OS eyes and the same cup-reach post-processing. Comparisons are paired by subject and eye. Complete-cohort uncertainty was estimated by resampling subjects rather than treating bilateral eyes as independent. Analyses are exploratory and were not adjusted for multiple comparisons.

> [!WARNING]
> **This is not an untouched test set.** Bi-Planar checkpoint selection optimized validation peripapillary MABE, whereas 3D checkpoint selection optimized validation Dice. The resulting comparison is appropriate for selecting a research candidate, but it cannot support a definitive clinical superiority claim.

</div>
<div class="column-panel" markdown="1">

## 2. Architecture and Resource Profile

| Characteristic | Dense Anisotropic 3D U-Net | Bi-Planar 2.5D Orthogonal Fusion |
| :--- | :--- | :--- |
| **Model** | `AnisotropicRNFLUNet3D` | `VolumetricRNFLNet` |
| **Parameters** | Approximately **20.90M** | Approximately **6.58M** |
| **Input context** | Sliding-window $64 \times 768 \times 64$ volumetric patches | Five-slice horizontal and vertical context with orthogonal fusion |
| **Training objective** | Binary cross-entropy plus soft Dice | Tversky/overlap, boundary regression, thickness consistency, and optical-edge alignment |
| **Inference** | Multiple overlapping 3D patches with Gaussian blending | Horizontal and vertical passes followed by fusion |
| **Expected advantage** | Native volumetric context | Explicit continuous boundary localization and lower computational footprint |

The architecture table describes engineering differences; it is not evidence of clinical superiority. Runtime, throughput, and peak memory were not measured under a controlled common inference protocol and are therefore not ranked here.

</div>
</div>

<!-- pagebreak -->

<div class="two-col-grid" markdown="1">
<div class="column-panel" markdown="1">

## 3. Complete-Cohort Paired Results

| Endpoint | Dense 3D | Bi-Planar |
| :--- | :---: | :---: |
| **Mean Dice ± SD** | $0.9446 \pm 0.0152$ | $0.9441 \pm 0.0309$ |
| **Higher-Dice volumes** | 12/40 | **28/40** |
| **Median MABE** | $5.50\,\mu\text{m}$ | **$4.71\,\mu\text{m}$** |
| **Mean MABE ± SD** | $5.87 \pm 2.13\,\mu\text{m}$ | $6.22 \pm 5.91\,\mu\text{m}$ |
| **Median $P_{95}$** | $18.93\,\mu\text{m}$ | **$16.85\,\mu\text{m}$** |
| **Mean cup IoU ± SD** | $0.9085 \pm 0.0464$ | **$0.9388 \pm 0.0204$** |
| **Maximum MABE** | **$17.02\,\mu\text{m}$** | $39.64\,\mu\text{m}$ |

**Paired interpretation**

- Dice means were effectively tied: difference $+0.0004$, subject-bootstrap 95% CI $[-0.0054,\ 0.0098]$, paired $t$-test $p=0.921$.
- Bi-Planar had lower MABE on 34/40 volumes and lower $P_{95}$ on 29/40.
- Cup IoU favored Bi-Planar: difference $-0.0304$ for 3D minus Bi-Planar, 95% CI $[-0.0518,\ -0.0130]$, $p<0.001$.
- The untrimmed MABE mean is dominated by `BEH0290 OD`; interpret it with the medians and sensitivity analysis.

</div>
<div class="column-panel" markdown="1">

## 4. Human-Corrected Reference Tier

Ten eyes from six subjects had a paired `bad` curve and expert-corrected `good` curve. This tier is clinically more informative than references accepted without editing.

| Endpoint | Dense 3D | Bi-Planar |
| :--- | :---: | :---: |
| **Mean Dice** | $0.9441$ | **$0.9485$** |
| **Mean MABE** | $6.12\,\mu\text{m}$ | **$5.13\,\mu\text{m}$** |
| **Mean $P_{95}$** | $22.26\,\mu\text{m}$ | **$18.12\,\mu\text{m}$** |
| **Mean cup IoU** | $0.8967$ | **$0.9407$** |

**Directional results**

- Bi-Planar had lower MABE on **10/10 corrected eyes** and all six audited subjects.
- Bi-Planar had lower $P_{95}$ on 9/10 eyes and higher Dice on 8/10.
- Subject-level MABE difference: $+0.94\,\mu\text{m}$ for 3D minus Bi-Planar; exact paired Wilcoxon $p=0.031$.

<div class="decision-box">
<strong>Selection signal:</strong> The corrected tier is small but directionally consistent. It provides the strongest available support for choosing Bi-Planar as the primary research model.
</div>

</div>
</div>

---

<!-- pagebreak -->

## 5. Sensitivity and Observed Failure Analysis

<img src="assets/model_comparison_3d_vs_biplanar_light_pdf.jpg" class="report-fig-wide-banner" alt="Paired model comparison showing MABE distributions, P95 error, and scan-level differences" />

<div class="fig-caption">
<strong>Figure 1: Paired error distributions across 40 validation volumes.</strong> The raw cohort mean is strongly influenced by the Bi-Planar failure on <code>BEH0290 OD</code>. Most scan-level differences favor Bi-Planar, while 3D provides a large advantage on that single case. The figure is descriptive; it does not establish a population-level maximum-error guarantee.
</div>

<div class="two-col-grid" markdown="1">
<div class="column-panel" markdown="1">

### BEH0290 OD

Bi-Planar reached $39.64\,\mu\text{m}$ MABE and $95.53\,\mu\text{m}$ $P_{95}$, whereas 3D reached $4.84\,\mu\text{m}$ MABE and $17.39\,\mu\text{m}$ $P_{95}$. This is a clinically important observed failure of the Bi-Planar pipeline and motivates explicit disagreement-based QC.

### Outlier sensitivity

With all 40 volumes, mean MABE was $5.87\,\mu\text{m}$ for 3D and $6.22\,\mu\text{m}$ for Bi-Planar. Excluding `BEH0290 OD`, mean MABE was $5.90\,\mu\text{m}$ for 3D and **$5.36\,\mu\text{m}$ for Bi-Planar** (paired $t$-test $p=0.032$). The lower complete-cohort mean for 3D is therefore not a stable general accuracy advantage.

The 3D model also had a difficult case: `BEH0335 OS` reached $17.02\,\mu\text{m}$ MABE and $62.54\,\mu\text{m}$ $P_{95}$. The available data support the statement that 3D avoided one severe Bi-Planar failure—not that 3D is immune to outliers.

<div class="caution-box">
<strong>Reliability interpretation:</strong> 3D avoided one severe Bi-Planar failure. Forty volumes are insufficient to estimate a guaranteed maximum error or population failure rate.
</div>

</div>
<div class="column-panel" markdown="1">

## 6. Qualitative Anatomy Review

<img src="assets/deep_dive_BEH0314_OD_pdf.jpg" class="report-fig-wide" alt="Dense 3D qualitative deep dive for BEH0314 OD" />

<div class="fig-caption">
<strong>Figure 2: Dense 3D qualitative review on audited scan BEH0314 OD.</strong> The panel was generated from the 3D evaluation output. It compares the corrected reference, commercial segmentation, and 3D prediction in representative B-scan and en face views. Because the corresponding Bi-Planar panel is not shown here, this figure supports anatomical plausibility but is not a visual head-to-head comparison.
</div>

On `BEH0314 OD`, both quantitative evaluations were strong, but Bi-Planar had lower MABE ($3.92$ versus $4.70\,\mu\text{m}$) and lower $P_{95}$ ($12.44$ versus $15.91\,\mu\text{m}$). The 3D visualization demonstrates preservation of the peripapillary ring and cup reach; the quantitative paired result should determine the comparative interpretation.

</div>
</div>

<!-- pagebreak -->

<div class="two-col-grid" markdown="1">
<div class="column-panel" markdown="1">

## 7. Limitations

1. **Validation reuse:** the 20 subjects were excluded from gradient-based training but were used for checkpoint selection, so this is not an independent test cohort.
2. **Different selection objectives:** Bi-Planar was selected primarily by validation MABE, while 3D was selected by validation Dice. This can favor each model on its own selection endpoint.
3. **Limited expert-corrected sample:** only 10 corrected eyes from six subjects were available.
4. **Reference heterogeneity:** 30 eyes lacked a paired pre-correction curve and were evaluated against segmentations accepted without manual editing.
5. **Rare failures:** one extreme Bi-Planar case cannot establish either a population failure rate or 3D outlier immunity.
6. **Exploratory inference:** multiple endpoints were examined without a prespecified multiplicity adjustment.
7. **No diagnostic endpoint:** segmentation metrics do not demonstrate sensitivity, specificity, false-alert rate, or clinical safety.
8. **No controlled efficiency benchmark:** latency, throughput, and memory were not measured under the same inference conditions.

The previously generated clinical-capability radar is excluded from the inferential results because its normalized axes were manually constructed from selected cases rather than estimated from a prespecified cohort analysis.

</div>
<div class="column-panel" markdown="1">

## 8. Model-Selection Decision and Next Steps

### Current decision

- **Primary research model:** **Bi-Planar 2.5D Orthogonal Fusion**, based on better typical MABE, stronger corrected-tier performance, and higher BMO cup IoU.
- **Complementary QC model:** **Dense 3D U-Net**, used to identify large inter-model disagreement and potential slice-decoupling failures.
- **Clinical deployment:** neither model is ready for autonomous clinical use on the basis of this validation study.

### Confirmatory evaluation plan

1. Freeze both checkpoints, inference code, preprocessing, post-processing, and decision thresholds before examining new outcomes.
2. Construct an untouched subject-level test cohort, prioritizing expert-corrected eyes and preserving all OD/OS acquisitions from each subject in one split.
3. Define subject-averaged corrected-eye MABE as the primary endpoint. Define $P_{95}$, Dice, cup IoU, prespecified failure rates, latency, and peak memory as secondary endpoints.
4. Use paired subject-level analysis with bootstrap confidence intervals. Report the number needed for manual review and all threshold failures, not only cohort averages.
5. Predefine a disagreement rule between the two models. Route high-disagreement cases to manual review and test whether this catches failures such as `BEH0290 OD` without excessive review burden.
6. If no adequate untouched cohort remains, use grouped nested cross-validation with checkpoint selection confined to inner folds and architecture comparison performed only on outer folds.

> [!NOTE]
> **Final interpretation:** Bi-Planar is the stronger accuracy candidate in the available validation data. Dense 3D contributes a credible complementary robustness signal. A new test cohort—not additional reinterpretation of this validation set—is the next step required to establish superiority.

</div>
</div>

<div class="source-strip" markdown="1">

### Evidence Sources

- 3D scan-level metrics: `../2026-10-04_rnfl_3d_expanded_18574378/assets/executive_cohort_report/cohort_evaluation_metrics.json`
- Bi-Planar scan-level metrics: `../2026-10-03_biplanar_expanded_18563914/assets/executive_cohort_report/cohort_evaluation_metrics.json`
- Frozen validation split: `train-cnn-models/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json`
- Phase 1 training cohort: `train-cnn-models/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json`

*Research validation report. Results are intended for model development and human-supervised evaluation, not autonomous clinical diagnosis.*

</div>
