<style>
span[style*="#d97706"] code, span[style*="#d97706"] {
    color: #ea580c !important;
}
@media print {
    body {
        line-height: 1.45 !important;
    }
    h2 {
        page-break-before: always !important;
        break-before: page !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
    }
    h2:last-of-type {
        page-break-before: auto !important;
        break-before: auto !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
    }
    h3 {
        margin-top: 12px !important;
        margin-bottom: 6px !important;
    }
    p, li {
        margin-bottom: 5px !important;
    }
    hr {
        margin: 8px 0 !important;
    }
    ul, ol {
        margin: 6px 0 10px 0 !important;
        padding-left: 24px !important;
        page-break-inside: auto !important;
    }
    li {
        margin-bottom: 4px !important;
        line-height: 1.45 !important;
        display: list-item !important;
        list-style-type: disc !important;
        page-break-inside: avoid !important;
    }
    ol li {
        list-style-type: decimal !important;
    }
    .kpi-grid {
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        margin: 10px 0 14px 0 !important;
        page-break-inside: avoid !important;
    }
    .kpi-card {
        flex: 1 !important;
        margin: 0 4px !important;
        padding: 8px 10px !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
    .challenge-grid {
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: space-between !important;
        page-break-inside: avoid !important;
    }
    .challenge-card {
        width: 48.5% !important;
        margin-bottom: 10px !important;
        box-sizing: border-box !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
    .table-container {
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        margin-bottom: 10px !important;
    }
    .evidence-context {
        page-break-inside: avoid !important;
        break-inside: avoid !important;
    }
    .evidence-table {
        table-layout: fixed !important;
        font-size: 9.25px !important;
        line-height: 1.28 !important;
        margin: 8px 0 10px 0 !important;
    }
    .evidence-table th, .evidence-table td {
        padding: 4px 5px !important;
        vertical-align: top !important;
        overflow-wrap: anywhere !important;
    }
    .evidence-note {
        background: #fff7ed !important;
        border-left: 4px solid #f97316 !important;
        padding: 7px 9px !important;
        margin: 8px 0 !important;
        font-size: 9.5px !important;
        line-height: 1.35 !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
    .gallery-grid {
        display: block !important;
        page-break-inside: auto !important;
        break-inside: auto !important;
    }
    .gallery-item {
        display: inline-block !important;
        width: 49% !important;
        vertical-align: top !important;
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        margin-bottom: 6px !important;
        box-sizing: border-box !important;
    }
    .gallery-item img {
        width: 100% !important;
        height: auto !important;
        display: block !important;
        margin: 2px 0 !important;
    }
    .deep-dive-item {
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        margin: 8px 0 14px 0 !important;
    }
    .deep-dive-item h3 {
        page-break-after: avoid !important;
        break-after: avoid !important;
    }
    .deep-dive-item img {
        margin-top: 4px !important;
    }
}
</style>

# Volumetric RNFL Segmentation (Volumetric Biplanar (6.6M) Model): Multi-Subject Cohort Report

**Cohort Scope**: 61 Subjects (`BEH0026` - `BEH0364`) | 122 OCT Volumes (61 OD + 61 OS)<br>
**Modality**: Optovue Solix OCT `Disc Cube` ($320 \times 768 \times 320$ voxels; $18.81\,\mu\text{m} \times 3.12\,\mu\text{m} \times 18.75\,\mu\text{m}$)<br>
**Checkpoint Provenance**: NYUAD HPC Jubail (SLURM Job `18266075`) | Checkpoint: `Best Biplanar (6.6M)`<br>
**Evaluation Runtime**: Local Apple MPS corrected-cohort rerun<br>
**Architecture / Variant**: **Volumetric Biplanar (6.6M) Architecture** (Volumetric 2.5D ResNet (6.6M parameters, base_channels=32, num_res_units=2) with dual-plane orthogonal fusion, continuous 1D surface regression, and surface-guided false-positive RPE clamping.)<br>
**Inference Policy**: Corrected OS native-coordinate restoration before horizontal/vertical biplanar fusion

**Evaluation Arms**:

- **<span style="color: #0284c7; font-weight: bold;">Cyan</span>**: Human-Corrected Reference (Good Arm)
- **<span style="color: #dc2626; font-weight: bold;">Red</span>**: Commercial Solix Heuristic Baseline (Bad Arm)
- **<span style="color: #16a34a; font-weight: bold;">Green</span>**: Multi-Task Volumetric U-Net (Volumetric Biplanar (6.6M) Model, 2.5D Context + Continuous 1D Boundary Regression)
- **<span style="color: #ea580c; font-weight: bold;">Orange</span>**: Held-Out Validation Cohort (<span style="color: #ea580c; font-weight: bold;">`BEH0086`, `BEH0090`, `BEH0174`, `BEH0284`, `BEH0310`, `BEH0314`, `BEH0335`, `BEH0354`</span>, 16 Scans Unseen During Training)

## 1. Executive Summary

This report delivers an automated cohort-wide comparative evaluation of the **Multi-Task Volumetric RNFL U-Net (<span style="color: #16a34a; font-weight: bold;">Green</span>)** against the **Human-Corrected Reference (<span style="color: #0284c7; font-weight: bold;">Cyan</span>)** and the **Commercial Solix Baseline (<span style="color: #dc2626; font-weight: bold;">Red</span>)** across 61 subjects (122 eye-level OCT volumes). The checkpoint was trained on NYUAD Jubail and this corrected cohort evaluation was executed locally on Apple MPS.

<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-title">Benchmark MABE</div>
        <div class="kpi-value">2.61 µm</div>
        <div class="kpi-sub">Median across 106 benchmark scans (OD + OS)</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-title">Held-Out MABE</div>
        <div class="kpi-value">5.12 µm</div>
        <div class="kpi-sub">Median across 16 unseen OD + OS scans</div>
    </div>
    <div class="kpi-card cyan">
        <div class="kpi-title">Benchmark Dice</div>
        <div class="kpi-value">0.8966</div>
        <div class="kpi-sub">Median across both eyes</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-title">NFL-Absence Cup-Region IoU</div>
        <div class="kpi-value">0.9517</div>
        <div class="kpi-sub">Mean: 0.9468 ± 0.0215</div>
    </div>
</div>

### High-Level Findings:

1. **Laterality Stability After Coordinate Correction**: Benchmark OD median MABE was **$2.59 \; \mu\text{m}$** with median Dice **$0.9041$**; benchmark OS median MABE was **$2.67 \; \mu\text{m}$** with median Dice **$0.8885$**. The previous cohort-wide OS collapse was an inference-coordinate defect, not a supported model finding.
2. **Held-Out Failure Is Subject-Specific**: Across the 16 held-out acquisitions (`BEH0086`, `BEH0090`, `BEH0174`, `BEH0284`, `BEH0310`, `BEH0314`, `BEH0335`, `BEH0354`), median MABE was **$5.12 \; \mu\text{m}$**. The worst held-out scan was **BEH0335 OS** at **$14.57 \; \mu\text{m}$**; pooled validation statistics therefore require scan-level review.
3. **Not Ready for Autonomous Clinical Use**: 45 of 106 assessed scans triggered at least one conservative operational review flag. These engineering thresholds are not clinically validated, but residual Dice, cup-IoU, and boundary-error failures require mandatory human review.
4. **Audit-Correction Performance**: Across 23 materially edited scans (15 benchmark training, 8 held-out validation), the U-Net reduced raw commercial error in 20 scans overall (and in 6 of 8 held-out validation scans, with validation median gain **+37.3%**). Median overall correction gain was **+70.1%** and median edited-column recovery rate was **61.3%**. This edit-focused analysis is primary; whole-mask commercial Dice is reference-dependent and descriptive only.

---

## 2. Methods & Evaluation Protocol

### 2.1 Cohort Architecture & Data Modality
- **Dataset**: 61 deidentified human subjects from the NYUAD / Rokers Lab Solix OCT repository.
- **Protocol**: Optovue Solix `Disc Cube` ($320 \times 768 \times 320$ voxels), covering $6.0 \times 6.0 \times 2.4\,\text{mm}^3$ centered on the optic nerve head (ONH).
- **Voxel Pitch**: $18.81\,\mu\text{m}$ (slow B-scan pitch) $\times 3.12\,\mu\text{m}$ (axial depth) $\times 18.75\,\mu\text{m}$ (fast A-scan pitch).

### 2.2 Evaluation Metrics
- **Dice Similarity Coefficient**: Spatial overlap between predicted and human-audited binary RNFL masks. It is a secondary consistency endpoint because unchanged regions dominate whole-mask overlap.
- **Mean Absolute Boundary Error (MABE)**: Mean axial displacement outside the optic cup cavity in $\mu\text{m}$ ($3.12367\,\mu\text{m}/\text{px}$).
- **95th Percentile Boundary Error ($P_{95}$)**: Localized worst-case boundary drift in $\mu\text{m}$.
- **Audit-Correction Gain**: One minus the ratio of U-Net error to raw error on columns where the raw and audited NFL boundaries differ by at least 1 px. Positive values indicate recovery of human corrections.
- **Unchanged-Region Preservation**: Fraction of human-accepted columns where the U-Net remains within 1 px of the audited NFL boundary.
- **NFL-Absence Cup-Region IoU**: Jaccard overlap of columns where the NFL boundary is absent. This is an annotation-derived cup-region endpoint, not a full anatomical cup segmentation.
- **Cup-Edge Localization**: Horizontal left-edge, right-edge, and width errors on slices where both reference and prediction contain a detectable NFL-absence region.

---

## 3. Cohort Quantitative Benchmark Results

### 3.1 Distribution and Individual Scan Profiles

The multi-panel plot stratifies benchmark and held-out scans by eye. Error metrics use logarithmic axes so the common 2-5 µm range remains visible alongside severe subject-level outliers. Dashed thresholds are operational report references, not validated clinical decision limits.

![Cohort Statistical Distributions](assets/executive_cohort_report/cohort_raincloud_distributions.png)

- **RNFL Dice Overlap**: Benchmark OD median Dice was **$0.9041$** versus **$0.8885$** for benchmark OS.
- **Peripapillary Boundary Error (MABE)**: Benchmark OD median MABE was **$2.59 \; \mu\text{m}$**, versus **$2.67 \; \mu\text{m}$** for benchmark OS. The $5.0 \; \mu\text{m}$ line is an operational reference.
- **NFL-Absence Cup-Region Detection**: The annotation-derived cup-region endpoint reached a median IoU of **$0.9517$**; this should not be interpreted as independent anatomical cup ground truth.

---

### 3.2 Complete 122-Scan Clinical Cohort Forest Chart

The chart shows all 122 eye-level scans ranked best to worst by MABE. MABE uses a logarithmic axis to preserve the 2-15 µm range while retaining severe outliers; Dice and Cup IoU are shown in separate aligned panels. `[MIRROR]` identifies an unedited machine copy rather than an independent commercial annotation.

![Complete 122-Scan Forest Plot](assets/executive_cohort_report/cohort_per_scan_forest_plot.png)

---

### 3.3 Audit-Correction Analysis: U-Net vs Raw Commercial Boundary

Whole-mask comparison is reference-dependent because the human-audited annotation was created by editing the raw commercial result. The primary comparator analysis therefore isolates columns with a raw-to-audit displacement of at least 1 px. Across 23 materially edited scans (15 benchmark training and 8 held-out validation across BEH0174, BEH0310, BEH0314, BEH0335, BEH0354), the U-Net reduced boundary error in 20 scans overall, and in 6 of 8 held-out validation scans. Across the unseen validation scans, median correction gain was **+37.3%**. Overall median correction gain was **+70.1%**, median edited-column recovery was **61.3%**, and median unchanged-region preservation was **81.8%**. Negative correction gain means the U-Net was farther from the audit than the raw boundary on edited columns. In the chart below, held-out validation scans are explicitly flagged with `[VAL]`.

![Audit-Correction Analysis](assets/executive_cohort_report/baseline_vs_unet_head_to_head.png)

---

## 4. External Evidence Context

The closest published evidence spans different OCT devices, scan geometries, pathologies, reference standards, and aggregation methods. The table therefore positions the model rather than ranking it. The current-project row uses the **subject-disjoint held-out cohort**; training-cohort benchmark Dice is intentionally excluded from the cross-study comparison.

<div class="evidence-context">
<table class="evidence-table">
<thead>
<tr><th style="width: 18%;">Evidence</th><th style="width: 24%;">Evaluation setting</th><th style="width: 14%;">RNFL Dice</th><th style="width: 19%;">Other error endpoint</th><th style="width: 25%;">Comparability note</th></tr>
</thead>
<tbody>
<tr><td><strong>Current model</strong></td><td>Optovue Solix Disc Cube; 8 subjects / 16 eyes; full volumes; subject-disjoint held-out set</td><td><strong>Median 0.948</strong></td><td>Boundary MABE: <strong>5.12 µm</strong> median; worst 14.57 µm</td><td>Most relevant generalization result, but the sample is too small for a superiority or safety claim.</td></tr>
<tr><td><a href="https://doi.org/10.1167/tvst.15.4.7">Arian et al., 2026</a></td><td>External Spectralis circular B-scans: Thailand glaucoma (n=157) and US edema (n=32)</td><td>Mean 0.858 / 0.845</td><td>Thickness MAE: 7.19 / 15.41 µm; lower-boundary MUE: 14.52 / 24.82 µm</td><td>Strong external clinical evidence, but 2D circles and thickness/boundary endpoints differ from the Solix volume evaluation.</td></tr>
<tr><td><a href="https://arxiv.org/abs/2207.14447">GOALS, 2022</a></td><td>Topcon DRI circumpapillary B-scans; patient-disjoint challenge tests</td><td>0.816 / 0.843</td><td>Boundary MED: 4.06 / 4.15 pixels</td><td>High anatomical relevance and multi-grader reference; single 2D circles and no defensible µm conversion.</td></tr>
<tr><td><a href="https://doi.org/10.1038/s41598-022-22135-x">Razaghi et al., 2022</a></td><td>Spectralis circular B-scans; 127 independent test eyes spanning healthy, NAION, and optic neuritis</td><td>0.870</td><td>Thickness MAE: 1.04-1.20 µm across groups</td><td>Independent same-device test; thickness MAE is not interchangeable with local boundary MABE.</td></tr>
<tr><td><a href="https://doi.org/10.3389/fcell.2026.1890734">Qiu et al., 2026 (M2D)</a></td><td>1,017 Heidelberg/TowardPi circumpapillary scans; expert-corrected subset</td><td>Mean 0.874</td><td>Expert-subset thickness MAD: 1.8 µm</td><td>Cross-device evidence, but large-scale overlap primarily used proprietary output as the reference.</td></tr>
<tr><td><a href="https://doi.org/10.18502/jovr.v18i1.12724">Razaghi et al., 2023</a></td><td>SD-OCT B-scans; 50-image internal test; subject separation unclear</td><td>0.910</td><td>Thickness MAE: 2.23 ± 2.10 µm</td><td>Favorable internal result with a small image-level test; weaker generalization evidence.</td></tr>
</tbody>
</table>

<div class="evidence-note"><strong>Interpretation:</strong> The held-out Dice of 0.948 lies within the approximately 0.82-0.88 range reported in external or difficult peripapillary RNFL evaluations. This is evidence of technical plausibility, not equivalence or superiority. No directly comparable external Optovue Solix Disc Cube benchmark was identified, and Dice alone does not resolve the 14.57 µm worst-case boundary failure.</div>
</div>

Values above retain each publication's original endpoint and aggregation. Mean and median values, full-volume and circular-scan evaluations, boundary and thickness errors, and human versus commercial-derived references must not be treated as interchangeable.

---

## 5. OD Subject Statistical Overview

The aligned dot plots summarize the 61 OD acquisitions only. Dice, MABE, and Cup IoU use separate axes and operational thresholds. Commercial points appear only where an independent comparator is available; blank comparator positions represent missing annotations, not zero values.

![Cohort Summary Chart](assets/executive_cohort_report/cohort_summary_chart.png)

---

## 6. OD Subject Gallery: Reference vs U-Net

Central peripapillary OD B-scans ($z = z_{\text{disc}}$) comparing the **Human-Corrected Reference (<span style="color: #0284c7; font-weight: bold;">Cyan</span>)** with the **Volumetric U-Net (<span style="color: #16a34a; font-weight: bold;">Green</span>)**. The gallery shows 24 representative OD views selected from 61, prioritizing held-out scans and high-MABE failures; OS acquisitions and the commercial baseline are not shown here. Complete scan-level metrics remain archived in the report assets.

<div class="gallery-grid">
<div class="gallery-item">
<p><strong>BEH0081 (OD)</strong> - Dice: <code>0.8947</code> | MABE: <code>2.88 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0081_OD.png" alt="Gallery BEH0081" />
</div>
<div class="gallery-item">
<p><strong>BEH0086 (OD)</strong> - Dice: <code>0.9566</code> | MABE: <code>4.30 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0086_OD.png" alt="Gallery BEH0086" />
</div>
<div class="gallery-item">
<p><strong>BEH0090 (OD)</strong> - Dice: <code>0.9351</code> | MABE: <code>6.88 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0090_OD.png" alt="Gallery BEH0090" />
</div>
<div class="gallery-item">
<p><strong>BEH0174 (OD)</strong> - Dice: <code>0.9478</code> | MABE: <code>5.92 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0174_OD.png" alt="Gallery BEH0174" />
</div>
<div class="gallery-item">
<p><strong>BEH0185 (OD)</strong> - Dice: <code>0.8909</code> | MABE: <code>2.89 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0185_OD.png" alt="Gallery BEH0185" />
</div>
<div class="gallery-item">
<p><strong>BEH0279 (OD)</strong> - Dice: <code>0.9055</code> | MABE: <code>2.43 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0279_OD.png" alt="Gallery BEH0279" />
</div>
<div class="gallery-item">
<p><strong>BEH0284 (OD)</strong> - Dice: <code>0.9551</code> | MABE: <code>4.16 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0284_OD.png" alt="Gallery BEH0284" />
</div>
<div class="gallery-item">
<p><strong>BEH0290 (OD)</strong> - Dice: <code>0.8812</code> | MABE: <code>2.67 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0290_OD.png" alt="Gallery BEH0290" />
</div>
<div class="gallery-item">
<p><strong>BEH0303 (OD)</strong> - Dice: <code>0.9308</code> | MABE: <code>2.62 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0303_OD.png" alt="Gallery BEH0303" />
</div>
<div class="gallery-item">
<p><strong>BEH0308 (OD)</strong> - Dice: <code>0.8806</code> | MABE: <code>2.53 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0308_OD.png" alt="Gallery BEH0308" />
</div>
<div class="gallery-item">
<p><strong>BEH0310 (OD)</strong> - Dice: <code>0.9509</code> | MABE: <code>4.27 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0310_OD.png" alt="Gallery BEH0310" />
</div>
<div class="gallery-item">
<p><strong>BEH0314 (OD)</strong> - Dice: <code>0.9504</code> | MABE: <code>5.12 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0314_OD.png" alt="Gallery BEH0314" />
</div>
<div class="gallery-item">
<p><strong>BEH0316 (OD)</strong> - Dice: <code>0.8356</code> | MABE: <code>2.95 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0316_OD.png" alt="Gallery BEH0316" />
</div>
<div class="gallery-item">
<p><strong>BEH0319 (OD)</strong> - Dice: <code>0.9339</code> | MABE: <code>2.38 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0319_OD.png" alt="Gallery BEH0319" />
</div>
<div class="gallery-item">
<p><strong>BEH0328 (OD)</strong> - Dice: <code>0.7957</code> | MABE: <code>2.91 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0328_OD.png" alt="Gallery BEH0328" />
</div>
<div class="gallery-item">
<p><strong>BEH0329 (OD)</strong> - Dice: <code>0.9226</code> | MABE: <code>2.46 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0329_OD.png" alt="Gallery BEH0329" />
</div>
<div class="gallery-item">
<p><strong>BEH0334 (OD)</strong> - Dice: <code>0.8418</code> | MABE: <code>2.87 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0334_OD.png" alt="Gallery BEH0334" />
</div>
<div class="gallery-item">
<p><strong>BEH0335 (OD)</strong> - Dice: <code>0.8708</code> | MABE: <code>11.53 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0335_OD.png" alt="Gallery BEH0335" />
</div>
<div class="gallery-item">
<p><strong>BEH0336 (OD)</strong> - Dice: <code>0.8282</code> | MABE: <code>3.17 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0336_OD.png" alt="Gallery BEH0336" />
</div>
<div class="gallery-item">
<p><strong>BEH0341 (OD)</strong> - Dice: <code>0.7980</code> | MABE: <code>3.04 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0341_OD.png" alt="Gallery BEH0341" />
</div>
<div class="gallery-item">
<p><strong>BEH0344 (OD)</strong> - Dice: <code>0.8742</code> | MABE: <code>2.58 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0344_OD.png" alt="Gallery BEH0344" />
</div>
<div class="gallery-item">
<p><strong>BEH0352 (OD)</strong> - Dice: <code>0.9368</code> | MABE: <code>2.83 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0352_OD.png" alt="Gallery BEH0352" />
</div>
<div class="gallery-item">
<p><strong>BEH0354 (OD)</strong> - Dice: <code>0.9570</code> | MABE: <code>4.12 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0354_OD.png" alt="Gallery BEH0354" />
</div>
<div class="gallery-item">
<p><strong>BEH0361 (OD)</strong> - Dice: <code>0.9036</code> | MABE: <code>2.93 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0361_OD.png" alt="Gallery BEH0361" />
</div>
</div>

---

## 7. 3-Arm Deep-Dive Panels: Validation & Archetype Subjects

Detailed cross-sectional analysis comparing optical intensity boundaries, vertical cut behavior, and local layer transitions across key clinical archetypes.

<div class="deep-dive-item">
<h3>Subject BEH0086 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0086_OD.png" alt="Deep Dive BEH0086" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0090 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0090_OD.png" alt="Deep Dive BEH0090" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0174 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0174_OD.png" alt="Deep Dive BEH0174" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0181 (OD) [Training / Benchmark]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0181_OD.png" alt="Deep Dive BEH0181" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0284 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0284_OD.png" alt="Deep Dive BEH0284" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0310 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0310_OD.png" alt="Deep Dive BEH0310" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0314 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0314_OD.png" alt="Deep Dive BEH0314" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0335 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0335_OD.png" alt="Deep Dive BEH0335" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0354 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0354_OD.png" alt="Deep Dive BEH0354" />
</div>

## 8. Algorithmic Mechanics Driving Boundary Adherence

<div class="challenge-grid">
    <div class="challenge-card">
        <div class="challenge-title">GCL Hyporeflective Wedge Penetration</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Plunges deeply into adjacent hyporeflective ganglion cell layer.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Continuous 1D head locks onto true hyperreflective optical gradient.</div>
        <div class="challenge-row"><span class="badge-cyan">Human-Corrected Reference</span> Manually reviewed anatomical transition interface.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Optic Cup Cavity Void & BMO Bridging</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Bridges straight across non-physiological empty cup void.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> 1D cup head accurately truncates margin at Bruch's Membrane Opening.</div>
        <div class="challenge-row"><span class="badge-cyan">Human-Corrected Reference</span> Reviewed peripapillary termination at the annotated margin.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Major Vessel Axial Shadowing</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Axial signal drop causes erratic vertical jumps and boundary loss.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Multi-slice 2.5D contextual slices interpolate across vessel shadows cleanly.</div>
        <div class="challenge-row"><span class="badge-cyan">Human-Corrected Reference</span> Reviewed continuous layer contours.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Pathological Disc Tilt & Steep Slope</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Steep regional gradients induce boundary distortion and clipping.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Continuous 1D regression preserves curvature continuity and slope fidelity.</div>
        <div class="challenge-row"><span class="badge-cyan">Human-Corrected Reference</span> Reviewed boundary conformity.</div>
    </div>
</div>

---

## 9. Clinical Significance & Conclusion

1. **Comparable OD and OS Cohort Performance**: Corrected native-coordinate restoration removes the systematic OS artifact. Benchmark medians are closely aligned by eye, but laterality stability does not eliminate individual failures.
2. **Residual Clinical Risk**: The worst held-out scan, BEH0335 OS, reached **$14.57 \; \mu\text{m}$** MABE, and additional scans miss Dice or cup-IoU operational limits. The model is suitable for research and human-supervised review, not autonomous clinical use.
3. **Comparator Evidence Is Reference-Dependent**: The edit-focused analysis shows whether the U-Net recovers human changes without allowing unchanged pixels to dominate. It still cannot establish clinical superiority because the audit is not an independent second-reader reference.
4. **External Positioning**: Held-out Dice is within published external or difficult-cohort RNFL ranges, but cross-study differences and the small held-out cohort prevent a direct ranking or superiority claim.
5. **Execution Summary**: Checkpoint `Best Biplanar (6.6M)` was evaluated using corrected biplanar inference. All visual assets, scan-level metrics, audit-correction fields, comparator missingness, and manual-review outputs are archived in `assets/executive_cohort_report`.

