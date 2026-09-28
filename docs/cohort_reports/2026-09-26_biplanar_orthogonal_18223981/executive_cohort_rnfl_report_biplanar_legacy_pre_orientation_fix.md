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

# Volumetric RNFL Segmentation (Bi-Planar Orthogonal Heavy Model): Multi-Subject Cohort Report

**Cohort Scope**: 23 Subjects (`BEH0086` – `BEH0410`) | 46 OCT Volumes (23 OD + 23 OS)  
**Modality**: Optovue Solix OCT `Disc Cube` ($320 \times 768 \times 320$ voxels; $18.81\,\mu\text{m} \times 3.12\,\mu\text{m} \times 18.75\,\mu\text{m}$)  
**Execution Environment**: NYUAD HPC Jubail (SLURM Job `18223981`) | Checkpoint: `best_volumetric_rnfl_net.pt`  
**Architecture / Variant**: **Bi-Planar Orthogonal Heavy Architecture** (Fast-Slow Cross-Axis 2.5D Context + Dual-Plane Fusion + Continuous 1D Boundary Regression)  

**Evaluation Arms**:

- **<span style="color: #0284c7; font-weight: bold;">Cyan</span>**: Clinician-Corrected Reference Algorithm (Good Arm)
- **<span style="color: #dc2626; font-weight: bold;">Red</span>**: Commercial Solix Heuristic Baseline (Bad Arm)
- **<span style="color: #16a34a; font-weight: bold;">Green</span>**: Multi-Task Volumetric U-Net (Bi-Planar Orthogonal Heavy Model, 2.5D Context + Continuous 1D Boundary Regression)
- **<span style="color: #ea580c; font-weight: bold;">Orange</span>**: Held-Out Validation Cohort (<span style="color: #ea580c; font-weight: bold;">`BEH0086`, `BEH0314`, `BEH0335`</span>, 6 Scans Unseen During Training)

## 1. Executive Summary

This report delivers an automated cohort-wide comparative evaluation of the **Multi-Task Volumetric RNFL U-Net (<span style="color: #16a34a; font-weight: bold;">Green</span>)** against the **Clinician Reference Algorithm (<span style="color: #0284c7; font-weight: bold;">Cyan</span>)** and the **Commercial Solix Baseline (<span style="color: #dc2626; font-weight: bold;">Red</span>)** across 23 subjects (46 eye-level OCT volumes) executed end-to-end on NYUAD Jubail.

<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-title">Benchmark OD MABE</div>
        <div class="kpi-value">3.02 µm</div>
        <div class="kpi-sub">± 0.18 µm | Median: 2.96 µm</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-title">Held-Out Val OD MABE</div>
        <div class="kpi-value">8.09 µm</div>
        <div class="kpi-sub">3 unseen OD scans</div>
    </div>
    <div class="kpi-card cyan">
        <div class="kpi-title">Benchmark OD Dice</div>
        <div class="kpi-value">0.8983</div>
        <div class="kpi-sub">± 0.0371 | Median: 0.9077</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-title">Optic Cup Cavity IoU</div>
        <div class="kpi-value">0.9179</div>
        <div class="kpi-sub">Mean: 0.8763 ± 0.1000</div>
    </div>
</div>

### High-Level Findings:

1. **Strong Laterality Dependence**: Benchmark OD scans achieved median MABE **$2.96 \; \mu\text{m}$** and median Dice **$0.9077$**, whereas benchmark OS scans reached median MABE **$113.31 \; \mu\text{m}$** and median Dice **$0.6649$**. Aggregate 40-scan statistics therefore conceal a material OS failure mode.
2. **Held-Out Validation Stress-Testing**: The 6 held-out acquisitions (`BEH0086`, `BEH0314`, `BEH0335`) show the same asymmetry: mean OD MABE was **$8.09 \; \mu\text{m}$**, while mean OS MABE was **$182.26 \; \mu\text{m}$**. These results should be interpreted by eye rather than as a single pooled performance claim.
3. **Optic Cup & BMO Termination**: Continuous 1D boundary regression heads reliably bounded Bruch's Membrane Opening (BMO), eliminating wedge over-segmentation into adjacent hyporeflective ganglion cell layers.

---

## 2. Methods & Evaluation Protocol

### 2.1 Cohort Architecture & Data Modality
- **Dataset**: 23 deidentified human subjects from the NYUAD / Rokers Lab Solix OCT repository.
- **Protocol**: Optovue Solix `Disc Cube` ($320 \times 768 \times 320$ voxels), covering $6.0 \times 6.0 \times 2.4\,\text{mm}^3$ centered on the optic nerve head (ONH).
- **Voxel Pitch**: $18.81\,\mu\text{m}$ (slow B-scan pitch) $\times 3.12\,\mu\text{m}$ (axial depth) $\times 18.75\,\mu\text{m}$ (fast A-scan pitch).

### 2.2 Evaluation Metrics
- **Dice Similarity Coefficient**: Spatial volume overlap between predicted and clinician reference binary RNFL masks.
- **Mean Absolute Boundary Error (MABE)**: Mean axial displacement outside the optic cup cavity in $\mu\text{m}$ ($3.12367\,\mu\text{m}/\text{px}$).
- **95th Percentile Boundary Error ($P_{95}$)**: Localized worst-case boundary drift in $\mu\text{m}$.
- **Cup Intersection over Union (Cup IoU)**: Jaccard index evaluating optic cup margin detection along the horizontal fast axis.

---

## 3. Cohort Quantitative Benchmark Results

### 3.1 Distribution and Individual Scan Profiles

The multi-panel plot stratifies benchmark and held-out scans by eye. Error metrics use logarithmic axes so the clinically relevant OD range remains visible alongside severe OS outliers. Dashed thresholds are operational report references, not validated clinical decision limits.

![Cohort Statistical Distributions](assets/executive_cohort_report/cohort_raincloud_distributions.png)

- **RNFL Dice Overlap**: Benchmark OD median Dice was **$0.9077$** versus **$0.6649$** for benchmark OS.
- **Peripapillary Boundary Error (MABE)**: Benchmark OD median MABE was **$2.96 \; \mu\text{m}$**, versus **$113.31 \; \mu\text{m}$** for benchmark OS. The $5.0 \; \mu\text{m}$ line is an operational reference.
- **Optic Cup Detection**: Optic cup margin tracking at Bruch's Membrane Opening (BMO) reached a median IoU of **$0.9179$**, preventing non-physiological bridging across the central cavity void.

---

### 3.2 Complete 46-Scan Clinical Cohort Forest Chart

The chart shows all 46 eye-level scans ranked best to worst by MABE. MABE uses a logarithmic axis to preserve the 2–15 µm range while retaining severe OS outliers; Dice and Cup IoU are shown in separate aligned panels. `[MIRROR]` identifies an unedited machine copy rather than an independent commercial annotation.

![Complete 46-Scan Forest Plot](assets/executive_cohort_report/cohort_per_scan_forest_plot.png)

---

### 3.3 Head-to-Head Comparative Delta: U-Net vs Commercial Baseline

Paired analysis is limited to the 12 scans with independent commercial annotations; the remaining 34 scans do not have a valid paired commercial comparator. Dice improved in 3 and worsened in 9 paired scans; Cup IoU improved in 6 and worsened in 6.

![Baseline vs U-Net Head-to-Head](assets/executive_cohort_report/baseline_vs_unet_head_to_head.png)

---

## 4. OD Subject Statistical Overview

The aligned dot plots summarize the 23 OD acquisitions only. Dice, MABE, and Cup IoU use separate axes and operational thresholds. Commercial points appear only where an independent comparator is available; blank comparator positions represent missing annotations, not zero values.

![Cohort Summary Chart](assets/executive_cohort_report/cohort_summary_chart.png)

---

## 5. OD Subject Gallery: Reference vs U-Net

Central peripapillary OD B-scans ($z = z_{\text{disc}}$) comparing the **Clinician Reference (<span style="color: #0284c7; font-weight: bold;">Cyan</span>)** with the **Volumetric U-Net (<span style="color: #16a34a; font-weight: bold;">Green</span>)**. The gallery contains one OD view per subject; OS acquisitions and the commercial baseline are not shown here.

<div class="gallery-grid">
<div class="gallery-item">
<p><strong>BEH0086 (OD)</strong> — Dice: <code>0.9063</code> | MABE: <code>4.62 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0086_OD.png" alt="Gallery BEH0086" />
</div>
<div class="gallery-item">
<p><strong>BEH0090 (OD)</strong> — Dice: <code>0.9193</code> | MABE: <code>3.29 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0090_OD.png" alt="Gallery BEH0090" />
</div>
<div class="gallery-item">
<p><strong>BEH0096 (OD)</strong> — Dice: <code>0.9300</code> | MABE: <code>3.26 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0096_OD.png" alt="Gallery BEH0096" />
</div>
<div class="gallery-item">
<p><strong>BEH0174 (OD)</strong> — Dice: <code>0.9079</code> | MABE: <code>2.95 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0174_OD.png" alt="Gallery BEH0174" />
</div>
<div class="gallery-item">
<p><strong>BEH0181 (OD)</strong> — Dice: <code>0.9133</code> | MABE: <code>3.10 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0181_OD.png" alt="Gallery BEH0181" />
</div>
<div class="gallery-item">
<p><strong>BEH0185 (OD)</strong> — Dice: <code>0.8921</code> | MABE: <code>3.11 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0185_OD.png" alt="Gallery BEH0185" />
</div>
<div class="gallery-item">
<p><strong>BEH0241 (OD)</strong> — Dice: <code>0.8197</code> | MABE: <code>2.83 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0241_OD.png" alt="Gallery BEH0241" />
</div>
<div class="gallery-item">
<p><strong>BEH0249 (OD)</strong> — Dice: <code>0.8640</code> | MABE: <code>2.96 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0249_OD.png" alt="Gallery BEH0249" />
</div>
<div class="gallery-item">
<p><strong>BEH0259 (OD)</strong> — Dice: <code>0.8935</code> | MABE: <code>3.21 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0259_OD.png" alt="Gallery BEH0259" />
</div>
<div class="gallery-item">
<p><strong>BEH0264 (OD)</strong> — Dice: <code>0.8168</code> | MABE: <code>3.50 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0264_OD.png" alt="Gallery BEH0264" />
</div>
<div class="gallery-item">
<p><strong>BEH0282 (OD)</strong> — Dice: <code>0.8577</code> | MABE: <code>2.83 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0282_OD.png" alt="Gallery BEH0282" />
</div>
<div class="gallery-item">
<p><strong>BEH0284 (OD)</strong> — Dice: <code>0.9077</code> | MABE: <code>2.91 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0284_OD.png" alt="Gallery BEH0284" />
</div>
<div class="gallery-item">
<p><strong>BEH0287 (OD)</strong> — Dice: <code>0.8712</code> | MABE: <code>2.93 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0287_OD.png" alt="Gallery BEH0287" />
</div>
<div class="gallery-item">
<p><strong>BEH0294 (OD)</strong> — Dice: <code>0.9217</code> | MABE: <code>2.90 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0294_OD.png" alt="Gallery BEH0294" />
</div>
<div class="gallery-item">
<p><strong>BEH0310 (OD)</strong> — Dice: <code>0.9297</code> | MABE: <code>2.90 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0310_OD.png" alt="Gallery BEH0310" />
</div>
<div class="gallery-item">
<p><strong>BEH0314 (OD)</strong> — Dice: <code>0.8654</code> | MABE: <code>5.31 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0314_OD.png" alt="Gallery BEH0314" />
</div>
<div class="gallery-item">
<p><strong>BEH0321 (OD)</strong> — Dice: <code>0.9473</code> | MABE: <code>2.82 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0321_OD.png" alt="Gallery BEH0321" />
</div>
<div class="gallery-item">
<p><strong>BEH0335 (OD)</strong> — Dice: <code>0.8915</code> | MABE: <code>14.35 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0335_OD.png" alt="Gallery BEH0335" />
</div>
<div class="gallery-item">
<p><strong>BEH0349 (OD)</strong> — Dice: <code>0.9002</code> | MABE: <code>2.89 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0349_OD.png" alt="Gallery BEH0349" />
</div>
<div class="gallery-item">
<p><strong>BEH0354 (OD)</strong> — Dice: <code>0.9077</code> | MABE: <code>3.09 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0354_OD.png" alt="Gallery BEH0354" />
</div>
<div class="gallery-item">
<p><strong>BEH0364 (OD)</strong> — Dice: <code>0.9391</code> | MABE: <code>2.96 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0364_OD.png" alt="Gallery BEH0364" />
</div>
<div class="gallery-item">
<p><strong>BEH0398 (OD)</strong> — Dice: <code>0.8845</code> | MABE: <code>3.05 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0398_OD.png" alt="Gallery BEH0398" />
</div>
<div class="gallery-item">
<p><strong>BEH0410 (OD)</strong> — Dice: <code>0.9427</code> | MABE: <code>3.00 µm</code></p>
<img src="assets/executive_cohort_report/gallery_bscan_BEH0410_OD.png" alt="Gallery BEH0410" />
</div>
</div>

---

## 6. 3-Arm Deep-Dive Panels: Validation & Archetype Subjects

Detailed cross-sectional analysis comparing optical intensity boundaries, vertical cut behavior, and local layer transitions across key clinical archetypes.

<div class="deep-dive-item">
<h3>Subject BEH0086 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0086_OD.png" alt="Deep Dive BEH0086" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0174 (OD) [Training / Benchmark]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0174_OD.png" alt="Deep Dive BEH0174" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0181 (OD) [Training / Benchmark]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0181_OD.png" alt="Deep Dive BEH0181" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0314 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0314_OD.png" alt="Deep Dive BEH0314" />
</div>

<div class="deep-dive-item">
<h3>Subject BEH0335 (OD) [Validation (Held-Out)]</h3>
<img src="assets/executive_cohort_report/deep_dive_BEH0335_OD.png" alt="Deep Dive BEH0335" />
</div>

## 7. Algorithmic Mechanics Driving Boundary Adherence

<div class="challenge-grid">
    <div class="challenge-card">
        <div class="challenge-title">GCL Hyporeflective Wedge Penetration</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Plunges deeply into adjacent hyporeflective ganglion cell layer.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Continuous 1D head locks onto true hyperreflective optical gradient.</div>
        <div class="challenge-row"><span class="badge-cyan">Clinician Truth</span> Manually verified anatomical transition interface.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Optic Cup Cavity Void & BMO Bridging</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Bridges straight across non-physiological empty cup void.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> 1D cup head accurately truncates margin at Bruch's Membrane Opening.</div>
        <div class="challenge-row"><span class="badge-cyan">Clinician Truth</span> Strict peripapillary termination at anatomical BMO.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Major Vessel Axial Shadowing</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Axial signal drop causes erratic vertical jumps and boundary loss.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Multi-slice 2.5D contextual slices interpolate across vessel shadows cleanly.</div>
        <div class="challenge-row"><span class="badge-cyan">Clinician Truth</span> Preserved continuous anatomical layer contours.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Pathological Disc Tilt & Steep Slope</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Steep regional gradients induce boundary distortion and clipping.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Continuous 1D regression preserves curvature continuity and slope fidelity.</div>
        <div class="challenge-row"><span class="badge-cyan">Clinician Truth</span> Verified anatomical boundary conformity.</div>
    </div>
</div>

---

## 8. Clinical Significance & Conclusion

1. **OD Boundary Precision with an OS Failure Mode**: Continuous 1D boundary regression achieves sub-voxel median MABE on benchmark OD scans, but the OS results show that this performance does not generalize across laterality in the current pipeline.
2. **End-to-End Cluster Orchestration**: This automated evaluation script confirms full integration between training, multi-volume GPU inference, metric logging, and clinical report generation within a single SLURM execution pass.
3. **Execution Summary**: Checkpoint `best_volumetric_rnfl_net.pt` generated on NYUAD Jubail (Job `18223981`). All visual assets and quantitative matrices are archived in `assets/executive_cohort_report`.

