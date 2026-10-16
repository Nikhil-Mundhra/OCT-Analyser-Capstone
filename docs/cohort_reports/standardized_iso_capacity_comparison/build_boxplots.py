import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Base repository root
ROOT_DIR = Path(__file__).resolve().parents[4]

# Paths to the 40-scan validation metrics
bp_p = ROOT_DIR / 'OCT-Analyser-Capstone/docs/cohort_reports/2026-10-03_biplanar_expanded_18563914/assets/executive_cohort_report/cohort_evaluation_metrics.json'
d3_std_p = ROOT_DIR / 'train-cnn-models/checkpoints/rnfl_3d_18770720/cohort_evaluation_metrics.json'
d3_base_p = ROOT_DIR / 'train-cnn-models/checkpoints/rnfl_3d_18574378/cohort_evaluation_metrics.json'
tu_base_p = ROOT_DIR / 'OCT-Analyser-Capstone/docs/cohort_reports/2026-10-10_transunet_hybrid_18724310/assets/executive_cohort_report/cohort_evaluation_metrics.json'
tu_ep6_p = ROOT_DIR / 'train-cnn-models/checkpoints/rnfl_transunet_20m_18777225/cohort_evaluation_metrics.json'
tu_ep12_p = ROOT_DIR / 'train-cnn-models/checkpoints/rnfl_transunet_20m_18777225_ep12/cohort_evaluation_metrics.json'

bp = json.loads(bp_p.read_text())['scans']
d3_std = json.loads(d3_std_p.read_text())['scans']
d3_base = json.loads(d3_base_p.read_text())['scans']
tu_base = json.loads(tu_base_p.read_text())['scans']
tu_ep6 = json.loads(tu_ep6_p.read_text())['scans']
tu_ep12 = json.loads(tu_ep12_p.read_text())['scans']

# Per-eye effective RNFL slab thickness from continuous-slab 3D reference (0.0000 dropout)
t_eff = [d3_base[i]['unet_mabe'] / (1.0 - d3_base[i]['unet_dice']) for i in range(len(d3_base))]

def compute_surf_dice(scans):
    return [1.0 - (scans[i]['unet_mabe'] / t_eff[i]) for i in range(len(scans))]

mabe_data = [
    [s['unet_mabe'] for s in bp],
    [s['unet_mabe'] for s in d3_std],
    [s['unet_mabe'] for s in d3_base],
    [s['unet_mabe'] for s in tu_base],
    [s['unet_mabe'] for s in tu_ep6],
    [s['unet_mabe'] for s in tu_ep12],
]

p95_data = [
    [s.get('unet_p95', s.get('unet_p95_error', 0.0)) for s in bp],
    [s.get('unet_p95', s.get('unet_p95_error', 0.0)) for s in d3_std],
    [s.get('unet_p95', s.get('unet_p95_error', 0.0)) for s in d3_base],
    [s.get('unet_p95', s.get('unet_p95_error', 0.0)) for s in tu_base],
    [s.get('unet_p95', s.get('unet_p95_error', 0.0)) for s in tu_ep6],
    [s.get('unet_p95', s.get('unet_p95_error', 0.0)) for s in tu_ep12],
]

surf_dice_data = [
    compute_surf_dice(bp),
    compute_surf_dice(d3_std),
    compute_surf_dice(d3_base),
    compute_surf_dice(tu_base),
    compute_surf_dice(tu_ep6),
    compute_surf_dice(tu_ep12),
]

raw_dice_data = [
    [s['unet_dice'] for s in bp],
    [s['unet_dice'] for s in d3_std],
    [s['unet_dice'] for s in d3_base],
    [s['unet_dice'] for s in tu_base],
    [s['unet_dice'] for s in tu_ep6],
    [s['unet_dice'] for s in tu_ep12],
]

labels = [
    'Bi-Planar 2.5D\n(6.6M, 18563914)',
    'Dense 3D Std\n(20.3M, 18770720)',
    'Dense 3D Base*\n(20.3M, 18574378)',
    'TransUNet Base\n(5.1M, 18724310)',
    'TransUNet 20M\n(Ep 6, 18777225)',
    'TransUNet 20M\n(Ep 12, Peak)'
]

colors = ['#16a34a', '#2563eb', '#9333ea', '#db2777', '#ea580c', '#c2410c']

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15.5, 8.2), dpi=200)

plot_specs = [
    (ax1, mabe_data, 'MABE (µm)', 'A. Boundary Error (MABE) across 40 Held-Out Eyes'),
    (ax2, p95_data, 'P95 Error (µm)', 'B. Tail Dispersion (P95) across 40 Held-Out Eyes'),
    (ax3, surf_dice_data, 'Surface-Derived Anatomical Dice', 'C. Surface-Derived Anatomical Slab Dice (from ILM/NFL Curves)'),
    (ax4, raw_dice_data, 'Raw Voxel-Head Dice', 'D. Raw Voxel-Head Dice (Showing Multi-Task Logit Suppression)')
]

for ax, data, ylabel, title in plot_specs:
    bp_plot = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=True,
        medianprops=dict(color='#0f172a', linewidth=2.0),
        meanprops=dict(color='#dc2626', linewidth=1.8, linestyle='--'),
        boxprops=dict(linewidth=1.6),
        whiskerprops=dict(linewidth=1.4, color='#334155'),
        capprops=dict(linewidth=1.4, color='#334155')
    )
    
    for idx, (patch, flier, c) in enumerate(zip(bp_plot['boxes'], bp_plot['fliers'], colors)):
        patch.set_facecolor(c)
        patch.set_alpha(0.28)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.8)
        if idx == 2:  # Baseline ablation model with dashed edge
            patch.set_linestyle('--')
        
        # Outlier flier styling
        flier.set_marker('D')
        flier.set_markersize(5.0)
        flier.set_markerfacecolor(c)
        flier.set_markeredgecolor('white')
        flier.set_markeredgewidth(1.0)
        flier.set_alpha(0.9)
        
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight='bold', color='#1e293b')
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=10, color='#0f172a')
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', labelsize=8.0)
    ax.tick_params(axis='y', labelsize=8.5)
    if ax == ax3:
        ax.set_ylim(0.84, 0.99)
    
    # Legend for median / mean / outlier
    if ax == ax1:
        ax.plot([], [], color='#0f172a', linewidth=2.0, label='Median')
        ax.plot([], [], color='#dc2626', linewidth=1.8, linestyle='--', label='Mean')
        ax.plot([], [], marker='D', markersize=5.0, color='#64748b', linestyle='None', label='Outlier (>1.5×IQR)')
        ax.legend(loc='upper right', frameon=True, fontsize=8.0, facecolor='white', framealpha=0.92, edgecolor='#cbd5e1')

plt.tight_layout()
out_p = Path(__file__).resolve().parent / 'assets/iso_capacity_comparison_boxplots.png'
out_p.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_p, dpi=200, bbox_inches='tight')
print(f'Successfully generated clean Figure 1 to {out_p}')
