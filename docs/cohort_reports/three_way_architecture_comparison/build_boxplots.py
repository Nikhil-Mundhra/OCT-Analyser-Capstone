import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[4]

# Paths to the 40-scan validation metrics
bp_p = ROOT_DIR / 'OCT-Analyser-Capstone/docs/cohort_reports/2026-10-03_biplanar_expanded_18563914/assets/executive_cohort_report/cohort_evaluation_metrics.json'
d3_p = ROOT_DIR / 'train-cnn-models/checkpoints/rnfl_3d_18770720/cohort_evaluation_metrics.json'
d3_base_p = ROOT_DIR / 'train-cnn-models/checkpoints/rnfl_3d_18574378/cohort_evaluation_metrics.json'
tu_p = ROOT_DIR / 'train-cnn-models/checkpoints/rnfl_transunet_20m_18777225_ep12/cohort_evaluation_metrics.json'

bp = json.loads(bp_p.read_text())['scans']
d3 = json.loads(d3_p.read_text())['scans']
d3_base = json.loads(d3_base_p.read_text())['scans']
tu = json.loads(tu_p.read_text())['scans']

t_eff = [d3_base[i]['unet_mabe'] / (1.0 - d3_base[i]['unet_dice']) for i in range(len(d3_base))]
def compute_surf_dice(scans):
    return [1.0 - (scans[i]['unet_mabe'] / t_eff[i]) for i in range(len(scans))]

mabe_data = [[s['unet_mabe'] for s in bp], [s['unet_mabe'] for s in d3], [s['unet_mabe'] for s in tu]]
surf_dice_data = [compute_surf_dice(bp), compute_surf_dice(d3), compute_surf_dice(tu)]
raw_dice_data = [[s['unet_dice'] for s in bp], [s['unet_dice'] for s in d3], [s['unet_dice'] for s in tu]]

labels = ['Bi-Planar 2.5D\n(6.6M, 18563914)', 'Dense 3D Std\n(20.3M, 18770720)', 'TransUNet 20M Peak\n(20.0M, Ep 12)']
colors = ['#16a34a', '#2563eb', '#ea580c']

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.3), dpi=200)

for ax, data, ylabel, title in [
    (ax1, mabe_data, 'MABE (µm)', 'A. Boundary Error (MABE, 40 Eyes)'),
    (ax2, surf_dice_data, 'Surface-Derived Anatomical Dice', 'B. Surface-Derived Anatomical Dice'),
    (ax3, raw_dice_data, 'Raw Voxel-Head Dice', 'C. Raw Voxel-Head Dice (Logit Dropout)')
]:
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
        
        flier.set_marker('D')
        flier.set_markersize(5.5)
        flier.set_markerfacecolor(c)
        flier.set_markeredgecolor('white')
        flier.set_markeredgewidth(1.0)
        flier.set_alpha(0.9)
        
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight='bold', color='#1e293b')
    ax.set_title(title, fontsize=10.2, fontweight='bold', pad=10, color='#0f172a')
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', labelsize=8.2)
    ax.tick_params(axis='y', labelsize=8.5)
    if ax == ax2:
        ax.set_ylim(0.84, 0.99)
    
    if ax == ax1:
        ax.plot([], [], color='#0f172a', linewidth=2.0, label='Median')
        ax.plot([], [], color='#dc2626', linewidth=1.8, linestyle='--', label='Mean')
        ax.plot([], [], marker='D', markersize=5.0, color='#64748b', linestyle='None', label='Outlier (>1.5×IQR)')
        ax.legend(loc='upper right', frameon=True, fontsize=7.8, facecolor='white', framealpha=0.92, edgecolor='#cbd5e1')

plt.tight_layout()
out_p = Path(__file__).resolve().parent / 'assets/three_way_comparison_boxplots.png'
plt.savefig(out_p, dpi=200, bbox_inches='tight')
print(f'Successfully generated clean Figure 1 to {out_p}')
