# OCT B-Scan Adaptive Preprocessing & Coupled Multi-Surface Engine

## Overview

The **OCT B-Scan Adaptive Preprocessing Engine** provides a modular, production-grade image preprocessing and coupled multi-surface anatomical tissue masking pipeline designed for high-resolution Optical Coherence Tomography (OCT) retinal B-scans.

The engine standardizes raw multi-vendor OCT scans (Heidelberg Spectralis, Topcon, Cirrus, Optovue, Bioptigen, NIDEK) into background-isolated, square-letterboxed $384 \times 384$ B-scans while strictly preserving **100% of pathological retinal tissue**, intraretinal fluid cysts, pigment epithelial detachments (PED), subretinal domes, and choroidal vascular lumens.

---

## Architecture & Module Organization

The preprocessing engine is organized as a modular, single-responsibility Python package under `training/classification/data/preprocessing/`:

```text
training/classification/data/preprocessing/
├── __init__.py           # Re-exports public API symbols for package access
├── multisurface.py       # Simultaneous Coupled Multi-Surface Graph Optimizer & B-Spline Regularizer
├── masking.py            # Dynamic dual-pass noise-floor binarization & SFCM stroma segmentation
├── outliers.py           # 1D trace Hampel outlier rejection & gradient support validation
├── params.py             # Single source of truth parameter manager (folder_params.json)
├── pipeline.py           # Orchestration workflow function (process_image)
├── white_bars.py         # Raycast scanner annotation bar & dynamic compass UI artifact removal
├── README.md             # Technical documentation
├── PLAN_2_NEURAL_SEGMENTATION.md # Roadmap for deep learning U-Net segmentation
└── tuning/               # Preprocessing Tuning & Verification Sub-Application
    ├── __init__.py       # Package init exporting run_server, FineTuningRequestHandler
    ├── __main__.py       # CLI entrypoint for `python -m data.preprocessing.tuning`
    ├── boundaries.py     # Coupled multi-surface & intelligent auto boundary algorithms
    ├── processor.py      # Batch folder & single-scan processing engine with overlay rendering
    ├── server.py         # HTTP Server, API endpoints, & Unified CLI dispatcher
    └── dashboard/        # Modular Web Application (index.html, css/, js/)
```

---

## Mathematical Formulation: Coupled Multi-Surface Graph Optimization & B-Spline Regularization

### 1. Joint Multi-Surface Energy Minimization
Instead of detecting boundaries sequentially where errors cascade, the engine solves for $\mathbf{S} = (S_{\text{ILM}}, S_{\text{RPE}}, S_{\text{Choroid}})$ simultaneously:

$$E(\mathbf{S}) = \sum_{k=1}^{3} E_{\text{data}}^{(k)}(S_k) + \lambda_{\text{smooth}} \sum_{k=1}^{3} E_{\text{smooth}}(S_k) + \sum_{k=1}^{2} E_{\text{couple}}(S_k, S_{k+1})$$

Where:
- **ILM Data Cost ($E_{\text{data}}^{(1)}$)**: Evaluates the downward dark-to-bright entry gradient normalized against vitreous cavity darkness:
  $$E_{\text{data}}^{(1)}(y, x) = 1.0 - \left(w_{\text{grad}} \cdot \nabla I_{\text{down}}(y, x)\right) + 0.50 \cdot \operatorname{clip}\left(\frac{\bar{I}_{\text{above}}(y, x) - 35}{25}, 0, 1\right)$$
  Where $\bar{I}_{\text{above}}(y, x) = \frac{1}{12}\sum_{k=1}^{12} I(y-k, x)$. If the region above is already bright, the cost penalizes internal retinal layers (such as the Inner Plexiform Layer), keeping the ILM strictly at the vitreous-retinal boundary.
- **RPE Data Cost ($E_{\text{data}}^{(2)}$)**: Evaluates the dominant hyperreflective melanin band intensity combined with the Bruch's membrane downward gradient ridge:
  $$E_{\text{data}}^{(2)}(y, x) = 1.0 - \left(0.50 \cdot \frac{I(y, x)}{\max(I)} + 0.50 \cdot \nabla I_{\text{down}}(y, x)\right)$$
- **Choroid CSI Cost ($E_{\text{data}}^{(3)}$)**: Solves the choroid-scleral interface (CSI) using Spatial Fuzzy C-Means (SFCM) stroma-to-sclera transition gradient and shadow compensation.

### 2. Hard Physiological Coupling Constraints
Inter-surface capacity edges enforce strict physical anatomical bounds across all columns:

$$\Delta_{\text{retina\_min}} \le S_{\text{RPE}}(x) - S_{\text{ILM}}(x) \le \Delta_{\text{retina\_max}} \quad \forall x \in [0, W-1]$$
$$\Delta_{\text{choroid\_min}} \le S_{\text{Choroid}}(x) - S_{\text{RPE}}(x) \le \Delta_{\text{choroid\_max}} \quad \forall x \in [0, W-1]$$

- $\Delta_{\text{retina\_min}} = 25\text{px}$, $\Delta_{\text{retina\_max}} = 270\text{px}$ (accommodates healthy foveal depression up to massive central DME cyst elevations).
- $\Delta_{\text{choroid\_min}} = 15\text{px}$, $\Delta_{\text{choroid\_max}} = 220\text{px}$.

### 3. Continuous B-Spline Curvature Regularization
Discrete boundary coordinates are projected onto a cubic B-spline basis $B_j(x)$ ($C^2$ continuous):

$$S(x) = \sum_{j=1}^{M} c_j B_j(x)$$

- Mathematically eliminates 1-pixel staircase teeth, local dropoffs, or sharp dips.
- Preserves smooth natural ocular curvature ($R > 10\text{mm}$) and dome-shaped pathological lesions.

---

## Unified Command-Line Interface (CLI)

The tuning and preprocessing engine can be invoked directly from the terminal without launching the browser UI:

### 1. Process a Single Image
```bash
python3 scripts/tuning_server.py --image "/path/to/scan.png" --out-dir ./output --save-mask
```

### 2. Batch Process Folder Samples
```bash
python3 scripts/tuning_server.py --folder "Chiu_BOE_2014-DME" --sample 10 --out-dir ./output --save-overlay
```

### 3. Extract JSON Metrics
```bash
python3 scripts/tuning_server.py --image "Subject_01_slice_010.png" --folder "Chiu_BOE_2014-DME" --json
```

### Available CLI Flags
- `--image` / `-i`: Path to an image file or filename within a dataset folder.
- `--folder` / `-f`: Subfolder name in the dataset (e.g. `Chiu_BOE_2014-DME`, `CHU_MH`, `AMD`).
- `--config` / `-c`: Inline JSON string or path to a `.json` configuration file.
- `--out-dir` / `-o`: Directory where processed images, raw scans, masks, and overlays are written (default: `./scratch/cli_output`).
- `--save-overlay` / `--no-overlay`: Generates diagnostic RGB overlays with cyan ILM, green RPE, orange Choroid, and purple vascular hole contours (default: true).
- `--save-mask` / `--no-mask`: Generates binary tissue mask `.png` files (default: false).
- `--sample`: Process N random sample images from the specified folder.
- `--json`: Dumps extracted 384x384 boundary vectors, layer thickness, and hole metrics to stdout as formatted JSON.

---

## 6-Point Anatomical Invariant Quality Gate

Every processed slice is verified against 6 invariant biological rules:
1. **Layer Hierarchy Order**: $y_{\text{ILM}}(x) < y_{\text{RPE}}(x) < y_{\text{Choroid}}(x) \quad \forall x$.
2. **Physiological Retinal Thickness**: $30\text{px} \le \bar{T}_{\text{retina}} \le 350\text{px}$.
3. **Physiological Choroid Thickness**: $15\text{px} \le \bar{T}_{\text{choroid}} \le 250\text{px}$.
4. **Boundary Clearance**: $\bar{y}_{\text{ILM}} > 5.0\text{px}$ (prevents clipping the scanner top).
5. **Continuous Trajectory**: $\max |\Delta y_{\text{ILM}}| \le 25.0\text{px}$ (forbids unphysical step discontinuities).
6. **Retinal Mass Centroid Proximity**: Anchors the search envelope to the primary tissue cross-section.

---

## Roadmap: Plan 2 Deep Learning Retinal Segmenter
For the upcoming deep learning transition using `OCT5K` ground-truth masks and semi-supervised active pseudo-labeling, refer to [PLAN_2_NEURAL_SEGMENTATION.md](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/training/classification/data/preprocessing/PLAN_2_NEURAL_SEGMENTATION.md).
