# OCT B-Scan Adaptive Preprocessing & Tissue Masking Engine

## Overview

The **OCT B-Scan Adaptive Preprocessing Engine** provides a modular, production-grade image preprocessing and anatomical tissue masking pipeline designed for high-resolution Optical Coherence Tomography (OCT) retinal B-scans.

The engine converts raw multi-vendor OCT scans (Heidelberg Spectralis, Topcon, Cirrus, Optovue, Bioptigen, NIDEK) into standardized, background-isolated, square-letterboxed B-scans while preserving **100% of pathological retinal tissue**, fluid cysts, subretinal domes, and low-reflectivity choroidal structures.

---

## Architecture & Module Organization

The preprocessing engine is organized as a modular, single-responsibility Python package under `image-classification-model-training/data/preprocessing/`:

```
image-classification-model-training/data/preprocessing/
├── __init__.py           # Re-exports public API symbols for package access
├── masking.py            # Dynamic dual-pass noise-floor binarization & SFCM stroma segmentation
├── outliers.py           # 1D trace Hampel outlier rejection & gradient support validation
├── params.py             # Single source of truth parameter manager (folder_params.json)
├── pipeline.py           # Orchestration workflow function (process_image)
├── white_bars.py         # Raycast scanner annotation bar & dynamic compass UI artifact removal
├── README.md             # Technical documentation
└── tuning/               # Dedicated Preprocessing Tuning Sub-Application
    ├── __init__.py       # Package init exporting run_server, FineTuningRequestHandler
    ├── __main__.py       # CLI entrypoint for `python -m data.preprocessing.tuning`
    ├── server.py         # HTTP Server & API endpoints
    └── dashboard/        # Modular Web Application (index.html, css/, js/)
```

For backwards compatibility, `image-classification-model-training/data/preprocessing.py` remains as a top-level module re-exporting all package functions, and `scripts/tuning_server.py` delegates to `data.preprocessing.tuning.server`.

---

## Core Algorithmic Mechanics

### 1. Dual-Pass Noise-Floor Adaptive Binarization
Global Otsu thresholding (~70-90) fits bright upper retinal layers well, but prematurely cuts off low-reflectivity choroidal tissue (intensity 30-50). The dual-pass engine resolves this using two adaptive passes:

- **Pass 1 ($Y_{\text{top}}$ ILM Surface)**: Evaluates Otsu thresholding strictly on pixels above the background noise floor ($\text{noise\_cutoff} = \max(25, \lfloor \mu_{\text{bg}} + 1.5 \sigma_{\text{bg}} \rfloor)$). This ignores vitreous haze and top border noise, fitting the true Inner Limiting Membrane (ILM) surface tightly.
- **Pass 2 ($Y_{\text{bottom}}$ Choroid Surface)**: Evaluates a tight background noise-floor threshold:
  $$\text{thresh\_bot\_val} = \max(\lfloor \mu_{\text{bg}} + 3.0 \cdot \sigma_{\text{bg}} + 5.0 \rfloor, \lfloor \text{thresh\_top\_val} \times 0.55 \rfloor)$$
  This captures 100% of low-reflectivity choroidal tissue down to the choroid floor while stopping strictly above background noise, eliminating dangling background space.

### 2. Dynamic Wide Shadow Bridging
Pathological fluid cysts and vessel shadows in RVO, DME, and CNV scans cast wide vertical attenuation beams (60-120px). The pipeline dynamically computes horizontal closing kernel width based on image resolution:
$$\text{effective\_sb\_px} = \max(121, \lfloor W \times 0.20 \rfloor)$$
This spans wide shadow beams across all scan resolutions without creating V-notch artifacts in the bottom boundary.

### 3. Organic Continuous Gaussian Boundary Smoothing
Discrete minimum filtering creates blocky $90^\circ$ staircase step corners that act as artificial high-frequency spatial impulse functions, polluting CNN convolutional feature maps. 

The pipeline replaces discrete step filters with **1D Gaussian filtering ($\sigma = 15.0$)** on both top and bottom vectors:
$$y_{\text{top\_final}} = \text{GaussianFilter}(y_{\text{top\_clean}}, \sigma = 15.0)$$
$$y_{\text{bottom\_final}} = \text{GaussianFilter}(y_{\text{bottom\_clean}}, \sigma = 15.0)$$
This produces $C^\infty$ continuously differentiable, organic anatomical curves with **zero sharp step squares**.

### 4. Scanner-Provenance UI Artifact Removal
Orientation compass wireframe boxes ('S', 'I', 'N', 'T') appear in Heidelberg Spectralis subfolders (e.g. `CHU_MH`).

- **Provenance Profiling (`is_spectralis_ui_candidate`)**: Restricts compass detection to Spectralis subfolders. Clean dataset folders (`RAO`, `OCT2017`, `CNV`, `DME`, `DRUSEN`, `AMD`) bypass UI rules completely, guaranteeing **$0\%$ false positives** on clean scans.
- **Topological Hierarchy Verification**: For Spectralis scans, candidate contours must contain **nested child contours (`n_children >= 2`)** (crosshair arrows and text letters) AND a high perimeter-to-area ratio ($\frac{P^2}{A} > 18.0$). Real choroidal tissue chunks (which have zero nested children) are never touched.

---

## ⚙️ Parameter Configuration (`folder_params.json`)

All 20 subfolders in the dataset are configured via [`image-classification-model-training/data/folder_params.json`](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/data/folder_params.json) (mirrored at [`data/folder_params.json`](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/data/folder_params.json)), which serves as the single source of truth across the interactive dashboard, server API, and processing scripts.

### Full Parameter Specification

| Parameter | Type | Default | Description |
|---|---|---|---|
| `top_noise_mult` | `float` | `1.5` | Pass 1 multiplier for ILM surface detection threshold ($\mu_{\text{bg}} + k \cdot \sigma_{\text{bg}}$) |
| `bot_noise_mult` | `float` | `3.0` | Pass 2 multiplier for choroid floor threshold ($\mu_{\text{bg}} + k \cdot \sigma_{\text{bg}}$) |
| `use_sfcm` | `bool` | `false` | When `true`, uses Spatial Fuzzy C-Means to segment the choroid stroma instead of Otsu |
| `shadow_bridge_top_pct` | `int` | `20` | Structuring element width (% of image width) for bridging vessel shadows across the ILM |
| `shadow_bridge_bot_pct` | `int` | `20` | Structuring element width (% of image width) for bridging vessel shadows across the choroid |
| `gaussian_sigma` | `int` | `15` | Gaussian smoothing filter standard deviation applied to boundary vectors |
| `margin_top` | `int` | `15` | Top padding margin in pixels above the ILM surface |
| `margin_bottom` | `int` | `15` | Bottom padding margin in pixels below the choroid boundary |
| `top_spike_suppress_px` | `int` | `0` | Peak threshold in pixels for detecting and interpolating upward spikes on the top boundary |
| `top_spike_window_px` | `int` | `80` | Rolling median window size for upward spike suppression |
| `top_dip_suppress_px` | `int` | `0` | Peak threshold in pixels for detecting and interpolating downward dips on the top boundary |
| `top_dip_window_px` | `int` | `80` | Rolling median window size for downward dip suppression |
| `sfcm_margin_bottom` | `int` | `15` | Extra padding margin in pixels applied to the SFCM choroid boundary |
| `sfcm_gaussian_sigma` | `int` | `15` | Gaussian smoothing standard deviation for the SFCM boundary |
| `sfcm_n_clusters` | `int` | `3` | Number of clusters for Fuzzy C-Means segmentation |
| `sfcm_fuzziness_m` | `float` | `2.0` | Fuzziness exponent $m$ for Fuzzy C-Means segmentation |
| `compass_ui_enabled` | `bool` | `false` | Toggles scanner-provenance circular compass UI artifact removal |
| `compass_location` | `str` | `"auto"` | Compass search region: `"auto"`, `"bottom_left"`, or `"bottom_right"` |

---

## Interactive Fine-Tuning Dashboard (`data.preprocessing.tuning`)

An interactive local web server enables visual folder-by-folder calibration with real-time SVG boundary overlay rendering.

### Launching the Dashboard Server
```bash
# Recommended module command
KMP_DUPLICATE_LIB_OK=TRUE python3 -m data.preprocessing.tuning

# Or via the forwarding script
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/tuning_server.py
```
Open **`http://localhost:8000`** in your browser.

### Interactive Features
- **Live SVG Vector Overlays**: Displays pixel-perfect cyan (Top/ILM), pink (Bottom/Choroid), and orange (SFCM Choroid) boundary vectors.
- **Interactive Control Handles**: Drag control points directly on the SVG overlay to adjust margins in real time.
- **Dual Mode Support**:
  - **Otsu Mode**: Dual-pass adaptive binarization for standard scans.
  - **SFCM Mode**: SOTA Dijkstra Dynamic Programming RPE detection + Spatial Fuzzy C-Means clustering to isolate choroidal stroma.
- **Dynamic Capping**: Automatically interpolates boundaries across compass UI boxes and enforces column blackout floor limits.
- **Save to JSON**: Click "Save Parameters for this Folder" to persist parameters directly to `data/folder_params.json`.
- **Parameter Tooltips**: Hover over `(i)` badges for medical and algorithmic definitions.

---

## End-to-End Workflow: Tuning to Full Batch Preprocessing

```
[ Raw Scans in Classified/ ]
             │
             ▼
[ Interactive Tuning Server (http://localhost:8000) ]
   ├── Visual inspection of sample scans
   ├── Slider tuning (Otsu / SFCM / Margins / Spikes)
   └── "Save Parameters" button
             │
             ▼
[ data/folder_params.json ]  (Single Source of Truth)
             │
             ▼
[ Batch Preprocessing Script (process_image) ]
   ├── Reads folder_params.json per subfolder
   ├── Detects & removes compass artifacts & white bars
   ├── Evaluates Otsu Top + (SFCM or Otsu) Bottom envelope
   ├── Applies letterbox square padding (384x384)
   └── Writes isolated tissue scans to Classified-preprocessed/
```

---

## Usage & Code Examples

### 1. Process an Image File
```python
from data.preprocessing import process_image

# Automatically loads parameters from folder_params.json based on parent folder name
success = process_image(
    src_path="/path/to/Classified/CHU_MH/MH_surgery_others_211_H.jpeg",
    dst_path="/path/to/Classified-preprocessed-Otsu/CHU_MH/MH_surgery_others_211_H.jpeg",
    frame=True,
    frame_size=384,
    quality=95
)
```

### 2. Generate a Tissue Mask for an Image Array
```python
import cv2
from data.preprocessing import generate_tissue_mask, get_folder_params

img = cv2.imread("scan.jpg", cv2.IMREAD_GRAYSCALE)
params = get_folder_params("CHU_MH")

mask = generate_tissue_mask(img, **params)
```

---

## 📊 Verification & Test Tools

- **Run Automated Test Suite**:
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_tuning_dashboard.py -v
  ```
- **Regenerate Representative Dataset & HTML Gallery**:
  ```bash
  python3 scripts/generate_otsu_dataset_and_gallery.py
  ```
- **HTML Gallery Output**: `Classified-preprocessed-Otsu/gallery.html`

