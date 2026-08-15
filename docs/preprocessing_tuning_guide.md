# OCT B-Scan Preprocessing: Tuning Server to Batch Pipeline Guide

Comprehensive technical documentation explaining the end-to-end architecture, configuration schema, and execution pipeline connecting the interactive tuning server to the production batch preprocessing engine.

---

## 1. Architectural Workflow

```
+-------------------------------------------------------------+
|                  Raw Scans in Classified/                   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|     Interactive Tuning Server (http://localhost:8000)       |
|     - Visual inspection of representative scans             |
|     - Real-time parameter sliders & vector dragging         |
|     - Mode toggle (Otsu Morphological vs SFCM Choroid)      |
|     - "Save Parameters for this Folder"                     |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|         Single Source of Truth Configuration Store          |
|                  (data/folder_params.json)                  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|            Production Batch Preprocessing Engine            |
|   (training/classification/data/preprocessing/)             |
|     1. Circular compass detection & zero-erasure            |
|     2. Scanner white annotation bar removal                 |
|     3. Dual-pass Otsu ILM top boundary                      |
|     4. SFCM or Otsu choroid bottom boundary                 |
|     5. Organic continuous envelope masking                  |
|     6. Letterbox square padding & resizing (384x384)        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|     Standardized Output (Classified-preprocessed-Otsu/)     |
+-------------------------------------------------------------+
```

---

## 2. Configuration Store Specification (`data/folder_params.json`)

All 20 subfolders in the dataset have dedicated parameter maps stored in [`data/folder_params.json`](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/data/folder_params.json) (mirrored in `image-classification-model-training/data/folder_params.json`).

### Parameter Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `top_noise_mult` | `float` | `1.5` | Top threshold cutoff multiplier: $\text{noise\_cutoff} = \mu_{\text{bg}} + k \cdot \sigma_{\text{bg}}$. Eliminates vitreous haze. |
| `bot_noise_mult` | `float` | `3.0` | Bottom threshold cutoff multiplier: captures low-reflectivity choroidal tissue down to the floor. |
| `use_sfcm` | `bool` | `false` | When `true`, enables Dijkstra DP RPE detection + Spatial Fuzzy C-Means for the bottom boundary. |
| `margin_top` | `int` | `15` | Upper padding margin (px) above the ILM to preserve vitreo-retinal interfaces and epiretinal membranes. |
| `margin_bottom` | `int` | `15` | Lower padding margin (px) below the choroid boundary. |
| `shadow_bridge_top_pct` | `int` | `20` | Structuring element width (% of image width) for bridging blood vessel shadows across the ILM. |
| `shadow_bridge_bot_pct` | `int` | `20` | Structuring element width (% of image width) for bridging attenuation beams across the choroid. |
| `gaussian_sigma` | `int` | `15` | Standard deviation $\sigma$ for 1D Gaussian smoothing filter on boundary vectors. |
| `top_spike_suppress_px` | `int` | `0` | Peak threshold (px) to detect and interpolate upward spikes on the ILM boundary. |
| `top_spike_window_px` | `int` | `80` | Rolling median window size (px) for upward spike detection. |
| `top_dip_suppress_px` | `int` | `0` | Peak threshold (px) to detect and interpolate downward dips on the ILM boundary. |
| `top_dip_window_px` | `int` | `80` | Rolling median window size (px) for downward dip detection. |
| `sfcm_margin_bottom` | `int` | `15` | Extra padding margin (px) applied specifically to the SFCM choroid boundary. |
| `sfcm_gaussian_sigma` | `int` | `15` | Gaussian smoothing $\sigma$ for the SFCM choroid boundary. |
| `sfcm_n_clusters` | `int` | `3` | Number of clusters for Fuzzy C-Means segmentation. |
| `sfcm_fuzziness_m` | `float` | `2.0` | Fuzziness exponent $m$ for Fuzzy C-Means clustering. |
| `compass_ui_enabled` | `bool` | `false` | Enables Heidelberg Spectralis orientation compass removal. |
| `compass_location` | `str` | `"auto"` | Compass search region: `"auto"`, `"bottom_left"`, or `"bottom_right"`. |

---

## 3. Interactive Tuning Dashboard (`data.preprocessing.tuning`)

The tuning sub-system is organized as a dedicated domain package within `training/classification/data/preprocessing/tuning/`:
- **Server:** [`data/preprocessing/tuning/server.py`](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/training/classification/data/preprocessing/tuning/server.py)
- **Web UI:** [`data/preprocessing/tuning/dashboard/`](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/training/classification/data/preprocessing/tuning/dashboard/) (`index.html`, `css/`, `js/`)
- **Forwarding Script:** [`scripts/tuning_server.py`](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/scripts/tuning_server.py)

### Launching the Tuning Server
```bash
# Direct module execution (Recommended)
KMP_DUPLICATE_LIB_OK=TRUE python3 -m data.preprocessing.tuning

# Or via the ergonomic root script
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/tuning_server.py
```
Open **`http://localhost:8000`** in any web browser.

### Key Capabilities
1. **Live Side-by-Side Comparison:** Compares raw scans directly against the letterboxed, masked output with SVG boundary vectors.
2. **Color Legend:**
   - **Cyan Line:** Top ILM Boundary vector ($y_{\text{top}} - \text{margin\_top}$).
   - **Pink Line:** Standard Otsu Bottom Boundary vector ($y_{\text{bottom}} + \text{margin\_bottom}$).
   - **Orange Line & Shaded Area:** Spatial Fuzzy C-Means (SFCM) Choroidal Stroma Boundary vector.
3. **Interactive Control Handles:** Drag control circles directly on the SVG overlay to interactively modify padding margins.
4. **Instant Persistence:** Clicking **"Save Parameters for this Folder"** sends a `POST /api/reprocess` request that updates `data/folder_params.json`.
5. **Contextual Tooltips:** Hover over any circular `(i)` badge next to a slider or toggle title to view its medical definition and algorithmic behavior.

---

## 4. Batch Preprocessing Execution (`data.preprocessing.pipeline`)

When the batch pipeline processes a folder (e.g. via [`scripts/generate_otsu_dataset_and_gallery.py`](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/scripts/generate_otsu_dataset_and_gallery.py) or `process_image`):

1. **Parameter Lookup:** `process_image` looks up the subfolder name in `data/folder_params.json` via `get_folder_params(folder_name)`.
2. **Scanner Artifact Removal:**
   - Circular compass boxes are detected and zero-erased using topological parent/child contour matching.
   - Horizontal white annotation bars ($\ge 190$ intensity) are detected and suppressed.
3. **Boundary Calculation:**
   - **ILM Top Surface:** Evaluated via Pass 1 adaptive Otsu above background noise, filtered with 1D Hampel outlier rejection and spike suppression, and smoothed with 1D Gaussian filtering ($\sigma = 15$).
   - **Choroid Bottom Surface:**
     - When `use_sfcm: false`: Uses Pass 2 Otsu bottom contour with wide shadow bridging and dynamic floor beam capping.
     - When `use_sfcm: true`: Evaluates Dijkstra Dynamic Programming RPE detection to lock onto Bruch's membrane, followed by Spatial Fuzzy C-Means clustering of the stroma layer.
   - **Blackout Box Capping:** Both boundaries are strictly capped at the top edge of any detected column blackout or compass regions.
4. **Organic Continuous Envelope Masking:** A continuous mask is rendered from $y_{\text{top}}[x] - \text{margin\_top}$ to $y_{\text{bottom}}[x] + \text{margin\_bottom}$. All background pixels outside this envelope are zeroed out (black).
5. **Square Letterboxing:** The masked image is padded symmetrically to a 1:1 square aspect ratio and resized to $384 \times 384$.

---

## 5. Python API Usage

### Processing a Single Scan File
```python
from data.preprocessing import process_image

# Reads parameters for the folder automatically from folder_params.json
success = process_image(
    src_path="/path/to/Classified/CHU_MH/scan_001.jpeg",
    dst_path="/path/to/Classified-preprocessed-Otsu/CHU_MH/scan_001.jpeg",
    frame=True,
    frame_size=384,
    quality=95
)
```

### Generating an Organic Mask Array Programmatically
```python
import cv2
from data.preprocessing import generate_tissue_mask, get_folder_params

img = cv2.imread("scan.jpg", cv2.IMREAD_GRAYSCALE)
params = get_folder_params("CHU_MH")

# Forwards calibrated parameters (margins, SFCM flags, sigma)
mask = generate_tissue_mask(img, **params)
```

---

## 6. Testing & Validation

Run the automated test suite to verify threshold calculations, vector downsampling, SFCM boundaries, and HTTP server endpoints:

```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_tuning_dashboard.py -v
```
