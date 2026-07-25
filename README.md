---
name: oct-analyser-agent-skill
description: "Comprehensive guide and agent skill for navigating, modifying, and understanding the OCT-Analyser-Capstone repository."
---

# Local OCT Analyzer MVP

Local OCT Analyzer MVP is a full-stack, local-first OCT/OCTA clinical workflow prototype. It combines a **FastAPI backend**, a **React wireframe-style clinical interface**, and a **Python OCT processing pipeline** for loading, preprocessing, flattening, previewing, and feature-extracting both **3D Optical Coherence Tomography (OCT) volumes** and **2D images**.

This file serves as both the **main project documentation** and a **Skill for AI Agents** assisting with the codebase. 

---

## Project Structure

```text
OCT-Analyser-Capstone/
├── .github/
│   └── workflows/
│       └── pytest.yml          # GitHub Actions test workflow
├── backend/                    # FastAPI backend and Pipeline
│   ├── Dockerfile
│   ├── pytest.ini
│   ├── requirements.txt
│   ├── oct_analyzer/           # Core ingestion, pipeline, and API 
│   └── tests/
├── frontend/                   # Next.js interface (clinical UI)
│   ├── index.html
│   ├── package.json
│   └── src/
├── image-classification-model-training/
│   ├── Documentation/
│   ├── models/
│   └── training/
├── image-segmentation-model-training/
│   ├── Documentation/
│   ├── models/
│   └── scripts/
├── OCT-Segmentation-Model/
│   ├── main.py                 # 15-layer Hierarchical U-Net inference
│   └── models/
└── vercel.json                 # Vercel deployment configuration
```

## Current Capabilities

### Backend and Pipeline
- Load Heidelberg `.vol` files with `eyepy`.
- Load DICOM `.dcm` files and extract pixel data, rescale metadata, and spacing.
- Load zipped image-stack exports containing sorted TIFF/BMP/PNG slices.
- Load 2D standard images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.bmp`) leveraging `PIL` for single-scan analysis.
- Normalize all MVP uploads into a shared `(Z, Y, X)` NumPy volume contract.
- Flatten OCT stacks relative to the retinal pigment epithelium (RPE).
- Run basic QC reporting: signal range, crop status, warnings, volume shape, source format, and metadata.
- Generate preview images for raw center slice, cropped slice, segmentation overlay, and CDF feature chart.
- Support 2D segmentation natively via PyTorch UNet models and classify using ResNet/EfficientNet models.
- **Generate deterministic demo 12-layer segmentation** for the MVP pipeline (with a clear upgrade path to the trained **15-layer Hierarchical U-Net model**).

### Local API
- `POST /api/scans` uploads one `.vol`, `.dcm`, `.zip` export, or a 2D image.
- `GET /api/scans/{scan_id}` returns status, diagnosis, confidence, QC, metadata, layer votes, CDF deciles, and preview URLs.
- `POST /api/segment_2d` runs a localized 2D segmentation script, invoking the trained 15-layer Hierarchical U-Net checkpoint.

### Frontend
- Uses the clinical wireframe-style interface as the main app at `http://127.0.0.1:3000`.
- Includes triage worklist, upload/QC, scan review, AI findings, human decision gate, and outcomes/audit screens.
- Upload/QC screen allows dragging/dropping of 3D volumes or 2D image datasets.
- Review screen displays generated scan previews and layer findings with dynamic SVG overlays for clinical evidence.

## Roadmap & Future Promises (To Be Delivered)

The application currently acts as a strong foundational MVP, but there are major components scheduled for future deployment:
- **Clinical Validation:** The app must undergo formal clinical trials and robust accuracy evaluations; it cannot currently be used for real patient diagnosis.
- **Trained 15-Layer Anatomical Segmentation:** The MVP currently uses a 12-layer deterministic placeholder. We will fully integrate our newly developed **15-layer Hierarchical U-Net model** (located in `OCT-Segmentation-Model/`) for robust anatomical structure extraction in the 3D pipeline.
- **Live Classification Inference:** 
  > **[WARNING] The live classification pipeline is currently running with a RANDOMLY INITIALIZED ConvNeXt V2 model because the trained weights (`multi_head.pth`) are missing from the `hf_space/weights/` directory.** The legacy models were only partially trained (last layers only). A full training run on real OCT data is required before classification outputs can be trusted.
- **Background Jobs & Database Persistence:** Very large scans will soon require a background job queue (e.g., Celery/Redis). Currently, scan state only lives in the local filesystem/process memory.
- **Enterprise Capabilities:** Future releases will introduce PDF report generation, HIPAA-compliant access controls, user authentication, and audit-grade clinical logging.
- **Proprietary Archive Support:** Expanding beyond standard `.dcm` and `.vol`, future ingestion targets may include reverse-engineering proprietary Solix archives (e.g. `.fds`).

## Requirements

- Python 3.11+
- PyTorch, MONAI, NumPy, SciPy, SimpleITK, eyepy
- FastAPI, Uvicorn, Pillow
- Next.js, React, Tailwind CSS v4
- pytest and pytest-cov for tests

Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
npm --prefix frontend install
```

## Usage

### Local MVP App

Run the full local MVP (Frontend, Backend, and Celery worker) from a fresh checkout with one command:
```bash
./start.sh
```
That script installs dependencies, boots up the Next.js frontend, the FastAPI backend, and the Redis-backed Celery worker concurrently, while tailing their outputs from the `logs/` directory.

Useful Make targets:
```bash
make install  # install Python and Node dependencies
make build    # build the frontend bundle
make test     # run the Python test suite
make clean    # remove runtime/cache artifacts
```

## Live Deployments (Vercel & Hugging Face)

### 1. Frontend (Vercel)
The repository uses Vercel's native Next.js integration. The Next.js frontend builds natively and handles routing, served from the edge/serverless functions. The static UI is deployed and live at:
[https://oct-analyser-capstone.vercel.app](https://oct-analyser-capstone.vercel.app)

To connect the hosted UI to a locally running backend, append the local backend API base URL:
```text
https://oct-analyser-capstone.vercel.app/?apiBase=http://127.0.0.1:8000
```

### 2. Model Inference (Hugging Face Spaces)
The heavy PyTorch and ML inference tasks are deployed as remote microservices on **Hugging Face Spaces**. The frontend client is configured to automatically communicate with these cloud deployments for Live AI findings:
*   **Classification Service**: [https://nmundhra-oct-image-classifier-model.hf.space](https://nmundhra-oct-image-classifier-model.hf.space)
*   **Segmentation Service**: [https://nmundhra-oct-segmentation-model.hf.space](https://nmundhra-oct-segmentation-model.hf.space)

## Testing & Docker

Run the test suite:
```bash
pytest -c backend/pytest.ini backend/tests
```

Build and run Docker image:
```bash
docker build -f backend/Dockerfile -t relaynet-3d .
docker run --rm relaynet-3d python -m backend.oct_analyzer.main
```

---

## Developer & Architecture Guide (Including AI Agent Skill Context)

If you are a new developer or an AI agent analyzing this repository, read this section carefully to understand the domain, architecture, data schemas, and implementation blueprint.

### 1. Domain Context
*   **OCT / OCTA**: Optical Coherence Tomography. These are volumetric medical scans of the retina.
*   **Volumetric Data (3D)**: Scans come as a 3D block of data (a stack of individual 2D cross-sectional images called B-scans). The project normalizes these to a `(Z, Y, X)` NumPy/PyTorch array.
*   **2D Image Data**: The system can also ingest single 2D scans/images. They are loaded and converted into a `(1, Y, X)` normalized array for standard processing, or handled via specific 2D endpoints (e.g., `/api/segment_2d`).
*   **Medical Formats**: The system ingests `.vol` (Heidelberg), `.dcm` (DICOM, via `pydicom`), `.zip` (TIFF/BMP/PNG stacks), and standard 2D images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`, `.bmp`). 

### 2. Codebase Architecture & Features
*   **`backend/` (FastAPI / PyTorch)**:
    *   **Data Ingestion (`oct_analyzer/data_loader.py`)**: Uses `eyepy`, `pydicom`, and `PIL` to extract raw pixels and spatial metadata (`Spacing`). Returns a normalized 3D array (even for 2D images).
    *   **Pipeline (`oct_analyzer/mvp_pipeline.py`)**: Responsible for:
        1. Fovea detection and auto-cropping (standardizing scan dimensions).
        2. RPE-based anatomical flattening.
        3. Feature extraction: Calculates **MGRF** (2nd-order reflectivity Gibbs energy) and extracts **CDF** (Cumulative Distribution Function) deciles.
        4. Classification.
    *   **API (`oct_analyzer/api.py`)**: Exposes `/api/scans` (upload), `/api/segment_2d` (2D UNet predictions), and preview endpoints.
*   **`frontend/` (Next.js / React)**:
    *   Provides a clinical drag-and-drop workflow. 
    *   **`src/app/`**: Core Next.js routing (Worklist, QC, Review, Human Decision Gate, Outcomes). Implements SVG overlays for segmentation masks (e.g. IRF, SRF lesions).
    *   **`src/api/octAnalyzerClient.js`**: Frontend API client.
*   **`OCT-Segmentation-Model/`** & **`image-classification-model-training/`**: 
    *   Standalone ML repositories for training the models.
    *   Classification utilizes a **Unified Multi-Head ConvNeXt V2** model. A shared feature backbone extracts global representations, which then feed into three specialized heads: **Head 1** (Normal vs. Abnormal), **Head 2** (5 Broad Pathology Families), and **Head 3** (11 Granular Biomarkers/Severities).
    *   **Segmentation utilizes a custom 15-layer Hierarchical U-Net model** (`n_granular_classes=15`, `n_coarse_classes=3`) trained in PyTorch. The inference API for this model is located in `OCT-Segmentation-Model/main.py`.

### 3. Detailed ML Documentation
To delve deeper into the training procedures, architectures, and datasets, refer to the detailed documentation in each respective ML module:

*   **Classification Model Docs (`image-classification-model-training/Documentation/`)**:
    *   [Architecture](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/Documentation/architecture.md)
    *   [Data Pipeline](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/Documentation/data_pipeline.md)
    *   [Dataset Info](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/Documentation/dataset.md)
    *   [Training Guide](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/Documentation/training.md)
*   **Segmentation Model Docs (`image-segmentation-model-training/Documentation/`)**:
    *   [API Reference](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-segmentation-model-training/Documentation/api_reference.md)
    *   [Architecture](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-segmentation-model-training/Documentation/architecture.md)
    *   [Training Guide](file:///Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-segmentation-model-training/Documentation/training_guide.md)

### 4. Data Schemas
*   **`NormalizedScan`**: Standard internal representation. Contains `volume` (Z, Y, X array), `spacing_mm`, `source_format`, and `metadata`.
*   **`ScanResult`**: API response schema. Contains `diagnosis`, `confidence`, `qc` (signal range, crop bounds), `layers` (layer votes, score, cdf deciles), `previews` (URLs), and `segmentation` polygons.

### 5. Code Modification Rules
1.  **Do not modify clinical diagnostic rules** without explicit user confirmation.
2.  **Maintain separation of concerns**: Backend handles all heavy lifting (ML, Array manipulation, OpenCV/PyTorch), Frontend is strictly a UI layer consuming JSON/Images.
3.  Ensure 100% test coverage for the backend (`pytest.ini`).
