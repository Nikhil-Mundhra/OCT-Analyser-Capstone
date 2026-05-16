# Local OCT Analyzer MVP

Local OCT Analyzer MVP is a local-first OCT/OCTA clinical workflow prototype.
It combines a FastAPI backend, a React wireframe-style clinical interface, and
the existing Python OCT processing pipeline for loading, preprocessing,
flattening, previewing, and feature-extracting 3D Optical Coherence Tomography
(OCT) volumes.

The app is currently an end-to-end demo shell: ingestion, preprocessing,
preview generation, feature extraction, and UI workflow are implemented, while
the medical segmentation and diagnosis logic remain deterministic placeholders
until real atlas/model assets are supplied.

## Current Capabilities

### Backend and Pipeline

- Load Heidelberg `.vol` files with `eyepy`.
- Load DICOM `.dcm` files and extract pixel data, rescale metadata, and spacing.
- Load zipped image-stack exports containing sorted TIFF/BMP/PNG slices plus
  optional XML/CSV spacing metadata.
- Normalize all MVP uploads into a shared `(Z, Y, X)` NumPy volume contract with
  spacing ordered as `(Z, Y, X)` millimetres.
- Normalize OCT volumes into channel-first PyTorch tensors.
- Flatten OCT stacks relative to the retinal pigment epithelium (RPE).
- Run basic QC reporting: signal range, crop status, warnings, volume shape,
  source format, and metadata.
- Generate preview images for raw center slice, cropped slice, segmentation
  overlay, and CDF feature chart.
- Extract real CDF-style texture features from layer masks using second-order
  reflectivity energy.
- Generate deterministic demo 12-layer segmentation, layer votes, and
  majority-vote diagnosis.
- Run IPN-V2 architecture inference as an optional OCTA/en face segmentation
  smoke test. Without a checkpoint this uses random weights and is marked
  `untrained_smoke`.
- Load a future IPN-V2 `.pth` checkpoint from the `IPNV2_CHECKPOINT`
  environment variable when available.
- Build a MONAI 3D U-Net when MONAI is installed.
- Fall back to a lightweight PyTorch 3D convolutional model when MONAI is not available.
- Train with an ordinal anatomical loss.
- Run pytest with 100% statement coverage enforced in CI.

### Local API

- `POST /api/scans` uploads one `.vol`, `.dcm`, or `.zip` scan export.
- `GET /api/scans/{scan_id}` returns status, diagnosis, confidence, QC,
  metadata, layer votes, CDF deciles, IPN-V2 metadata, and preview URLs.
- `GET /api/scans/{scan_id}/preview/{kind}` serves generated PNG previews.
- API responses include `is_demo_model: true` to make the placeholder model
  status explicit.

### Frontend

- Uses the clinical wireframe-style interface as the main app at
  `http://127.0.0.1:5173`.
- Includes triage worklist, upload/QC, scan review, AI findings, human decision
  gate, and outcomes/audit screens.
- Upload/QC screen is wired to the local FastAPI backend.
- Review screen displays generated scan previews and layer findings.
- Review screen includes an “OCTA/IPN-V2” toggle when IPN-V2 previews are
  available, with a visible warning for untrained smoke mode.
- Human decision gate supports clinician decision selection, required rationale,
  saved sign-off state, and case JSON export.
- Outcomes/audit screen reflects the active scan and saved decision.

## Not Implemented Yet

- The app is not clinically validated and must not be used for diagnosis.
- 12-layer segmentation is a deterministic placeholder, not anatomical
  segmentation.
- Diagnosis and layer votes come from a deterministic placeholder classifier,
  not a trained ANN.
- IPN-V2 output is currently test-drive plumbing only unless a compatible
  checkpoint is supplied through `IPNV2_CHECKPOINT`.
- No real retinal atlas or non-rigid registration is connected yet.
- Fovea detection is a simple center/brightness heuristic.
- Proprietary Solix archives such as `.fds` are not supported.
- There is no background job queue for very large scans.
- Cases are not persisted in a database; runtime scan state is local process and
  filesystem state.
- Authentication, HIPAA controls, audit-grade logging, and deployment hardening
  are not implemented.
- PDF report generation and clinical accuracy evaluation are not implemented.

## Project Structure

```text
OCT-Analyser-Capstone/
├── .github/
│   └── workflows/
│       └── pytest.yml          # GitHub Actions test workflow
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI upload, status, and preview endpoints
│   ├── data_loader.py          # .vol, .dcm, and .zip stack loading
│   ├── ipnv2_adapter.py        # IPN-V2 architecture wrapper and smoke inference
│   ├── mvp_pipeline.py         # MVP QC, preprocessing, features, demo classifier
│   ├── model.py                # MONAI/PyTorch 3D segmentation model factory
│   ├── pre_processing.py       # Preprocessing pipeline and fallback normalizer
│   ├── preview.py              # PNG preview rendering helpers
│   ├── runtime.py              # macOS scientific-stack runtime guards
│   ├── scan_types.py           # Normalized scan data contract
│   └── train.py                # Single training-step helper
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py        # Unit tests for 100% statement coverage
├── app.js                      # Main React clinical workflow frontend
├── demo/
│   ├── app.bundle.js           # Built MVP frontend bundle
│   ├── index.html              # Original standalone wireframe demo page
│   └── wireframe.bundle.js     # Original wireframe bundle
├── anatomical_flattener.py     # RPE-based flattening utilities
├── functions.py                # Spatial continuity loss
├── losses.py                   # Ordinal anatomical loss
├── main.py                     # Example execution entry point
├── Dockerfile                  # Containerized runtime
├── requirements.txt            # Runtime and test dependencies
├── pytest.ini                  # Pytest and coverage settings
└── .coveragerc                 # Coverage configuration
```

## Requirements

- Python 3.11+
- PyTorch
- MONAI
- NumPy
- SciPy
- SimpleITK
- eyepy
- FastAPI
- Uvicorn
- Pillow
- React
- esbuild
- pytest and pytest-cov for tests

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Place an OCT file named `patient_001_baseline.vol` in the project root, or edit
the path in `main.py`, then run:

```bash
python main.py
```

The entry point performs:

1. Volume loading from `.vol` or `.dcm`.
2. Intensity preprocessing and channel-first tensor conversion.
3. RPE-based anatomical flattening.

If the sample file is missing, the script exits with a clear file-not-found
message.

## Local MVP App

Run the full local MVP from a fresh checkout with one command:

```bash
make run
```

That command creates or updates `.venv`, installs Python dependencies, installs
Node dependencies, builds the React bundle, starts the FastAPI backend at
`http://127.0.0.1:8000`, and serves the frontend at
`http://127.0.0.1:5173`. Press `Ctrl-C` in the Make process to stop both
servers.

Useful Make targets:

```bash
make install  # install Python and Node dependencies
make build    # build the frontend bundle
make test     # run the Python test suite
make clean    # remove runtime/cache artifacts
```

You can override ports when needed:

```bash
API_PORT=8001 WEB_PORT=5174 make run
```

Manual equivalent:

```bash
.venv/bin/python -m pip install -r requirements.txt
npm install
npm run build
.venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
python3 -m http.server 5173
```

Open `http://127.0.0.1:5173` to use the clinical workflow frontend. It includes
the original wireframe-style worklist, upload/QC, scan review, human decision
gate, and audit/outcomes screens, now wired to the local API for `.vol`, `.dcm`,
or `.zip` stack uploads. The current classifier and 12-layer segmentation are
deterministic placeholders and the API response includes `is_demo_model: true`;
outputs are for local demonstration only, not clinical use.

To test-drive IPN-V2 with real weights later, set:

```bash
export IPNV2_CHECKPOINT=/path/to/checkpoint.pth
```

Without that variable, the app still generates IPN-V2 probability/overlay
previews, but labels them as untrained smoke-mode output.

The original standalone wireframe demo is still available at
`http://127.0.0.1:5173/demo/`.

## Testing

Run the test suite:

```bash
pytest
```

Coverage is enforced by `pytest.ini`:

```bash
pytest --cov=. --cov-report=term-missing --cov-fail-under=100
```

The GitHub Actions workflow in `.github/workflows/pytest.yml` runs the same
pytest suite on every push and pull request.

## Docker

Build the image:

```bash
docker build -t relaynet-3d .
```

Run the container:

```bash
docker run --rm relaynet-3d python main.py
```

The Dockerfile copies `requirements.txt` first so dependency installation can be
cached between source-code changes.

## Notes

- MONAI and SciPy are preferred when installed.
- Local fallback paths keep imports and smoke tests working in lighter
  environments.
- On macOS, runtime guards reduce known OpenMP conflicts between NumPy and
  PyTorch.
