# ReLayNet-3D: Volumetric OCT Segmentation

ReLayNet-3D is a Python pipeline for loading, preprocessing, flattening, and
segmenting 3D Optical Coherence Tomography (OCT) volumes. The project includes
format-aware volume loading, a MONAI-backed preprocessing path with local
fallbacks, an RPE-based anatomical flattener, a 3D segmentation model factory,
and anatomical loss helpers for training.

## Current Capabilities

- Load Heidelberg `.vol` files with `eyepy`.
- Load DICOM `.dcm` files with `SimpleITK`.
- Normalize OCT volumes into channel-first PyTorch tensors.
- Flatten OCT stacks relative to the retinal pigment epithelium (RPE).
- Build a MONAI 3D U-Net when MONAI is installed.
- Fall back to a lightweight PyTorch 3D convolutional model when MONAI is not available.
- Train with an ordinal anatomical loss.
- Run pytest with 100% statement coverage enforced in CI.

## Project Structure

```text
OCT-Analyser-Capstone/
├── .github/
│   └── workflows/
│       └── pytest.yml          # GitHub Actions test workflow
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # .vol and .dcm volume loading
│   ├── model.py                # MONAI/PyTorch 3D segmentation model factory
│   ├── pre_processing.py       # Preprocessing pipeline and fallback normalizer
│   ├── runtime.py              # macOS scientific-stack runtime guards
│   └── train.py                # Single training-step helper
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py        # Unit tests for 100% statement coverage
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
