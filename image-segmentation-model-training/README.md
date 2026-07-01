# OCT Image Segmentation Pipeline

This repository contains the data extraction, model architecture, training loop, and inference modules for building a Hierarchical Multi-Head Segmentation Model for Optical Coherence Tomography (OCT) B-scans.

## Directory Structure

- **`data/`**: PyTorch Dataset definitions (e.g., `segmentation_dataset.py`) for loading and augmenting OCT scans and their hierarchical masks.
- **`Documentation/`**: Contains architectural details and frontend API integration guides.
- **`models/`**: Neural network architectures (e.g., `unet.py` containing the `HierarchicalUNet`).
- **`scripts/`**: Utility scripts for data extraction (`extract_oct5k_segmentation.py`) and verification testing.
- **`src/inference/`**: Object-Oriented post-processing module that converts dense pixel masks into vector polygons and clinical metrics.
- **`training/`**: Scripts containing the training loop (`train.py`) and validation logic.

## Pipeline Overview

1. **Data Extraction**:
   Run `python scripts/extract_oct5k_segmentation.py` to extract 1,672 OCT5k scans. This script parses three distinct expert manual gradings and performs a pixel-wise majority vote to create an ensembled, highly robust ground truth mask.
   
2. **Model Training**:
   Install dependencies with `pip install -r requirements.txt`. 
   Run `python training/train.py --data /path/to/dataset --epochs 50 --batch-size 4 --lr 1e-4 --amp` to train the `HierarchicalUNet`. It splits the ensembled dataset, trains simultaneously on coarse classes (3 total) and granular classes (15 total), uses mixed-precision training (`--amp`), and saves model checkpoints to `checkpoints/`.
   
3. **Inference & Vectorization**:
   The `src/inference/` module contains a `SegmentationAnalyzer` that converts the trained model's raw dense output into structured `LesionInstance` and `RetinalLayer` vector objects, and calculates clinical metrics. See `Documentation/api_reference.md` for details.

## Environment & Setup

Ensure the following dependencies are installed in your environment:
- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `opencv-python-headless`
- `albumentations`
- `imageio` (v2)
