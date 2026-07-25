# Training HRF Segmentation Models

This directory contains the training scripts for both U-Net and Attention U-Net models.

## Files

- `hrfunet.py` - Complete training pipeline for U-Net
- `hrf-aunet.py` - Complete training pipeline for Attention U-Net
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your data in the following structure:

```
dataset/
├── HRF_IMAGES/
│   ├── image001.jpeg
│   ├── image002.jpeg
│   └── ...
└── HRF_MASKS/
    ├── image001_HRF.ome.tiff
    ├── image002_HRF.ome.tiff
    └── ...
```

### 3. Configure Paths

Edit the `Config` class at the top of each script to set your data and checkpoint directories:

```python
class Config:
    DATA_DIR = '/path/to/your/dataset'
    CHECKPOINT_DIR = '/path/to/save/checkpoints'
```

### 4. Train U-Net

```bash
python hrfunet.py
```

### 5. Train Attention U-Net

```bash
python hrf-aunet.py
```

## Training Details

### U-Net (`hrfunet.py`)

- **Architecture:** Classic U-Net with 5 encoder-decoder levels (base filters: 64)
- **Preprocessing:** CLAHE + 0-255 normalization
- **Loss:** Focal Tversky Loss (α=0.3, β=0.7, γ=4/3)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **LR Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Augmentation:** HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, RandomGamma
- **Early Stopping:** 25 epochs
- **Output:** `best_model.pth` in checkpoint directory

### Attention U-Net (`hrf-aunet.py`)

- **Architecture:** U-Net with Attention Gates (base filters: 64)
- **Preprocessing:** CLAHE + Z-Score normalization (ImageNet statistics)
- **Loss:** Dice Loss (smooth=1.0)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **LR Scheduler:** CosineAnnealing (T_max=100, eta_min=1e-6)
- **Augmentation:** None (disabled)
- **Early Stopping:** 15 epochs
- **Output:** `best_model.pth` in checkpoint directory

### Common Settings

- **Batch Size:** 4
- **Epochs:** 100
- **Mixed Precision (AMP):** Enabled
- **Data Split:** 70% Train / 15% Validation / 15% Test
- **Random Seed:** 42

## Hardware Requirements

- GPU: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- RAM: 16GB+ recommended
- Storage: ~10GB for checkpoints and logs

## Monitoring Training

Both scripts print training progress including:
- Epoch number and learning rate
- Training loss and Dice coefficient
- Validation loss and Dice coefficient
- Best model checkpointing with early stopping

## Outputs

After training, you'll find in the checkpoint directory:
- `best_model.pth` - Best model checkpoint (based on validation Dice)
- `latest_model.pth` - Latest epoch checkpoint
- Test set evaluation results (metrics, ROC curves, confusion matrices)
- Prediction visualizations

## Troubleshooting

### Out of Memory Errors
- Reduce `BATCH_SIZE` in the Config class
- Set `USE_AMP = True` to enable mixed precision training

### Poor Convergence
- Check data normalization matches the model's preprocessing
- Verify masks are binary (0 and 1)
- Try adjusting the learning rate

## Pre-trained Models

Don't want to train from scratch? Download our pre-trained weights:

[**Google Drive - Pre-trained Models**](https://drive.google.com/drive/folders/1J78I28EzEXMD3jNrusWfuQoCFeFhtZsT)
