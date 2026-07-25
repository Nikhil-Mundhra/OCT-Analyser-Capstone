# Quick Start Guide - HRF Segmentation

Get started with HRF segmentation in 3 minutes!

## Choose Your Path

### Path 1: Try the Live Demo (Fastest)

**No installation required!**

[Open HuggingFace Space](https://huggingface.co/spaces/AI-RESEARCHER-2024/HRF_Segmentation)

1. Upload an OCT image
2. Select model (U-Net or Attention U-Net)
3. Click "Run Segmentation"
4. View and download results

---

### Path 2: Train Your Own Model (Advanced)

**Train from scratch on your dataset**

```bash
# Clone repository
git clone https://github.com/ai-research-2025/hrf-segmentation.git
cd hrf-segmentation

# Install dependencies
cd training
pip install -r requirements.txt

# Prepare your data (HRF_IMAGES/ and HRF_MASKS/ folders)
# Edit Config class in the script to set your dataset paths
# Then train!
python hrfunet.py        # For U-Net
python hrf-aunet.py      # For Attention U-Net
```

**Requirements:**
- NVIDIA GPU (8GB+ VRAM)
- Python 3.8+
- Training dataset with images (`HRF_IMAGES/`) and masks (`HRF_MASKS/`)

---

## Download Pre-trained Models

**All trained models available here:**

[**Google Drive - Model Weights**](https://drive.google.com/drive/folders/1J78I28EzEXMD3jNrusWfuQoCFeFhtZsT)

Contains:
- `Unet/best_model.pth` - U-Net checkpoint (~355 MB)
- `Attention-Unet/best_model.pth` - Attention U-Net checkpoint (~361 MB)
- Test metrics, prediction visualizations, and ROC curves

---

## Learn More

- **Full Documentation**: See main [README.md](README.md)
- **Training Guide**: [training/README.md](training/README.md)

---

## Need Help?

- Check the [main README](README.md) for detailed instructions
- [Open an issue](https://github.com/ai-research-2025/hrf-segmentation/issues)
- [Start a discussion](https://github.com/ai-research-2025/hrf-segmentation/discussions)

---

## Star This Repo

Found this helpful? Give it a star to show your support!
