# Data & Augmentation Strategy

The preprocessing pipeline leverages the **MONAI (Medical Open Network for AI)** framework to apply specialized, robust transformations that standardize heterogeneous OCT scans without destroying critical biological anomalies.

## 1. CLAHE Preprocessing
OCT scans vary wildly in illumination, contrast, and noise profiles depending on the specific hardware scanner used by the clinic. 

We wrap OpenCV's **Contrast Limited Adaptive Histogram Equalization (CLAHE)** into a custom MONAI Transform. This equalizes the structural tissue contrast across the entire dataset, forcing the ConvNeXt backbone to learn structural biomarker patterns rather than overfitting to the brightness signature of a specific scanner.

## 2. Safe Spatial Transformations
> [!WARNING]
> **Strict Augmentation Policy**
> We explicitly prohibit the use of `RandomResizedCrop` or any aggressive spatial shearing/cropping. Many retinal pathologies (e.g., peripheral cysts, fluid pockets) exist at the very edges of the scan. Random crops act as spatial dropout, physically erasing the disease and severely corrupting the ground-truth label.

Our approved training augmentations:
- **`Resize(384x384)`**: Standardizes resolution.
- **`RandFlip`**: Vertical and Horizontal flips are perfectly biologically plausible for retinal structures.
- **`RandRotate`**: Rotation is strictly bounded to ~15 degrees (`range_x=0.26`), and `keep_size=True` ensures no tissue is cropped off the edges.
- **`RandGaussianNoise`**: Simulates inherent scanner speckle noise.
- **`RandCoarseDropout`**: Randomly drops a single `32x32` patch to force the network to rely on global structural context rather than local artifacts.

## 3. Standardization
All images are explicitly scaled between `[0, 1]` using `ScaleIntensity()`, and then standardized using the ConvNeXt-required ImageNet Mean (`[0.485, 0.456, 0.406]`) and Std (`[0.229, 0.224, 0.225]`) using `NormalizeIntensity()`.
