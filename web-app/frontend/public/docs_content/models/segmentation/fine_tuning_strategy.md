# Advanced Fine-Tuning Strategy

This document details the architectural mechanisms, loss masking protocols, and sampling strategies implemented to safely fine-tune the Hierarchical U-Net on the OCT5K dataset without causing catastrophic forgetting of its foundational 15-class knowledge.

## The Core Challenge
The model was initially trained on a large, high-fidelity proprietary dataset (~90,000 images) containing 15 distinct granular classes (retinal layers and pathologies). The goal of this fine-tuning phase is to teach the model to recognize extremely precise **Drusen boundaries** (a specific pathology) using the open-source **OCT5K** dataset. 

However, fine-tuning introduces three fatal traps:
1. **The Taxonomy Clash**: OCT5K only provides 6 coarse classes. If we force the model to predict these 6 classes, it will actively unlearn the 15-class precision it already acquired (e.g., merging the ILM, OPL, and IS/OS layers into a single blob).
2. **The Imbalance Trap**: Training on a 90,000-image base dataset alongside a 1,200-image OCT5K dataset via naive concatenation would cause the model to either completely ignore OCT5K or overfit to the same 1,200 images 60 times per epoch.
3. **The Morphological Mismatch**: Drusen is often thought of as a line boundary, but neural networks learn dense volumes. Extracting 1-pixel thick boundaries from the OCT5K masks provides almost zero gradient signal.

## 1. Targeted Loss Masking (The `255` Epiphany)

To prevent catastrophic forgetting, we utilize PyTorch's `ignore_index = 255`. 

Instead of extracting 1-pixel lines from the OCT5K masks, we map the native filled regions directly to our model's taxonomy:
- **OCT5K Region 4** (the physical volume of the RPE layer encompassing Drusen) is mapped directly to **Model Class 8**.
- The inner retinal layers (OCT5K Regions 1, 2, and 3) are lumped too coarsely to map to our 15-class schema. We explicitly mask them with `255`.

In our `FocalLoss` and `DiceLoss` implementations, we apply a `valid_mask = (targets != 255)`. This effectively neutralizes the gradients for those pixels. The model is allowed to predict whatever it wants in those ignored regions based on its prior knowledge, focusing its learning exclusively on optimizing the RPE/Drusen boundaries.

## 2. Virtual Epochs and Stratified Sampling

To solve the 90k vs 1.2k dataset disparity, we abandoned standard `shuffle=True` and implemented **Virtual Epochs**.

1. **Independent Splitting**: We run `random_split` on the Base and OCT5K datasets *independently* before wrapping them in transforms. This prevents PyTorch `Subset` index mapping errors.
2. **ConcatDataset & WeightedRandomSampler**: We concatenate the isolated training sets and apply a strict `WeightedRandomSampler` configured to pull batches at a **75% Base / 25% OCT5K** probability ratio.
3. **The 15,000 Sample Hard-Cap**: If left unchecked, the sampler would pull 90,000 images per epoch. At a 25% ratio, the model would see every OCT5K image ~18 times per epoch before a validation loop ever runs, triggering latent space collapse. We hard-capped the virtual epoch length to `15,000` samples to enforce rapid, frequent validation checkpoints.

## 3. Adaptive Gradient Suppression (Focal Loss)

We replaced standard `CrossEntropyLoss` with **Focal Loss** (`gamma=2.0`). 
Medical images (especially OCT) are dominated by "easy" background pixels (Vitreous, deep Choroid). Standard Cross-Entropy allows these massive regions to drown out the gradients of tiny, complex structures like Drusen. Focal Loss scales the loss by `(1 - p_t)^gamma`, effectively muting gradients from highly confident background predictions and aggressively amplifying the signal from the hard, ambiguous Drusen interfaces.

## 4. Asymmetric Freezing

Fine-tuning often destroys generalized feature extraction. Instead of a flat learning rate across the entire U-Net, we split the parameter groups in the `AdamW` optimizer:
- **Encoder Backbone**: Locked to `10%` of the base learning rate. This preserves the generalized edge-detection and structural knowledge learned from the massive base dataset.
- **Decoder & Heads**: Run at `100%` of the base learning rate. This allows the upsampling layers and spatial attention gates to aggressively adapt to the new, hyper-precise Drusen definitions without damaging the foundational feature extractors.

## 5. Aggressive Domain Homogenization

OCT scanners (e.g., Topcon vs. Spectralis) have vastly different physical signatures (speckle noise, contrast curves). To ensure the model learns *anatomy* rather than *scanner artifacts*, we apply heavy augmentations to the training pipeline using Albumentations:
- `A.ElasticTransform`: Simulates severe fluid distortion and macular buckling.
- `A.MultiplicativeNoise` & `A.GaussNoise`: Simulates inherent coherence speckle.
- `A.CLAHE`: Simulates post-processing contrast equalization applied by specific proprietary viewer software.

## 6. Independent Checkpointing

Because we have two distinct distributions (Base and OCT5K), a blended validation average is dangerous. The model could achieve a "good" average score by perfecting the Base dataset while completely collapsing on OCT5K.

Our validation loop evaluates `base_val_loader` and `oct5k_val_loader` completely independently. It explicitly saves:
- `unet_hierarchical_best_base.pth`
- `unet_hierarchical_best_oct5k.pth`

This guarantees that we preserve the exact state of the weights where the model achieved peak performance on the specific pathology without regression on the general structural task.
