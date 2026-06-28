# OCT Pipeline — Data & Imbalance Strategy

## Dataset Composition

The pipeline aggregates data from three disparate sources to form a master dataset of **86,120 OCT/OCTA scans**:
1. Kermany/UCSD dataset
2. OCTDL dataset
3. OCTID dataset

This multi-source approach increases generalisation but introduces significant variations in labelling schemas and class distributions. `config/hierarchy.yaml` serves as the single source of truth, mapping raw directory paths to standardized `L1`, `L2`, and `L3` indices.

## The Imbalance Problem

In the medical imaging domain, rare diseases mean scarce data. The class distribution across all 12 fine classes highlights a critical long-tail imbalance:

- **Dominant:** `CNV` (37,205 images), `NORMAL` (26,853 images)
- **Scarce:** `CSR` (102 images), `VID` (76 images)
- **Critical:** `RAO` (22 images — a 1,691x minority compared to CNV)

Training a standard model on this distribution results in the network collapsing: it learns to constantly predict "CNV" and achieves high accuracy by completely ignoring the rare pathologies.

## Three-Tiered Imbalance Mitigation

To force the network to learn the rare classes, we implement a defensive stack across three layers: the DataLoader, the Loss Function, and the Augmentation Pipeline.

### 1. Stratified K-Fold + WeightedRandomSampler (Data Layer)
- **Stratified K-Fold:** We use `sklearn.model_selection.StratifiedKFold(n_splits=5)`. This ensures that even the rarest classes (like the 22 RAO images) are proportionally distributed across all 5 folds. Without stratification, a fold might contain zero RAO images, crashing the AUROC metric.
- **WeightedRandomSampler:** In the training dataloader, we calculate the inverse frequency of each class (`1.0 / count`). We feed these weights to PyTorch's `WeightedRandomSampler`. This changes the sampling probability so that the dataloader oversamples minority classes and undersamples dominant classes, ensuring that *within any given epoch*, the model sees a balanced number of examples from every class.

### 2. Focal Loss + Alpha Weighting (Loss Layer)
Even with balanced sampling, "easy" examples (like classic NORMAL scans) can overwhelm the gradient flow compared to "hard" border-case examples.
- **Focal Loss (γ=2.0):** We use Focal Loss instead of standard CrossEntropy. The `(1 - p_t)^gamma` modulating factor heavily penalizes confident, correct predictions (driving their loss near zero), while preserving the loss for uncertain or incorrect predictions. This forces the optimizer to spend its energy on the hardest examples.
- **Alpha Weighting:** We pass the normalized class weights as the `alpha` parameter to the Focal Loss to give a structural baseline boost to minority classes.
- **Label Smoothing (ε=0.1):** Applied at Level 2 and 3. Because the datasets come from three different annotator pools, there is inherent label noise (e.g. one dataset's "AMD" might overlap with another's "DRUSEN"). Label smoothing prevents the model from becoming overconfident on noisy annotations.

### 3. Dual-Resolution Augmentation (Augmentation Layer)
Since we are oversampling classes with very few images, the model runs a high risk of memorizing them (overfitting).
- **L1/L2 (Standard Augmentation, 224px):** Uses standard random flips, subtle rotations (±10°), and minor brightness/contrast shifts to prevent overfitting on the broad classification tasks.
- **L3 (Heavy Augmentation, 384px):** Specialist models (especially Vascular and Structural) employ aggressive augmentation:
  - Stronger affine transformations (rotations, translations, scaling)
  - `RandomErasing(p=0.2)`: Cuts out small rectangular patches of the image. This forces the network to look at the *entire* structural context of the retina rather than memorizing a single localized artifact in the scarce RAO/CSR images.

By combining sampling probabilities, loss modulation, and heavy synthetic variation, we maximize the predictive signal extracted from the long-tail minority classes.
