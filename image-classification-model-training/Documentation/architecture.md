# Unified Multi-Head Architecture

The classification pipeline utilizes a highly optimized **Multi-Head ConvNeXt V2** architecture. This unified network extracts robust image features via a shared backbone and branches into specialized streams to perform both binary health screening and granular pathology routing simultaneously.

## 1. Shared Backbone
- **Model**: `convnextv2_tiny` (pre-trained on ImageNet).
- **Function**: Extracts high-dimensional semantic features from the grayscale OCT scans.
- **Output**: The unpooled feature map from Stage 4 (shape: `[B, 1024, H, W]`).

## 2. Dual-Stream Feature Processing
The network splits the unpooled feature map into two distinct streams:

### Head 1: The Gatekeeper (Binary Classification)
- **Objective**: Classify the scan as **Normal (0)** or **Abnormal (1)**.
- **Flow**: The raw feature map undergoes standard Global Average Pooling (GAP). The flattened `[1024]` vector is fed directly into a shallow MLP (`Linear -> GELU -> Dropout -> Linear`).
- **Output**: A single logit `(B, 1)` representing the probability of pathology.

### Head 2: The Router (Multi-Class Pathology)
- **Objective**: Identify the specific disease category (e.g., AMD, DRUSEN, CSR, MH).
- **Attention Filtering**: The unpooled features first pass through a **CBAM (Convolutional Block Attention Module)**. This forces the network to spatially and channel-wise attend to the actual biological lesions rather than background noise.
- **Flow**: The attended features undergo GAP, resulting in a filtered `[1024]` vector.

## 3. Hierarchical Conditioning
The architecture implements a cascading logic: the probability of a scan being abnormal (from Head 1) is inherently useful for determining the specific pathology (Head 2). 

1. We apply a sigmoid to the Head 1 logit to get a raw probability.
2. We concatenate this `[1]` probability with the `[1024]` attended feature vector from Head 2, resulting in a `[1025]` feature vector.
3. This is fed into the Router MLP to predict the specific pathology class.

> [!WARNING]
> **Gradient Isolation (`.detach()`)**
> The H1 probability passed to H2 is explicitly detached from the computation graph. If it wasn't detached, the H2 Multi-Class loss would propagate backward and inadvertently alter the H1 Gatekeeper's carefully learned boundary for Normal/Abnormal classification.
