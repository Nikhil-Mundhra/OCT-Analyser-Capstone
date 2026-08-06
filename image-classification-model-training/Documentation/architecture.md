# Multi-Head ConvNeXt V2 System Architecture

This document specifies the architectural design of the unified multi-head classification network for OCT retinal scan analysis.

---

## 1. System Architecture Overview

The network utilizes a shared **ConvNeXt V2 Base** backbone operating on RGB-converted single-channel OCT B-scans (`224x224x3`). High-dimensional feature maps are extracted across multiple spatial depths and fed into dual processing streams:
1. **Head 1 (Gatekeeper)**: Binary classification for Normal vs Abnormal screening.
2. **Head 2 (Granular Pathology Router)**: Multi-scale attention-filtered pathology identification across 12 disease categories.

```mermaid
graph TD
    subgraph Input ["Input Stage"]
        A["OCT B-Scan (B, 3, 224, 224)"]
    end

    subgraph Backbone ["ConvNeXt V2 Base Feature Extractor"]
        A --> Stem["Stem & Stage 0 (Frozen)"]
        Stem --> S1["Stage 1 Feature Map (B, 256, 28, 28)"]
        S1 --> S2["Stage 2 Feature Map (B, 512, 14, 14)"]
        S2 --> S3["Stage 3 Feature Map (B, 1024, 7, 7)"]
    end

    subgraph Stream1 ["Head 1: Gatekeeper (Binary Classification)"]
        S3 --> GAP1["Adaptive Avg Pool (GAP)"]
        GAP1 --> H1_Flat["Flatten (B, 1024)"]
        H1_Flat --> H1_MLP["MLP: Linear(1024, 512) -> GELU -> Dropout(0.2) -> Linear(512, 1)"]
        H1_MLP --> H1_Logit["Normal / Abnormal Logit (B, 1)"]
        H1_Logit --> H1_Sigmoid["Sigmoid Function"]
        H1_Sigmoid --> P_H1["p_H1 Probability"]
    end

    subgraph Conditioning ["Gradient-Isolated Conditioning"]
        P_H1 --> Detach[".detach() Operator"]
    end

    subgraph Stream2 ["Head 2: Granular Pathology Router (Multi-Scale CBAM)"]
        S1 --> CBAM1["CBAM Attention (Stage 1)"]
        S2 --> CBAM2["CBAM Attention (Stage 2)"]
        S3 --> CBAM3["CBAM Attention (Stage 3)"]
        
        CBAM1 --> GAP_S1["GAP -> Flatten (B, 256)"]
        CBAM2 --> GAP_S2["GAP -> Flatten (B, 512)"]
        CBAM3 --> GAP_S3["GAP -> Flatten (B, 1024)"]
        
        GAP_S1 --> Concat_MS["Multi-Scale Feature Concat (B, 1792)"]
        GAP_S2 --> Concat_MS
        GAP_S3 --> Concat_MS
        
        Concat_MS --> Concat_H2["Concatenate [MultiScale, p_H1.detach()] (B, 1793)"]
        Detach --> Concat_H2
        
        Concat_H2 --> H2_MLP["MLP: Linear(1793, 512) -> GELU -> Dropout(0.2) -> Linear(512, 12)"]
        H2_MLP --> H2_Logits["12 Granular Pathology Logits (B, 12)"]
    end
```

---

## 2. Layer Pipeline & Tensor Dimensionality

The multi-scale encoder aggregation extracts complementary spatial representations: fine-grained structural features from earlier layers (`Stage 1`) and deep semantic context from later layers (`Stage 3`).

```mermaid
graph LR
    subgraph Shapes ["Tensor Shape Progression"]
        direction LR
        In["Input B-Scan<br/>(B, 3, 224, 224)"] --> S1_Shape["Stage 1<br/>(B, 256, 28, 28)"]
        S1_Shape --> S2_Shape["Stage 2<br/>(B, 512, 14, 14)"]
        S2_Shape --> S3_Shape["Stage 3<br/>(B, 1024, 7, 7)"]
        
        S1_Shape -->|CBAM + GAP| F1["Vector 1<br/>(B, 256)"]
        S2_Shape -->|CBAM + GAP| F2["Vector 2<br/>(B, 512)"]
        S3_Shape -->|CBAM + GAP| F3["Vector 3<br/>(B, 1024)"]
        S3_Shape -->|GAP| F_H1["H1 Context<br/>(B, 1024)"]
        
        F_H1 -->|MLP| H1_Out["H1 Logit<br/>(B, 1)"]
        H1_Out -->|Sigmoid + Detach| P_H1_Vec["p_H1<br/>(B, 1)"]
        
        F1 --> Concat["Concatenation Layer<br/>(B, 1793)"]
        F2 --> Concat
        F3 --> Concat
        P_H1_Vec --> Concat
        
        Concat -->|MLP| H2_Out["H2 Logits<br/>(B, 12)"]
    end
```

---

## 3. Execution Sequence & Gradient Isolation

Hierarchical conditioning passes the predicted binary abnormal probability ($p_{H1}$) as a scalar feature to Head 2.

```mermaid
sequenceDiagram
    autonumber
    participant Input as Input Tensor (B, 3, 224, 224)
    participant Backbone as ConvNeXt V2 Backbone
    participant Head1 as Head 1 (Gatekeeper)
    participant CBAM as Multi-Scale CBAM Blocks
    participant Detach as Autograd Detach (.detach())
    participant Head2 as Head 2 (Router)

    Input->>Backbone: Forward Pass
    Backbone-->>Head1: Stage 3 Features (B, 1024, 7, 7)
    Backbone-->>CBAM: Stages 1, 2, 3 Features
    
    Head1->>Head1: GAP + Linear Layer Pass
    Head1-->>Detach: H1 Normal/Abnormal Logits (B, 1)
    
    Detach->>Detach: Compute Sigmoid(h1_logits)
    Note over Detach: Explicitly detach tensor from Autograd Graph<br/>Prevents H2 loss backprop from altering H1 decision boundary
    
    CBAM->>CBAM: Spatial & Channel Attention Filtering
    CBAM-->>Head2: Aggregated Multi-Scale Vector (B, 1792)
    Detach-->>Head2: Detached p_H1 Scalar (B, 1)
    
    Head2->>Head2: Concatenate Features -> (B, 1793)
    Head2-->>Input: Return Dict { 'normal_abnormal': (B,1), 'pathology': (B,12) }
```

---

## 4. Architectural Rules & Constraints

### 4.1 Gradient Isolation Requirement
The $p_{H1}$ scalar concatenated into Head 2 must be explicitly detached via `.detach()`. Without `.detach()`, gradients from the multi-label focal loss on Head 2 would flow back through the binary probability and distort Head 1's decision boundary between healthy and pathological scans.

### 4.2 Multi-Scale Feature Pooling
Head 2 does not rely solely on the final bottleneck feature map. By pooling across Stage 1 (256 channels), Stage 2 (512 channels), and Stage 3 (1024 channels), the classification branch retains fine-grained spatial details (such as focal micro-drusen or small cystoid maculopathy) that are typically lost in deep bottleneck layers.

### 4.3 Strict Hierarchical Inference Rule
During evaluation and inference (`return_probs=True`), the joint pathology probability is explicitly conditioned on the binary health state:

$$P(\text{Pathology}_k) = P(\text{Pathology}_k \mid \text{Abnormal}) \times P(\text{Abnormal})$$

$$\mathbf{p}_{\text{final\_H2}} = \text{Softmax}(\mathbf{z}_{H2}) \times \text{Sigmoid}(z_{H1})$$

This mathematical conditioning guarantees zero probability leakage for granular pathologies when a scan is classified as healthy.
