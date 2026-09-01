# Hierarchical Classification, Open-Set Recognition, and Tri-State Clinical Triage Architecture

## Executive Summary

This document details the mathematical formulation, structural constraints, error propagation mechanisms, and clinical triage protocols for the Multi-Head ConvNeXt V2 architecture in Optical Coherence Tomography (OCT) retinal image analysis.

The architecture addresses disjoint medical data constraints where normal/abnormal triage labels and multi-class disease labels exist in hierarchical tiers. This document establishes the formal foundations of the joint probability distribution, deconstructs its core failure modes, and specifies a calibrated tri-state decision-support layer.

---

## 1. Mathematical Formulation

### 1.1 Hierarchical Decomposition
Let $x \in \mathcal{X}$ denote an input OCT B-scan image. The diagnosis space is partitioned into two hierarchical questions:
1. **Screening Gatekeeper ($H1$):** Binary status $y_{\text{abnormal}} \in \{0, 1\}$.
2. **Granular Pathology ($H2$):** Specific pathology category $y_{\text{pathology}} \in \{D_1, D_2, \dots, D_K\}$, defined strictly on the abnormal subspace $y_{\text{abnormal}} = 1$.

The shared backbone extracts a universal multi-scale feature representation:
$$\mathbf{f}_{\text{visual}} = \left[\mathbf{f}_{S2}, \mathbf{f}_{S3}, \mathbf{f}_{S4}\right] \in \mathbb{R}^{256 + 512 + 1024} = \mathbb{R}^{1792}$$

The heads output:
$$z_{H1} = \mathbf{w}_{H1}^T \mathbf{f}_{S4} + b_{H1} \in \mathbb{R}$$
$$\mathbf{z}_{H2} = \mathbf{W}_{H2}^T \left[\mathbf{f}_{\text{visual}}, \;\sigma(z_{H1})\right] + \mathbf{b}_{H2} \in \mathbb{R}^K$$

### 1.2 Probability Factorization
Under the chain rule of conditional probability, the joint marginal probability of disease $D_i$ in the patient population is:
$$P(D_i \mid x) = P(D_i \mid \text{abnormal}, x) \cdot P(\text{abnormal} \mid x)$$

Where:
$$P(\text{abnormal} \mid x) = \sigma(z_{H1}) = \frac{1}{1 + e^{-z_{H1}}}$$
$$P(D_i \mid \text{abnormal}, x) = \operatorname{softmax}(\mathbf{z}_{H2})_i = \frac{e^{z_{i, H2}}}{\sum_{j=1}^K e^{z_{j, H2}}}$$

The total probability mass distributed across all $K$ known diseases satisfies:
$$\sum_{i=1}^K P(D_i \mid x) = \sum_{i=1}^K \left[ P(D_i \mid \text{abnormal}, x) \cdot P(\text{abnormal} \mid x) \right] = P(\text{abnormal} \mid x) \sum_{i=1}^K P(D_i \mid \text{abnormal}, x) = P(\text{abnormal} \mid x)$$

The complement represents the posterior probability of a normal retina:
$$P(\text{Normal} \mid x) = 1 - P(\text{abnormal} \mid x)$$

---

## 2. Deconstruction of Core Architectural Limitations

The conditional hierarchy is mathematically sound under its training assumptions, but exhibits two distinct vulnerabilities when transitioning to open-world clinical environments:

### Problem 1: Error Propagation Through the Hierarchy (Cascading Gatekeeper Error)
* **Root Cause:** $H2$'s loss is masked out for normal training samples ($\mathcal{L}_{H2} = 0$ when $y_{\text{abnormal}} = 0$).
* **Mechanism:** While the shared backbone features encounter normal scans during $H1$ training, the $H2$ linear head is never supervised on normal tissue manifolds and possesses no learned decision boundary for normal features.
* **Failure Cascade:** If $H1$ produces a false positive ($P(\text{abnormal}) = 0.85$) due to noise, tilt, or decentration, $H2$'s closed softmax operator is forced to distribute that probability mass across the $K$ disease classes, yielding a false positive disease attribution (e.g., $P(\text{Drusen}) = 0.85 \times 0.65 = 55.2\%$).

### Problem 2: Closed-Set Bound in Open-World Clinical Practice (Open-Set Limitation)
* **Root Cause:** $H2$ optimizes a closed-set simplex over $K$ fixed categories:
  $$\sum_{i=1}^K P(D_i \mid \text{abnormal}, x) = 1.0$$
* **Mechanism:** Real-world ophthalmology encounters rare or unmodeled pathologies $x \notin \{D_1, \dots, D_K\}$ (e.g., Stargardt disease, Best vitelliform dystrophy, retinal capillary hemangioma).
* **Failure Cascade:** $H1$ correctly identifies abnormality ($P(\text{abnormal}) \to 1.0$), but $H2$ cannot express "None of the Above". Softmax overconfidently attributes the anomaly to whichever known category shares the closest superficial latent projection.

---

## 3. Critical Analysis of Architectural Hypotheses

| Proposed Mechanism | Common Assumption | Rigorous Scientific Reality |
| :--- | :--- | :--- |
| **Independent Sigmoids** | Sigmoids naturally output low probabilities for unseen pathologies. | Sigmoids *allow* a "none confident" state ($\sum P(D_i) \neq 1$), but do *not guarantee* it. An unseen disease sharing visual features with Drusen can easily produce $P(\text{Drusen}) = 0.95$. Reliable OOD detection requires explicit calibration and abstention objectives. |
| **Normal Representation** | H2 has never observed normal tissue. | The *shared backbone* has extracted features from normal tissue via $H1$ supervision. The limitation is specifically that the *H2 classification head* has no supervised decision boundary or negative calibration on normal features. |
| **`.detach()` on $P(H1)$** | Detaching $H1$ probability is an architectural flaw. | Detaching is a valid regularizer that prevents $H2$ gradients from corrupting the semantic probability $P(\text{Abnormal} \mid x)$. The limitation is that masking normal samples denies $H2$ any gradient signal on how to behave when $P(H1) \to 0$. |
| **Entropy / Max Softmax Probability** | High entropy reliably identifies out-of-distribution inputs. | Deep networks are frequently overconfident on OOD samples ("confidence $\neq$ familiarity"). A novel pathology can yield Max Probability $> 0.95$. Entropy is an uncertainty heuristic, not an OOD guarantee. |
| **Explicit "Other" Class** | Adding an "Other" class solves open-set recognition. | An "Other" class expands the closed set from $K$ to $K+1$ ($K + \text{known outliers}$). It does not resolve open-set recognition for novel pathologies absent from the "Other" training split. |

---

## 4. The Tri-State Clinical Triage Architecture

To ensure clinical safety without destabilizing model training, the system incorporates a post-hoc, statistically calibrated **Tri-State Clinical Triage Layer**.

```
                           [ Input OCT B-Scan x ]
                                     │
                             [ Shared Backbone ]
                                     │
                      ┌──────────────┴──────────────┐
                      ▼                             ▼
               [ H1 Output ]                  [ H2 Output ]
            z_h1 ──► P(Abnormal)          Raw Logits z_h2 ∈ R^K
                      │                             │
                      │               ┌─────────────┼─────────────┐
                      │               ▼             ▼             ▼
                      │           Energy E(z;T)   MSP(z;T)   Entropy H(z;T)
                      │               └─────────────┬─────────────┘
                      │                             ▼
                      │                    [ Calibrated OOD Gate ]
                      │                      is_ood(x) ∈ {0, 1}
                      │                             │
          ┌───────────┴─────────────────────────────┴───────────┐
          ▼                                                     ▼
     P(A) ≤ τ_N                                            P(A) ≥ τ_A
          │                                                     │
   ┌──────┴──────┐                                       ┌──────┴──────┐
   ▼             ▼                                       ▼             ▼
[ NORMAL ]  [ REVIEW_REQUIRED ]                    [ REVIEW_REQUIRED ] [ KNOWN_PATHOLOGY ]
            (H1_AMBIGUOUS)                         (H2_OOD / LOW_CONF) (P(D_i) = P(A)·p_i)
```

### 4.1 Dual $H1$ Thresholding ($\tau_N < \tau_A$)
Rather than an arbitrary $0.50$ boundary, two operating thresholds are calibrated on patient-disjoint validation cohorts:
* $\tau_A$: Set to achieve $\ge 98\%$ sensitivity on abnormal scans (minimizes false-negative disease omissions).
* $\tau_N$: Set to achieve $\ge 95\%$ specificity on normal scans (minimizes false alarms).

### 4.2 Raw-Logit Free Energy OOD Scoring
Free Energy is computed directly from raw $H2$ logits $\mathbf{z}$ before softmax:
$$E(\mathbf{z}; T) = -T \cdot \log \sum_{i=1}^K \exp\left(\frac{z_i}{T}\right)$$

Where $T$ is the temperature parameter optimized via Negative Log-Likelihood minimization.

### 4.3 Triage Decision Rule
$$\hat{\mathcal{S}}(x) = \begin{cases} 
\text{NORMAL} & \text{if } P(\text{abnormal} \mid x) \le \tau_N \\
\text{REVIEW\_REQUIRED} \quad [\text{H1\_AMBIGUOUS}] & \text{if } \tau_N < P(\text{abnormal} \mid x) < \tau_A \\
\text{REVIEW\_REQUIRED} \quad [\text{H2\_OOD / LOW\_CONF}] & \text{if } P(\text{abnormal} \mid x) \ge \tau_A \;\wedge\; \text{is\_ood}(x) \\
\text{KNOWN\_PATHOLOGY} \quad [D_{\arg\max}] & \text{if } P(\text{abnormal} \mid x) \ge \tau_A \;\wedge\; \neg\text{is\_ood}(x)
\end{cases}$$

### 4.4 State-Dependent Interpretability (Grad-CAM)
* **`NORMAL`:** Renders $H1$ context verifying the integrity of retinal layers.
* **`KNOWN_PATHOLOGY`:** Renders $H1$ abnormality attention and $H2$ disease-specific activations.
* **`REVIEW_REQUIRED`:** Prominently displays $H1$ abnormality attention while suppressing unconfirmed $H2$ disease maps to prevent diagnostic anchoring.

---

## 5. Summary Statement

> *"The architecture operates as a hierarchical closed-set classifier with probabilistic gating. While mathematically coherent under conditional independence, clinical deployment requires explicit abstention boundaries to manage gatekeeper error propagation and out-of-distribution pathologies. The Tri-State Clinical Triage Layer transforms forced classifications into safe, calibrated clinical referrals."*
