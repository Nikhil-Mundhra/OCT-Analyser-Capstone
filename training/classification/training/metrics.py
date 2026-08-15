"""
training/metrics.py

Evaluation metrics for the hierarchical OCT classification pipeline.

MetricAccumulator collects batch-level predictions across an entire epoch
and computes the following epoch-level metrics in one pass:

  - Accuracy          (overall correct / total)
  - Macro F1          (unweighted mean F1 per class — penalises minority class failure)
  - Weighted F1       (weighted by support — reflects overall clinical performance)
  - AUROC             (binary: standard; multi-class: One-vs-Rest macro average)
  - Per-class F1      (individual F1 per class — critical for minority monitoring)
  - Confusion Matrix  (numpy array for TensorBoard / matplotlib)
  - Classification Report (full precision/recall/F1 string for logging)

Macro F1 is the primary model-selection metric because it gives equal weight
to every class, including RAO (22 samples) and CSR (102 samples). A model
that ignores these classes would have a low macro F1 even if its overall
accuracy is high.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class MetricAccumulator:
    """
    Accumulates batch predictions and labels for epoch-level metric computation.

    Usage::

        acc = MetricAccumulator()

        for images, labels in val_loader:
            logits = model(images.to(device))
            acc.update(logits.cpu(), labels.cpu())

        metrics = acc.compute(class_names=['NORMAL', 'ABNORMAL'], prefix='val_')
        print(metrics['val_accuracy'], metrics['val_auroc'])
    """

    def __init__(self) -> None:
        self._logits: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def reset(self) -> None:
        """Clear accumulated state — call at the start of each epoch."""
        self._logits = []
        self._labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Record one batch of predictions.

        Args:
            logits: Raw model output, shape ``(B, C)``. Should be on CPU.
            labels: Ground-truth integer indices, shape ``(B,)``. CPU.
        """
        self._logits.append(logits.detach().cpu().float())
        self._labels.append(labels.detach().cpu())

    def compute(
        self,
        class_names: Optional[List[str]] = None,
        prefix: str = "",
    ) -> Dict:
        """
        Compute all epoch-level metrics from the accumulated batches.

        Args:
            class_names: Ordered list of class name strings. Used for
                         per-class F1 keys and the classification report.
            prefix:      String prefix for all returned dict keys.
                         Use ``'train_'`` or ``'val_'`` to distinguish phases.

        Returns:
            Dict with the following keys (after applying prefix):
              - ``{prefix}accuracy``       : float
              - ``{prefix}macro_f1``       : float
              - ``{prefix}weighted_f1``    : float
              - ``{prefix}auroc``          : float (NaN if computation fails)
              - ``{prefix}f1_{ClassName}`` : float per class
              - ``{prefix}confusion_matrix``: np.ndarray [C × C]
              - ``{prefix}report``         : str (classification report)
        """
        if not self._logits:
            logger.warning("MetricAccumulator.compute() called with no data.")
            return {}

        all_logits = torch.cat(self._logits, dim=0)   # (N, C)
        all_labels = torch.cat(self._labels, dim=0).numpy()  # (N,)

        all_probs  = F.softmax(all_logits, dim=1).numpy()    # (N, C)
        all_preds  = np.argmax(all_probs, axis=1)            # (N,)
        num_classes = all_logits.shape[1]

        names = class_names or [str(i) for i in range(num_classes)]

        out: Dict = {}

        # ── Core metrics ──────────────────────────────────────────────────────
        out[f"{prefix}accuracy"] = float(
            accuracy_score(all_labels, all_preds)
        )
        out[f"{prefix}macro_f1"] = float(
            f1_score(all_labels, all_preds, average="macro", zero_division=0)
        )
        out[f"{prefix}weighted_f1"] = float(
            f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        )

        # ── AUROC ─────────────────────────────────────────────────────────────
        try:
            if num_classes == 2:
                out[f"{prefix}auroc"] = float(
                    roc_auc_score(all_labels, all_probs[:, 1])
                )
            else:
                # OvR macro AUROC — robust to class imbalance
                out[f"{prefix}auroc"] = float(
                    roc_auc_score(
                        all_labels,
                        all_probs,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except ValueError as exc:
            # Fails when some classes have no positive samples in this fold/split
            logger.debug("AUROC computation failed: %s", exc)
            out[f"{prefix}auroc"] = float("nan")

        # ── Per-class F1 ──────────────────────────────────────────────────────
        per_class_f1 = f1_score(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),   # Enumerate all classes explicitly
            average=None,
            zero_division=0,
        )
        for name, f1 in zip(names, per_class_f1):
            out[f"{prefix}f1_{name}"] = float(f1)

        # ── Confusion matrix ──────────────────────────────────────────────────
        out[f"{prefix}confusion_matrix"] = confusion_matrix(
            all_labels, all_preds, labels=list(range(num_classes))
        )

        # ── Classification report ─────────────────────────────────────────────
        out[f"{prefix}report"] = classification_report(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),   # Always enumerate all expected classes
            target_names=names,
            zero_division=0,
        )

        return out

    def __len__(self) -> int:
        return sum(t.shape[0] for t in self._labels)
