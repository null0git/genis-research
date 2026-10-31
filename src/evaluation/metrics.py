"""Classification metrics for intrusion detection evaluation.

Every metric here is documented, in Section 8 of
STAGE2_PROPOSAL_AND_METHODOLOGY.md, as to why it is included -- this module
just computes them. Accuracy is computed and reported (for comparability
with the existing GeNIS benchmark) but macro_f1 is the metric this
project's confirmatory hypothesis tests (RQ1/RQ2) actually use.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
) -> Dict[str, Any]:
    """Compute the full metric set for one set of predictions.

    Args:
        y_true: Ground-truth integer-encoded labels.
        y_pred: Predicted integer-encoded labels.
        labels: Optional explicit label ordering (e.g. [0, 1] for binary
            benign/malicious); if None, inferred from the union of y_true
            and y_pred.

    Returns:
        Dict with accuracy, macro_f1, weighted_f1, macro_precision,
        macro_recall, per_class (dict of per-label precision/recall/f1),
        false_positive_rate (binary case only, assumes label 1 = positive
        / malicious, label 0 = negative / benign), confusion_matrix (as a
        nested list, JSON-serializable), and labels (the label ordering
        used).
    """
    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class = {
        str(label): {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(np.sum(np.array(y_true) == label)),
        }
        for i, label in enumerate(labels)
    }

    result: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": [str(l) for l in labels],
        "n_samples": int(len(y_true)),
    }

    if len(labels) == 2:
        # Convention: labels[1] is treated as the positive/malicious class.
        # This assumes a binary label encoding where 1 = malicious, 0 =
        # benign; callers must ensure y_true/y_pred follow this encoding
        # (src/training/train.py enforces this at label-encoding time).
        tn, fp, fn, tp = cm.ravel()
        result["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
        result["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")

    return result


def aggregate_metrics_across_runs(metric_dicts: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of per-run/per-fold metric dicts into mean +/- SD.

    Args:
        metric_dicts: List of dicts as returned by
            compute_classification_metrics, one per seed or fold.

    Returns:
        Dict with mean and std for each scalar metric present in every
        input dict (macro_f1, accuracy, etc.), plus n_runs.
    """
    scalar_keys = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "false_positive_rate",
        "false_negative_rate",
    ]
    agg: Dict[str, Any] = {"n_runs": len(metric_dicts)}
    for key in scalar_keys:
        values = [m[key] for m in metric_dicts if key in m and not np.isnan(m[key])]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return agg
