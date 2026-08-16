"""Classification metrics and audit-ready diagnostic tables."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _as_1d(values: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def classification_metrics(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute primary and diagnostic classification metrics.

    Balanced accuracy and macro-F1 are the PAIR-BST primary classification
    outcomes.  Supplying ``labels`` fixes the class universe, so per-class rows
    and confusion matrices retain a stable order across folds.
    """

    true = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    class_labels = np.asarray(labels if labels is not None else np.unique(true))
    if class_labels.ndim != 1 or class_labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence.")

    raw_cm = confusion_matrix(true, pred, labels=class_labels)
    denominators = raw_cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(
        raw_cm,
        denominators,
        out=np.zeros_like(raw_cm, dtype=np.float64),
        where=denominators != 0,
    )

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    per_class = [
        {
            "label": label.item() if isinstance(label, np.generic) else label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(class_labels)
    ]

    return {
        "accuracy": float(accuracy_score(true, pred)),
        # Mean recall over the explicitly frozen class universe. This equals
        # sklearn balanced_accuracy_score when every class is present and stays
        # well-defined (zero recall) for an absent class in an audit subset.
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(
            f1_score(
                true,
                pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true,
                pred,
                labels=class_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "labels": class_labels.tolist(),
        "per_class": per_class,
        "confusion_matrix": raw_cm,
        "confusion_matrix_row_normalized": normalized_cm,
        "n_samples": int(true.shape[0]),
    }


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    """Return only scalar fields from :func:`classification_metrics`."""

    names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "n_samples")
    return {name: metrics[name] for name in names}
