
"""Classification metrics for the CUB-200-2011 project."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

NUM_CLASSES = 200


def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    """Compute aggregate and per-class classification metrics.

    The required aggregate keys are:
        accuracy, macro_f1, macro_precision, macro_recall

    By default, macro metrics include all 200 classes. Classes without
    true or predicted examples receive a score of zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length.")
    if not isinstance(num_classes, (int, np.integer)) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2.")

    for name, labels in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{name} must contain integer class labels.")
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError(
                f"{name} labels must be in [0, {num_classes - 1}]."
            )

    labels = np.arange(num_classes)

    per_class_precision = precision_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0,
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0,
    )
    per_class_f1 = f1_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(per_class_f1)),
        "macro_precision": float(np.mean(per_class_precision)),
        "macro_recall": float(np.mean(per_class_recall)),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "normalized_confusion_matrix": normalized_cm,
    }
