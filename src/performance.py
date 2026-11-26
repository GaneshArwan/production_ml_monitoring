"""Model performance evaluation for labeled production batches."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.schemas import PerformanceMetrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float = 0.5,
) -> PerformanceMetrics:
    """Compute AUC-ROC, precision, recall, F1, and class balance rates safely."""
    if not np.isfinite(decision_threshold) or not (0.0 <= decision_threshold <= 1.0):
        raise ValueError("decision_threshold must be finite and within [0.0, 1.0].")

    y_raw = np.asarray(y_true)
    p_arr = np.asarray(y_prob, dtype=float)
    if y_raw.ndim != 1 or p_arr.ndim != 1 or y_raw.size == 0 or y_raw.shape != p_arr.shape:
        raise ValueError("y_true and y_prob must be non-empty 1D arrays of equal length.")
    if not np.isfinite(p_arr).all():
        raise ValueError("y_prob must contain only finite probabilities.")
    if not np.isin(y_raw, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels (0 or 1).")

    y_arr = y_raw.astype(int)
    y_pred = (p_arr >= decision_threshold).astype(int)
    auc = float(roc_auc_score(y_arr, p_arr)) if np.unique(y_arr).size > 1 else 0.5

    return PerformanceMetrics(
        auc_roc=round(auc, 4),
        precision=round(float(precision_score(y_arr, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_arr, y_pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_arr, y_pred, zero_division=0)), 4),
        positive_rate=round(float(np.mean(y_arr)), 4),
        prediction_rate=round(float(np.mean(y_pred)), 4),
    )
