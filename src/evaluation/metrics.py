"""
Evaluation metrics for anomaly detection.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from typing import Dict


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int = 100
) -> Dict[str, float]:
    """
    Compute anomaly detection metrics.

    Parameters:
        labels: Ground truth (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)
        k: Top-k for P@k metric

    Returns:
        Dictionary with AUROC, AUPRC, P@k
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # Handle edge cases
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {
            'auroc': np.nan,
            'auprc': np.nan,
            'p_at_k': np.nan
        }

    # AUROC
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = np.nan

    # AUPRC (Average Precision)
    try:
        auprc = average_precision_score(labels, scores)
    except ValueError:
        auprc = np.nan

    # P@k: Precision in top-k scored samples
    k = min(k, len(labels))
    top_k_idx = np.argsort(scores)[-k:]
    p_at_k = labels[top_k_idx].mean()

    return {
        'auroc': auroc,
        'auprc': auprc,
        'p_at_k': p_at_k
    }


def check_score_direction(labels: np.ndarray, scores: np.ndarray) -> bool:
    """
    Check if scores have correct direction (higher = more anomalous).

    Returns True if anomalies have higher scores on average.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]

    return anomaly_scores.mean() > normal_scores.mean()
