"""Evaluation Metrics — computes classification and regression performance metrics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    y_true: np.ndarray | list, y_pred: np.ndarray | list
) -> dict[str, Any]:
    """Calculate classification performance metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.

    Returns:
        Dictionary of calculated metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        return {}

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        classes = np.unique(y_true_arr)
        avg = "binary" if len(classes) <= 2 else "macro"

        metrics = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(
                precision_score(y_true_arr, y_pred_arr, average=avg, zero_division=0)
            ),
            "recall": float(recall_score(y_true_arr, y_pred_arr, average=avg, zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred_arr, average=avg, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
            "classification_report": classification_report(y_true_arr, y_pred_arr, zero_division=0),
        }
        return metrics

    except ImportError:
        # Fallback to pure numpy implementation if sklearn is not available
        logger.warning(
            "scikit-learn is not available. Falling back to pure numpy classification metrics."
        )
        correct = np.sum(y_true_arr == y_pred_arr)
        total = len(y_true_arr)
        acc = correct / total if total > 0 else 0.0

        # Simple binary fallback
        if len(np.unique(y_true_arr)) <= 2:
            tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
            fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
            fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
            tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            }
        else:
            return {
                "accuracy": float(acc),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }


def calculate_regression_metrics(
    y_true: np.ndarray | list, y_pred: np.ndarray | list
) -> dict[str, Any]:
    """Calculate regression performance metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary of calculated metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        return {}

    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mse = float(mean_squared_error(y_true_arr, y_pred_arr))
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
            "r2": float(r2_score(y_true_arr, y_pred_arr)),
        }
        return metrics

    except ImportError:
        logger.warning(
            "scikit-learn is not available. Falling back to pure numpy regression metrics."
        )
        errors = y_true_arr - y_pred_arr
        mse = float(np.mean(errors**2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))

        # R2 score
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
