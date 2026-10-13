"""Single-output regression metrics and classification metrics."""

import numpy as np
from numpy.typing import ArrayLike

from minilearn.utils import FloatArray, validate_binary_y, validate_y


def _paired_targets(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[FloatArray, FloatArray]:
    actual = validate_y(y_true)
    predicted = validate_y(y_pred)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return actual, predicted


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return MSE = mean((y_true - y_pred)^2)."""
    actual, predicted = _paired_targets(y_true, y_pred)
    return float(np.mean((actual - predicted) ** 2))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return RMSE = sqrt(MSE), expressed in the target's original units."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return R² = 1 - residual_sum_squares / total_sum_squares.

    For constant targets return 1 for exact predictions and 0 otherwise, as in
    scikit-learn's default force_finite behavior. At least two samples are required.
    """
    actual, predicted = _paired_targets(y_true, y_pred)
    if len(actual) < 2:
        raise ValueError("R² requires at least two samples.")
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total == 0:
        return 1.0 if residual == 0 else 0.0
    return 1.0 - residual / total


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of matching 1D labels, including string labels."""
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.size == 0:
        raise ValueError("Labels must be nonempty 1D arrays.")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    for array in (actual, predicted):
        if array.dtype.kind in "biufc" and not np.isfinite(array).all():
            raise ValueError("Labels must not contain NaN or infinity.")
    return float(np.mean(actual == predicted))


def _binary_counts(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int, int]:
    actual = validate_binary_y(y_true)
    predicted = validate_binary_y(y_pred)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    true_positive = int(np.count_nonzero((actual == 1) & (predicted == 1)))
    false_positive = int(np.count_nonzero((actual == 0) & (predicted == 1)))
    false_negative = int(np.count_nonzero((actual == 1) & (predicted == 0)))
    return true_positive, false_positive, false_negative


def precision_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return TP / (TP + FP) for positive label 1; return 0 if undefined."""
    tp, fp, _ = _binary_counts(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return TP / (TP + FN) for positive label 1; return 0 if undefined."""
    tp, _, fn = _binary_counts(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return 2 TP / (2 TP + FP + FN); return 0 if undefined."""
    tp, fp, fn = _binary_counts(y_true, y_pred)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
