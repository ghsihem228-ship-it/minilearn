"""Compare regression and classification metrics to scikit-learn."""

import numpy as np
import pytest
from sklearn import metrics as reference

from minilearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


@pytest.mark.parametrize(
    "metric, reference_metric",
    [
        (mean_squared_error, reference.mean_squared_error),
        (root_mean_squared_error, reference.root_mean_squared_error),
        (r2_score, reference.r2_score),
    ],
)
def test_regression_metrics_match_sklearn(metric, reference_metric):
    y = np.array([2.0, -1.0, 4.0, 7.0])
    prediction = np.array([2.5, -2.0, 3.0, 8.0])
    assert metric(y, prediction) == pytest.approx(reference_metric(y, prediction))


@pytest.mark.parametrize("prediction, expected", [([2, 2], 1), ([1, 1], 0)])
def test_constant_target_r2(prediction, expected):
    assert r2_score([2, 2], prediction) == expected


def test_r2_can_be_negative_and_requires_two_samples():
    assert r2_score([0, 1], [10, 10]) < 0
    with pytest.raises(ValueError, match="at least two"):
        r2_score([1], [1])


@pytest.mark.parametrize(
    "actual, predicted",
    [([0, 1, 1], [0, 0, 1]), (["cat", "dog", "cat"], ["cat", "cat", "cat"])],
)
def test_accuracy_matches_sklearn(actual, predicted):
    assert accuracy_score(actual, predicted) == pytest.approx(
        reference.accuracy_score(actual, predicted)
    )


@pytest.mark.parametrize("metric", [mean_squared_error, root_mean_squared_error, r2_score])
@pytest.mark.parametrize(
    "actual, predicted, message",
    [
        ([], [], "nonempty"),
        ([1, 2], [1], "same shape"),
        ([[1], [2]], [1, 2], "1D"),
        ([1, np.nan], [1, 2], "finite"),
    ],
)
def test_invalid_regression_inputs(metric, actual, predicted, message):
    with pytest.raises(ValueError, match=message):
        metric(actual, predicted)


@pytest.mark.parametrize(
    "actual, predicted, message",
    [([], [], "nonempty"), ([1, 2], [1], "same shape"), ([np.nan], [0], "NaN")],
)
def test_invalid_accuracy_inputs(actual, predicted, message):
    with pytest.raises(ValueError, match=message):
        accuracy_score(actual, predicted)
