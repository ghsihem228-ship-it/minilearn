"""Check scaling against scikit-learn and splitting against invariants."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from minilearn.utils import StandardScaler, make_rng, train_test_split, validate_X, validate_y


def test_scaling_matches_sklearn_and_round_trips():
    X = np.array([[1, 2, 7], [4, 8, 7], [7, 14, 7]], dtype=float)
    original = X.copy()
    reference = SklearnStandardScaler().fit(X)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    np.testing.assert_allclose(scaled, reference.transform(X))
    np.testing.assert_allclose(scaler.mean_, reference.mean_)
    np.testing.assert_allclose(scaler.var_, reference.var_)
    np.testing.assert_allclose(scaler.scale_, reference.scale_)
    np.testing.assert_allclose(scaler.inverse_transform(scaled), X)
    holdout = np.array([[10, 20, 7]], dtype=float)
    np.testing.assert_allclose(scaler.transform(holdout), reference.transform(holdout))
    np.testing.assert_array_equal(X, original)


def test_scaler_validation():
    with pytest.raises(ValueError, match="not fitted"):
        StandardScaler().transform([[1]])
    scaler = StandardScaler().fit([[1, 2]])
    with pytest.raises(ValueError, match="2 features"):
        scaler.transform([[1]])
    with pytest.raises(ValueError, match="finite"):
        scaler.inverse_transform([[np.inf, 1]])


@pytest.mark.parametrize("test_size, expected", [(0.25, 3), (3, 3)])
def test_split_preserves_pairs_and_is_reproducible(test_size, expected):
    X = np.arange(20).reshape(10, 2)
    y = X[:, 0] * 10
    first = train_test_split(X, y, test_size=test_size, random_state=4)
    second = train_test_split(X, y, test_size=test_size, random_state=4)
    X_train, X_test, y_train, y_test = first
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert len(X_test) == expected and len(X_train) == 10 - expected
    np.testing.assert_array_equal(y_train, X_train[:, 0] * 10)
    np.testing.assert_array_equal(y_test, X_test[:, 0] * 10)
    assert not set(y_train) & set(y_test)
    assert set(y_train) | set(y_test) == set(y)


def test_unshuffled_split_preserves_order():
    X = np.arange(10).reshape(-1, 1)
    result = train_test_split(X, X[:, 0], test_size=3, shuffle=False)
    np.testing.assert_array_equal(result[0], X[:7])
    np.testing.assert_array_equal(result[1], X[7:])


@pytest.mark.parametrize("test_size", [0, -1, 10, 1.0, 0.0, np.nan, True, "0.2"])
def test_invalid_test_size(test_size):
    with pytest.raises(ValueError, match="test_size"):
        train_test_split(np.ones((10, 2)), np.ones(10), test_size=test_size)


def test_split_rejects_unusable_inputs():
    with pytest.raises(ValueError, match="training and one test"):
        train_test_split([[1]], [1])
    with pytest.raises(ValueError, match="same number"):
        train_test_split([[1], [2]], [1])
    with pytest.raises(ValueError, match="shuffle"):
        train_test_split([[1], [2]], [1, 2], shuffle=1)


def test_rng_does_not_modify_global_state():
    np.random.seed(10)
    before = np.random.get_state()
    make_rng(42).normal(size=20)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(make_rng(42).normal(size=5), make_rng(42).normal(size=5))


@pytest.mark.parametrize("seed", [-1, 1.5, True, "42"])
def test_invalid_random_state(seed):
    with pytest.raises(ValueError, match="random_state"):
        make_rng(seed)


def test_validation_rejects_empty_features_and_nonnumeric_targets():
    with pytest.raises(ValueError, match="nonempty 2D"):
        validate_X(np.empty((3, 0)))
    with pytest.raises(ValueError, match="real numeric"):
        validate_y(["not a number"])
