"""Validation, seeded splitting, and scaling shared by MiniLearn estimators."""

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike, name: str) -> FloatArray:
    """Convert real numeric input and reject NaN, infinity, and complex values."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must contain real numeric values.")
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numeric values.") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values; NaN and infinity are invalid.")
    return array


def validate_X(X: ArrayLike) -> FloatArray:
    """Return finite 2D features of shape (n_samples, n_features)."""
    array = _as_float_array(X, "X")
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("X must be a nonempty 2D array of shape (n_samples, n_features).")
    return array


def validate_y(y: ArrayLike, n_samples: int | None = None) -> FloatArray:
    """Return a finite, nonempty 1D target; multioutput targets are unsupported."""
    array = _as_float_array(y, "y")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("y must be a nonempty 1D array of shape (n_samples,).")
    if n_samples is not None and len(array) != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    return array


def make_rng(random_state: int | None = None) -> np.random.Generator:
    """Create a local generator without changing NumPy's global random state."""
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0
    ):
        raise ValueError("random_state must be None or a nonnegative integer.")
    return np.random.default_rng(random_state)


def train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    *,
    test_size: float | int = 0.2,
    shuffle: bool = True,
    random_state: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split paired arrays, returning X_train, X_test, y_train, y_test.

    Fractional test sizes round up, matching scikit-learn's size convention.
    Stratification and multiple arrays are deliberately outside this initial API.
    """
    features = validate_X(X)
    target = validate_y(y, len(features))
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a boolean.")
    rng = make_rng(random_state)
    n_samples = len(features)
    if isinstance(test_size, bool):
        raise ValueError("test_size must be a fraction in (0, 1) or a positive integer.")
    if isinstance(test_size, Integral):
        n_test = int(test_size)
    elif isinstance(test_size, Real) and np.isfinite(test_size) and 0 < test_size < 1:
        n_test = int(np.ceil(test_size * n_samples))
    else:
        raise ValueError("test_size must be a fraction in (0, 1) or a positive integer.")
    if not 0 < n_test < n_samples:
        raise ValueError("test_size must leave at least one training and one test sample.")
    indices = rng.permutation(n_samples) if shuffle else np.arange(n_samples)
    if shuffle:
        test_indices, train_indices = indices[:n_test], indices[n_test:]
    else:
        train_indices, test_indices = indices[:-n_test], indices[-n_test:]
    return (
        features[train_indices],
        features[test_indices],
        target[train_indices],
        target[test_indices],
    )


class StandardScaler:
    """Standardize each feature as z = (x - mean) / standard deviation.

    Use population variance (ddof=0). Constant columns have scale 1.
    Fit only on training data to avoid leaking information from the test set.
    """

    def fit(self, X: ArrayLike) -> "StandardScaler":
        """Learn feature means and scales."""
        features = validate_X(X)
        self.mean_ = features.mean(axis=0)
        self.var_ = features.var(axis=0)
        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        self.n_features_in_ = features.shape[1]
        return self

    def _validate_transform(self, X: ArrayLike) -> FloatArray:
        if not hasattr(self, "mean_"):
            raise ValueError("StandardScaler is not fitted; call fit before transform.")
        features = validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features.")
        return features

    def transform(self, X: ArrayLike) -> FloatArray:
        """Apply the training-data scaling statistics."""
        return (self._validate_transform(X) - self.mean_) / self.scale_

    def inverse_transform(self, X: ArrayLike) -> FloatArray:
        """Recover original feature units: x = z * scale + mean."""
        return self._validate_transform(X) * self.scale_ + self.mean_

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        """Fit and transform the same training data."""
        return self.fit(X).transform(X)
