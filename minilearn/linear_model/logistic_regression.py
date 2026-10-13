"""Binary logistic regression and cross-entropy using only NumPy."""

import warnings
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from minilearn.metrics import accuracy_score
from minilearn.utils import (
    FloatArray,
    _as_float_array,
    make_rng,
    validate_binary_y,
    validate_X,
    validate_y,
)


def sigmoid(logits: ArrayLike) -> FloatArray:
    """Return sigma(z) = 1 / (1 + exp(-z)) without exponential overflow."""
    values = _as_float_array(logits, "logits")
    return np.exp(-np.logaddexp(0.0, -values))


def binary_cross_entropy(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Return mean -[y log(p) + (1-y) log(1-p)] with clipped probabilities.

    Probabilities must be a finite 1D array in [0, 1]. Clip to [eps, 1-eps]
    before evaluating logarithms. Training uses an exact logit-based loss so
    saturation clipping does not change the optimization gradient.
    """
    target = validate_binary_y(y_true)
    probability = validate_y(y_prob)
    if target.shape != probability.shape:
        raise ValueError("y_true and y_prob must have the same shape.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")
    eps = np.finfo(np.float64).eps
    probability = np.clip(probability, eps, 1 - eps)
    return float(-np.mean(target * np.log(probability) + (1 - target) * np.log1p(-probability)))


class LogisticRegression:
    """Minimize mean binary cross-entropy + alpha * ||w||² / (2n).

    For z = Xw + b and p = sigma(z), the coefficient gradient is
    [X.T @ (p-y) + alpha*w] / n; the unpenalized intercept gradient is mean(p-y).
    Uses vectorized full-batch gradient descent and supports labels {0, 1} only.

    Parameters
    ----------
    alpha : float, default=0.0
        Nonnegative L2 strength. Matches sklearn LogisticRegression's C=1/alpha;
        alpha=0 corresponds to C=infinity. The intercept is never penalized.
    fit_intercept : bool, default=True
        Learn an intercept; otherwise keep it fixed at zero.
    learning_rate : float, default=0.1
        Positive step size. Standardize features for faster convergence.
    max_iter : int, default=10000
        Maximum number of gradient updates.
    tol : float, default=1e-6
        Stop when the maximum absolute coefficient/intercept gradient is <= tol.
    random_state : int or None, default=None
        Seed for small random initial coefficients; intercept starts at zero.

    Learned attributes: coef_ (1D), intercept_, classes_ ([0, 1]), n_features_in_,
    n_iter_, converged_, and loss_history_ (initial loss plus one per update).
    """

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        max_iter: int = 10000,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        for name, value, allow_zero in (
            ("alpha", self.alpha, True),
            ("learning_rate", self.learning_rate, False),
            ("tol", self.tol, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or (value < 0 if allow_zero else value <= 0)
            ):
                bound = "nonnegative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a finite {bound} number.")
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, Integral)
            or self.max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("fit_intercept must be a boolean.")
        make_rng(self.random_state)

    def _loss_and_gradient(
        self, X: FloatArray, y: FloatArray, coef: FloatArray, intercept: float
    ) -> tuple[float, FloatArray, float]:
        """Compute the exact objective and gradient used by fit.

        logaddexp(0, -(2y-1)z) equals cross-entropy while avoiding log(0),
        exponential overflow, and subtraction of large nearly equal values.
        """
        try:
            with np.errstate(over="raise", invalid="raise"):
                logits = X @ coef + intercept
                loss = float(
                    np.mean(np.logaddexp(0.0, -(2 * y - 1) * logits))
                    + (self.alpha * (coef @ coef) / (2 * len(y)) if self.alpha else 0.0)
                )
                error = sigmoid(logits) - y
                gradient = (X.T @ error + self.alpha * coef) / len(y)
                intercept_gradient = float(error.mean()) if self.fit_intercept else 0.0
        except FloatingPointError as error:
            raise ValueError("Optimization overflowed; scale X or reduce learning_rate.") from error
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError("Optimization overflowed; scale X or reduce learning_rate.")
        return loss, gradient, intercept_gradient

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """Fit on finite 2D features and 1D labels containing both 0 and 1."""
        self._validate_parameters()
        features = validate_X(X)
        target = validate_binary_y(y, len(features), require_both_classes=True)
        coef = make_rng(self.random_state).normal(0, 0.01, features.shape[1])
        intercept = 0.0
        loss, gradient, intercept_gradient = self._loss_and_gradient(
            features, target, coef, intercept
        )
        history = [loss]
        n_iter = 0
        for _ in range(self.max_iter):
            if max(float(np.max(np.abs(gradient))), abs(intercept_gradient)) <= self.tol:
                break
            try:
                with np.errstate(over="raise", invalid="raise"):
                    coef = coef - self.learning_rate * gradient
                    intercept -= self.learning_rate * intercept_gradient
            except FloatingPointError as error:
                raise ValueError(
                    "Optimization overflowed; scale X or reduce learning_rate."
                ) from error
            loss, gradient, intercept_gradient = self._loss_and_gradient(
                features, target, coef, intercept
            )
            history.append(loss)
            n_iter += 1
        converged = max(float(np.max(np.abs(gradient))), abs(intercept_gradient)) <= self.tol
        if not converged:
            warnings.warn(
                "Gradient descent did not converge; scale X, adjust learning_rate, "
                "increase max_iter, or add L2 for separable data.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.coef_ = coef
        self.intercept_ = float(intercept)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = features.shape[1]
        self.n_iter_ = n_iter
        self.converged_ = bool(converged)
        self.loss_history_ = np.asarray(history)
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return (n_samples, 2) probabilities in classes_ order, clipped to eps."""
        if not hasattr(self, "coef_"):
            raise ValueError("LogisticRegression is not fitted; call fit before predict_proba.")
        features = validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features.")
        try:
            with np.errstate(over="raise", invalid="raise"):
                logits = features @ self.coef_ + self.intercept_
        except FloatingPointError as error:
            raise ValueError("Prediction overflowed; scale X.") from error
        eps = np.finfo(np.float64).eps
        probability = np.clip(sigmoid(logits), eps, 1 - eps)
        return np.column_stack([1 - probability, probability])

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Predict class 1 when P(y=1) > 0.5; ties go to class 0."""
        probability = self.predict_proba(X)
        return self.classes_[np.argmax(probability, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on binary labels."""
        target = validate_binary_y(y)
        return accuracy_score(target, self.predict(X))
