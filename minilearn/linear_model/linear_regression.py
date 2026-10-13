"""Ordinary least squares and ridge regression using only NumPy."""

import warnings
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from minilearn.metrics import r2_score
from minilearn.utils import FloatArray, make_rng, validate_X, validate_y


class LinearRegression:
    """Minimize J(w) = (||Xw + b - y||² + alpha * ||w||²) / (2n).

    Parameters
    ----------
    solver : {"normal", "gd"}, default="normal"
        "normal" solves the normal-equation objective with stable least squares;
        "gd" uses vectorized full-batch gradient descent.
    alpha : float, default=0.0
        Nonnegative L2 penalty, matching scikit-learn Ridge's alpha convention.
        The intercept is never penalized.
    fit_intercept : bool, default=True
        Center X and y during fitting and recover b = mean(y) - mean(X) @ w.
    learning_rate : float, default=0.01
        Positive gradient-descent step size. Scale features for faster convergence.
    max_iter : int, default=10000
        Maximum gradient-descent updates.
    tol : float, default=1e-8
        Stop when the infinity norm of the objective gradient is at most tol.
    random_state : int or None, default=None
        Seed for small random initial coefficients in gradient descent.

    Learned attributes include coef_, intercept_, n_features_in_, n_iter_,
    converged_, and loss_curve_. This estimator supports a single target only.
    """

    def __init__(
        self,
        *,
        solver: Literal["normal", "gd"] = "normal",
        alpha: float = 0.0,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        max_iter: int = 10000,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        self.solver = solver
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if self.solver not in ("normal", "gd"):
            raise ValueError("solver must be 'normal' or 'gd'.")
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

    def _loss(self, X: FloatArray, y: FloatArray, coef: FloatArray) -> float:
        residual = X @ coef - y
        return float((residual @ residual + self.alpha * (coef @ coef)) / (2 * len(y)))

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """Fit coefficients; do not modify X or y. Return this estimator."""
        self._validate_parameters()
        features = validate_X(X)
        target = validate_y(y, len(features))
        x_mean = features.mean(axis=0) if self.fit_intercept else np.zeros(features.shape[1])
        y_mean = float(target.mean()) if self.fit_intercept else 0.0
        centered_X, centered_y = features - x_mean, target - y_mean

        if self.solver == "normal":
            design, response = centered_X, centered_y
            if self.alpha > 0:
                # Augmentation solves ridge without explicitly forming X.T @ X.
                design = np.vstack([centered_X, np.sqrt(self.alpha) * np.eye(features.shape[1])])
                response = np.concatenate([centered_y, np.zeros(features.shape[1])])
            coef = np.linalg.lstsq(design, response, rcond=None)[0]
            n_iter, converged = 0, True
            losses = [self._loss(centered_X, centered_y, coef)]
        else:
            coef = make_rng(self.random_state).normal(0, 0.01, features.shape[1])
            losses = [self._loss(centered_X, centered_y, coef)]
            converged, n_iter = False, 0
            for _ in range(self.max_iter):
                gradient = (
                    centered_X.T @ (centered_X @ coef - centered_y) + self.alpha * coef
                ) / len(target)
                if np.linalg.norm(gradient, ord=np.inf) <= self.tol:
                    converged = True
                    break
                with np.errstate(over="ignore", invalid="ignore"):
                    coef = coef - self.learning_rate * gradient
                    loss = self._loss(centered_X, centered_y, coef)
                if not np.isfinite(coef).all() or not np.isfinite(loss):
                    raise ValueError("Gradient descent diverged; scale X or reduce learning_rate.")
                losses.append(loss)
                n_iter += 1
            if not converged:
                gradient = (
                    centered_X.T @ (centered_X @ coef - centered_y) + self.alpha * coef
                ) / len(target)
                converged = bool(np.linalg.norm(gradient, ord=np.inf) <= self.tol)
            if not converged:
                warnings.warn(
                    "Gradient descent did not converge; scale X, adjust learning_rate, "
                    "or increase max_iter.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.coef_ = coef
        self.intercept_ = float(y_mean - x_mean @ coef)
        self.n_features_in_ = features.shape[1]
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.loss_curve_ = np.asarray(losses)
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Return predictions y_hat = X @ coef_ + intercept_."""
        if not hasattr(self, "coef_"):
            raise ValueError("LinearRegression is not fitted; call fit before predict.")
        features = validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features.")
        return features @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R² on the supplied data."""
        return r2_score(y, self.predict(X))
