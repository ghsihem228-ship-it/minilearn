"""Compare both solvers to independent scikit-learn references."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge

from minilearn import LinearRegression


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    y = X @ np.array([2.0, -3.0, 0.5, 4.0, 0.0]) + 7 + rng.normal(0, 0.1, len(X))
    return X, y


@pytest.mark.parametrize("solver", ["normal", "gd"])
@pytest.mark.parametrize("alpha", [0.0, 2.5])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_matches_sklearn(regression_data, solver, alpha, fit_intercept):
    X, y = regression_data
    reference = (
        Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="svd")
        if alpha
        else SklearnLinearRegression(fit_intercept=fit_intercept)
    ).fit(X, y)
    model = LinearRegression(
        solver=solver,
        alpha=alpha,
        fit_intercept=fit_intercept,
        learning_rate=0.1,
        tol=1e-10,
        random_state=42,
    )
    assert model.fit(X, y) is model
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-8, rtol=0)
    np.testing.assert_allclose(model.intercept_, reference.intercept_, atol=1e-8, rtol=0)
    holdout = np.random.default_rng(100).normal(size=(30, 5))
    np.testing.assert_allclose(
        model.predict(holdout), reference.predict(holdout), atol=1e-8, rtol=0
    )
    assert model.score(X, y) == pytest.approx(reference.score(X, y), abs=1e-10)
    assert model.converged_
    assert model.n_features_in_ == 5
    if solver == "gd":
        assert 0 < model.n_iter_ < model.max_iter
        assert len(model.loss_curve_) == model.n_iter_ + 1
        assert np.all(np.diff(model.loss_curve_) <= 1e-12)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_rank_deficient_design(alpha):
    X = np.array([[1, 2, 1], [2, 4, 1], [3, 6, 1], [4, 8, 1]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)
    reference = (Ridge(alpha=alpha, solver="svd") if alpha else SklearnLinearRegression()).fit(X, y)
    model = LinearRegression(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-10, rtol=0)
    np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=1e-10, rtol=0)


def test_underdetermined_design():
    X = np.random.default_rng(4).normal(size=(4, 10))
    y = np.arange(4, dtype=float)
    reference = SklearnLinearRegression().fit(X, y)
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-10, rtol=0)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-10, rtol=0)


@pytest.mark.parametrize("solver", ["normal", "gd"])
def test_intercept_is_not_regularized(solver):
    X = np.ones((12, 2))
    y = np.full(12, 9.0)
    model = LinearRegression(solver=solver, alpha=100, learning_rate=0.1, random_state=0).fit(X, y)
    np.testing.assert_allclose(model.coef_, 0, atol=1e-8, rtol=0)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8, rtol=0)
    assert model.intercept_ == pytest.approx(9)


def test_reproducible_and_does_not_mutate_input(regression_data):
    X, y = regression_data
    original_X, original_y = X.copy(), y.copy()
    first = LinearRegression(solver="gd", learning_rate=0.1, random_state=3).fit(X, y)
    second = LinearRegression(solver="gd", learning_rate=0.1, random_state=3).fit(X, y)
    np.testing.assert_array_equal(first.coef_, second.coef_)
    np.testing.assert_array_equal(first.loss_curve_, second.loss_curve_)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


def test_refit_updates_learned_state(regression_data):
    X, y = regression_data
    model = LinearRegression().fit(X, y)
    model.fit(X[:, :2], 2 * y)
    assert model.n_features_in_ == 2
    assert model.coef_.shape == (2,)
    with pytest.raises(ValueError, match="2 features"):
        model.predict(X)


def test_nonconvergence_is_reported(regression_data):
    X, y = regression_data
    model = LinearRegression(solver="gd", max_iter=1, random_state=0)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model.fit(X, y)
    assert not model.converged_
    assert model.n_iter_ == 1
    assert np.isfinite(model.predict(X)).all()


def test_divergence_has_actionable_error():
    with pytest.raises(ValueError, match="diverged.*scale X"):
        LinearRegression(solver="gd", learning_rate=1e200, random_state=0).fit(
            [[0], [1], [2]], [0, 1, 2]
        )


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"solver": "invalid"}, "solver"),
        ({"alpha": -1}, "alpha"),
        ({"alpha": np.nan}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": np.inf}, "learning_rate"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"tol": -1}, "tol"),
        ({"fit_intercept": 1}, "fit_intercept"),
        ({"random_state": -1}, "random_state"),
    ],
)
def test_invalid_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        LinearRegression(**parameters).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize(
    "X, y, message",
    [
        ([1, 2], [1, 2], "2D"),
        ([], [], "nonempty"),
        ([[1], [2]], [[1], [2]], "1D"),
        ([[1], [2]], [1], "same number"),
        ([[np.nan], [2]], [1, 2], "finite"),
        ([[1], [2]], [1, np.inf], "finite"),
        ([[1j], [2j]], [1, 2], "real numeric"),
    ],
)
def test_invalid_training_data(X, y, message):
    with pytest.raises(ValueError, match=message):
        LinearRegression().fit(X, y)


def test_prediction_validation():
    with pytest.raises(ValueError, match="not fitted"):
        LinearRegression().predict([[1]])
    model = LinearRegression().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="1 features"):
        model.predict([[1, 2]])
    with pytest.raises(ValueError, match="finite"):
        model.predict([[np.nan]])


def test_single_sample_fit_and_r2_limitation():
    model = LinearRegression().fit([[3]], [7])
    np.testing.assert_allclose(model.predict([[3]]), [7])
    with pytest.raises(ValueError, match="at least two"):
        model.score([[3]], [7])
