"""Reference comparisons, numerical gradients, and stability checks."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import log_loss

from minilearn import LogisticRegression
from minilearn.linear_model.logistic_regression import binary_cross_entropy, sigmoid


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 4))
    probability = sigmoid(X @ np.array([0.8, -1.2, 0.6, 1.0]) + 0.4)
    y = rng.binomial(1, probability)
    return X, y


@pytest.mark.parametrize("alpha", [0.0, 2.5])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_matches_sklearn(classification_data, alpha, fit_intercept):
    X, y = classification_data
    reference = SklearnLogisticRegression(
        C=1 / alpha if alpha else np.inf,
        fit_intercept=fit_intercept,
        solver="lbfgs",
        tol=1e-12,
        max_iter=10000,
    ).fit(X[:400], y[:400])
    model = LogisticRegression(
        alpha=alpha,
        fit_intercept=fit_intercept,
        learning_rate=0.5,
        tol=1e-10,
        random_state=42,
    )
    assert model.fit(X[:400], y[:400]) is model
    np.testing.assert_allclose(model.coef_, reference.coef_[0], atol=1e-6, rtol=0)
    np.testing.assert_allclose(model.intercept_, reference.intercept_[0], atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        model.predict_proba(X[400:]), reference.predict_proba(X[400:]), atol=1e-7, rtol=0
    )
    np.testing.assert_array_equal(model.predict(X[400:]), reference.predict(X[400:]))
    assert model.score(X[400:], y[400:]) == reference.score(X[400:], y[400:])
    np.testing.assert_array_equal(model.classes_, reference.classes_)
    assert model.converged_ and 0 < model.n_iter_ < model.max_iter
    assert len(model.loss_history_) == model.n_iter_ + 1
    assert np.all(np.diff(model.loss_history_) <= 1e-14)


@pytest.mark.parametrize("alpha", [0.0, 3.5])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_analytical_gradient_matches_central_differences(alpha, fit_intercept):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 3))
    y = np.array([0, 1] * 6, dtype=float)
    coef = np.array([0.3, -0.7, 0.2])
    intercept = 0.4 if fit_intercept else 0.0
    model = LogisticRegression(alpha=alpha, fit_intercept=fit_intercept)
    loss, gradient, intercept_gradient = model._loss_and_gradient(X, y, coef, intercept)
    expected_loss = binary_cross_entropy(y, sigmoid(X @ coef + intercept))
    expected_loss += alpha * np.dot(coef, coef) / (2 * len(y))
    assert loss == pytest.approx(expected_loss, abs=1e-12)
    step = 1e-6
    numerical = np.empty_like(coef)
    for feature in range(len(coef)):
        delta = np.zeros_like(coef)
        delta[feature] = step
        above = model._loss_and_gradient(X, y, coef + delta, intercept)[0]
        below = model._loss_and_gradient(X, y, coef - delta, intercept)[0]
        numerical[feature] = (above - below) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=1e-7)
    if fit_intercept:
        above = model._loss_and_gradient(X, y, coef, intercept + step)[0]
        below = model._loss_and_gradient(X, y, coef, intercept - step)[0]
        assert intercept_gradient == pytest.approx((above - below) / (2 * step), abs=1e-9)
    else:
        assert intercept_gradient == 0


def test_extreme_logits_are_finite_and_loss_preserves_bad_predictions():
    logits = np.array([-1000.0, 0.0, 1000.0])
    np.testing.assert_array_equal(sigmoid(logits), [0, 0.5, 1])
    X = np.array([[-1000.0], [1000.0]])
    model = LogisticRegression()
    wrong_loss, gradient, intercept_gradient = model._loss_and_gradient(
        X, np.array([1.0, 0.0]), np.array([1.0]), 0.0
    )
    assert wrong_loss == pytest.approx(1000)
    np.testing.assert_allclose(gradient, [1000])
    assert intercept_gradient == 0
    correct_loss = model._loss_and_gradient(X, np.array([0.0, 1.0]), np.array([1.0]), 0.0)[0]
    assert correct_loss == pytest.approx(0)
    assert np.isfinite(binary_cross_entropy([1, 0], [0, 1]))


def test_cross_entropy_matches_sklearn_and_clips_endpoints():
    target, probability = [0, 1, 0, 1], [0, 1, 0.2, 0.8]
    assert binary_cross_entropy(target, probability) == pytest.approx(log_loss(target, probability))
    eps = np.finfo(float).eps
    assert binary_cross_entropy([1, 0], [0, 1]) == pytest.approx(-np.log(eps))


def test_unregularized_loss_does_not_square_large_coefficients():
    # The logits and objective are finite even though squaring w would overflow.
    X = np.array([[1e-155], [-1e-155]])
    model = LogisticRegression(alpha=0)
    loss, gradient, intercept_gradient = model._loss_and_gradient(
        X, np.array([1.0, 0.0]), np.array([1e155]), 0.0
    )
    assert loss == pytest.approx(np.logaddexp(0, -1))
    assert np.isfinite(gradient).all() and np.isfinite(intercept_gradient)


@pytest.mark.parametrize(
    "target, probability, message",
    [
        ([0, 1], [0.2], "same shape"),
        ([0, 2], [0.2, 0.3], "0 or 1"),
        ([0, 1], [-0.1, 0.8], "between"),
        ([0, 1], [0.1, 1.1], "between"),
        ([0, 1], [0.1, np.nan], "finite"),
        ([0, 1], [[0.1], [0.8]], "1D"),
    ],
)
def test_invalid_cross_entropy_inputs(target, probability, message):
    with pytest.raises(ValueError, match=message):
        binary_cross_entropy(target, probability)


def test_intercept_is_not_regularized_and_ties_choose_zero():
    X = np.zeros((8, 2))
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    model = LogisticRegression(alpha=100, learning_rate=0.05, tol=1e-9, random_state=0).fit(X, y)
    np.testing.assert_allclose(model.coef_, 0, atol=1e-9, rtol=0)
    assert model.intercept_ == pytest.approx(np.log(3), abs=1e-7)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], 0.75, atol=1e-8)
    balanced = LogisticRegression(fit_intercept=False).fit(np.zeros((4, 1)), [0, 1, 0, 1])
    assert balanced.n_iter_ == 0
    np.testing.assert_array_equal(balanced.predict([[0]]), [0])


def test_prediction_probability_shape_sum_and_saturation(classification_data):
    X, y = classification_data
    model = LogisticRegression(learning_rate=0.5, random_state=0).fit(X, y)
    probability = model.predict_proba(X[:7])
    assert probability.shape == (7, 2)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert ((probability > 0) & (probability < 1)).all()
    extremes = np.vstack([-1e6 * model.coef_, 1e6 * model.coef_])
    extreme_probability = model.predict_proba(extremes)
    assert np.isfinite(extreme_probability).all()
    assert ((extreme_probability > 0) & (extreme_probability < 1)).all()
    np.testing.assert_array_equal(model.predict(extremes), [0, 1])


def test_reproducibility_refit_and_input_immutability(classification_data):
    X, y = classification_data
    original_X, original_y = X.copy(), y.copy()
    first = LogisticRegression(learning_rate=0.5, random_state=3).fit(X, y)
    second = LogisticRegression(learning_rate=0.5, random_state=3).fit(X, y)
    np.testing.assert_array_equal(first.coef_, second.coef_)
    np.testing.assert_array_equal(first.loss_history_, second.loss_history_)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    first.fit(X[:, :2], y)
    assert first.coef_.shape == (2,)
    assert first.n_features_in_ == 2
    with pytest.raises(ValueError, match="2 features"):
        first.predict(X)


def test_nonconvergence_emits_warning_and_retains_loss_history(classification_data):
    X, y = classification_data
    model = LogisticRegression(max_iter=1, random_state=0)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model.fit(X, y)
    assert not model.converged_
    assert model.n_iter_ == 1 and len(model.loss_history_) == 2
    assert np.isfinite(model.predict_proba(X)).all()


def test_overflow_has_actionable_error():
    with pytest.raises(ValueError, match="overflowed.*scale X"):
        LogisticRegression(learning_rate=1e308, random_state=0).fit([[-10.0], [10.0]], [1, 0])


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"alpha": -1}, "alpha"),
        ({"alpha": np.nan}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": np.inf}, "learning_rate"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"tol": 0}, "tol"),
        ({"tol": np.inf}, "tol"),
        ({"fit_intercept": 1}, "fit_intercept"),
        ({"random_state": -1}, "random_state"),
    ],
)
def test_invalid_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        LogisticRegression(**parameters).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize(
    "X, y, message",
    [
        ([1, 2], [0, 1], "2D"),
        ([], [], "nonempty"),
        ([[1], [2]], [[0], [1]], "1D"),
        ([[1], [2]], [0], "same number"),
        ([[np.nan], [2]], [0, 1], "finite"),
        ([[1], [2]], [0, np.nan], "finite"),
        ([[1], [2]], [0, np.inf], "finite"),
        ([[1j], [2j]], [0, 1], "real numeric"),
        ([[1], [2]], [0, 2], "0 or 1"),
        ([[1], [2]], [-1, 1], "0 or 1"),
        ([[1], [2]], [0, 0], "both binary classes"),
        ([[1], [2]], [1, 1], "both binary classes"),
        ([[1], [2], [3]], [0, 0.5, 1], "0 or 1"),
    ],
)
def test_invalid_training_data(X, y, message):
    with pytest.raises(ValueError, match=message):
        LogisticRegression().fit(X, y)


@pytest.mark.parametrize("method", ["predict", "predict_proba", "score"])
def test_unfitted_errors(method):
    args = ([[0]], [0]) if method == "score" else ([[0]],)
    with pytest.raises(ValueError, match="not fitted"):
        getattr(LogisticRegression(), method)(*args)


def test_prediction_and_score_validation(classification_data):
    X, y = classification_data
    model = LogisticRegression(learning_rate=0.5, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="4 features"):
        model.predict_proba([[0]])
    with pytest.raises(ValueError, match="finite"):
        model.predict([[0, 0, np.nan, 0]])
    with pytest.raises(ValueError, match="2D"):
        model.predict([0, 0, 0, 0])
    with pytest.raises(ValueError, match="same shape"):
        model.score(X[:2], [0])
    with pytest.raises(ValueError, match="0 or 1"):
        model.score(X[:2], [0, 2])
