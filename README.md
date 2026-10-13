# MiniLearn

Core machine learning algorithms implemented with NumPy, with explained math,
notebook demos, and tests against scikit-learn. Built for students learning the
derivations and recruiters reviewing the implementation.

![Fitted regression lines and gradient-descent convergence](docs/assets/linear_regression.png)

## Quick start

Requires Python 3.12 or newer. NumPy is the only library runtime dependency.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

```python
import numpy as np
from minilearn import LinearRegression

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])
model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)  # [2.] 1.0 (up to floating-point rounding)
print(model.predict([[4.0]]))  # [9.]
```

For full-batch gradient descent, use `solver="gd"`, standardize training features
with `minilearn.utils.StandardScaler`, and choose an appropriate `learning_rate`.
Set `alpha>0` to enable ridge, using the same penalty convention as scikit-learn.
Use `random_state` for reproducible initial coefficients. The API supports
single-output regression; learned attributes end with `_`.

## Algorithm status

| Algorithm | Status | Math | Notebook |
| --- | --- | --- | --- |
| Linear regression: least squares, gradient descent, L2 | Implemented | [Derivation](docs/linear_regression.md) | [Demo](notebooks/01_linear_regression.ipynb) |
| Binary logistic regression | Next | — | — |
| k-means with k-means++ | Planned | — | — |
| PCA via SVD | Planned | — | — |
| Decision tree classifier | Planned | — | — |
| MLP with backpropagation and mini-batch SGD | Planned | — | — |

Shared utilities include input validation, seeded train/test splitting, and
standard scaling. Metrics include MSE, RMSE, R², and classification accuracy.

## Validation and demo

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
jupyter nbconvert --to notebook --execute notebooks/01_linear_regression.ipynb \
  --output-dir /tmp --ExecutePreprocessor.timeout=120
```

The notebook plots fitted lines and gradient-descent loss and prints a comparison
table against scikit-learn. It uses generated data and needs no downloads.

Seeded notebook results (160 generated samples, 20% held out):

| Solver | L2 alpha | Test MSE | Test R² | Max coefficient difference vs. scikit-learn |
| --- | --- | --- | --- | --- |
| Least squares | 0 | 0.240614 | 0.981088 | `8.882e-16` |
| Gradient descent | 0 | 0.240614 | 0.981088 | `9.264e-11` |
| Least squares | 2.5 | 0.224650 | 0.982343 | `8.882e-16` |
| Gradient descent | 2.5 | 0.224650 | 0.982343 | `9.377e-11` |

| Comparison | Acceptance criterion |
| --- | --- |
| Both solvers vs. ordinary least squares | Coefficients and held-out predictions within `1e-8` absolute tolerance |
| Both solvers vs. Ridge, `alpha=2.5` | Coefficients and held-out predictions within `1e-8` absolute tolerance |
| Intercept on/off | R² within `1e-10` absolute tolerance |
| Rank-deficient and underdetermined designs | Least-squares results within `1e-10` absolute tolerance |

These are reproducible correctness checks on seeded synthetic data, not speed
benchmarks or claims about generalization on real-world datasets. CI runs pytest,
Ruff, packaging, and notebook execution on every push and pull request.

## Roadmap

Add logistic regression next, including a numerical gradient check; then k-means,
PCA, decision trees, and the MLP in separate feature branches and PRs. Later:
k-NN, naive Bayes, random forests, Adam, and decision tree regression.

See [contributing](CONTRIBUTING.md) for development conventions. Licensed under
the [MIT License](LICENSE). scikit-learn is used only in tests and notebooks.
