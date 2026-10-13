# Contributing to MiniLearn

Use Python 3.12 or newer. Work on one algorithm per feature branch and pull request.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m pytest
python -m build
```

Keep `minilearn/` dependent only on NumPy and the Python standard library.
scikit-learn belongs in tests and notebooks. Write vectorized operations, validate
inputs, document the objective and gradients, and use local random generators.

Each algorithm needs comparisons against scikit-learn, edge-case tests, an
executable plotting notebook, a math note, and an updated README status table.
Logistic regression and the neural network will also need numerical gradient checks.

Use conventional commits such as `feat: add linear regression`,
`test: compare ridge solvers with scikit-learn`, and `docs: explain ridge gradients`.
Run the commands above before opening a PR. Review the PR before merging.
