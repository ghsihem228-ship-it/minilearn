# Linear regression and ridge

For $X \in \mathbb{R}^{n\times d}$, target vector $y$, weights $w$, and intercept
$b$, predictions are $\hat y=Xw+b$. MiniLearn minimizes

$$J(w,b)=\frac{\|Xw+b-y\|_2^2+\alpha\|w\|_2^2}{2n}.$$

The intercept is not regularized. `alpha=0` gives ordinary least squares.
Multiplying the objective by $2n$ shows that this uses scikit-learn Ridge's
penalty convention, rather than an L2 penalty independent of sample count.

## Centering and the normal equation

With an intercept, define $X_c=X-\bar X$ and $y_c=y-\bar y$. Setting the derivative
with respect to $b$ to zero gives $b=\bar y-\bar Xw$. Substitute and solve

$$ (X_c^T X_c+\alpha I)w=X_c^T y_c. $$

MiniLearn does **not** explicitly invert this matrix: forming $X^TX$ squares the
condition number. Ordinary least squares uses `np.linalg.lstsq(X_c, y_c)`.
Ridge solves the equivalent augmented least-squares problem

$$
\begin{bmatrix}X_c\\\sqrt{\alpha}I\end{bmatrix} w
\approx \begin{bmatrix}y_c\\0\end{bmatrix}.
$$

This handles dependent columns and underdetermined designs. At `alpha=0`, least
squares chooses the minimum-norm solution if coefficients are nonunique.
With `fit_intercept=False`, use the original arrays and set $b=0$.

## Gradient descent

$$\nabla_w J=\frac{X_c^T(X_cw-y_c)+\alpha w}{n}.$$

Every update uses the entire batch: $w\leftarrow w-\eta\nabla_w J$.
The implementation loops over optimization steps, not individual samples.
The intercept is recovered after training from the means above.

If $L=\lambda_{\max}((X_c^TX_c+\alpha I)/n)$, a step size $0<\eta<2/L$
converges for a strictly convex quadratic. A conservative choice is $\eta\leq1/L$.
Standardize training features before gradient descent and apply the same
statistics to held-out data. Scaling changes the meaning of the ridge penalty,
so compare models using the same transformed features.

Training stops when $\|\nabla_w J\|_\infty\leq\text{tol}$. `loss_curve_` includes
the initial objective and one entry per update. `n_iter_` counts updates.
`converged_` reports whether the stopping criterion was reached. Reaching
`max_iter` without convergence emits a warning; nonfinite updates raise an error.

## Evaluation

$$\mathrm{MSE}=\frac1n\sum_i(y_i-\hat y_i)^2,\qquad
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.$$

$R^2$ can be negative. For constant targets return 1 for exact predictions and 0
otherwise, matching scikit-learn's default behavior. MiniLearn requires two
samples for $R^2$ and supports only a one-dimensional target.

See [the demo notebook](../notebooks/01_linear_regression.ipynb) and
[comparison tests](../tests/test_linear_regression.py).
