# Binary logistic regression

For $X\in\mathbb{R}^{n\times d}$, coefficients $w$, and intercept $b$, define

$$z_i=x_i^T w+b,\qquad p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}.$$

MiniLearn requires labels in $\{0,1\}$, with both classes present during fitting.
`predict_proba` returns two columns in `classes_ = [0, 1]` order. `predict` chooses
class 1 when its probability exceeds 0.5; ties choose class 0. `score` is accuracy.
Coefficients are stored in a one-dimensional `coef_`, as in MiniLearn's linear
regression API (scikit-learn stores binary coefficients as shape `(1, d)`).

## Cross-entropy and the gradient

The Bernoulli likelihood for a single observation is $p_i^{y_i}(1-p_i)^{1-y_i}$.
Taking its negative logarithm gives binary cross-entropy:

$$\ell_i=-y_i\log p_i-(1-y_i)\log(1-p_i).$$

Differentiate with respect to $p_i$ and use $dp_i/dz_i=p_i(1-p_i)$:

$$\frac{\partial\ell_i}{\partial p_i}
=-\frac{y_i}{p_i}+\frac{1-y_i}{1-p_i},\qquad
\frac{\partial\ell_i}{\partial z_i}
=\left(-\frac{y_i}{p_i}+\frac{1-y_i}{1-p_i}\right)p_i(1-p_i)=p_i-y_i.$$

Since $\partial z_i/\partial w=x_i$ and $\partial z_i/\partial b=1$, the regularized
mean loss and its gradients are

$$J(w,b)=\frac1n\sum_i\ell_i+\frac{\alpha}{2n}\|w\|_2^2,$$

$$\nabla_w J=\frac{X^T(p-y)+\alpha w}{n},\qquad
\frac{\partial J}{\partial b}=\frac1n\sum_i(p_i-y_i).$$

The intercept is unpenalized. With `fit_intercept=False`, keep $b=0$ and update
only $w$. Multiplying $J$ by $n$ shows that `alpha>0` matches scikit-learn's
`LogisticRegression(C=1/alpha)` with L2; `alpha=0` corresponds to `C=infinity`.
This penalty convention also matches MiniLearn linear regression.

## Numerical stability

Compute the sigmoid as `exp(-logaddexp(0, -z))`, so no exponential has a large
positive argument. For probability-based cross-entropy, clip $p$ to
$[\epsilon,1-\epsilon]$ and use `log1p(-p)` before evaluating the logs.
`binary_cross_entropy` and `predict_proba` use double-precision machine epsilon.

Training evaluates the mathematically equivalent logit loss

$$\ell_i=\log\left(1+\exp(-(2y_i-1)z_i)\right)$$

with `logaddexp`. This avoids overflow and cancellation and preserves the true
loss for confidently wrong predictions. For example, $y=0,z=1000$ has loss 1000,
while probability clipping caps its reported probability-based loss near 36.
Using this exact training loss keeps the analytical gradient consistent with
the objective instead of silently changing it at clipped probabilities.

## Optimization and stopping

Full-batch gradient descent updates $w\leftarrow w-\eta\nabla_w J$ and
$b\leftarrow b-\eta\partial J/\partial b$. Operations are vectorized over samples.
The loop runs over optimization steps only. Use `random_state` to reproduce the
small random coefficient initialization; the initial intercept is zero.

Stop when the maximum absolute gradient component is at most `tol`.
`loss_history_` includes the initial objective followed by one loss per update;
`n_iter_` counts updates and `converged_` records whether the tolerance was met.
Reaching `max_iter` emits a warning. Floating-point overflow produces an error
suggesting feature scaling or a smaller step size.

Standardize features using training statistics only. If $A=[X,\mathbf{1}]$ and
$D=\mathrm{diag}(\alpha,\ldots,\alpha,0)$, a Lipschitz bound is
$L\leq\lambda_{\max}(A^TA/(4n)+D/n)$; choosing $0<\eta<2/L$ ensures descent for
this smooth convex objective. Without an intercept, omit the last column.
L2 helps control large weights, but feature scaling changes the penalty's meaning.

Perfectly separable data with `alpha=0` have no finite maximum-likelihood optimum;
gradient tolerance is a stopping criterion, not proof that finite optimal weights
exist. Increase regularization for such data.

## Gradient verification and metrics

Tests approximate each coefficient derivative using central differences:

$$\frac{\partial J}{\partial w_j}\approx
\frac{J(w+h e_j,b)-J(w-h e_j,b)}{2h},\qquad h=10^{-6}.$$

The same check covers the intercept, L2 on/off, and intercept on/off, calling the
same `_loss_and_gradient` method used by fitting.

Precision is $TP/(TP+FP)$, recall is $TP/(TP+FN)$, and F1 is
$2TP/(2TP+FP+FN)$. Positive label is 1. Undefined ratios return 0, matching
scikit-learn with `zero_division=0`; accuracy remains suitable for other labels.

See [the plotting notebook](../notebooks/02_logistic_regression.ipynb) and
[tests](../tests/test_logistic_regression.py).
