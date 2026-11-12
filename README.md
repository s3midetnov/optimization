# Penalized least squares from scratch

Final project for *Optimization with applications I* (Université de Genève, fall 2026).
A small Python package that solves

$$
\min_{\beta \in \mathbb{R}^p} \; \frac{1}{2n}\|X\beta - y\|_2^2 + \lambda P(\beta), \qquad P \in \{0,\ \|\beta\|_2^2,\ \|\beta\|_1\}
$$

with solvers implemented by hand in NumPy (no regression libraries). SciPy and scikit-learn are used only to check the results.

## Quick start

Requires Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # install dependencies
uv run pytest      # run the test suite
```

```python
import numpy as np
from semester_project.data_fitting import fit
from semester_project.utils import Penalty

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 10))
y = X @ rng.standard_normal(10) + 0.1 * rng.standard_normal(200)

res = fit(X, y, lam=0.05, penalty=Penalty.Lasso, beta_0=np.zeros(10))
print(res.beta, res.n_iter, res.converged)
```

## Methods

| Problem | Solver | Where |
| --- | --- | --- |
| OLS (λ = 0) | gradient descent, Newton's method, normal equations | `data_fitting.py` |
| Ridge, P = ‖β‖² | gradient descent | `data_fitting.py` |
| Lasso, P = ‖β‖₁ | proximal gradient (ISTA) with soft-thresholding | `data_fitting.py` |

Every solver returns an `OptimizationResult` with the estimate, the objective history, the number of iterations and a convergence flag.

## Project structure

- [`src/semester_project/loss.py`](src/semester_project/loss.py): least-squares loss ½ₙ‖Xβ − y‖², its gradient and Hessian, and the penalized objective.
- [`src/semester_project/penalty.py`](src/semester_project/penalty.py): ridge and lasso penalties, the ridge gradient, and the soft-thresholding operator (the proximal map of λ‖·‖₁).
- [`src/semester_project/gradient_descent.py`](src/semester_project/gradient_descent.py): generic solvers that know nothing about regression: `smooth_descent` (fixed step), `proximal_gradient_descent` (ISTA) and `newton_descent`.
- [`src/semester_project/data_fitting.py`](src/semester_project/data_fitting.py): plugs the losses and penalties into the generic solvers (OLS, ridge, lasso, Newton), plus the analytic OLS solution, `predict` and the `fit(..., penalty=...)` entry point.
- [`src/semester_project/utils.py`](src/semester_project/utils.py): `OptimizationResult`, the `Penalty` enum and `norm_squared`.
- [`src/semester_project/type_alisases.py`](src/semester_project/type_alisases.py): the `FloatArray` type used for annotations.
- [`src/semester_project/experiments/tp2.ipynb`](src/semester_project/experiments/tp2.ipynb): exercise session 2, with Newton's method and numerical derivatives.
- [`tests/`](tests/): pytest suite (17 tests). It checks known values of the loss, the gradient against finite differences, penalties and soft-thresholding, and the solvers against `scipy.optimize.minimize`.

## Status

- [x] Loss, penalties, gradient descent, proximal gradient, Newton
- [x] Unit tests against analytic and SciPy solutions
- [ ] Experiments on the course datasets: coefficient paths, choosing λ on the validation set, test-set comparison with scikit-learn
- [ ] Extensions: FISTA, coordinate descent, gradient-based selection of λ
