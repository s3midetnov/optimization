from collections.abc import Callable

import numpy as np

from .type_alisases import FloatArray
from .utils import OptimizationResult, norm_squared


def smooth_descent(
    objective: Callable[[FloatArray], float],
    gradient: Callable[[FloatArray], FloatArray],
    beta_0: FloatArray,
    eta: float = 0.1,
    epsi: float = 1.0e-6,
    max_steps: int = 1000,
) -> OptimizationResult:
    if eta <= 0:
        raise ValueError("eta must be positive")
    if epsi <= 0:
        raise ValueError("epsi must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    history = [objective(beta_0)]
    b = beta_0.copy()
    grad = gradient(beta_0)
    steps = 0
    converged = False
    for steps in range(max_steps):
        b = beta_0 - eta * grad
        grad = gradient(b)
        history.append(objective(b))

        if norm_squared(grad) < epsi ** 2: 
            converged = True
            break
        beta_0 = b

    return OptimizationResult(
        beta=b, objective_history=history, n_iter=steps + 1, converged=converged
    )


def proximal_gradient_descent(
    objective_diff: Callable[[FloatArray], float],
    objective_diff_grad: Callable[[FloatArray], float],
    objective_non_diff: Callable[[FloatArray], float],
    proximal_fun: Callable[[FloatArray], FloatArray],
    beta_0: FloatArray,
    eta: float = 0.1,
    epsi: float = 1.0e-6,
    max_steps: int = 1000,
) -> OptimizationResult:
    if eta <= 0:
        raise ValueError("eta must be positive")
    if epsi <= 0:
        raise ValueError("epsi must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    def objective(beta: FloatArray) -> float:
        return objective_diff(beta) + objective_non_diff(beta)

    history = [objective(beta_0)]
    b = beta_0.copy()

    steps = 0
    converged = False

    for steps in range(max_steps):
        z = beta_0 - eta * objective_diff_grad(beta_0)  # gradient step
        b = proximal_fun(z)  # proximal step
        history.append(objective(b))
        if np.linalg.norm(b - beta_0, ord=2) < epsi:
            converged = True
            break
        beta_0 = b

    return OptimizationResult(
        beta=b, objective_history=history, n_iter=steps + 1, converged=converged
    )


def newton_descent(
    objective: Callable[[FloatArray], float],
    gradient: Callable[[FloatArray], FloatArray],
    hessian: Callable[[FloatArray], FloatArray],
    beta_0: FloatArray,
    epsi: float = 1.0e-6,
    max_steps: int = 1000,
) -> OptimizationResult:
    if epsi <= 0:
        raise ValueError("epsi must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    history = [objective(beta_0)]
    if max_steps == 0:
        return OptimizationResult(
            beta=beta_0, objective_history=history, n_iter=0, converged=False
        )

    b = beta_0.copy()
    grad = gradient(beta_0)
    steps = 0
    converged = False

    for steps in range(max_steps):
        hess = hessian(b)
        step = np.linalg.solve(hess, grad)
        b = beta_0 - step

        grad = gradient(b)

        history.append(objective(b))
        if norm_squared(grad) < epsi**2:
            converged = True
            break
        beta_0 = b

    return OptimizationResult(
        beta=b, objective_history=history, n_iter=steps + 1, converged=converged
    )
