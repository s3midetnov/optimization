from collections.abc import Callable

import numpy as np

from .type_alisases import FloatArray
from .utils import OptimizationResult, norm_squared
from .loss import least_squares_loss, least_squares_loss_gradient
from .penalty import lasso_penalty, soft_threshold


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
        if norm_squared(grad) < epsi**2:
            converged = True
            break
        beta_0 = b

    return OptimizationResult(
        beta=b, objective_history=history, n_iter=steps, converged=converged
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

    def objective(beta : FloatArray) -> float: 
        objective_diff(beta) + objective_non_diff(beta)
    
    history = [objective(beta_0)]
    b = beta_0.copy()
    
    steps = 0
    converged = False

    for steps in range(max_steps):        
        z = beta_0 - eta * objective_diff_grad(beta_0) # gradient step
        b = proximal_fun(z)                            # proximal step
        history.append(objective(b))
        if np.linalg.norm(b - beta_0, ord=2) < epsi:
            converged = True
            break 
        beta_0 = b

    return OptimizationResult(
        beta=b, objective_history=history, n_iter=steps, converged=converged
    )


def lasso_gradient_descent(
        X : FloatArray,
        y : FloatArray,        
        lam : float,
        beta_0: FloatArray,
        eta: float = 0.1,
        epsi: float = 1.0e-6,
        max_steps: int = 1000,
) -> OptimizationResult: 
    def objective_diff (beta : FloatArray) -> float:
        return least_squares_loss(X, y, beta)

    def objective_diff_grad (beta : FloatArray) -> FloatArray:
        return least_squares_loss_gradient(X, y, beta)

    def objective_non_diff (beta : FloatArray) -> float: 
        return lam * lasso_penalty(beta)

    def proximal_fun (z : FloatArray) -> FloatArray:
        return soft_threshold(z, eta)
    
    return proximal_gradient_descent(
        objective_diff=objective_diff,
        objective_diff_grad=objective_diff_grad,
        objective_non_diff=objective_non_diff,
        proximal_fun=proximal_fun,
        beta_0=beta_0,
        eta=eta,
        epsi=epsi,
        max_steps=max_steps
    )