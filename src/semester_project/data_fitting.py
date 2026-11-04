from .gradient_descent import newton_descent, proximal_gradient_descent
from .loss import (
    least_squares_loss,
    least_squares_loss_gradient,
    least_squares_loss_hessian,
)
from .penalty import lasso_penalty, soft_threshold
from .utils import FloatArray, OptimizationResult


def lasso_gradient_descent(
    X: FloatArray,
    y: FloatArray,
    lam: float,
    beta_0: FloatArray,
    eta: float = 0.1,
    epsi: float = 1.0e-6,
    max_steps: int = 1000,
) -> OptimizationResult:
    def objective_diff(beta: FloatArray) -> float:
        return least_squares_loss(X, y, beta)

    def objective_diff_grad(beta: FloatArray) -> FloatArray:
        return least_squares_loss_gradient(X, y, beta)

    def objective_non_diff(beta: FloatArray) -> float:
        return lam * lasso_penalty(beta)

    def proximal_fun(z: FloatArray) -> FloatArray:
        return soft_threshold(z, eta * lam)

    return proximal_gradient_descent(
        objective_diff=objective_diff,
        objective_diff_grad=objective_diff_grad,
        objective_non_diff=objective_non_diff,
        proximal_fun=proximal_fun,
        beta_0=beta_0,
        eta=eta,
        epsi=epsi,
        max_steps=max_steps,
    )


def newton_method(
    X: FloatArray,
    y: FloatArray,
    beta_0: FloatArray,
    epsi: float = 1.0e-6,
    max_steps: int = 1000,
) -> OptimizationResult:
    def objective(beta: FloatArray) -> float:
        return least_squares_loss(X, y, beta)

    def objective_grad(beta: FloatArray) -> FloatArray:
        return least_squares_loss_gradient(X, y, beta)

    def objective_hess(beta: FloatArray) -> FloatArray:
        return least_squares_loss_hessian(X, y, beta)

    return newton_descent(
        objective=objective,
        gradient=objective_grad,
        hessian=objective_hess,
        beta_0=beta_0,
        epsi=epsi,
        max_steps=max_steps,
    )
