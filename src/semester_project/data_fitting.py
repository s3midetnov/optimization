import numpy as np

from .gradient_descent import newton_descent, proximal_gradient_descent, smooth_descent
from .loss import (
    least_squares_loss,
    least_squares_loss_gradient,
    least_squares_loss_hessian,
)
from .penalty import lasso_penalty, soft_threshold, ridge_penalty, ridge_penalty_gradient
from .utils import FloatArray, OptimizationResult, Penalty




def ordinary_gradient_desent(        
    X: FloatArray,
    y: FloatArray,
    beta_0: FloatArray,
    eta: float = 0.1,
    epsi: float = 1.0e-6,
    max_steps: int = 1000,
) -> OptimizationResult:
    def objective(beta: FloatArray) -> float:
        return least_squares_loss(X, y, beta)

    def objective_grad(beta: FloatArray) -> FloatArray: 
        return least_squares_loss_gradient(X, y, beta)

    return smooth_descent(
        objective=objective,
        gradient=objective_grad,
        beta_0=beta_0,
        eta=eta,
        epsi=epsi,
        max_steps=max_steps
    )

def ridge_gradient_descent(
    X: FloatArray,
    y: FloatArray,
    lam: float,
    beta_0: FloatArray,
    eta: float = 0.1,
    epsi: float = 1.0e-6,
    max_steps: int = 1000,
) -> OptimizationResult:
    def objective(beta: FloatArray) -> float:
        return least_squares_loss(X, y, beta) + lam * ridge_penalty(beta)

    def objective_grad(beta: FloatArray) -> FloatArray: 
        return least_squares_loss_gradient(X, y, beta) + lam * ridge_penalty_gradient(beta)

    return smooth_descent(
        objective=objective,
        gradient=objective_grad,
        beta_0=beta_0,
        eta=eta,
        epsi=epsi,
        max_steps=max_steps
    )


def analytic_solution(
        X: FloatArray,
        y: FloatArray,
) -> FloatArray:
    if X.ndim != 2:
        raise ValueError("Incorrect dimensions: first argument must be a matrix")

    if y.ndim != 1:
            raise ValueError("Incorrect dimensions: second argument must be a vector")

    n, _ = X.shape
    m = y.shape[0]

    if m != n: 
        raise ValueError(f"Incorrect dimensions, first dimension of  must be the same as dimension of y")
    
    return np.linalg.solve(X.T @ X, X.T @ y)


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


def predict(
        X: FloatArray,
        beta: FloatArray
) -> FloatArray:
    if X.ndim >= 3:
        raise ValueError("X must be a matrix")
    if beta.ndim >= 2:
        raise ValueError("beta must be a vector")
    n, p_X = X.shape 
    p_beta = beta.shape[0]
    if p_X != p_beta: 
        raise ValueError("dimensions of X, beta are wrong, one should be able to multiply X @ beta ")
    return X @ beta 


def fit(
        X: FloatArray,
        y: FloatArray,
        lam: float,
        penalty: Penalty,
        beta_0: FloatArray,
        eta: float = 0.1,
        epsi: float = 1.0e-6,
        max_steps: int = 1000,
) -> OptimizationResult:
    match penalty: 
        case Penalty.Lasso: 
            return lasso_gradient_descent(X, y, lam, beta_0, eta, epsi, max_steps)
        case Penalty.Ridge:
            return ridge_gradient_descent(X, y, lam, beta_0, eta, epsi, max_steps)