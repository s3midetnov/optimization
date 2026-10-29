from collections.abc import Callable
from .type_alisases import FloatArray
from .utils import norm_squared
from .penalty import ridge_penalty, ridge_penalty_gradient, lasso_penalty, soft_threshold

def least_squares_loss(
    X: FloatArray,
    y: FloatArray,
    beta: FloatArray
) -> float:
    """
    Compute least_squares_loss

    Args:
        X: Feature matrix of shape (n, p)
        y: Target vector of shape (n,).
        beta: Coefficient vector of shape (p,).

    Returns:
        The loss value: ||X @ beta - y||² / (2 * n).
    """
    n, _ = X.shape
    return norm_squared(X @ beta - y) / (2 * n)


def least_squares_loss_gradient(
        X: FloatArray,
        y: FloatArray,
        beta: FloatArray
) -> FloatArray:
    n, _ = X.shape
    return X.T @ (X @ beta - y) / n


def objective(
        X: FloatArray, 
        y: FloatArray, 
        beta: FloatArray, 
        penalty: Callable[[FloatArray], float] , 
        lam: float 
) -> float: 
    return least_squares_loss(X, y, beta) + lam * penalty(beta)
