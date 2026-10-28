from .type_alisases import *
from .utils import * 

def ridge_penalty(
        beta: FloatArray
) -> float:
    return norm_squared(beta)


def ridge_penalty_gradient(
        beta: FloatArray
) -> FloatArray: 
    return beta * 2


def lasso_penalty(
        beta: FloatArray
) -> float: 
    return abs(beta).sum()


def soft_threshold(
        z: FloatArray,
        threshold: float 
) -> FloatArray:
    return np.sign(z) * np.maximum(abs(z) - threshold, 0)

