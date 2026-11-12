from dataclasses import dataclass
from enum import Enum


from .type_alisases import FloatArray


def norm_squared(x: FloatArray) -> float:
    return float((x**2).sum())


@dataclass
class OptimizationResult:
    beta: FloatArray
    objective_history: list[float]
    n_iter: int
    converged: bool


class Penalty(Enum):
    Lasso = 1
    Ridge = 2

