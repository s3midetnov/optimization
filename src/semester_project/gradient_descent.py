from .type_alisases import FloatArray
from collections.abc import Callable
from .utils import norm_squared


def smooth_descent(
        objective: Callable[[FloatArray], float],
        gradient: Callable[[FloatArray], FloatArray],        
        beta_0: FloatArray, 
        eta: float = 0.1,
        epsi: float = 1.e-6, 
        max_steps: int =1000,        
) -> tuple[FloatArray, list[float]]:
    if eta <= 0:
        raise ValueError("eta must be positive")
    if epsi <= 0:
        raise ValueError("epsi must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")


    history = [objective(beta_0)]
    b = beta_0.copy()    
    grad = gradient(beta_0)
    for _ in range(max_steps):
        b = beta_0 - eta * grad
        grad = gradient(b)
        history.append(objective(b))
        if norm_squared(grad) < epsi ** 2: 
            break        
        beta_0 = b    
    return b, history 