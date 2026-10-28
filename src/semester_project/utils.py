from .type_alisases import FloatArray

def norm_squared(
        x : FloatArray
) -> float:
    return (x**2).sum()