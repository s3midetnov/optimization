import numpy as np

from semester_project.loss import norm_squared

rng = np.random.default_rng(42)


def test_norm_zero() -> None:
    for n in range(10):
        assert norm_squared(np.zeros(n)) == 0


def test_norm_pythagorean() -> None:
    assert norm_squared(np.array([3.0, 4.0])) == 25.0

    assert norm_squared(np.array([5.0, 12.0])) == 13.0**2

    assert norm_squared(np.array([7.0, 24.0])) == 25.0**2

    assert norm_squared(np.array([9.0, 40.0])) == 41.0**2


def test_norm_is_positive() -> None:
    for _ in range(10):
        n = rng.integers(0, 100)
        v = rng.standard_normal((n,))
        assert norm_squared(v) >= 0
