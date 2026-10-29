from semester_project.gradient_descent import smooth_descent
from semester_project.type_alisases import FloatArray
import numpy as np
rng = np.random.default_rng(42)


def test_parabola() -> None: 
    def parab(x : FloatArray) -> FloatArray :
        return x ** 2
    def parab_grad(x : FloatArray) -> FloatArray:
        return 2 * x
    for _ in range(10):
        beta_start = rng.uniform(-5, 5)
        assert abs(smooth_descent(parab, parab_grad, np.array([beta_start]))[0]) < 0.01
    


def test_parabola_history() -> None:
    def objective(x: FloatArray) -> float:
        return float(np.sum(x**2))

    def gradient(x: FloatArray) -> FloatArray:
        return 2 * x

    beta_0 = np.array([4.0])

    beta, history = smooth_descent(
        objective,
        gradient,
        beta_0,
        eta=0.25,
        epsi=1e-8,
        max_steps=3,
    )

    expected_beta = np.array([0.5])
    expected_history = [16.0, 4.0, 1.0, 0.25]

    np.testing.assert_allclose(beta, expected_beta)
    np.testing.assert_allclose(history, expected_history)

    assert len(history) == 4