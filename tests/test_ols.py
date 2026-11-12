import numpy as np

from semester_project.gradient_descent import proximal_gradient_descent, smooth_descent
from semester_project.loss import least_squares_loss, least_squares_loss_gradient
from semester_project.penalty import (
    ridge_penalty,
    ridge_penalty_gradient,
    soft_threshold,
)
from semester_project.type_alisases import FloatArray

rng = np.random.default_rng(42)
import pytest
from scipy.optimize import minimize


def test_parabola() -> None:
    def parab(x: FloatArray) -> FloatArray:
        return x**2

    def parab_grad(x: FloatArray) -> FloatArray:
        return 2 * x

    for _ in range(10):
        beta_start = rng.uniform(-5, 5)
        assert (
            abs(smooth_descent(parab, parab_grad, np.array([beta_start])).beta) < 0.01
        )


def test_parabola_history() -> None:
    def objective(x: FloatArray) -> float:
        return float(np.sum(x**2))

    def gradient(x: FloatArray) -> FloatArray:
        return 2 * x

    beta_0 = np.array([4.0])

    results = smooth_descent(
        objective,
        gradient,
        beta_0,
        eta=0.25,
        epsi=1e-8,
        max_steps=3,
    )
    beta = results.beta
    history = results.objective_history

    expected_beta = np.array([0.5])
    expected_history = [16.0, 4.0, 1.0, 0.25]

    np.testing.assert_allclose(beta, expected_beta)
    np.testing.assert_allclose(history, expected_history)

    assert len(history) == 4


def test_np_scipy_alignment_simple() -> None:
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )

    beta_true = np.array([2.0, -1.0])

    beta_start = np.array([5.0, 5.0])

    y = X @ beta_true

    beta_np, *_ = np.linalg.lstsq(X, y, rcond=None)

    def objective(beta: FloatArray) -> float:
        return least_squares_loss(X, y, beta)

    def objective_gradient(beta: FloatArray) -> FloatArray:
        return least_squares_loss_gradient(X, y, beta)

    def objective_and_grad(beta: FloatArray) -> tuple[float, FloatArray]:
        return objective(beta), objective_gradient(beta)

    beta_return = smooth_descent(objective, objective_gradient, beta_start).beta

    beta_scipy = minimize(objective_and_grad, beta_start, jac=True).x

    assert np.allclose(beta_scipy, beta_return)

    assert np.allclose(beta_np, beta_return)


def test_np_scipy_alignment_enhanced() -> None:
    X_0 = rng.integers(5, size=(10, 4))
    noise = rng.standard_normal(size=(10, 4))
    X = X_0 + noise

    beta_true = rng.integers([1.0, 2.0, 3.0, 4.0])

    beta_start = np.array([7.0, 4.0, 2.0, 15.0])

    y = X @ beta_true

    beta_np, *_ = np.linalg.lstsq(X, y, rcond=None)

    def objective(beta: FloatArray) -> float:
        return least_squares_loss(X, y, beta)

    def objective_gradient(beta: FloatArray) -> FloatArray:
        return least_squares_loss_gradient(X, y, beta)

    def objective_and_grad(beta: FloatArray) -> tuple[float, FloatArray]:
        return objective(beta), objective_gradient(beta)

    results = smooth_descent(
        objective, objective_gradient, beta_start, eta=0.001, max_steps=10_000
    )
    beta_return = results.beta

    beta_scipy = minimize(objective_and_grad, beta_start, jac=True).x

    assert results.converged

    np.testing.assert_allclose(
        beta_return,
        beta_np,
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        beta_return,
        beta_scipy,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("lam", [3.0, 0.1, 0.01])
def test_ridge_analytical(lam: float) -> None:
    X_0 = rng.integers(5, size=(10, 4))
    noise = rng.standard_normal(size=(10, 4))
    X = X_0 + noise
    n, p = X.shape

    beta_true = rng.integers([1.0, 2.0, 3.0, 4.0])

    beta_start = np.array([7.0, 4.0, 2.0, 15.0])

    y = X @ beta_true

    def objective(beta: FloatArray) -> float:
        return least_squares_loss(X, y, beta) + lam * ridge_penalty(beta)

    def objective_gradient(beta: FloatArray) -> FloatArray:
        return least_squares_loss_gradient(X, y, beta) + lam * ridge_penalty_gradient(
            beta
        )

    L = np.linalg.norm(X, ord=2) ** 2 / n + 2 * lam
    eta = 1.0 / L

    results = smooth_descent(
        objective=objective,
        gradient=objective_gradient,
        beta_0=beta_start,
        eta=eta,
        max_steps=100_000,
    )

    assert results.converged

    beta_analytical = np.linalg.solve(X.T @ X + 2 * n * lam * np.eye(p), X.T @ y)

    np.testing.assert_allclose(results.beta, beta_analytical, atol=1e-6, rtol=1e-6)


def test_lasso_1dim_1() -> None:
    eta = 0.01

    def parab(x: FloatArray) -> FloatArray:
        return x**2

    def parab_grad(x: FloatArray) -> FloatArray:
        return 2 * x

    def non_smooth_penalty(x: FloatArray) -> FloatArray:
        return abs(x)

    def proximal_fun(z: FloatArray) -> FloatArray:
        return soft_threshold(z, eta)

    for _ in range(10):
        beta_start = np.array([rng.uniform(-5, 5)])

        result_optimize = proximal_gradient_descent(
            objective_diff=parab,
            objective_diff_grad=parab_grad,
            objective_non_diff=non_smooth_penalty,
            proximal_fun=proximal_fun,
            beta_0=beta_start,
            eta=eta,
        )
        np.testing.assert_allclose(result_optimize.beta, np.array([0.0]))


# def test_lasso_1dim_2() -> None:
#     eta = 0.001

#     def parab(x: FloatArray) -> FloatArray:
#         return (x - 3) ** 2

#     def parab_grad(x: FloatArray) -> FloatArray:
#         return 2 * (x - 3)

#     def non_smooth_penalty(x: FloatArray) -> FloatArray:
#         return abs(x)

#     def proximal_fun(z: FloatArray) -> FloatArray:
#         return soft_threshold(z, eta)

#     for _ in range(10):
#         beta_start = np.array([rng.uniform(-5, 5)])

#         result_optimize = proximal_gradient_descent(
#             objective_diff=parab,
#             objective_diff_grad=parab_grad,
#             objective_non_diff=non_smooth_penalty,
#             proximal_fun=proximal_fun,
#             beta_0=beta_start,
#             eta=eta,
#             max_steps=10_000,
#         )
        
#         np.testing.assert_allclose(result_optimize.beta, np.array([2.5]))
