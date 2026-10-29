from semester_project.loss import least_squares_loss, least_squares_loss_gradient
import numpy as np
rng = np.random.default_rng(42)


def test_loss_perfect_fit() -> None:
    for _ in range(10):
        X = rng.random((3, 2))
        beta = rng.random((2, ))
        y = X @ beta

        assert np.isclose(least_squares_loss(X, y, beta), 0.0)


def test_loss_known_value() -> None:
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
    ])
    beta = np.array([2.0, 3.0])
    y = np.array([4.0, 10.0])

    assert np.isclose(least_squares_loss(X, y, beta), 1.25)


def test_gradient_numerical() -> None: 
    n = rng.integers(2, 100)
    m = rng.integers(2, 100)
    X = rng.random((n, m))
    beta = rng.random((m, ))
    y = rng.random((n, ))

    prog_grad = least_squares_loss_gradient(X, y, beta)
    for j in range(m):
        vector_j = np.eye(m)[j]
        h = 1e-4
        prog_grad_j = prog_grad[j]
        comp_grad_j = (least_squares_loss(X, y, beta + vector_j * h) - least_squares_loss(X, y, beta - vector_j * h)) / (2 * h)
        assert abs(prog_grad_j - comp_grad_j) < 1e-5
