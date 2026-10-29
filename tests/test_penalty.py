from semester_project.penalty import soft_threshold, ridge_penalty, ridge_penalty_gradient
import numpy as np 


def test_ridge_penalty():
    beta = np.array([1.0, -2.0, 3.0])
    assert ridge_penalty(beta) == 14.0


def test_ridge_penalty_gradient():
    beta = np.array([1.0, -2.0, 3.0])
    expected = np.array([2.0, -4.0, 6.0])

    np.testing.assert_allclose(
        ridge_penalty_gradient(beta),
        expected,
    )


def test_soft_threshold() -> None:
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
    result = soft_threshold(z, threshold=1.0)
    np.testing.assert_allclose(result, expected)
