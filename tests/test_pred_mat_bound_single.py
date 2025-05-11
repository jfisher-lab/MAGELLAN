import numpy as np
import pandas as pd
import pytest

from magellan.sci_opt import pred_mat_bound_single


@pytest.fixture
def sample_data():
    """Create sample input data"""
    X = pd.DataFrame([[1, 0], [0, 1]])  # 2x2 matrix
    W = np.array([[0.5, 0.3], [0.2, 0.7]])
    A = np.array([[1, -1], [-1, 1]])
    eye_correct = np.array([[1, 0], [0, 1]])
    zero_correct = np.array([[0, 0], [0, 0]])

    return X, W, A, eye_correct, zero_correct


def test_basic_prediction(sample_data):
    """Test basic prediction functionality"""
    X, W, A, eye_correct, zero_correct = sample_data
    result = pred_mat_bound_single(X, W, A, eye_correct, zero_correct, t=1)

    # Check output shape
    assert result.shape == X.shape
    # Check values are within bounds
    assert np.all(result >= 0)
    assert np.all(result <= 2)
    # Check values are integers (after rounding)
    assert np.all(np.mod(result, 1) == 0)


def test_value_bounds():
    """Test that values are correctly bounded between min_val and max_val"""
    X = pd.DataFrame([[3, -1], [-2, 4]])  # Values outside normal bounds
    W = np.array([[1, 1], [1, 1]])
    A = np.array([[1, 1], [1, 1]])
    eye_correct = np.array([[1, 0], [0, 1]])
    zero_correct = np.array([[0, 0], [0, 0]])

    result = pred_mat_bound_single(
        X, W, A, eye_correct, zero_correct, t=1, min_val=0, max_val=2
    )

    # Check bounds
    assert np.all(result >= 0)
    assert np.all(result <= 2)


def test_difference_limiting():
    """Test that differences between steps are limited to ±1"""
    X = pd.DataFrame([[0, 0], [0, 0]])
    W = np.array([[2, 2], [2, 2]])  # Large weights to force big changes
    A = np.array([[1, 1], [1, 1]])
    eye_correct = np.array([[1, 0], [0, 1]])
    zero_correct = np.array([[0, 0], [0, 0]])

    result = pred_mat_bound_single(X, W, A, eye_correct, zero_correct, t=2)

    # Initial values are 0, so after one step values should be at most 1
    intermediate = pred_mat_bound_single(X, W, A, eye_correct, zero_correct, t=1)
    assert np.all(np.abs(intermediate) <= 1)

    # After second step, values should be at most 2
    assert np.all(np.abs(result) <= 2)


def test_iterations():
    """Test that multiple iterations work correctly"""
    # Start with an initial state that will drive changes
    X = pd.DataFrame([[2, 0], [0, 0]])  # Max value in first position

    # Weights that will propagate changes
    W = np.array([[1.0, 0.8], [0.8, 1.0]])

    # Adjacency matrix with positive connections
    A = np.array([[1.0, 0.5], [0.5, 1.0]])

    # Identity matrices that won't interfere
    eye_correct = np.array([[1, 0], [0, 1]])
    zero_correct = np.array([[0, 0], [0, 0]])

    # Run for different numbers of iterations
    result1 = pred_mat_bound_single(X, W, A, eye_correct, zero_correct, t=1)
    result2 = pred_mat_bound_single(X, W, A, eye_correct, zero_correct, t=2)

    # Print intermediate results for debugging
    print(f"Initial X:\n{X}")
    print(f"Result after 1 iteration:\n{result1}")
    print(f"Result after 2 iterations:\n{result2}")

    # Check that we see iteration effects
    assert not np.array_equal(result1, result2), (
        "Results should differ between iterations"
    )

    # Additional checks
    assert np.any(result1 > 0), "First iteration should produce some non-zero values"
    assert np.any(result2 > 0), "Second iteration should produce some non-zero values"
    assert not np.array_equal(result1, X), (
        "First iteration should change initial values"
    )

    # Verify changes are bounded
    diff = np.abs(result2 - result1)
    assert np.all(diff <= 1), "Changes between iterations should be limited to 1"


def test_different_bounds():
    """Test with different min and max values"""
    X = pd.DataFrame([[1, 2], [3, 4]])
    W = np.array([[1, 1], [1, 1]])
    A = np.array([[1, 1], [1, 1]])
    eye_correct = np.array([[1, 0], [0, 1]])
    zero_correct = np.array([[0, 0], [0, 0]])

    result = pred_mat_bound_single(
        X, W, A, eye_correct, zero_correct, t=1, min_val=1, max_val=3
    )

    # Check bounds
    assert np.all(result >= 1)
    assert np.all(result <= 3)


def test_zero_weights():
    """Test behavior with zero weights"""
    X = pd.DataFrame([[1, 1], [1, 1]])
    W = np.zeros((2, 2))
    A = np.array([[1, 1], [1, 1]])
    eye_correct = np.array([[1, 0], [0, 1]])
    zero_correct = np.array([[0, 0], [0, 0]])

    result = pred_mat_bound_single(X, W, A, eye_correct, zero_correct)

    # With zero weights, output should be rounded to nearest integer within bounds
    assert np.all(result >= 0)
    assert np.all(result <= 2)


def test_input_validation():
    """Test that invalid inputs raise appropriate errors"""
    X = pd.DataFrame([[1]])  # 1x1 matrix
    W = np.array([[1, 1], [1, 1]])  # 2x2 matrix
    A = np.array([[1]])  # 1x1 matrix
    eye_correct = np.array([[1]])  # 1x1 matrix
    zero_correct = np.array([[0]])  # 1x1 matrix

    with pytest.raises(ValueError):
        pred_mat_bound_single(X, W, A, eye_correct, zero_correct)
