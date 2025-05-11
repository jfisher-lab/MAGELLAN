import numpy as np
import pytest

from magellan.sci_opt import get_loss


def test_mse_no_reg():
    """Test MSE loss without regularization"""
    diff = np.array([1, -2, 3])
    W = np.array([0.1, 0.2, -0.3])
    result = get_loss(diff, W, lambd=0.1, err="mse", reg=None)
    expected = np.sum(np.power(diff, 2))
    assert np.isclose(result, expected)

def test_mae_no_reg():
    """Test MAE loss without regularization"""
    diff = np.array([1, -2, 3])
    W = np.array([0.1, 0.2, -0.3])
    result = get_loss(diff, W, lambd=0.1, err="mae", reg=None)
    expected = np.sum(np.abs(diff))
    assert np.isclose(result, expected)

def test_mse_with_l1():
    """Test MSE loss with L1 regularization"""
    diff = np.array([1, -2, 3])
    W = np.array([0.1, 0.2, -0.3])
    lambd = 0.1
    result = get_loss(diff, W, lambd, err="mse", reg=1)
    expected = np.sum(np.power(diff, 2)) + lambd * np.linalg.norm(W, 1)
    assert np.isclose(result, expected)

def test_mse_with_l2():
    """Test MSE loss with L2 regularization"""
    diff = np.array([1, -2, 3])
    W = np.array([0.1, 0.2, -0.3])
    lambd = 0.1
    result = get_loss(diff, W, lambd, err="mse", reg=2)
    expected = np.sum(np.power(diff, 2)) + lambd * np.linalg.norm(W, 2)
    assert np.isclose(result, expected)

def test_invalid_error():
    """Test that invalid error type raises KeyError"""
    diff = np.array([1, -2, 3])
    W = np.array([0.1, 0.2, -0.3])
    with pytest.raises(KeyError, match="err must be mse or mae"):
        get_loss(diff, W, lambd=0.1, err="invalid")

def test_zero_diff():
    """Test with zero differences"""
    diff = np.zeros(3)
    W = np.array([0.1, 0.2, -0.3])
    result = get_loss(diff, W, lambd=0.1, err="mse")
    expected = 0.1 * np.linalg.norm(W, 1)
    assert np.isclose(result, expected)

def test_zero_weights():
    """Test with zero weights"""
    diff = np.array([1, -2, 3])
    W = np.zeros(3)
    result = get_loss(diff, W, lambd=0.1, err="mse")
    expected = np.sum(np.power(diff, 2))
    assert np.isclose(result, expected)

def test_empty_arrays():
    """Test with empty arrays"""
    diff = np.array([])
    W = np.array([])
    result = get_loss(diff, W, lambd=0.1, err="mse")
    assert result == 0

def test_different_shapes():
    """Test arrays of different shapes"""
    diff = np.array([1, -2, 3])
    W = np.array([0.1, 0.2])  # Different shape from diff
    result = get_loss(diff, W, lambd=0.1, err="mse")
    # The function should still work as the arrays are used independently
    expected = np.sum(np.power(diff, 2)) + 0.1 * np.linalg.norm(W, 1)
    assert np.isclose(result, expected)