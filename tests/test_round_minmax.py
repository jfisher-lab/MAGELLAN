import numpy as np
import pytest

from magellan.sci_opt import round_minmax


def test_round_minmax_basic():
    """Test basic rounding functionality within bounds"""
    X = np.array([1.2, 1.7, 0.3, 1.5])
    expected = np.array([1.0, 2.0, 0.0, 2.0])
    result = round_minmax(X)
    np.testing.assert_array_equal(result, expected)

def test_round_minmax_custom_bounds():
    """Test with custom min and max bounds"""
    X = np.array([2.2, 3.7, 1.3, 4.5])
    expected = np.array([2.0, 4.0, 1.0, 4.0])
    result = round_minmax(X, min_val=1, max_val=4)
    np.testing.assert_array_equal(result, expected)

def test_round_minmax_exact_values():
    """Test with values exactly at min, max, and integers"""
    X = np.array([0.0, 1.0, 2.0, 1.5])
    expected = np.array([0.0, 1.0, 2.0, 2.0])
    result = round_minmax(X)
    np.testing.assert_array_equal(result, expected)

def test_round_minmax_out_of_bounds():
    """Test with values outside the bounds"""
    X = np.array([-1.2, 3.7, 0.3, -0.5])
    expected = np.array([0.0, 2.0, 0.0, 0.0])
    result = round_minmax(X)
    np.testing.assert_array_equal(result, expected)

def test_round_minmax_empty():
    """Test with empty array"""
    X = np.array([])
    result = round_minmax(X)
    assert len(result) == 0

def test_round_minmax_2d():
    """Test with 2D array"""
    X = np.array([[1.2, 1.7], [0.3, 2.5]])
    expected = np.array([[1.0, 2.0], [0.0, 2.0]])
    result = round_minmax(X)
    np.testing.assert_array_equal(result, expected)

def test_round_minmax_invalid_bounds():
    """Test with min_val greater than max_val"""
    X = np.array([1.2, 1.7])
    with pytest.raises(ValueError):
        round_minmax(X, min_val=3, max_val=1)

def test_round_minmax_dtype():
    """Test that function preserves integer dtype"""
    X = np.array([1, 2, 0, 3], dtype=np.int32)
    result = round_minmax(X)
    assert result.dtype == X.dtype