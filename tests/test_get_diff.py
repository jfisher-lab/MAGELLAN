import numpy as np
import pytest

from magellan.sci_opt import get_diff


def test_get_diff_basic():
    y = [1, 2, 3]
    pred = [0.5, 1.5, 2.5]
    expected = np.array([0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(get_diff(y, pred), expected)

def test_get_diff_different_types():
    # Test with lists, numpy arrays, and mixed inputs
    cases = [
        ([1, 2], [1, 1], [0, 1]),
        (np.array([1, 2]), [1, 1], [0, 1]),
        ([1, 2], np.array([1, 1]), [0, 1]),
        (np.array([1, 2]), np.array([1, 1]), [0, 1])
    ]
    
    for y, pred, expected in cases:
        np.testing.assert_array_almost_equal(get_diff(y, pred), expected)

def test_get_diff_empty():
    y, pred = [], []
    assert len(get_diff(y, pred)) == 0
    assert isinstance(get_diff(y, pred), np.ndarray)

def test_get_diff_different_dtypes():
    y = [1.0, 2.0]
    pred = [1, 1]  # integers
    expected = np.array([0.0, 1.0])
    np.testing.assert_array_almost_equal(get_diff(y, pred), expected)

def test_get_diff_shapes():
    with pytest.raises(ValueError):
        get_diff([1, 2], [1])  # Different lengths should raise ValueError