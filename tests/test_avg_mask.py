import numpy as np
import pandas as pd
import pytest

from magellan.sci_opt import update_function_mask


@pytest.fixture
def sample_adjacency_matrix():
    """Create a sample adjacency matrix for testing."""
    return np.array(
        [
            [0, 1, -1, 0],
            [-1, 0, 1, 1],
            [1, -1, 0, 0],
            [0, 0, 1, -1],
        ]
    )


def test_avg_mask_activator(sample_adjacency_matrix):
    """Test averaging mask for activator edges (val=1)."""
    result = update_function_mask(pd.DataFrame(sample_adjacency_matrix), 1)

    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0.5, 0.5],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_avg_mask_inhibitor(sample_adjacency_matrix):
    """Test averaging mask for inhibitor edges (val=-1)."""
    result = update_function_mask(pd.DataFrame(sample_adjacency_matrix), -1)

    expected = np.array(
        [
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, -1],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_avg_mask_empty():
    """Test averaging mask with empty matrix."""
    empty_matrix = np.zeros((3, 3))
    result = update_function_mask(pd.DataFrame(empty_matrix), 1)
    expected = np.zeros((3, 3))
    np.testing.assert_array_equal(result, expected)


def test_avg_mask_single_node():
    """Test averaging mask with single node."""
    single_node = np.array([[1]])
    result = update_function_mask(pd.DataFrame(single_node), 1)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)


def test_avg_mask_all_connections():
    """Test averaging mask when all nodes are connected."""
    all_connected = np.ones((3, 3))
    result = update_function_mask(pd.DataFrame(all_connected), 1)
    expected = np.ones((3, 3)) / 3
    np.testing.assert_array_almost_equal(result, expected)


def test_sum_mask_activator(sample_adjacency_matrix):
    """Test sum mask for activator edges (val=1)."""
    result = update_function_mask(
        pd.DataFrame(sample_adjacency_matrix), 1, method="sum"
    )

    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_sum_mask_inhibitor(sample_adjacency_matrix):
    """Test sum mask for inhibitor edges (val=-1)."""
    result = update_function_mask(
        pd.DataFrame(sample_adjacency_matrix), -1, method="sum"
    )

    expected = np.array(
        [
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, -1],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_sum_mask_empty():
    """Test sum mask with empty matrix."""
    empty_matrix = np.zeros((3, 3))
    result = update_function_mask(pd.DataFrame(empty_matrix), 1, method="sum")
    expected = np.zeros((3, 3))
    np.testing.assert_array_equal(result, expected)


def test_sum_mask_single_node():
    """Test sum mask with single node."""
    single_node = np.array([[1]])
    result = update_function_mask(pd.DataFrame(single_node), 1, method="sum")
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)


def test_sum_mask_all_connections():
    """Test sum mask when all nodes are connected."""
    all_connected = np.ones((3, 3))
    result = update_function_mask(pd.DataFrame(all_connected), 1, method="sum")
    expected = np.ones((3, 3))
    np.testing.assert_array_almost_equal(result, expected)


def test_invalid_method():
    """Test that invalid method raises ValueError."""
    matrix = np.ones((3, 3))
    with pytest.raises(
        ValueError,
        match="Invalid method: invalid, currently supported are 'avg' and 'sum'",
    ):
        update_function_mask(pd.DataFrame(matrix), 1, method="invalid")
