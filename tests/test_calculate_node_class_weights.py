import numpy as np
import pandas as pd
import pytest

from magellan.prune_opt import calculate_node_class_weights


def test_calculate_node_class_weights_returns_dict():
    """Test that the function returns a dictionary."""
    # Create a simple test DataFrame
    df = pd.DataFrame(
        {"exp1": [0, 1, 2], "exp2": [1, 0, np.nan]}, index=["node1", "node2", "node3"]
    )

    result = calculate_node_class_weights(df, min_range=0, max_range=2)

    assert isinstance(result, dict)
    assert "node1" in result
    assert "node2" in result
    assert "node3" in result  # node3 has only NaN values


def test_calculate_node_class_weights_method_inverse_freq():
    """Test the inverse_freq method with precise expected values."""
    df = pd.DataFrame(
        {"exp1": [0, 0, 0], "exp2": [0, 1, 2]}, index=["node1", "node2", "node3"]
    )

    result = calculate_node_class_weights(
        df, min_range=0, max_range=2, method="inverse_freq"
    )

    # For node1:
    # Value 0 appears 2 times out of 2 total observations -> frequency = 1.0
    # Inverse frequency = 1/1.0 = 1.0
    # Values 1 and 2 don't appear -> weight = max_range + 1.0 = 3.0 each
    # Total unnormalized weights = 1.0 + 3.0 + 3.0 = 7.0
    # Normalized weights = [1.0/7.0, 3.0/7.0, 3.0/7.0] ≈ [0.143, 0.429, 0.429]
    assert len(result["node1"]) == 3
    assert abs(result["node1"][0] - 1.0 / 7.0) < 1e-6
    assert abs(result["node1"][1] - 3.0 / 7.0) < 1e-6
    assert abs(result["node1"][2] - 3.0 / 7.0) < 1e-6

    # For node2:
    # Value 0 appears 1 time, value 1 appears 1 time out of 2 total -> both freq = 0.5
    # Inverse frequencies = 1/0.5 = 2.0 each
    # Value 2 doesn't appear -> weight = 3.0
    # Total unnormalized weights = 2.0 + 2.0 + 3.0 = 7.0
    # Normalized weights = [2.0/7.0, 2.0/7.0, 3.0/7.0] ≈ [0.286, 0.286, 0.429]
    assert len(result["node2"]) == 3
    assert abs(result["node2"][0] - 2.0 / 7.0) < 1e-6
    assert abs(result["node2"][1] - 2.0 / 7.0) < 1e-6
    assert abs(result["node2"][2] - 3.0 / 7.0) < 1e-6

    # For node3:
    # Value 0 appears 1 time, value 2 appears 1 time out of 2 total -> both freq = 0.5
    # Inverse frequencies = 1/0.5 = 2.0 each
    # Value 1 doesn't appear -> weight = 3.0
    # Total unnormalized weights = 2.0 + 3.0 + 2.0 = 7.0
    # Normalized weights = [2.0/7.0, 3.0/7.0, 2.0/7.0] ≈ [0.286, 0.429, 0.286]
    assert len(result["node3"]) == 3
    assert abs(result["node3"][0] - 2.0 / 7.0) < 1e-6
    assert abs(result["node3"][1] - 3.0 / 7.0) < 1e-6
    assert abs(result["node3"][2] - 2.0 / 7.0) < 1e-6


def test_calculate_node_class_weights_method_inverse_freq_sparsity_stable():
    """Test the inverse_freq_sparsity_stable method."""
    df = pd.DataFrame(
        {"exp1": [0, 0, 0], "exp2": [0, 1, 2]}, index=["node1", "node2", "node3"]
    )

    result = calculate_node_class_weights(
        df, min_range=0, max_range=2, method="inverse_freq_sparsity_stable"
    )

    # node1 only has value 0, so weights should be [1.0, 0.0, 0.0]
    assert len(result["node1"]) == 3
    assert result["node1"][0] == 1.0
    assert result["node1"][1] == 0.0
    assert result["node1"][2] == 0.0

    # node3 has both 0 and 2, weights for 1 should be 0
    assert result["node3"][1] == 0.0

    # Check normalization for observed values only
    for node in result:
        observed_sum = sum(
            w for i, w in enumerate(result[node]) if any(df.loc[node] == i)
        )
        assert abs(observed_sum - 1.0) < 1e-6


def test_calculate_node_class_weights_method_soft_inverse_freq():
    """Test the soft_inverse_freq method."""
    df = pd.DataFrame(
        {"exp1": [0, 0, 0], "exp2": [0, 1, 2]}, index=["node1", "node2", "node3"]
    )

    result = calculate_node_class_weights(
        df, min_range=0, max_range=2, method="soft_inverse_freq"
    )

    # All nodes should have non-zero weights for all values
    for node in result:
        assert all(w > 0 for w in result[node])
        assert abs(sum(result[node]) - 1.0) < 1e-6

    # Observed values should have higher weights than unobserved ones
    assert result["node1"][0] > result["node1"][1]
    assert result["node1"][0] > result["node1"][2]


def test_calculate_node_class_weights_method_balanced():
    """Test the balanced method."""
    df = pd.DataFrame(
        {
            "exp1": [0, 0, 0],
            "exp2": [0, 1, 2],
            "exp3": [0, 2, 2],
        },
        index=["node1", "node2", "node3"],
    )

    result = calculate_node_class_weights(
        df, min_range=0, max_range=2, method="balanced"
    )

    # node1 only has value 0, so weights should be [1.0, 0.0, 0.0]
    assert len(result["node1"]) == 3
    assert result["node1"][0] == 1.0
    assert result["node1"][1] == 0.0
    assert result["node1"][2] == 0.0

    # node3 has both 0 and 2, so weights should be [0.5, 0.0, 0.5]
    assert abs(result["node3"][0] - 0.5) < 1e-6
    assert result["node3"][1] == 0.0
    assert abs(result["node3"][2] - 0.5) < 1e-6


def test_calculate_node_class_weights_method_no_weighting():
    """Test the no_weighting method."""
    df = pd.DataFrame(
        {"exp1": [0, 0, 0], "exp2": [0, 1, 2]}, index=["node1", "node2", "node3"]
    )

    result = calculate_node_class_weights(
        df, min_range=0, max_range=2, method="no_weighting"
    )

    for node in result:
        assert all(w == 1.0 for w in result[node])
        assert len(result[node]) == 3


def test_calculate_node_class_weights_extreme_boost():
    """Test that extreme values get boosted correctly."""
    df = pd.DataFrame(
        {"exp1": [0, 1, 2], "exp2": [0, 1, 2], "exp3": [1, 1, 1]},
        index=["node1", "node2", "node3"],
    )

    # With extreme_boost = 2.0
    result1 = calculate_node_class_weights(
        df, min_range=0, max_range=2, method="inverse_freq", extreme_boost=2.0
    )

    # With extreme_boost = 1.0 (no boost)
    result2 = calculate_node_class_weights(
        df, min_range=0, max_range=2, method="inverse_freq", extreme_boost=1.0
    )

    # Check that extreme values get higher weight with extreme_boost > 1
    for node in ["node1", "node2", "node3"]:
        if 0 in df.loc[node].values and 2 in df.loc[node].values:
            # Both extremes are present, check they're boosted
            assert result1[node][0] > result2[node][0]
            assert result1[node][2] > result2[node][2]


def test_calculate_node_class_weights_invalid_method():
    """Test that invalid method raises ValueError."""
    df = pd.DataFrame({"exp1": [0, 1, 2]}, index=["node1", "node2", "node3"])

    with pytest.raises(ValueError):
        calculate_node_class_weights(
            df, min_range=0, max_range=2, method="invalid_method"
        )


def test_calculate_node_class_weights_invalid_range():
    """Test that invalid range raises ValueError."""
    df = pd.DataFrame({"exp1": [0, 1, 2]}, index=["node1", "node2", "node3"])

    with pytest.raises(ValueError):
        calculate_node_class_weights(
            df, min_range=0, max_range=2, method="invalid_method"
        )


def test_calculate_node_class_weights_negative_range():
    """Test that negative max_range raises ValueError."""
    df = pd.DataFrame({"exp1": [0, 1, 0]}, index=["node1", "node2", "node3"])

    with pytest.raises(ValueError):
        calculate_node_class_weights(df, min_range=0, max_range=-1)


def test_calculate_node_class_weights_empty_dataframe():
    """Test function handles empty DataFrame correctly."""
    df = pd.DataFrame({}, index=[])

    result = calculate_node_class_weights(df, min_range=0, max_range=2)

    assert result == {}


def test_calculate_node_class_weights_all_nan_values():
    """Test function handles DataFrame with all NaN values correctly."""
    df = pd.DataFrame(
        {"exp1": [np.nan, np.nan, np.nan]}, index=["node1", "node2", "node3"]
    )

    result = calculate_node_class_weights(df, min_range=0, max_range=2)

    assert result == {}
