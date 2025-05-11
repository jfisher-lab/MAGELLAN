from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from magellan.sci_opt import pred_all_bound_single


@pytest.fixture
def sample_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame], nx.DiGraph, dict
]:
    """Create sample test data"""
    # Create sample X DataFrame (3 nodes, 2 experiments)
    X = pd.DataFrame(
        {"exp1": [1.0, 0.0, 0.0], "exp2": [0.0, 1.0, 0.0]},
        index=["node1", "node2", "node3"],
    )

    # Create sample y DataFrame (expected results)
    y = pd.DataFrame(
        {"exp1": [1.0, 0.5, -1.0], "exp2": [-1.0, 1.0, 0.5]},
        index=["node1", "node2", "node3"],
    )

    # Create sample weight matrix
    W = pd.DataFrame(
        [[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0]],
        index=["node1", "node2", "node3"],
        columns=["node1", "node2", "node3"],
    )

    # Create sample adjacency matrices (one DataFrame per experiment)
    A = [
        pd.DataFrame(
            [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
            index=["node1", "node2", "node3"],
            columns=["node1", "node2", "node3"],
        ),
        pd.DataFrame(
            [[1, 0, 1], [1, 1, 0], [0, 1, 1]],
            index=["node1", "node2", "node3"],
            columns=["node1", "node2", "node3"],
        ),
    ]

    # Create sample perturbation dictionary
    pert_dic = {
        "exp1": {"pert": {"node1": 1.0}, "exp": {"node2": 0.5, "node3": 0.0}},
        "exp2": {"pert": {"node2": 1.0}, "exp": {"node1": 0.0, "node3": 0.5}},
    }
    # Create networkx DiGraph from weight matrix W

    G = nx.DiGraph()
    nodes = ["node1", "node2", "node3"]
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if W.loc[source, target] > 0:
                G.add_edge(source, target, weight=W.loc[source, target])

    return X, y, W, A, G, pert_dic


@pytest.fixture
def sample_data_min_val_one() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame], nx.DiGraph, dict
]:
    """Create sample test data"""
    # Create sample X DataFrame (3 nodes, 2 experiments)
    X = pd.DataFrame(
        {"exp1": [2.0, 1.0, 1.0], "exp2": [1.0, 2.0, 1.0]},
        index=["node1", "node2", "node3"],
    )

    # Create sample y DataFrame (expected results)
    y = pd.DataFrame(
        {"exp1": [1.0, 1.5, 1.0], "exp2": [1.0, 1.0, 1.5]},
        index=["node1", "node2", "node3"],
    )

    # Create sample weight matrix
    W = pd.DataFrame(
        [[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0]],
        index=["node1", "node2", "node3"],
        columns=["node1", "node2", "node3"],
    )

    # Create sample adjacency matrices (one DataFrame per experiment)
    A = [
        pd.DataFrame(
            [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
            index=["node1", "node2", "node3"],
            columns=["node1", "node2", "node3"],
        ),
        pd.DataFrame(
            [[1, 0, 1], [1, 1, 0], [0, 1, 1]],
            index=["node1", "node2", "node3"],
            columns=["node1", "node2", "node3"],
        ),
    ]

    # Create sample perturbation dictionary
    pert_dic = {
        "exp1": {"pert": {"node1": 1.0}, "exp": {"node2": 1.5, "node3": 1.0}},
        "exp2": {"pert": {"node2": 1.0}, "exp": {"node1": 1.0, "node3": 1.5}},
    }
    # Create networkx DiGraph from weight matrix W

    G = nx.DiGraph()
    nodes = ["node1", "node2", "node3"]
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if W.loc[source, target] > 0:
                G.add_edge(source, target, weight=W.loc[source, target])
    return X, y, W, A, G, pert_dic


def test_basic_prediction(sample_data):
    """Test basic prediction functionality"""
    X, y, W, A, G, pert_dic = sample_data
    time_step = 3

    result = pred_all_bound_single(X, y, W, A, G, pert_dic, time_step=time_step)

    assert isinstance(result, np.ndarray)
    assert result.shape == y.shape
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_extract_exp_false(sample_data):
    """Test when extract_exp is False - should return predictions for all nodes"""
    X, y, W, A, G, pert_dic = sample_data
    time_step = 3

    result = pred_all_bound_single(
        X, y, W, A, G, pert_dic, time_step=time_step, extract_exp=False
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == y.shape

    # Check that perturbed nodes maintain their perturbation values
    for exp_idx, (exp_name, exp_data) in enumerate(pert_dic.items()):
        for node, value in exp_data["pert"].items():
            node_idx = list(X.index).index(node)
            assert result[node_idx, exp_idx] == value

    # Check that values are within bounds
    assert np.all(result >= 0.0)
    assert np.all(result <= 2.0)


def test_value_bounds(sample_data_min_val_one):
    """Test that predictions respect min_val and max_val bounds for predicted nodes only"""
    X, y, W, A, G, pert_dic = sample_data_min_val_one
    time_step = 3
    min_val = 1
    max_val = 3

    result = pred_all_bound_single(
        X, y, W, A, G, pert_dic, time_step=time_step, min_val=min_val, max_val=max_val
    )

    # For each experiment, check bounds only for nodes in that experiment's 'exp' dictionary
    for exp_idx, (exp_name, exp_data) in enumerate(pert_dic.items()):
        predicted_nodes = exp_data["exp"].keys()
        for node in predicted_nodes:
            node_idx = list(X.index).index(node)
            node_value = result[node_idx, exp_idx]
            assert min_val <= node_value <= max_val, (
                f"Node {node} in experiment {exp_name} has value {node_value} outside bounds [{min_val}, {max_val}]"
            )

        # For non-predicted nodes, verify they are zero
        non_predicted_nodes = set(X.index) - set(predicted_nodes)
        for node in non_predicted_nodes:
            node_idx = list(X.index).index(node)
            assert result[node_idx, exp_idx] == 0, (
                f"Non-predicted node {node} in experiment {exp_name} has non-zero value {result[node_idx, exp_idx]}"
            )


def test_weight_matrix_types(sample_data):
    """Test both numpy array and pandas DataFrame weight matrices"""
    X, y, W, A, G, pert_dic = sample_data
    time_step = 3

    # Test with numpy array
    result_np = pred_all_bound_single(X, y, W, A, G, pert_dic, time_step=time_step)

    # Test with pandas DataFrame
    W_df = pd.DataFrame(W, index=X.index, columns=X.index)
    result_pd = pred_all_bound_single(X, y, W_df, A, G, pert_dic, time_step=time_step)

    np.testing.assert_array_almost_equal(result_np, result_pd)


def test_no_dummy_nodes(sample_data):
    """Test that the function works correctly without dummy nodes"""
    X, y, W, A, G, pert_dic = sample_data
    time_step = 3

    result = pred_all_bound_single(X, y, W, A, G, pert_dic, time_step=time_step)

    assert isinstance(result, np.ndarray)
    assert result.shape == y.shape
    assert not np.any(np.isnan(result))


def test_with_dummy_nodes(sample_data):
    """Test that the function works correctly with dummy nodes"""
    X, y, W, A, G, pert_dic = sample_data
    time_step = 3

    # Add a dummy node
    X_with_dummy = X.copy()
    X_with_dummy.loc["dummy_node4"] = 0.0
    y_with_dummy = y.copy()
    y_with_dummy.loc["dummy_node4"] = 0.0

    # Create corresponding node for dummy
    X_with_dummy.loc["node4"] = 0.0
    y_with_dummy.loc["node4"] = 0.0

    # Extend matrices
    W_with_dummy = pd.DataFrame(
        np.pad(W, ((0, 2), (0, 2)), mode="constant"),
        index=X_with_dummy.index,
        columns=X_with_dummy.index,
    )

    # Update adjacency matrices with new dimensions
    A_with_dummy = [
        pd.DataFrame(
            np.pad(a, ((0, 2), (0, 2)), mode="constant"),
            index=X_with_dummy.index,
            columns=X_with_dummy.index,
        )
        for a in A
    ]

    # Add dummy nodes to the graph
    G_with_dummy = G.copy()
    G_with_dummy.add_node("dummy_node4")
    G_with_dummy.add_node("node4")

    # Update pert_dic to include dummy node
    pert_dic_with_dummy = pert_dic.copy()
    pert_dic_with_dummy["exp1"]["pert"]["dummy_node4"] = 1.0

    result = pred_all_bound_single(
        X_with_dummy,
        y_with_dummy,
        W_with_dummy,
        A_with_dummy,
        G_with_dummy,  # Use the updated graph
        pert_dic_with_dummy,
        time_step=time_step,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == y_with_dummy.shape
    assert not np.any(np.isnan(result))


def test_invalid_inputs():
    """Test error handling for invalid inputs"""
    with pytest.raises(ValueError):
        # Empty DataFrames
        pred_all_bound_single(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            [pd.DataFrame(), pd.DataFrame()],
            nx.DiGraph(),
            OrderedDict(),
            time_step=3,
        )

    with pytest.raises(ValueError):
        # Mismatched dimensions
        X = pd.DataFrame({"exp1": [1.0]})
        y = pd.DataFrame({"exp1": [1.0, 2.0]})
        W = pd.DataFrame([[1.0]])
        A = [pd.DataFrame([[[1.0]]])]
        G = nx.DiGraph()
        pert_dic = OrderedDict({"exp1": {"pert": {}, "exp": {}}})

        pred_all_bound_single(X, y, W, A, G, pert_dic, time_step=3)


def test_time_step_zero(sample_data):
    """Test behavior with time_step=0"""
    X, y, W, A, G, pert_dic = sample_data

    with pytest.raises(ValueError):
        pred_all_bound_single(X, y, W, A, G, pert_dic, time_step=0)


def test_empty_perturbation_dict(sample_data):
    """Test behavior with empty perturbation dictionary"""
    X, y, W, A, G, _ = sample_data
    empty_pert_dic = OrderedDict()

    with pytest.raises(ValueError):
        pred_all_bound_single(X, y, W, A, G, empty_pert_dic, time_step=3)


def test_result_reproducibility(sample_data):
    """Test that results are reproducible"""
    X, y, W, A, G, pert_dic = sample_data
    time_step = 3

    result1 = pred_all_bound_single(X, y, W, A, G, pert_dic, time_step=time_step)

    result2 = pred_all_bound_single(X, y, W, A, G, pert_dic, time_step=time_step)

    np.testing.assert_array_equal(result1, result2)
