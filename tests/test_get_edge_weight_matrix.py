import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from magellan.gnn_model import Net, get_edge_weight_matrix


@pytest.fixture
def simple_network():
    """Create a simple test network with 3 nodes."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    # node_list = ["A", "B", "C"]
    edge_idx = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)

    # Create a model with predetermined edge weights
    edge_weight = torch.tensor([0.5, 0.7, 0.3], dtype=torch.float32)
    model = Net(edge_weight=edge_weight, min_val=0, max_val=2)

    return model, edge_idx, G


@pytest.fixture
def network_with_self_loops():
    """Create a test network that includes self-loops."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "A")
    G.add_edge("B", "B")
    # node_list = ["A", "B"]
    # Include self-loops in edge index
    edge_idx = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.int64)

    edge_weight = torch.tensor([0.1, 0.5, 0.3, 0.2], dtype=torch.float32)
    model = Net(edge_weight=edge_weight, min_val=0, max_val=2)

    return model, edge_idx, G


@pytest.fixture
def empty_network():
    """Create an empty network."""
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    # node_list = ["A", "B"]
    # A_mult = pd.DataFrame(
    #     [[0, 0],
    #      [0, 0]],
    #     index=node_list,
    #     columns=node_list
    # )
    edge_idx = torch.tensor([[], []], dtype=torch.int64)

    model = Net(edge_weight=torch.tensor([]), min_val=0, max_val=2)

    return model, edge_idx, G


@pytest.fixture
def network_with_dummy():
    """Create a test network that includes dummy nodes and self-loops."""
    # node_list = ["A", "dummy_1", "B"]
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("A", "A"),
            ("A", "dummy_1"),
            ("dummy_1", "dummy_1"),
            ("dummy_1", "B"),
            ("B", "B"),
            ("A", "B"),
        ],
    )
    # Include dummy nodes and self-loops in edge index
    edge_idx = torch.tensor([[0, 0, 1, 1, 2, 0], [0, 1, 1, 2, 2, 2]], dtype=torch.int64)

    edge_weight = torch.tensor([0.1, 0.5, 0.3, 0.4, 0.2, 0.6], dtype=torch.float32)
    model = Net(edge_weight=edge_weight, min_val=0, max_val=2)

    return model, edge_idx, G


def test_basic_functionality(simple_network):
    """Test basic functionality with a simple network."""
    model, edge_idx, G = simple_network

    result = get_edge_weight_matrix(model, edge_idx, G)

    # Check that result is a DataFrame with correct shape and indices
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert list(result.index) == list(G.nodes())
    assert list(result.columns) == list(G.nodes())

    # Check specific weight values
    assert result.loc["B", "A"] == pytest.approx(0.5)
    assert result.loc["C", "B"] == pytest.approx(0.7)
    assert result.loc["A", "C"] == pytest.approx(0.3)


def test_self_loops(network_with_self_loops):
    """Test that self-loops are correctly handled (ignored)."""
    model, edge_idx, node_list = network_with_self_loops

    result = get_edge_weight_matrix(
        model, edge_idx, node_list, remove_dummy_and_self_loops=True
    )

    # Self-loops should be zero
    assert result.loc["A", "A"] == 0
    assert result.loc["B", "B"] == 0

    # Check non-self-loop edges
    assert result.loc["B", "A"] == pytest.approx(0.5)
    assert result.loc["A", "B"] == pytest.approx(0.3)


def test_self_loops_not_removed(network_with_self_loops):
    """Test that self-loops are correctly handled (ignored)."""
    model, edge_idx, node_list = network_with_self_loops

    result = get_edge_weight_matrix(
        model, edge_idx, node_list, remove_dummy_and_self_loops=False
    )

    # Self-loops should be zero
    assert result.loc["A", "A"] == pytest.approx(0.1)
    assert result.loc["B", "B"] == pytest.approx(0.2)

    # Check non-self-loop edges
    assert result.loc["B", "A"] == pytest.approx(0.5)
    assert result.loc["A", "B"] == pytest.approx(0.3)


def test_empty_network(empty_network):
    """Test behavior with an empty network."""
    model, edge_idx, G = empty_network

    result = get_edge_weight_matrix(model, edge_idx, G)

    # Should return zero matrix with correct shape and indices
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == list(G.nodes())
    assert list(result.columns) == list(G.nodes())
    assert (result == 0).all().all()


def test_edge_index_clone(simple_network):
    """Test that original edge_idx is not modified."""
    model, edge_idx, node_list = simple_network
    original_edge_idx = edge_idx.clone()

    _ = get_edge_weight_matrix(model, edge_idx, node_list)

    assert torch.equal(edge_idx, original_edge_idx)


def test_weight_matrix_symmetry(simple_network):
    """Test that weight matrix properly reflects edge directionality."""
    model, edge_idx, node_list = simple_network

    result = get_edge_weight_matrix(model, edge_idx, node_list)

    # Check that zeros appear in expected positions
    assert result.loc["A", "B"] == 0
    assert result.loc["B", "C"] == 0
    assert result.loc["C", "A"] == 0


def test_datatype_consistency(simple_network):
    """Test that output maintains correct datatypes."""
    model, edge_idx, node_list = simple_network

    result = get_edge_weight_matrix(model, edge_idx, node_list)

    # Check that all columns have float64 dtype
    assert all(dtype == np.dtype("float64") for dtype in result.dtypes)

    # Additional check to verify actual values are float64
    assert result.values.dtype == np.dtype("float64")


def test_dummy_nodes_and_self_loops(network_with_dummy):
    """Test that dummy nodes and self-loops are correctly handled when specified."""
    model, edge_idx, G = network_with_dummy

    # Test with remove_dummy_and_self_loops=True
    result_cleaned = get_edge_weight_matrix(
        model, edge_idx, G, remove_dummy_and_self_loops=True
    )

    # Self-loops should be zero
    assert result_cleaned.loc["A", "A"] == 0
    assert result_cleaned.loc["dummy_1", "dummy_1"] == 0
    assert result_cleaned.loc["B", "B"] == 0

    # Dummy node connections should be zero
    assert (result_cleaned["dummy_1"] == 0).all()
    assert (result_cleaned.loc["dummy_1"] == 0).all()
    # Non-dummy, non-self-loop edges should remain
    assert result_cleaned.loc["B", "A"] == pytest.approx(0.5)

    # Test with remove_dummy_and_self_loops=False (default)
    result_original = get_edge_weight_matrix(model, edge_idx, G)

    # Self-loops should remain
    assert result_original.loc["A", "A"] == pytest.approx(0.1)
    assert result_original.loc["dummy_1", "dummy_1"] == pytest.approx(0.2)
    assert result_original.loc["B", "B"] == pytest.approx(0.3)

    # Dummy node connections should remain
    assert result_original.loc["dummy_1", "A"] == pytest.approx(0.6)
    assert result_original.loc["B", "dummy_1"] == pytest.approx(0.0)
