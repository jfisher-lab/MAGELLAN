from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from magellan.prune import (
    get_adjacency_matrix_mult,
    get_experiment_array,
    get_inh,
    get_sorted_node_list,
    graph_node_list_checks,
)


def test_get_adjacency_matrix_mult():
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")  # +1 edge
    G.add_edge("B", "C", sign="Inhibitor")  # -1 edge
    G.add_edge("C", "A", sign="Activator")  # +1 edge

    node_list = ["A", "B", "C"]

    # Get the adjacency matrix
    result = get_adjacency_matrix_mult(G)

    # Expected matrix:
    expected = pd.DataFrame(
        [
            [0.0, 0.0, 1.0],  # C -> A: Activator (+1)
            [1.0, 0.0, 0.0],  # A -> B: Activator (+1)
            [0.0, -1.0, 0.0],  # B -> C: Inhibitor (-1)
        ],
        index=node_list,
        columns=node_list,
    )

    # Check that result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check dimensions
    assert result.shape == (3, 3)

    # Check index and columns
    assert list(result.index) == node_list
    assert list(result.columns) == node_list

    # Check values
    pd.testing.assert_frame_equal(result, expected)


#############################################################
# Test for graph_node_list_checks
#############################################################


def test_graph_node_list_checks_valid():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")

    # Should not raise an exception
    graph_node_list_checks(G)


def test_graph_node_list_checks_empty_graph():
    G = nx.DiGraph()

    with pytest.raises(ValueError, match="The graph G is empty."):
        graph_node_list_checks(G)


#############################################################
# Test for get_inh
#############################################################


@pytest.fixture
def sample_graph_with_inhibitors():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")
    G.add_edge("D", "C", sign="Inhibitor")
    G.add_edge("C", "A", sign="Activator")

    return G, ["A", "B", "C", "D"]


def test_get_inh(sample_graph_with_inhibitors, monkeypatch):
    G, _ = sample_graph_with_inhibitors
    node_list = get_sorted_node_list(G)

    # Mock base_adj to return a predetermined adjacency matrix
    def mock_base_adj(G):
        return pd.DataFrame(
            [
                [0, 0, 1, 0],  # A has one activator parent: C
                [1, 0, 0, 0],  # B has one activator parent: A
                [0, -1, 0, -1],  # C has two inhibitor parents: B and D
                [0, 0, 0, 0],  # D has no parents
            ],
            index=node_list,
            columns=node_list,
        )

    monkeypatch.setattr("magellan.prune.base_adj", mock_base_adj)

    # C should be identified as an inhibitor-only node
    result = get_inh(G)
    assert result == ["C"]


def test_get_inh_no_inhibitors(monkeypatch):
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Activator")
    node_list = get_sorted_node_list(G)

    # Mock base_adj
    def mock_base_adj(G):
        return pd.DataFrame(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            index=node_list,
            columns=node_list,
        )

    monkeypatch.setattr("magellan.prune.base_adj", mock_base_adj)

    result = get_inh(G)
    assert result == []


def test_get_inh_mixed_parents(monkeypatch):
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("C", "B", sign="Inhibitor")
    node_list = get_sorted_node_list(G)

    # Mock base_adj
    def mock_base_adj(G):
        return pd.DataFrame(
            [
                [0, 0, 0],
                [1, 0, -1],  # B has mixed parents (activator and inhibitor)
                [0, 0, 0],
            ],
            index=node_list,
            columns=node_list,
        )

    monkeypatch.setattr("magellan.prune.base_adj", mock_base_adj)

    result = get_inh(G)
    assert result == []  # B has mixed parents, so not inhibitor-only


#############################################################
# Test for get_adjacency_matrix_mult
#############################################################


def test_get_adjacency_matrix_mult_basic(monkeypatch):
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")  # +1 edge
    G.add_edge("B", "C", sign="Inhibitor")  # -1 edge
    G.add_edge("C", "A", sign="Activator")  # +1 edge

    node_list = get_sorted_node_list(G)

    # Mock the functions we need
    def mock_graph_node_list_checks(G):
        return None

    def mock_base_adj(G):
        return pd.DataFrame(
            [
                [0, 0, 1],  # C -> A
                [1, 0, 0],  # A -> B
                [0, -1, 0],  # B -> C
            ],
            index=node_list,
            columns=node_list,
        )

    def mock_update_function_mask(A, val, method):
        if val == 1:  # Activator
            return pd.DataFrame(
                [
                    [0, 0, 1],  # Only C -> A is active
                    [1, 0, 0],  # Only A -> B is active
                    [0, 0, 0],  # No activator for C
                ],
                index=node_list,
                columns=node_list,
            )
        else:  # Inhibitor
            return pd.DataFrame(
                [
                    [0, 0, 0],  # No inhibitor for A
                    [0, 0, 0],  # No inhibitor for B
                    [0, -1, 0],  # Only B -> C is inhibitor
                ],
                index=node_list,
                columns=node_list,
            )

    monkeypatch.setattr(
        "magellan.prune.graph_node_list_checks", mock_graph_node_list_checks
    )
    monkeypatch.setattr("magellan.prune.base_adj", mock_base_adj)
    monkeypatch.setattr(
        "magellan.prune.update_function_mask", mock_update_function_mask
    )

    # Call the function
    result = get_adjacency_matrix_mult(G)

    # Expected result based on the mocks
    expected = pd.DataFrame(
        [
            [0, 0, 1],  # C -> A: Activator (+1)
            [1, 0, 0],  # A -> B: Activator (+1)
            [0, -1, 0],  # B -> C: Inhibitor (-1)
        ],
        index=node_list,
        columns=node_list,
    )

    # Check that result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check dimensions
    assert result.shape == (3, 3)

    # Check index and columns
    assert list(result.index) == node_list
    assert list(result.columns) == node_list

    # Check values
    pd.testing.assert_frame_equal(result, expected)


def test_get_adjacency_matrix_mult_method_sum(monkeypatch):
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    node_list = get_sorted_node_list(G)

    # Mock functions
    def mock_graph_node_list_checks(G):
        return None

    def mock_base_adj(G):
        return pd.DataFrame([[0, 0], [1, 0]], index=node_list, columns=node_list)

    def mock_update_function_mask(A, val, method):
        if method == "sum":
            if val == 1:
                return pd.DataFrame(
                    [[0, 0], [1, 0]], index=node_list, columns=node_list
                )
            else:
                return pd.DataFrame(
                    [[0, 0], [0, 0]], index=node_list, columns=node_list
                )
        else:
            raise ValueError(f"Unexpected method: {method}")

    monkeypatch.setattr(
        "magellan.prune.graph_node_list_checks", mock_graph_node_list_checks
    )
    monkeypatch.setattr("magellan.prune.base_adj", mock_base_adj)
    monkeypatch.setattr(
        "magellan.prune.update_function_mask", mock_update_function_mask
    )

    result = get_adjacency_matrix_mult(G, method="sum")

    expected = pd.DataFrame([[0, 0], [1, 0]], index=node_list, columns=node_list)

    pd.testing.assert_frame_equal(result, expected)


def test_get_adjacency_matrix_mult_error_propagation(monkeypatch):
    G = nx.DiGraph()
    # node_list = ["A"]  # Doesn't match G nodes

    # Mock graph_node_list_checks to raise the expected error
    def mock_graph_node_list_checks(G):
        raise ValueError("The node_list does not match the nodes in the graph G.")

    monkeypatch.setattr(
        "magellan.prune.graph_node_list_checks", mock_graph_node_list_checks
    )

    with pytest.raises(
        ValueError, match="The node_list does not match the nodes in the graph G."
    ):
        get_adjacency_matrix_mult(G)


#############################################################
# Test for get_experiment_array
#############################################################


def test_get_experiment_array_basic():
    # Create a simple test setup
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")

    node_list = get_sorted_node_list(G)
    inh = ["C"]  # C is inhibitor-only

    # Simple perturbation dictionary
    pert_dic_all = OrderedDict({"exp1": {"pert": {"A": 1}}, "exp2": {"pert": {"B": 1}}})

    # Create a base adjacency matrix
    A_mult = pd.DataFrame(
        [
            [0, 0, 0],
            [1, 0, 0],  # A -> B
            [0, -1, 0],  # B -> C
        ],
        index=node_list,
        columns=node_list,
    )

    # Call the function
    result = get_experiment_array(A_mult, inh, pert_dic_all, G)

    # We should get a 3D array with shape (2, 4, 4) for 2 experiments, 4 nodes (A, B, C, dummy_C)
    assert isinstance(result, list)
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.DataFrame)
    assert result[0].shape == (4, 4)
    assert result[1].shape == (4, 4)

    # For exp1 (A perturbed):
    # - A gets self-loop (had no parents to lose)
    # - B keeps A as parent (A being perturbed doesn't remove its outgoing edges)
    # - C keeps B as inhibitor, gets dummy_C parent
    # - dummy_C gets self-loop
    expected_exp1 = np.array(
        [
            [1, 0, 0, 0],  # A: self-loop only
            [1, 0, 0, 0],  # B: keeps A as parent
            [0, -1, 0, 1],  # C: B inhibitor and dummy_C parent
            [0, 0, 0, 1],  # dummy_C: self-loop
        ]
    )
    np.testing.assert_array_equal(result[0], expected_exp1)

    # For exp2 (B perturbed):
    # - A keeps its original connections (none)
    # - B gets self-loop, loses any parents
    # - C loses B as parent, keeps dummy_C parent
    # - dummy_C keeps self-loop
    expected_exp2 = np.array(
        [
            [0, 0, 0, 0],  # A: no parents
            [0, 1, 0, 0],  # B: self-loop only
            [0, -1, 0, 1],  # C: B inhibitor and dummy_C parent
            [0, 0, 0, 1],  # dummy_C: self-loop
        ]
    )
    np.testing.assert_array_equal(result[1], expected_exp2)


def test_get_experiment_array_no_perturbations():
    # Empty perturbation dictionary
    pert_dic_all = {}

    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    node_list = get_sorted_node_list(G)

    # Create a simple adjacency matrix
    A_mult = pd.DataFrame([[0, 1], [0, 0]], index=node_list, columns=node_list)

    inh = []

    # Function should return an empty array with shape (0, n, n)
    result = get_experiment_array(A_mult, inh, pert_dic_all, G)

    assert isinstance(result, list)
    assert len(result) == 0


def test_get_experiment_array_with_multiple_inhibitors():
    # Create test graph
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")
    G.add_edge("C", "D", sign="Inhibitor")

    node_list = get_sorted_node_list(G)
    inh = ["C", "D"]  # Two inhibitor-only nodes

    # Simple perturbation dictionary with one experiment
    pert_dic_all = OrderedDict({"exp1": {"pert": {"A": 1}}})

    # Create a base adjacency matrix
    A_mult = pd.DataFrame(
        [
            [0, 0, 0, 0],  # A has no parents
            [1, 0, 0, 0],  # B has A parent
            [0, -1, 0, 0],  # C has B parent (inhibitor)
            [0, 0, -1, 0],  # D has C parent (inhibitor)
        ],
        index=node_list,
        columns=node_list,
    )

    # Call the function
    result = get_experiment_array(A_mult, inh, pert_dic_all, G)

    # We should get a 3D array with shape (1, 6, 6) for 1 experiment, 6 nodes
    assert isinstance(result, list)
    assert isinstance(result[0], pd.DataFrame)
    assert result[0].shape == (6, 6)

    # When A is perturbed:
    # - A gets self-loop (perturbed)
    # - B keeps A as parent (A being perturbed doesn't remove outgoing edges)
    # - C keeps B as inhibitor, gets dummy_C parent
    # - D keeps C as inhibitor, gets dummy_D parent
    # - dummy_C and dummy_D get self-loops
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],  # A: self-loop only (perturbed)
            [1, 0, 0, 0, 0, 0],  # B: keeps A as parent
            [0, -1, 0, 0, 1, 0],  # C: B inhibitor and dummy_C parent
            [0, 0, -1, 0, 0, 1],  # D: C inhibitor and dummy_D parent
            [0, 0, 0, 0, 1, 0],  # dummy_C: self-loop
            [0, 0, 0, 0, 0, 1],  # dummy_D: self-loop
        ]
    )
    np.testing.assert_array_equal(result[0], expected)
