import networkx as nx
import numpy as np
import pytest

from magellan.sci_opt import get_adj


@pytest.mark.skip(reason="Function get_adj is no longer implemented")
def test_get_adj_basic():
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")

    node_list = ["A", "B", "C"]
    pert_dic_all = {
        "exp1": {"pert": {"A": 1}, "exp": {"C": 0}},
        "exp2": {"pert": {"B": 1}, "exp": {"C": 1}},
    }

    A, inh = get_adj(G, pert_dic_all, node_list)

    # Check output shapes
    assert isinstance(A, np.ndarray)
    assert A.shape == (2, 4, 4)  # 2 experiments, 4x4 matrix (includes dummy node)
    assert isinstance(inh, list)
    assert len(inh) == 1  # C is an inhibitor-only node
    assert inh[0] == "C"


@pytest.mark.skip(reason="Function get_adj is no longer implemented")
def test_get_adj_empty_graph():
    G = nx.DiGraph()
    node_list = []
    pert_dic_all = {}

    with pytest.raises(ValueError):
        get_adj(G, pert_dic_all, node_list)


@pytest.mark.skip(reason="Function get_adj is no longer implemented")
def test_get_adj_invalid_edge_sign():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="InvalidSign")
    node_list = ["A", "B"]
    pert_dic_all = {"exp1": {"pert": {}, "exp": {}}}

    with pytest.raises(ValueError, match="Invalid edge sign"):
        get_adj(G, pert_dic_all, node_list)


@pytest.mark.skip(reason="Function get_adj is no longer implemented")
def test_get_adj_complex_network():
    # Create a more complex test graph
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("A", "C", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")
    G.add_edge("C", "D", sign="Inhibitor")
    G.add_edge("A", "D", sign="Activator")

    node_list = ["A", "B", "C", "D"]
    pert_dic_all = {
        "exp1": {"pert": {"A": 1}, "exp": {"D": 0}},
        "exp2": {"pert": {"B": 1}, "exp": {"C": 1}},
        "exp3": {"pert": {"C": 1}, "exp": {"D": 1}},
    }

    A, inh = get_adj(G, pert_dic_all, node_list)

    # Check output shapes
    assert A.shape == (3, 4, 4)  # 3 experiments, 4x4 matrix
    assert len(inh) == 0  # No nodes are inhibitor-only


@pytest.mark.skip(reason="Function get_adj is no longer implemented")
def test_get_adj_complex_network_inh_only():
    # Create a more complex test graph
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")
    G.add_edge("C", "D", sign="Inhibitor")
    G.add_edge("A", "D", sign="Activator")

    node_list = ["A", "B", "C", "D"]
    pert_dic_all = {
        "exp1": {"pert": {"A": 1}, "exp": {"D": 0}},
        "exp2": {"pert": {"B": 1}, "exp": {"C": 1}},
        "exp3": {"pert": {"C": 1}, "exp": {"D": 1}},
    }

    A, inh = get_adj(G, pert_dic_all, node_list)

    # Check output shapes
    # Note that the shape of A is 5x5 because we add a dummy node, due to C being inhibitor-only
    assert A.shape == (3, 5, 5)  # 3 experiments, 5x5 matrix
    assert len(inh) == 1  # C is inhibitor-only
    assert inh[0] == "C"


@pytest.mark.skip(reason="Function get_adj is no longer implemented")
def test_get_adj_node_list_mismatch():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")

    # Test with node_list missing a node from the graph
    node_list = ["A"]
    pert_dic_all = {"exp1": {"pert": {"A": 1}, "exp": {"B": 0}}}

    with pytest.raises(
        ValueError, match="The node_list does not match the nodes in the graph"
    ):
        get_adj(G, pert_dic_all, node_list)
