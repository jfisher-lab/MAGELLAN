from collections import OrderedDict

import networkx as nx
import pytest

from magellan.prune import add_dummy_nodes_and_generate_A_inh


@pytest.fixture
def sample_graph():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")
    return G


@pytest.fixture
def sample_graph_complex():
    G = nx.DiGraph()
    # Add multiple activator edges to the same node
    G.add_edge("A", "B", sign="Activator", edge_weight=1)
    G.add_edge("D", "B", sign="Activator", edge_weight=1)  # Add another activator to B
    G.add_edge("B", "C", sign="Inhibitor", edge_weight=1)
    G.add_edge("A", "C", sign="Activator", edge_weight=0.5)
    return G


@pytest.fixture
def sample_pert_dic():
    return {"exp1": {"pert": {"A": 1}, "exp": {"C": 0}}}


def test_add_dummy_nodes_and_generate_A_inh(sample_graph, sample_pert_dic):
    max_range = 2
    G, _, inh, inh_dic = add_dummy_nodes_and_generate_A_inh(
        sample_graph, sample_pert_dic, max_range
    )

    # Test that initial dummy node was added
    assert "0A_node00" in G.nodes

    # Test that dummy nodes were added for inhibitors
    assert "dummy_C" in G.nodes  # C is an inhibitor only node

    # Test that dummy nodes have correct edges
    assert G.has_edge("dummy_C", "C")
    assert G.has_edge("dummy_C", "dummy_C")
    assert G["dummy_C"]["C"]["sign"] == "Activator"

    # Test that node_list contains all nodes in correct order
    expected_nodes = sorted(["0A_node00", "A", "B", "C", "dummy_C"])
    assert sorted(G.nodes()) == expected_nodes

    # Test that inh list contains dummy nodes
    assert inh == ["dummy_C"]

    # Test that inh_dic has correct values
    assert inh_dic == {"dummy_C": max_range}

    # Test that pert_dic was updated with dummy nodes
    assert sample_pert_dic["exp1"]["pert"]["dummy_C"] == max_range


def test_empty_graph():
    G = nx.DiGraph()
    pert_dic = OrderedDict({"exp1": {"pert": {}, "exp": {}}})
    max_range = 2

    with pytest.raises(ValueError, match="The graph G has no edges."):
        add_dummy_nodes_and_generate_A_inh(G, pert_dic, max_range)


def test_multiple_inhibitors(sample_graph):
    # Add another inhibitor edge
    sample_graph.add_edge("C", "D", sign="Inhibitor")

    pert_dic = OrderedDict({"exp1": {"pert": {"A": 1}, "exp": {"D": 0}}})
    max_range = 2

    G, _, inh, inh_dic = add_dummy_nodes_and_generate_A_inh(
        sample_graph, pert_dic, max_range
    )

    # Test that dummy nodes were added for both inhibitors
    assert "dummy_C" in G.nodes
    assert "dummy_D" in G.nodes

    # Test that both inhibitors are in inh list
    assert set(inh) == {"dummy_C", "dummy_D"}

    # Test that both inhibitors are in inh_dic
    assert inh_dic == {"dummy_C": max_range, "dummy_D": max_range}


def test_add_dummy_nodes_and_generate_A_inh_methods(
    sample_graph_complex, sample_pert_dic
):
    """Test different methods (avg and sum) for add_dummy_nodes_and_generate_A_inh"""
    max_range = 2

    # Create deep copies of the inputs to avoid modification between calls
    sample_graph_avg = sample_graph_complex.copy()
    sample_graph_sum = sample_graph_complex.copy()
    pert_dic_avg = OrderedDict(
        {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in sample_pert_dic.items()}
    )
    pert_dic_sum = OrderedDict(
        {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in sample_pert_dic.items()}
    )

    # Test avg method
    G_avg, A_avg, inh_avg, inh_dic_avg = add_dummy_nodes_and_generate_A_inh(
        sample_graph_avg, pert_dic_avg, max_range, tf_method="avg"
    )

    # Test sum method
    G_sum, A_sum, inh_sum, inh_dic_sum = add_dummy_nodes_and_generate_A_inh(
        sample_graph_sum, pert_dic_sum, max_range, tf_method="sum"
    )

    # Test that graph structure is identical for both methods
    assert set(G_avg.nodes()) == set(G_sum.nodes())
    assert set(G_avg.edges()) == set(G_sum.edges())

    # Test that node lists are identical
    assert sorted(G_avg.nodes()) == sorted(G_sum.nodes())

    # Test that inhibitor lists and dictionaries are identical
    assert inh_avg == inh_sum
    assert inh_dic_avg == inh_dic_sum

    # Test that adjacency matrices are different for avg and sum methods
    assert [a.equals(b) for a, b in zip(A_avg, A_sum)]


def test_invalid_method(sample_graph, sample_pert_dic):
    """Test that invalid method raises ValueError"""
    max_range = 2

    with pytest.raises(
        ValueError,
        match="Invalid method: invalid, currently supported are 'avg' and 'sum'",
    ):
        add_dummy_nodes_and_generate_A_inh(
            sample_graph, sample_pert_dic, max_range, tf_method="invalid"
        )
