import networkx as nx
import pytest

from magellan.prune import make_node_dic, make_pert_idx


def test_make_node_dic():
    # Test empty list
    G = nx.DiGraph()
    assert make_node_dic(G) == {}

    # Test single node
    G = nx.DiGraph()
    G.add_node("A")
    assert make_node_dic(G) == {"A": 0}

    # Test multiple nodes
    G = nx.DiGraph()
    G.add_nodes_from(["A", "B", "C"])
    assert make_node_dic(G) == {"A": 0, "B": 1, "C": 2}

    # Removed, as nx.DiGraph() does not allow duplicate nodes
    # G = nx.DiGraph()
    # G.add_nodes_from(["A", "A", "B"])
    # with pytest.raises(ValueError):
    #    make_node_dic(G)


def test_make_pert_idx():
    # Test data
    pert_list = ["node1", "node2", "node3"]
    node_dic = {"node1": 0, "node2": 1, "node3": 2, "node4": 3}

    result = make_pert_idx(pert_list, node_dic)
    expected = [0, 1, 2]

    assert result == expected


def test_make_pert_idx_empty():
    node_dic = {"node1": 0}
    assert make_pert_idx([], node_dic) == []


def test_make_pert_idx_missing_key():
    pert_list = ["missing"]
    node_dic = {"node1": 0}

    with pytest.raises(KeyError):
        make_pert_idx(pert_list, node_dic)
