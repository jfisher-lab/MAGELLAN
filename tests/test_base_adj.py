import networkx as nx
import pandas as pd
import pytest

from magellan.sci_opt import base_adj


@pytest.fixture
def simple_graph():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")
    G.add_edge("C", "A", sign="Activator")
    return G


def test_base_adj_basic(simple_graph):
    result = base_adj(simple_graph)

    expected = pd.DataFrame(
        [
            [0, 0, 1],  # A's row: only C activates A
            [1, 0, 0],  # B's row: only A activates B
            [0, -1, 0],  # C's row: B inhibits C
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    pd.testing.assert_frame_equal(result, expected)


def test_base_adj_empty_graph():
    G = nx.DiGraph()
    with pytest.raises(ValueError, match="The graph G is empty."):
        base_adj(G)


def test_base_adj_invalid_sign():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="InvalidSign")

    with pytest.raises(ValueError):
        base_adj(G)
