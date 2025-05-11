from dataclasses import dataclass

import networkx as nx
import pytest

from magellan.synthetic import add_spurious_edges


# Mock PruningTestConfig class
@dataclass
class PruningTestConfig:
    n_spurious_edges: int
    inhibition_fraction: float
    min_edge_weight: float
    max_edge_weight: float


@pytest.fixture
def basic_graph():
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0, sign="Activator")
    G.add_edge("B", "C", weight=1.0, sign="Inhibitor")
    G.add_edge("A", "D", weight=1.0, sign="Activator")
    return G


@pytest.fixture
def config():
    return PruningTestConfig(
        n_spurious_edges=2,
        inhibition_fraction=0.5,
        min_edge_weight=0.1,
        max_edge_weight=1.0,
    )


def test_basic_functionality(basic_graph, config):
    G_noisy, spurious = add_spurious_edges(basic_graph, config)

    # Check number of spurious edges added
    assert len(spurious) == config.n_spurious_edges

    # Check total number of edges
    assert len(G_noisy.edges()) == len(basic_graph.edges()) + config.n_spurious_edges

    # Check spurious edges aren't in original graph
    for edge in spurious:
        assert edge not in basic_graph.edges()


def test_inhibition_fraction(basic_graph, config):
    G_noisy, spurious = add_spurious_edges(basic_graph, config)

    # Count inhibitory edges in spurious edges
    inhibitory_count = sum(
        1 for edge in spurious if G_noisy.edges[edge]["sign"] == "Inhibitor"
    )

    # Check if matches expected fraction
    expected_inhibitory = int(config.n_spurious_edges * config.inhibition_fraction)
    assert inhibitory_count == expected_inhibitory


def test_edge_weights(basic_graph, config):
    G_noisy, spurious = add_spurious_edges(basic_graph, config)

    # Check all spurious edge weights are within bounds
    for edge in spurious:
        weight = G_noisy.edges[edge]["weight"]
        assert config.min_edge_weight <= weight <= config.max_edge_weight


def test_no_self_loops(basic_graph, config):
    G_noisy, spurious = add_spurious_edges(basic_graph, config)

    # Check no self-loops in spurious edges
    for u, v in spurious:
        assert u != v


def test_no_duplicate_edges(basic_graph, config):
    G_noisy, spurious = add_spurious_edges(basic_graph, config)

    # Check no duplicates in spurious edges
    assert len(spurious) == len(set(spurious))

    # Check no overlap with original edges
    original_edges = set(basic_graph.edges())
    assert not (original_edges & spurious)


def test_not_enough_available_edges():
    # Create a small complete graph
    G = nx.complete_graph(3, create_using=nx.DiGraph)
    nx.set_edge_attributes(G, "Activator", "sign")
    nx.set_edge_attributes(G, 1.0, "weight")

    config = PruningTestConfig(
        n_spurious_edges=10,  # More than possible
        inhibition_fraction=0.5,
        min_edge_weight=0.1,
        max_edge_weight=1.0,
    )

    with pytest.raises(ValueError):
        add_spurious_edges(G, config)  # type: ignore


def test_node_pairs_considered(basic_graph, config):
    """Test that all valid node pairs are considered for spurious edges"""
    G_noisy, spurious = add_spurious_edges(basic_graph, config)

    # Get all possible node pairs
    nodes = list(basic_graph.nodes())
    all_possible = set((u, v) for u in nodes for v in nodes if u != v)

    # Remove original edges
    all_possible -= set(basic_graph.edges())

    # Check that spurious edges are subset of possible edges
    assert spurious.issubset(all_possible)


def test_reference_count(basic_graph, config):
    G_noisy, spurious = add_spurious_edges(basic_graph, config)

    # Check reference count for spurious edges
    for edge in spurious:
        assert G_noisy.edges[edge]["n_references"] == config.min_edge_weight


def test_large_graph_performance():
    """Test performance with larger graph"""
    G = nx.scale_free_graph(10000, seed=42)
    G = nx.DiGraph(G)  # Convert to DiGraph
    nx.set_edge_attributes(G, "Activator", "sign")
    nx.set_edge_attributes(G, 1.0, "weight")

    config = PruningTestConfig(
        n_spurious_edges=50,
        inhibition_fraction=0.3,
        min_edge_weight=0.1,
        max_edge_weight=1.0,
    )

    G_noisy, spurious = add_spurious_edges(G, config)  # type: ignore
    assert len(spurious) == config.n_spurious_edges
