from dataclasses import dataclass

import networkx as nx
import numpy as np
import pytest

from magellan.graph import Graph


@dataclass
class MockConfig:
    """Mock config class for testing"""

    n_nodes: int = 100
    forward_edge_probability: float = 0.05
    backward_edge_probability: float = 0.025
    skip_layer_probability: float = 0.025
    inhibition_fraction: float = 0.3
    min_edge_weight: float = 0.1
    max_edge_weight: float = 1.0
    network_density: float = 1.0


@pytest.fixture
def config():
    """Create a mock config for testing"""
    np.random.seed(42)  # Set seed for reproducibility
    return MockConfig()


def test_graph_basic_properties(config):
    """Test basic properties of the generated graph"""
    from magellan.synthetic import generate_base_graph

    G = generate_base_graph(config, verbose=False)

    # Test graph type and properties
    assert isinstance(G, Graph)
    assert G.number_of_nodes() <= config.n_nodes
    assert G.number_of_edges() > 0
    assert nx.is_directed(G)

    # Test node names
    node_names = list(G.nodes())
    assert all(isinstance(node, str) for node in node_names)
    # Check for duplicates by comparing length of list vs set
    assert len(node_names) == len(set(node_names)), "Duplicate node names found"


def test_edge_weights_and_signs(config):
    """Test edge weights and signs in the generated graph"""
    from magellan.synthetic import generate_base_graph

    G = generate_base_graph(config, verbose=False)

    # Test edge properties
    for u, v in G.edges():
        edge_data = G[u][v]

        # Check required attributes
        assert "edge_weight" in edge_data
        assert "sign" in edge_data
        assert "n_references" in edge_data

        # Check weight bounds
        weight = edge_data["edge_weight"]
        assert (
            config.min_edge_weight <= weight <= config.max_edge_weight * 1.2
        )  # Account for inhibitor boost

        # Check sign validity
        assert edge_data["sign"] in ["Activator", "Inhibitor"]

        # Check n_references matches edge_weight
        # assert abs(edge_data["n_references"] - edge_data["edge_weight"]) < 1e-10


def test_layered_structure(config):
    """Test the layered structure of the generated graph"""
    from magellan.synthetic import generate_base_graph

    G = generate_base_graph(config, verbose=False)

    # Get nodes as integers for layer checking
    nodes = [int(node) for node in G.nodes()]
    nodes.sort()

    # Check node distribution across layers
    n_layers = 10
    _ = config.n_nodes // n_layers

    # Test some forward connections exist (we can't test exact numbers due to randomness)
    forward_connections = 0
    for u, v in G.edges():
        if int(u) < int(v):  # Forward connection
            forward_connections += 1

    assert forward_connections > 0, "No forward connections found"


def test_inhibition_fraction(config):
    """Test the fraction of inhibitory edges"""
    from magellan.synthetic import generate_base_graph

    G = generate_base_graph(config, verbose=False)

    # Count inhibitory edges
    inhibitory_edges = sum(
        1 for _, _, data in G.edges(data=True) if data["sign"] == "Inhibitor"
    )
    total_edges = G.number_of_edges()

    # Allow for some random variation (±5%)
    expected_inhibitory = total_edges * config.inhibition_fraction
    assert abs(inhibitory_edges - expected_inhibitory) <= max(2, total_edges * 0.05)


def test_no_self_loops(config):
    """Test that the graph has no self-loops"""
    from magellan.synthetic import generate_base_graph

    G = generate_base_graph(config, verbose=False)
    assert len(list(nx.selfloop_edges(G))) == 0


def test_edge_probability_influence(config):
    """Test the influence of edge probabilities on network structure"""
    from magellan.synthetic import generate_base_graph

    # Create two graphs with different probabilities
    config1 = MockConfig(forward_edge_probability=0.1)
    config2 = MockConfig(forward_edge_probability=0.05)

    G1 = generate_base_graph(config1, verbose=False)  # type: ignore
    G2 = generate_base_graph(config2, verbose=False)  # type: ignore

    # The graph with higher probability should have more edges
    assert G1.number_of_edges() > G2.number_of_edges()


def test_different_network_sizes():
    """Test the function with different network sizes"""
    from magellan.synthetic import generate_base_graph

    sizes = [50, 100, 200]
    for size in sizes:
        config = MockConfig(n_nodes=size)
        G = generate_base_graph(config, verbose=False)  # type: ignore
        assert G.number_of_nodes() <= size


def test_edge_weight_distribution(config):
    """Test the distribution of edge weights"""
    from magellan.synthetic import generate_base_graph

    G = generate_base_graph(config, verbose=False)

    # Collect weights
    forward_weights = []
    backward_weights = []

    for u, v in G.edges():
        if int(v) > int(u):  # Forward edge
            forward_weights.append(G[u][v]["edge_weight"])
        else:  # Backward edge
            backward_weights.append(G[u][v]["edge_weight"])

    if forward_weights:  # Only test if we have forward edges
        assert np.mean(forward_weights) >= config.min_edge_weight
        assert np.mean(forward_weights) <= config.max_edge_weight

    if backward_weights:  # Only test if we have backward edges
        assert np.mean(backward_weights) >= config.min_edge_weight
        assert np.mean(backward_weights) <= config.max_edge_weight


def test_invalid_config():
    """Test that invalid configurations raise appropriate errors"""
    from magellan.synthetic import generate_base_graph

    # Test negative nodes
    with pytest.raises(ValueError):
        invalid_config = MockConfig(n_nodes=-10)
        generate_base_graph(invalid_config, verbose=False)  # type: ignore

    # Test invalid probabilities
    with pytest.raises(ValueError):
        invalid_config = MockConfig(forward_edge_probability=1.5)
        generate_base_graph(invalid_config, verbose=False)  # type: ignore

    # Test invalid edge weights
    with pytest.raises(ValueError):
        invalid_config = MockConfig(min_edge_weight=1.0, max_edge_weight=0.5)
        generate_base_graph(invalid_config, verbose=False)  # type: ignore
