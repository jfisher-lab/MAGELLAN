import networkx as nx
import pandas as pd
import pytest

from magellan.prune import get_trained_network


def test_basic_weight_assignment():
    """Test basic weight assignment for a simple graph."""
    # Create original graph
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G.add_edge("B", "C", sign="Inhibitor")
    
    # Create weight matrix
    W = pd.DataFrame([
        [0, 0.3, 0],  # A's row
        [0.5, 0, 0.2],  # B's row 
        [0, 0.7, 0]     # C's row
    ], index=["A", "B", "C"], columns=["A", "B", "C"])
    
    G_trained = get_trained_network(G, W)
    
    # Check edge weights are assigned correctly
    assert G_trained["A"]["B"]["edge_weight"] == 0.5
    assert G_trained["B"]["C"]["edge_weight"] == 0.7
    
    # Check original attributes are preserved
    assert G_trained["A"]["B"]["sign"] == "Activator"
    assert G_trained["B"]["C"]["sign"] == "Inhibitor"

def test_preserve_graph_structure():
    """Test that graph structure is preserved."""
    G = nx.DiGraph()
    edges = [("A", "B"), ("B", "C"), ("A", "C")]
    G.add_edges_from(edges)
    
    W = pd.DataFrame(0.5, index=list("ABC"), columns=list("ABC"))
    
    G_trained = get_trained_network(G, W)
    
    # Check nodes and edges are preserved
    assert set(G_trained.nodes()) == set(G.nodes())
    assert set(G_trained.edges()) == set(G.edges())

def test_empty_graph():
    """Test behavior with empty graph."""
    G = nx.DiGraph()
    W = pd.DataFrame()
    
    G_trained = get_trained_network(G, W)
    
    assert isinstance(G_trained, nx.DiGraph)
    assert len(G_trained.nodes()) == 0
    assert len(G_trained.edges()) == 0

def test_single_node():
    """Test graph with single self-loop."""
    G = nx.DiGraph()
    G.add_edge("A", "A", sign="Activator")
    
    W = pd.DataFrame([[0.3]], index=["A"], columns=["A"])
    
    G_trained = get_trained_network(G, W)
    
    assert G_trained["A"]["A"]["edge_weight"] == 0.3
    assert G_trained["A"]["A"]["sign"] == "Activator"

def test_weight_matrix_mismatch():
    """Test error handling for mismatched weight matrix."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    
    # Weight matrix missing node
    W = pd.DataFrame([[0.5]], index=["A"], columns=["A"])
    
    with pytest.raises(KeyError):
        get_trained_network(G, W)

def test_original_graph_unchanged():
    """Test that original graph is not modified."""
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator")
    G_original = G.copy()
    
    W = pd.DataFrame([[0, 0.5], [0, 0]], index=["A", "B"], columns=["A", "B"])
    
    _ = get_trained_network(G, W)
    
    # Check original graph hasn't changed
    assert "edge_weight" not in G["A"]["B"]
    assert dict(G.edges()) == dict(G_original.edges())