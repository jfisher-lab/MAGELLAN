import networkx as nx
import pytest

from magellan.prune import check_dummy_nodes


def create_graph(nodes: list[str]) -> nx.DiGraph:
    """Helper function to create a DiGraph from a list of nodes"""
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    return G

def test_check_dummy_nodes_valid():
    """Test that valid DiGraphs without dummy nodes pass"""
    valid_lists = [
        ["node1", "node2", "node3"],
        ["ABC", "DEF", "GHI"],
        ["1", "2", "3"],
        [],  # Empty graph should be valid
        ["node_with_underscore"],
        ["UPPERCASE", "lowercase", "MixedCase"],
        ["very_long_node_name_without"],
        ["n1", "n2", "n3"]
    ]
    
    for nodes in valid_lists:
        G = create_graph(nodes)
        check_dummy_nodes(G)  # Should not raise any exception

def test_check_dummy_nodes_invalid():
    """Test that DiGraphs with dummy nodes raise ValueError"""
    invalid_lists = [
        ["node1", "dummy_node", "node3"],
        ["dummy"],
        ["dummy_"],
        ["prefix_dummy_suffix"],
        ["node1", "node2", "dummy"],
        ["DUMMY", "node1"],  # Test case insensitivity
        ["Dummy", "node1"],  # Test case insensitivity
        ["node1", "dUmMy", "node2"]  # Test mixed case
    ]
    
    for nodes in invalid_lists:
        G = create_graph(nodes)
        with pytest.raises(ValueError):
            check_dummy_nodes(G)

def test_check_dummy_nodes_edge_cases():
    """Test edge cases and special characters"""
    edge_cases = [
        ["dummy123"],
        ["123dummy"],
        ["_dummy_"],
        ["node1", "not_a_dummy_just_contains_word"],
        ["dummynode"],  # Word 'dummy' at start
        ["duMmynode"],  # Word 'dummy' at start with mixed case
        ["nodedummy"],  # Word 'dummy' at end
    ]
    
    for nodes in edge_cases:
        G = create_graph(nodes)
        with pytest.raises(ValueError):
            check_dummy_nodes(G)

def test_check_dummy_nodes_types():
    """Test with different input types"""
    # Should raise TypeError for non-DiGraph inputs
    with pytest.raises(TypeError):
        check_dummy_nodes(None)  # type: ignore
        
    with pytest.raises(TypeError):
        check_dummy_nodes(42)  # type: ignore
        
    with pytest.raises(TypeError):
        check_dummy_nodes(["node1", "node2"])  # type: ignore
        
    with pytest.raises(TypeError):
        # Undirected graph
        check_dummy_nodes(nx.Graph())  # type: ignore

def test_check_dummy_nodes_empty_strings():
    """Test with empty strings and whitespace"""
    valid_empty = [
        ["", "node1", "node2"],  # Empty string
        [" ", "node1", "node2"],  # Space
        ["\t", "node1", "node2"],  # Tab
        ["\n", "node1", "node2"],  # Newline
        ["   ", "node1", "node2"]  # Multiple spaces
    ]
    
    for nodes in valid_empty:
        G = create_graph(nodes)
        check_dummy_nodes(G)  # Should not raise any exception

def test_check_dummy_nodes_unicode():
    """Test with unicode characters"""
    valid_unicode = [
        ["नोड", "节点", "ノード"],  # Different scripts
        ["λ", "π", "θ"],  # Greek letters
        ["node1", "ñōdę2", "nödé3"],  # Accented characters
        ["🔬", "🧬", "🔭"]  # Emojis
    ]
    
    for nodes in valid_unicode:
        G = create_graph(nodes)
        check_dummy_nodes(G)  # Should not raise any exception

def test_check_dummy_nodes_special_characters():
    """Test with special characters"""
    valid_special = [
        ["node#1", "node@2", "node$3"],
        ["node+", "node-", "node*"],
        ["node&1", "node|2", "node^3"],
        ["node(1)", "node[2]", "node{3}"]
    ]
    
    for nodes in valid_special:
        G = create_graph(nodes)
        check_dummy_nodes(G)  # Should not raise any exception

def test_check_dummy_nodes_mixed_content():
    """Test with mixed content types"""
    # Test each invalid node type separately since networkx will raise ValueError
    # for None and some other types before we get to our type check
    
    # First test cases where networkx allows node creation but our function should reject
    numeric_nodes = ["node1", 42, "node3"]
    float_nodes = ["node1", 3.14, "node3"]
    bool_nodes = ["node1", True, "node3"]
    
    for nodes in [numeric_nodes, float_nodes, bool_nodes]:
        G = create_graph(nodes)
        with pytest.raises(TypeError):
            check_dummy_nodes(G)
    
    # Then test cases where networkx itself will reject the nodes
    invalid_nodes = [
        ["node1", None, "node3"],  # None not allowed as node
        ["node1", [], "node3"],    # List not hashable
        ["node1", {}, "node3"]     # Dict not hashable
    ]
    
    for nodes in invalid_nodes:
        with pytest.raises(ValueError):
            G = create_graph(nodes)