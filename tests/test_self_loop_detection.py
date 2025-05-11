
import hypothesis.strategies as st
import networkx as nx
import pytest
from hypothesis import given

from magellan.prune import check_no_self_loops, has_self_loop, identify_self_loops


@pytest.fixture
def empty_graph() -> nx.DiGraph:
    return nx.DiGraph()

@pytest.fixture
def simple_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    return G

@pytest.fixture
def self_loop_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([('A', 'A'), ('B', 'C'), ('C', 'B')])
    return G

@pytest.fixture
def multiple_self_loops_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([('A', 'A'), ('B', 'B'), ('C', 'D')])
    return G

def test_has_self_loop_empty_graph(empty_graph: nx.DiGraph) -> None:
    assert not has_self_loop(empty_graph, 'A')

def test_has_self_loop_nonexistent_node(simple_graph: nx.DiGraph) -> None:
    assert not has_self_loop(simple_graph, 'nonexistent')

def test_has_self_loop_no_loop(simple_graph: nx.DiGraph) -> None:
    assert not has_self_loop(simple_graph, 'A')

def test_has_self_loop_with_loop(self_loop_graph: nx.DiGraph) -> None:
    assert has_self_loop(self_loop_graph, 'A')
    assert not has_self_loop(self_loop_graph, 'B')

def test_identify_self_loops_empty_graph(empty_graph: nx.DiGraph) -> None:
    assert identify_self_loops(empty_graph) == []

def test_identify_self_loops_no_loops(simple_graph: nx.DiGraph) -> None:
    assert identify_self_loops(simple_graph) == []

def test_identify_self_loops_single_loop(self_loop_graph: nx.DiGraph) -> None:
    assert identify_self_loops(self_loop_graph) == ['A']

def test_identify_self_loops_multiple_loops(multiple_self_loops_graph: nx.DiGraph) -> None:
    result = identify_self_loops(multiple_self_loops_graph)
    assert sorted(result) == ['A', 'B']

def test_check_no_self_loops_valid_graph(simple_graph: nx.DiGraph) -> None:
    check_no_self_loops(simple_graph)  # Should not raise any exception

def test_check_no_self_loops_empty_graph(empty_graph: nx.DiGraph) -> None:
    check_no_self_loops(empty_graph)  # Should not raise any exception

def test_check_no_self_loops_with_loop(self_loop_graph: nx.DiGraph) -> None:
    with pytest.raises(ValueError) as exc_info:
        check_no_self_loops(self_loop_graph)
    assert "contains self-loops" in str(exc_info.value)
    assert "A" in str(exc_info.value)

def test_check_no_self_loops_multiple_loops(multiple_self_loops_graph: nx.DiGraph) -> None:
    with pytest.raises(ValueError) as exc_info:
        check_no_self_loops(multiple_self_loops_graph)
    assert "contains self-loops" in str(exc_info.value)
    assert all(node in str(exc_info.value) for node in ['A', 'B'])

# Property-based tests using hypothesis



@given(st.text(min_size=1))
def test_has_self_loop_property(node_name: str) -> None:
    G = nx.DiGraph()
    # Test empty graph
    assert not has_self_loop(G, node_name)
    
    # Test with self-loop
    G.add_edge(node_name, node_name)
    assert has_self_loop(G, node_name)

@given(st.lists(st.text(min_size=1), min_size=2, unique=True))
def test_identify_self_loops_property(node_names: list[str]) -> None:
    G = nx.DiGraph()
    # Add self loops to half of nodes (rounded up)
    mid = (len(node_names) + 1) // 2  # Round up division
    for node in node_names[:mid]:
        G.add_edge(node, node)
    
    # Add regular edges for remaining nodes
    for node in node_names[mid:]:
        G.add_edge(node, node_names[0])
    
    result = identify_self_loops(G)
    assert sorted(result) == sorted(node_names[:mid])