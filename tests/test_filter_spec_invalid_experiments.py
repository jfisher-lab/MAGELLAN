from copy import deepcopy

import networkx as nx
import pytest

from magellan.prune import (
    filter_spec_by_children,
    filter_spec_by_parents,
    filter_spec_invalid_experiments,
)


@pytest.fixture
def sample_graph():
    """Create a simple graph for testing."""
    G = nx.DiGraph()

    # Create a small network:
    # A -> B -> C
    # |    |
    # v    v
    # D    E
    G.add_edge("A", "B")
    G.add_edge("A", "D")
    G.add_edge("B", "C")
    G.add_edge("B", "E")

    # Add isolated nodes
    G.add_node("F")  # No parents or children

    return G


@pytest.fixture
def sample_pert_dic():
    """Create a sample perturbation dictionary."""
    return {
        "exp1": {"pert": {"A": 1}, "exp": {"C": 0, "D": 1}},
        "exp2": {"pert": {"B": 1}, "exp": {"C": 0, "E": 1}},
        "exp3": {
            "pert": {"C": 1},  # C has no children
            "exp": {"E": 0},
        },
        "exp4": {
            "pert": {"A": 1},
            "exp": {"F": 0},  # F has no parents
        },
        "exp5": {
            "pert": {"F": 1},  # F has no children
            "exp": {"C": 0},
        },
    }


def test_filter_spec_by_parents(sample_graph, sample_pert_dic):
    """Test filtering specs by removing nodes with no parents."""
    filtered = filter_spec_by_parents(sample_pert_dic, sample_graph)

    # Check that F was removed as an expectation node
    assert "exp4" not in filtered, (
        "Experiment with only isolated expectation nodes should be removed"
    )

    # Check other experiments survived
    assert len(filtered) == 4, f"Expected 4 experiments, got {len(filtered)}"

    # Check specific experiments
    assert "exp1" in filtered, "exp1 should survive (all nodes have parents)"
    assert "exp2" in filtered, "exp2 should survive (all nodes have parents)"
    assert "exp3" in filtered, "exp3 should survive (all nodes have parents)"
    assert "exp5" in filtered, "exp5 should survive (C has parents)"

    # Check that all expectation nodes in filtered experiments have parents
    for exp, specs in filtered.items():
        for node in specs["exp"]:
            assert len(list(sample_graph.predecessors(node))) > 0, (
                f"Node {node} in {exp} has no parents"
            )


def test_filter_spec_by_children(sample_graph, sample_pert_dic):
    """Test filtering specs by removing nodes with no children."""
    filtered = filter_spec_by_children(sample_pert_dic, sample_graph)

    # Check that experiments with isolated perturbation nodes are removed
    assert "exp3" not in filtered, "Experiment with C as perturbation should be removed"
    assert "exp5" not in filtered, "Experiment with F as perturbation should be removed"

    # Check other experiments survived
    assert len(filtered) == 3, f"Expected 3 experiments, got {len(filtered)}"
    assert "exp1" in filtered, "exp1 should survive (A has children)"
    assert "exp2" in filtered, "exp2 should survive (B has children)"
    assert "exp4" in filtered, "exp4 should survive (A has children)"

    # Check that all perturbation nodes in filtered experiments have children
    for exp, specs in filtered.items():
        for node in specs["pert"]:
            assert len(list(sample_graph.successors(node))) > 0, (
                f"Node {node} in {exp} has no children"
            )


def test_filter_spec_invalid_experiments(sample_graph, sample_pert_dic):
    """Test the combined filtering function."""
    # Create a combined filter with both parent and children checks
    filtered = filter_spec_invalid_experiments(
        sample_pert_dic, sample_graph, verbose=False
    )

    # Only exp1 and exp2 should survive the filtering
    assert len(filtered) == 2, f"Expected 2 experiments, got {len(filtered)}"
    assert "exp1" in filtered, "exp1 should survive all filters"
    assert "exp2" in filtered, "exp2 should survive all filters"
    assert "exp3" not in filtered, "exp3 should be filtered out"
    assert "exp4" not in filtered, "exp4 should be filtered out"
    assert "exp5" not in filtered, "exp5 should be filtered out"


def test_filters_dont_modify_original(sample_graph, sample_pert_dic):
    """Test that the filtering functions don't modify the original dictionary."""
    original = deepcopy(sample_pert_dic)

    # Apply all filters
    filter_spec_by_parents(sample_pert_dic, sample_graph)
    filter_spec_by_children(sample_pert_dic, sample_graph)
    filter_spec_invalid_experiments(sample_pert_dic, sample_graph)

    # Check that original wasn't modified
    assert sample_pert_dic == original, "Original dictionary was modified"


def test_empty_input():
    """Test handling of empty inputs."""
    G = nx.DiGraph()
    G.add_node("A")

    empty_dict = {}

    # All functions should handle empty dicts gracefully
    assert filter_spec_by_parents(empty_dict, G) == {}, (
        "Should return empty dict for empty input"
    )
    assert filter_spec_by_children(empty_dict, G) == {}, (
        "Should return empty dict for empty input"
    )
    assert filter_spec_invalid_experiments(empty_dict, G) == {}, (
        "Should return empty dict for empty input"
    )


def test_verbose_output(sample_graph, sample_pert_dic, capsys):
    """Test that verbose output is correctly generated."""
    # Test parent filter verbose output
    filter_spec_by_parents(sample_pert_dic, sample_graph, verbose=True)
    captured = capsys.readouterr()
    assert "Removed" in captured.out
    assert "F" in captured.out

    # Test children filter verbose output
    filter_spec_by_children(sample_pert_dic, sample_graph, verbose=True)
    captured = capsys.readouterr()
    assert "Removed" in captured.out
    assert "C" in captured.out
    assert "F" in captured.out


def test_filter_spec_by_children_non_aggressive(sample_graph, sample_pert_dic):
    """Test filtering specs by removing nodes with no children (non-aggressive mode)."""
    filtered = filter_spec_by_children(sample_pert_dic, sample_graph, aggressive=False)

    # In non-aggressive mode, only experiments where ALL perturbations have no children are removed
    assert "exp3" not in filtered, "Experiment with C as perturbation should be removed"
    assert "exp5" not in filtered, "Experiment with F as perturbation should be removed"

    # Check other experiments survived
    assert len(filtered) == 3, f"Expected 3 experiments, got {len(filtered)}"
    assert "exp1" in filtered, "exp1 should survive (A has children)"
    assert "exp2" in filtered, "exp2 should survive (B has children)"
    assert "exp4" in filtered, "exp4 should survive (A has children)"


def test_filter_spec_by_children_aggressive(sample_graph, sample_pert_dic):
    """Test filtering specs by removing nodes with no children (aggressive mode)."""
    # Add more complex test cases
    complex_pert_dic = deepcopy(sample_pert_dic)
    complex_pert_dic["exp6"] = {
        "pert": {"A": 1, "C": 1},  # A has children, C doesn't
        "exp": {"E": 0},
    }
    complex_pert_dic["exp7"] = {
        "pert": {"A": 1, "B": 1},  # Both have children
        "exp": {"D": 0},
    }

    filtered = filter_spec_by_children(complex_pert_dic, sample_graph, aggressive=True)

    # In aggressive mode, experiments with ANY perturbation without children are removed
    assert "exp3" not in filtered, "exp3 should be removed (C has no children)"
    assert "exp5" not in filtered, "exp5 should be removed (F has no children)"
    assert "exp6" not in filtered, (
        "exp6 should be removed (has C which has no children)"
    )

    # Only experiments where ALL perturbations have children remain
    assert len(filtered) == 4, f"Expected 4 experiments, got {len(filtered)}"
    assert "exp1" in filtered, "exp1 should survive (A has children)"
    assert "exp2" in filtered, "exp2 should survive (B has children)"
    assert "exp4" in filtered, "exp4 should survive (A has children)"
    assert "exp7" in filtered, "exp7 should survive (both A and B have children)"


def test_filter_spec_invalid_experiments_non_aggressive(sample_graph, sample_pert_dic):
    """Test the combined filtering function in non-aggressive mode."""
    filtered = filter_spec_invalid_experiments(
        sample_pert_dic, sample_graph, aggressive=False, verbose=False
    )

    # Only exp1 and exp2 should survive all filters
    assert len(filtered) == 2, f"Expected 2 experiments, got {len(filtered)}"
    assert "exp1" in filtered, "exp1 should survive all filters"
    assert "exp2" in filtered, "exp2 should survive all filters"
    assert "exp3" not in filtered, "exp3 should be filtered out (C has no children)"
    assert "exp4" not in filtered, "exp4 should be filtered out (F has no parents)"
    assert "exp5" not in filtered, "exp5 should be filtered out (F has no children)"


def test_filter_spec_invalid_experiments_aggressive(sample_graph, sample_pert_dic):
    """Test the combined filtering function in aggressive mode."""
    # Add a more complex test case
    complex_pert_dic = deepcopy(sample_pert_dic)
    complex_pert_dic["exp6"] = {
        "pert": {"A": 1, "C": 1},  # A has children, C doesn't
        "exp": {"E": 0},  # E has parents
    }

    filtered = filter_spec_invalid_experiments(
        complex_pert_dic, sample_graph, aggressive=True, verbose=False
    )

    # Only exp1 and exp2 should survive all filters
    assert len(filtered) == 2, f"Expected 2 experiments, got {len(filtered)}"
    assert "exp1" in filtered, "exp1 should survive all filters"
    assert "exp2" in filtered, "exp2 should survive all filters"
    assert "exp3" not in filtered, "exp3 should be filtered out (C has no children)"
    assert "exp4" not in filtered, "exp4 should be filtered out (F has no parents)"
    assert "exp5" not in filtered, "exp5 should be filtered out (F has no children)"
    assert "exp6" not in filtered, (
        "exp6 should be filtered out in aggressive mode (C has no children)"
    )
