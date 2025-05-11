from magellan.prune import filter_specification


def test_filter_specification_basic():
    """Test basic filtering with simple input."""
    # Define a sample graph nodes set
    graph_nodes = {"A", "B", "C", "D"}

    # Define a sample perturbation dictionary
    pert_dic_small = {
        "exp1": {"pert": {"A": 1, "B": 0}, "exp": {"C": 2, "D": 1}},
        "exp2": {"pert": {"A": 0, "C": 1}, "exp": {"B": 1, "D": 2}},
    }

    # Filter the specification
    filtered_spec = filter_specification(pert_dic_small, graph_nodes)

    # Verify the result
    assert "exp1" in filtered_spec
    assert "exp2" in filtered_spec
    assert filtered_spec["exp1"]["pert"] == {"A": 1, "B": 0}
    assert filtered_spec["exp1"]["exp"] == {"C": 2, "D": 1}
    assert filtered_spec["exp2"]["pert"] == {"A": 0, "C": 1}
    assert filtered_spec["exp2"]["exp"] == {"B": 1, "D": 2}


def test_filter_specification_missing_perturbation():
    """Test filtering when perturbation nodes are missing from graph."""
    graph_nodes = {"A", "B", "C"}

    pert_dic_small = {
        "exp1": {"pert": {"A": 1, "B": 0}, "exp": {"C": 2}},
        "exp2": {
            "pert": {"A": 0, "D": 1},  # D is not in graph_nodes
            "exp": {"B": 1, "C": 2},
        },
    }

    filtered_spec = filter_specification(pert_dic_small, graph_nodes)

    # exp2 should be filtered out as it has a perturbation node not in graph_nodes
    assert "exp1" in filtered_spec
    assert "exp2" not in filtered_spec
    assert filtered_spec["exp1"]["pert"] == {"A": 1, "B": 0}
    assert filtered_spec["exp1"]["exp"] == {"C": 2}


def test_filter_specification_missing_expectation():
    """Test filtering when expectation nodes are missing from graph."""
    graph_nodes = {"A", "B", "C"}

    pert_dic_small = {
        "exp1": {
            "pert": {"A": 1, "B": 0},
            "exp": {"C": 2, "D": 1},  # D is not in graph_nodes
        },
        "exp2": {
            "pert": {"A": 0, "B": 1},
            "exp": {"D": 2, "E": 1},  # D and E are not in graph_nodes
        },
    }

    filtered_spec = filter_specification(pert_dic_small, graph_nodes)

    # exp1 should have D filtered out from "exp"
    # exp2 should be completely filtered out as all its expectation nodes are not in graph_nodes
    assert "exp1" in filtered_spec
    assert "exp2" not in filtered_spec
    assert filtered_spec["exp1"]["pert"] == {"A": 1, "B": 0}
    assert filtered_spec["exp1"]["exp"] == {"C": 2}


def test_filter_specification_empty_input():
    """Test filtering with empty input."""
    graph_nodes = {"A", "B", "C"}
    pert_dic_small = {}

    filtered_spec = filter_specification(pert_dic_small, graph_nodes)

    assert filtered_spec == {}


def test_filter_specification_empty_graph():
    """Test filtering with empty graph nodes."""
    graph_nodes = set()

    pert_dic_small = {"exp1": {"pert": {"A": 1, "B": 0}, "exp": {"C": 2}}}

    filtered_spec = filter_specification(pert_dic_small, graph_nodes)

    # All experiments should be filtered out
    assert filtered_spec == {}


def test_filter_specification_verbose():
    """Test filtering with verbose option enabled."""
    graph_nodes = {"A", "B", "C"}

    pert_dic_small = {
        "exp1": {
            "pert": {"A": 1, "B": 0},
            "exp": {"C": 2, "D": 1},  # D is not in graph_nodes
        },
        "exp2": {
            "pert": {"A": 0, "D": 1},  # D is not in graph_nodes
            "exp": {"B": 1, "C": 2},
        },
    }

    # This just tests that the function runs without error when verbose is True
    # In a real implementation, we'd possibly capture stdout and verify output
    filtered_spec = filter_specification(pert_dic_small, graph_nodes, verbose=True)

    assert "exp1" in filtered_spec
    assert "exp2" not in filtered_spec


def test_filter_specification_no_valid_expectations():
    """Test filtering when there are no valid expectation nodes after filtering."""
    graph_nodes = {"A", "B", "C"}

    pert_dic_small = {
        "exp1": {
            "pert": {"A": 1, "B": 0},
            "exp": {"D": 2, "E": 1},  # D and E are not in graph_nodes
        }
    }

    filtered_spec = filter_specification(pert_dic_small, graph_nodes)

    # exp1 should be filtered out as all its expectation nodes are not in graph_nodes
    assert filtered_spec == {}


def test_filter_specification_preserves_input():
    """Test that the original input is not modified."""
    graph_nodes = {"A", "B", "C"}

    pert_dic_small = {
        "exp1": {
            "pert": {"A": 1, "B": 0},
            "exp": {"C": 2, "D": 1},  # D is not in graph_nodes
        }
    }

    original_pert_dic = pert_dic_small.copy()
    filtered_spec = filter_specification(pert_dic_small, graph_nodes)

    # Verify the original dictionary was not modified
    assert pert_dic_small == original_pert_dic
    # Verify the filtered dictionary is different
    assert filtered_spec != original_pert_dic
