from collections import OrderedDict

import networkx as nx
import pandas as pd
import pytest

from magellan.sci_opt import get_data


def test_get_data_basic():
    """Test basic functionality with simple input."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    # node_list = ["A", "B", "C"]
    pert_dic_all = {
        "exp1": {
            "pert": {"A": 1.0},
            "exp": {"B": 0.5},
        },
    }
    pert_dic_all = OrderedDict(sorted(pert_dic_all.items(), key=lambda t: t[0]))

    X, y = get_data(pert_dic_all, G)

    # Check dataframe shapes and types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert X.shape == (3, 1)
    assert y.shape == (3, 1)

    # Check specific values
    assert X.loc["A", "exp1"] == 1.0
    assert X.loc["B", "exp1"] == 0.0
    assert y.loc["B", "exp1"] == 0.5
    assert y.loc["A", "exp1"] == 0.0


def test_get_data_multiple_experiments():
    """Test with multiple experiments and perturbations."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    # node_list = ["A", "B", "C"]
    pert_dic_all = {
        "exp1": {
            "pert": {"A": 1.0},
            "exp": {"B": 0.5},
        },
        "exp2": {
            "pert": {"B": -1.0, "C": 2.0},
            "exp": {"A": 0.7},
        },
    }
    pert_dic_all = OrderedDict(sorted(pert_dic_all.items(), key=lambda t: t[0]))
    X, y = get_data(pert_dic_all, G)

    # Check experiment ordering
    assert list(X.columns) == ["exp1", "exp2"]

    # Check perturbation values
    assert X.loc["A", "exp1"] == 1.0
    assert X.loc["B", "exp2"] == -1.0
    assert X.loc["C", "exp2"] == 2.0

    # Check expectation values
    assert y.loc["B", "exp1"] == 0.5
    assert y.loc["A", "exp2"] == 0.7


def test_get_data_empty_input():
    """Test with empty perturbation dictionary."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    pert_dic_all = {}
    pert_dic_all = OrderedDict(sorted(pert_dic_all.items(), key=lambda t: t[0]))

    with pytest.raises(ValueError, match="pert_dic_all cannot be empty"):
        get_data(pert_dic_all, G)


def test_get_data_empty_node_list():
    """Test that get_data raises ValueError when given empty node list."""
    G = nx.DiGraph()
    pert_dic_all = {"exp1": {"pert": {}, "exp": {}}}  # Simplified test data
    pert_dic_all = OrderedDict(sorted(pert_dic_all.items(), key=lambda t: t[0]))

    with pytest.raises(
        ValueError, match="node_list cannot be None or empty"
    ):  # Added expected message
        get_data(pert_dic_all, G)


def test_get_data_empty_pert_dic_all():
    """Test that get_data raises ValueError when given empty pert_dic_all."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    pert_dic_all = {"exp1": {"pert": {}, "exp": {}}}  # Simplified test data
    pert_dic_all = OrderedDict(sorted(pert_dic_all.items(), key=lambda t: t[0]))

    with pytest.raises(ValueError, match="pert_dic_all contains empty experiment data"):
        get_data(pert_dic_all, G)


def test_get_data_completely_empty_dict():
    """Test that get_data raises ValueError when given completely empty dictionary."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    pert_dic_all = {}
    pert_dic_all = OrderedDict(sorted(pert_dic_all.items(), key=lambda t: t[0]))

    with pytest.raises(ValueError, match="pert_dic_all cannot be empty"):
        get_data(pert_dic_all, G)
