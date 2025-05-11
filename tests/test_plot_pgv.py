import os
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import networkx as nx
import pytest

import magellan.plot as visualization


def test_plot_pgv_with_pygraphviz():
    """Test plot_pgv when pygraphviz is available."""
    # First test that pygraphviz is actually importable
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        pytest.skip("pygraphviz not installed, skipping test")

    # Setup test data
    G = nx.DiGraph()
    G.add_edge("GeneA", "GeneB")
    G.add_edge("GeneB", "GeneC")
    G.add_edge("GeneA", "GeneC")

    gene_sets = {
        "type1": {"GeneA", "GeneD"},
        "type2": {"GeneB", "GeneE"},
        "type3": {"GeneC", "GeneF"},
        "deg": {"GeneA", "GeneC"},  # Should be skipped in the networkx approach
    }

    # Create a mock for pygraphviz
    mock_pgv = MagicMock()
    mock_agraph = MagicMock()
    mock_pgv.agraph.AGraph.return_value = mock_agraph

    # Mock the pgv_dot function
    mock_dot_result = "digraph G { /* mocked DOT graph */ }"
    with (
        patch.object(visualization, "pgv", mock_pgv),
        patch.object(
            visualization, "pgv_dot", return_value=mock_dot_result
        ) as mock_pgv_dot,
    ):
        # Create a temporary directory for the output
        test_dir = "test_output"
        os.makedirs(test_dir, exist_ok=True)

        # Call the function
        visualization.plot_pgv(
            G=G,
            gene_sets=gene_sets,
            G_dot=None,
            thre=3,
            path=test_dir,
            file_name="test_with_pgv",
        )

        # Verify that pygraphviz was used
        mock_pgv_dot.assert_called_once_with(gene_sets, G, 3)
        mock_pgv.agraph.AGraph.assert_called_once_with(mock_dot_result)
        mock_agraph.draw.assert_called_once_with(
            f"{test_dir}/test_with_pgv.png", prog="dot"
        )

        # Clean up
        if os.path.exists(f"{test_dir}/test_with_pgv.png"):
            os.remove(f"{test_dir}/test_with_pgv.png")
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


def test_plot_pgv_without_pygraphviz():
    """Test plot_pgv when pygraphviz is not available."""
    # Setup test data
    G = nx.DiGraph()
    G.add_edge("GeneA", "GeneB")
    G.add_edge("GeneB", "GeneC")
    G.add_edge("GeneA", "GeneC")

    gene_sets = {
        "type1": {"GeneA", "GeneD"},
        "type2": {"GeneB", "GeneE"},
        "type3": {"GeneC", "GeneF"},
        "deg": {"GeneA", "GeneC"},  # Should be skipped
    }

    # Patch pygraphviz to be None to simulate not installed
    with (
        patch.object(visualization, "pgv", None),
        patch.object(plt, "figure") as mock_figure,
        patch.object(nx, "spring_layout", return_value={}) as mock_layout,
        patch.object(nx, "draw_networkx_nodes") as mock_draw_nodes,
        patch.object(nx, "draw_networkx_edges") as mock_draw_edges,
        patch.object(nx, "draw_networkx_labels") as mock_draw_labels,
        patch.object(plt, "title") as mock_title,
        patch.object(plt, "legend") as mock_legend,
        patch.object(plt, "axis") as mock_axis,
        patch.object(plt, "tight_layout") as mock_tight_layout,
        patch.object(plt, "savefig") as mock_savefig,
        patch.object(plt, "close") as mock_close,
        patch("builtins.print") as mock_print,
    ):
        # Create a temporary directory for the output
        test_dir = "test_output"
        os.makedirs(test_dir, exist_ok=True)

        # Call the function
        visualization.plot_pgv(
            G=G,
            gene_sets=gene_sets,
            thre=3,
            path=test_dir,
            file_name="test_without_pgv",
        )

        # Verify that networkx plotting functions were called
        mock_print.assert_called_once_with(
            "pygraphviz is not installed, falling back to networkx drawing"
        )
        mock_figure.assert_called_once()
        mock_layout.assert_called_once_with(G)

        # Verify that draw_networkx_nodes was called for each gene type except 'deg'
        assert mock_draw_nodes.call_count == 3  # For 'type1', 'type2', 'type3'

        # Verify that other matplotlib functions were called
        mock_draw_edges.assert_called_once()
        mock_draw_labels.assert_called_once()
        mock_title.assert_called_once_with("Network Visualization")
        mock_legend.assert_called_once()
        mock_axis.assert_called_once_with("off")
        mock_tight_layout.assert_called_once()
        mock_savefig.assert_called_once_with(
            f"{test_dir}/test_without_pgv.png", dpi=300, bbox_inches="tight"
        )
        mock_close.assert_called_once()

        # Clean up
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


def test_plot_pgv_with_custom_dot():
    """Test plot_pgv when a custom DOT graph string is provided."""
    # Setup test data
    G = nx.DiGraph()
    gene_sets = {
        "type1": {"GeneA", "GeneD"},
        "type2": {"GeneB", "GeneE"},
    }

    custom_dot = "digraph G { /* custom DOT graph */ }"

    # Mock pygraphviz
    mock_pgv = MagicMock()
    mock_agraph = MagicMock()
    mock_pgv.agraph.AGraph.return_value = mock_agraph

    with (
        patch.object(visualization, "pgv", mock_pgv),
        patch.object(visualization, "pgv_dot") as mock_pgv_dot,
    ):
        # Create a temporary directory for the output
        test_dir = "test_output"
        os.makedirs(test_dir, exist_ok=True)

        # Call the function with custom DOT
        visualization.plot_pgv(
            G=G,
            gene_sets=gene_sets,
            G_dot=custom_dot,
            path=test_dir,
            file_name="test_custom_dot",
        )

        # Verify that the custom DOT was used instead of calling pgv_dot
        mock_pgv_dot.assert_not_called()
        mock_pgv.agraph.AGraph.assert_called_once_with(custom_dot)
        mock_agraph.draw.assert_called_once()

        # Clean up
        if os.path.exists(f"{test_dir}/test_custom_dot.png"):
            os.remove(f"{test_dir}/test_custom_dot.png")
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


def test_plot_pgv_creates_directory():
    """Test that plot_pgv creates the output directory if it doesn't exist."""
    # Setup test data
    G = nx.DiGraph()
    gene_sets = {"type1": {"GeneA"}}

    # Use a temporary directory that doesn't exist yet
    test_dir = "test_output_nonexistent"
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

    try:
        # Patch everything to avoid actual drawing
        with (
            patch.object(visualization, "pgv", None),
            patch.object(plt, "figure"),
            patch.object(nx, "spring_layout", return_value={}),
            patch.object(nx, "draw_networkx_nodes"),
            patch.object(nx, "draw_networkx_edges"),
            patch.object(nx, "draw_networkx_labels"),
            patch.object(plt, "savefig"),
            patch.object(plt, "close"),
        ):
            # Call the function
            visualization.plot_pgv(
                G=G, gene_sets=gene_sets, path=test_dir, file_name="test_dir_creation"
            )

            # Verify the directory was created
            assert os.path.exists(test_dir)

    finally:
        # Clean up
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
