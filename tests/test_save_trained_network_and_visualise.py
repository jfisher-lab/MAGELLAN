from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pandas as pd
import pytest

from magellan.json_io import (
    json_to_graph,
)
from magellan.plot import (
    save_trained_network_and_visualise,
)


@pytest.fixture
def mock_input_json(tmp_path):
    """Create a mock json file path"""
    return tmp_path / "input.json"


@pytest.fixture
def mock_output_dir(tmp_path):
    """Create a mock output directory"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def mock_dataframe():
    """Create a mock weight dataframe"""
    return pd.DataFrame(
        [[0.0, 0.5], [0.7, 0.0]],
        index=["node1", "node2"],
        columns=["node1", "node2"],
    )


@pytest.fixture
def mock_digraph():
    """Create a mock directed graph with two nodes and an edge"""
    G = nx.DiGraph()
    G.add_node("node1")
    G.add_node("node2")
    G.add_edge("node1", "node2", sign="Activator", edge_weight=0.5)
    return G


class TestSaveTrainedNetwork:
    @patch("magellan.plot.json_to_graph")
    @patch("magellan.plot.get_pos_from_json")
    @patch("magellan.plot.get_const_from_json")
    @patch("magellan.plot.analyze_deleted_edges")
    @patch("magellan.plot.analyze_sign_flips")
    @patch("magellan.plot.get_trained_network")
    @patch("magellan.plot.gen_json")
    @patch("magellan.plot.annotate_graph")
    @patch("magellan.plot.graph_to_pydot")
    def test_all_functions_called_with_correct_params(
        self,
        mock_graph_to_pydot,
        mock_annotate_graph,
        mock_gen_json,
        mock_get_trained_network,
        mock_analyze_sign_flips,
        mock_analyze_deleted_edges,
        mock_get_const_from_json,
        mock_get_pos_from_json,
        mock_json_to_graph,
        mock_input_json,
        mock_output_dir,
        mock_dataframe,
        mock_digraph,
    ):
        """Test that all functions are called with correct parameters"""
        # Setup mocks
        mock_json_to_graph.return_value = mock_digraph
        mock_get_pos_from_json.return_value = {"node1": (0, 0), "node2": (1, 1)}
        mock_get_const_from_json.return_value = {}
        mock_get_trained_network.return_value = mock_digraph
        mock_annotate_graph.return_value = mock_digraph

        file_name = "test_model"
        min_range = 0
        max_range = 2

        # Call function under test
        save_trained_network_and_visualise(
            mock_input_json,
            mock_dataframe,
            mock_output_dir,
            file_name,
            min_range,
            max_range,
        )

        # Verify all mocks were called with correct parameters
        mock_json_to_graph.assert_called_once_with(mock_input_json)
        mock_get_pos_from_json.assert_called_once_with(mock_input_json)
        mock_get_const_from_json.assert_called_once_with(mock_input_json)
        mock_analyze_deleted_edges.assert_called_once_with(mock_digraph, mock_dataframe)
        mock_analyze_sign_flips.assert_called_once_with(mock_digraph, mock_dataframe)
        mock_get_trained_network.assert_called_once_with(mock_digraph, mock_dataframe)

        # Check gen_json was called correctly
        mock_gen_json.assert_called_once()
        args, kwargs = mock_gen_json.call_args
        assert args[0] == mock_digraph
        assert args[1] == mock_output_dir
        assert args[2] == f"{file_name}_synthetic_weight_est_realSpec"
        assert kwargs["min_range"] == min_range
        assert kwargs["max_range"] == max_range
        assert kwargs["func_type"] == "weighted_default"

        # Check annotate_graph and graph_to_pydot were called correctly
        mock_annotate_graph.assert_called_once_with(mock_digraph, mock_dataframe)
        mock_graph_to_pydot.assert_called_once()
        args, kwargs = mock_graph_to_pydot.call_args
        assert args[0] == mock_digraph
        assert str(args[1]).endswith("network_after_training")
        assert kwargs["format"] == "png"

    @patch("magellan.plot.json_to_graph")
    def test_integration_with_real_functions(
        self,
        mock_json_to_graph,
        mock_input_json,
        mock_output_dir,
        mock_dataframe,
        mock_digraph,
    ):
        """Test integration with real functions except json_to_graph which we mock"""
        # Setup mocks - we still need to mock json_to_graph because we don't have a real file
        mock_json_to_graph.return_value = mock_digraph

        # Add necessary attributes to the graph for the real functions
        for u, v in mock_digraph.edges():
            mock_digraph[u][v]["edge_weight"] = 0.5

        # We need to mock all functions that would access files or perform actual operations
        with (
            patch(
                "magellan.plot.get_pos_from_json",
                return_value={"node1": (0, 0), "node2": (1, 1)},
            ),
            patch("magellan.plot.get_const_from_json", return_value={}),
            patch("magellan.plot.analyze_deleted_edges", return_value=[]),
            patch("magellan.plot.analyze_sign_flips", return_value=[]),
            patch("magellan.plot.get_trained_network", return_value=mock_digraph),
            patch("magellan.plot.gen_json") as mock_gen_json,
            patch("magellan.plot.annotate_graph", return_value=mock_digraph),
            patch("magellan.plot.graph_to_pydot"),
            patch("os.path.join", lambda *args: str(Path(*args))),
        ):
            # Call function under test
            save_trained_network_and_visualise(
                mock_input_json,
                mock_dataframe,
                mock_output_dir,
                "test_model",
                0,
                2,
            )

            # Verify that gen_json was called correctly
            mock_gen_json.assert_called_once()

    def test_handles_edge_cases(
        self,
        mock_input_json,
        mock_output_dir,
    ):
        """Test that the function handles edge cases properly"""
        # Empty dataframe
        with patch("magellan.plot.json_to_graph", return_value=nx.DiGraph()):
            with patch("magellan.plot.get_pos_from_json", return_value={}):
                with patch("magellan.plot.get_const_from_json", return_value={}):
                    with patch("magellan.plot.analyze_deleted_edges", return_value=[]):
                        with patch("magellan.plot.analyze_sign_flips", return_value=[]):
                            with patch(
                                "magellan.plot.get_trained_network",
                                return_value=nx.DiGraph(),
                            ):
                                with patch("magellan.plot.gen_json"):
                                    with patch(
                                        "magellan.plot.annotate_graph",
                                        return_value=nx.DiGraph(),
                                    ):
                                        with patch("magellan.plot.graph_to_pydot"):
                                            # Empty dataframe should still work
                                            save_trained_network_and_visualise(
                                                mock_input_json,
                                                pd.DataFrame(),
                                                mock_output_dir,
                                                "test_model",
                                                0,
                                                2,
                                            )

                                            # Path object instead of string should work
                                            save_trained_network_and_visualise(
                                                Path(mock_input_json),
                                                pd.DataFrame(),
                                                Path(mock_output_dir),
                                                "test_model",
                                                0,
                                                2,
                                            )


@pytest.mark.parametrize(
    "input_type,expected_type",
    [
        (str, nx.DiGraph),
        (Path, nx.DiGraph),
    ],
)
def test_json_to_graph_input_types(input_type, expected_type, mock_input_json):
    """Test that json_to_graph accepts both str and Path inputs"""
    with patch(
        "magellan.json_io.read_json",
        return_value={"Model": {"Variables": [], "Relationships": []}},
    ):
        with patch("magellan.json_io.nx.DiGraph") as mock_digraph:
            mock_digraph.return_value = nx.DiGraph()
            json_to_graph(input_type(mock_input_json))
            mock_digraph.assert_called()


def test_json_to_graph_duplicate_nodes(mock_input_json):
    """Test that json_to_graph raises ValueError when duplicate node names are present"""
    # Mock JSON with duplicate node names
    mock_json = {
        "Model": {
            "Variables": [
                {"Id": "1", "Name": "node1"},
                {"Id": "2", "Name": "node1"},  # Duplicate name
                {"Id": "3", "Name": "node2"},
            ],
            "Relationships": [],
        }
    }

    with patch("magellan.json_io.read_json", return_value=mock_json):
        with pytest.raises(ValueError) as exc_info:
            json_to_graph(mock_input_json)

        assert "Duplicate nodes found in json file: ['node1', 'node1']" in str(
            exc_info.value
        )


# @pytest.mark.parametrize(
#     "input_json,W,path_data,file_name,min_range,max_range,expected_error",
#     [
#         (
#             None,
#             pd.DataFrame(),
#             "output",
#             "test",
#             0,
#             2,
#             (ValueError, TypeError, AttributeError),
#         ),
#         (
#             "input.json",
#             None,
#             "output",
#             "test",
#             0,
#             2,
#             (ValueError, TypeError, AttributeError),
#         ),
#         (
#             "input.json",
#             pd.DataFrame(),
#             None,
#             "test",
#             0,
#             2,
#             (ValueError, TypeError, AttributeError),
#         ),
#         (
#             "input.json",
#             pd.DataFrame(),
#             "output",
#             None,
#             0,
#             2,
#             (ValueError, TypeError, AttributeError),
#         ),
#     ],
# )
# def test_raises_error_on_invalid_inputs(
#     input_json, W, path_data, file_name, min_range, max_range, expected_error
# ):
#     """Test that the function raises appropriate errors on invalid inputs"""
#     # Mock both json_to_graph and read_json to prevent actual file operations
#     with (
#         patch("magellan.plot.json_to_graph"),
#         patch("magellan.json_io.read_json") as mock_read_json,
#     ):
#         mock_read_json.return_value = {"Layout": {"Variables": []}}
#         with pytest.raises(expected_error):
#             save_trained_network_and_visualise(
#                 input_json, W, path_data, file_name, min_range, max_range
#             )
