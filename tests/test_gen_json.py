from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from magellan.json_io import gen_json, read_json


@pytest.fixture
def sample_graph():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign="Activator", edge_weight=0.5)
    G.add_edge("B", "C", sign="Inhibitor", edge_weight=0.8)
    return G


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "from (shortest path)": ["A", "B"],
            "to (shortest path)": ["B", "C"],
            "sign": ["Activator", "Inhibitor"],
            "edge_weight": [0.5, 0.8],
        }
    )


@pytest.fixture
def gene_sets():
    return {"mut": {"A"}, "pheno": {"C"}, "deg": {"B"}}


def test_gen_json_basic(tmp_path, sample_graph):
    output_file = tmp_path / "test_model.json"
    gen_json(sample_graph, tmp_path, "test_model")

    assert output_file.exists()
    json_data = read_json(output_file)
    assert json_data["Model"]["Name"] == "test_model"
    assert len(json_data["Model"]["Variables"]) == 3
    assert len(json_data["Model"]["Relationships"]) == 2


@pytest.mark.skip(reason="DataFrame input not yet implemented")
def test_gen_json_from_dataframe(tmp_path, sample_dataframe):
    output_file = tmp_path / "test_model_df.json"
    gen_json(sample_dataframe, tmp_path, "test_model_df")
    json_data = read_json(output_file)
    assert len(json_data["Model"]["Relationships"]) == 2


def test_gen_json_with_gene_sets(tmp_path, sample_graph, gene_sets):
    output_file = tmp_path / "test_model_colors.json"
    gen_json(sample_graph, tmp_path, "test_model_colors", gene_sets=gene_sets)

    json_data = read_json(output_file)
    variables = json_data["Layout"]["Variables"]
    color_assignments = {var["Name"]: var.get("Fill", None) for var in variables}
    assert color_assignments["A"] == "BMA_Green"
    assert color_assignments["B"] == "BMA_Orange"
    assert color_assignments["C"] == "BMA_Purple"


def test_gen_json_with_custom_ranges(tmp_path, sample_graph):
    range_dic = {"A": (0, 3), "B": (1, 4), "C": (0, 2)}
    output_file = tmp_path / "test_model_ranges.json"
    gen_json(sample_graph, tmp_path, "test_model_ranges", range_dic=range_dic)

    json_data = read_json(output_file)
    variables = {
        var["Name"]: (var["RangeFrom"], var["RangeTo"])
        for var in json_data["Model"]["Variables"]
    }

    assert variables["A"] == (0, 3)
    assert variables["B"] == (1, 4)
    assert variables["C"] == (0, 2)


def test_gen_json_with_constant_nodes(tmp_path, sample_graph):
    const_dic = {"A": 1}
    output_file = tmp_path / "test_model_const.json"
    gen_json(sample_graph, tmp_path, "test_model_const", const_dic=const_dic)

    json_data = read_json(output_file)
    variables = {var["Name"]: var["Formula"] for var in json_data["Model"]["Variables"]}
    assert variables["A"] == "1"


def test_gen_json_invalid_input():
    with pytest.raises(TypeError, match="G must be a directed graph"):
        gen_json("invalid", Path("test"), "test_model") # type: ignore


def test_gen_json_invalid_func_type(tmp_path, sample_graph):
    with pytest.raises(
        NotImplementedError,
        match="func_type must be 'weighted_default', 'default', or None for now, invalid passed",
    ):
        gen_json(sample_graph, tmp_path, "test_model", func_type="invalid")


def test_gen_json_custom_scale(tmp_path, sample_graph):
    output_file = tmp_path / "test_model_scale.json"
    gen_json(sample_graph, tmp_path, "test_model_scale", scale=5.0)

    json_data = read_json(output_file)
    pos_values = [
        (var["PositionX"], var["PositionY"]) for var in json_data["Layout"]["Variables"]
    ]
    assert any(abs(x) > 2 or abs(y) > 2 for x, y in pos_values)
