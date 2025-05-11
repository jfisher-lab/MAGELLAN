from pathlib import Path

import networkx as nx


def graph_to_pydot(
    g: nx.DiGraph | nx.MultiDiGraph,
    out_path: str | Path,
    mut: list[str] | None = None,
    pheno: list[str] | None = None,
    deg: list[str] | None = None,
    format: str = "png",
):
    py_g = nx.drawing.nx_pydot.to_pydot(g)

    if mut or pheno or deg:
        for node in py_g.get_nodes():
            if mut and node.get_name() in mut:
                node.set_fillcolor("red")
                node.set_style("filled")
            elif pheno and node.get_name() in pheno:
                node.set_fillcolor("green")
                node.set_style("filled")
            elif deg and node.get_name() in deg:
                node.set_fillcolor("blue")
                node.set_style("filled")
            else:
                node.set_fillcolor("white")
    if isinstance(out_path, str):
        if format == "png":
            if out_path[-4:] != ".png":
                out_path += ".png"
            py_g.write_png(out_path)
        elif format == "svg":
            if out_path[-4:] != ".svg":
                out_path += ".svg"
            py_g.write_svg(out_path)
    elif isinstance(out_path, Path):
        if format == "png":
            out_path = out_path.with_suffix(".png")
            py_g.write_png(out_path)
        elif format == "svg":
            out_path = out_path.with_suffix(".svg")
            py_g.write_svg(out_path)

    return None
