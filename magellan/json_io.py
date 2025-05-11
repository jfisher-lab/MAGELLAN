import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from magellan.utils.file_io import read_json, save_json


def nested_max(id_list):
    """
    Return nested max as BMA only processes max with TWO elements in the function

    :param id_list: list, each element is an integer, corresponding to node ID
    :return: str, function in the format max(var[-1], max(var[-2], max(...)))

    """

    if len(id_list) > 2:
        return "max(var(%s), %s)" % (id_list[0], nested_max(id_list[1:]))
    else:
        return "max(var(%s), var(%s))" % (id_list[0], id_list[1])


def get_varList(id_list):
    return ["var(%d)" % i for i in id_list]


def get_sum(id_list):
    return "(%s)" % ("+".join(get_varList(id_list)))


def get_prod(id_list, plus=1):
    var_list = get_varList(id_list)
    var_list = ["(%s+%d)" % (i, plus) for i in var_list]

    return "(%s)" % ("*".join(var_list))


def weighted_avg(id_list, weight_list):
    """
    Return weighted average

    :param id_list: list, each element is an integer, corresponding to node ID
    :param weight_list: list, has the same length as id_list, each element is a float, corresponding to a node weight

    :return: str, function of weighted average in the format avg(var(x1)*w1, var(x2)*w2, ...)

    """

    var_list = get_varList(id_list)
    nnz_weight = np.sum(
        np.abs(np.asarray(weight_list)) >= 10**-3
    )  # only count nonzero weights (|w| >= 10**-3)
    var_list = [
        "%s*%d/%d"
        % (
            var_list[i],
            int(weight_list[i] * 10**3),
            (nnz_weight * 10**3),
        )  # convert fraction
        for i in range(len(var_list))
    ]

    # var_str = ','.join(var_list)
    # return 'avg(%s)' % var_str

    var_str = "+".join(var_list)

    return var_str


def get_avg(id_list, weight_list):
    """
    Return average over a list of variables

    :param id_list: list, each element is an integer, corresponding to node ID
    :param weight_list: False or list. If list, each element us a float, correponding to an edge weight

    :return: str, function in the format avg(var[0], var[1], ...)

    """

    if len(id_list) == 1:  # not applicable to 'prod'
        if weight_list:
            return "var(%d)*%d/%d" % (id_list[0], weight_list[0] * 10**3, 10**3)
        else:
            return "var(%d)" % id_list[0]

    if weight_list:  # weighted average
        return weighted_avg(id_list, weight_list)
    else:
        var_str = ",".join(get_varList(id_list))
        return "avg(%s)" % var_str


def get_max(var, const=1):
    """
    Return a max function between var and const such as max(const, var)

    :param var: str, variables to be maximised over
    :param const: int, constant to be maximised over
    :return: str, function in the format max(const, var)

    """

    return "max(%s, %d)" % (var, const)


def extract_node(n: str, G: nx.DiGraph, sign_type: str) -> list[str]:
    """
    Extract parent nodes of node n by sign type

    :param n: str, target node name
    :param G: networkx.DiGraph
    :param sign_type: str, sign type, either 'Activator' or 'Inhibitor'
    :return: list, elements are parent nodes of node n with sign_type

    """

    if isinstance(G, nx.MultiDiGraph):
        return [
            n_s
            for n_s in G.predecessors(n)
            for i in G[n_s][n]
            if G[n_s][n][i]["sign"] == sign_type
        ]
    elif isinstance(G, nx.DiGraph):
        return [n_s for n_s in G.predecessors(n) if G[n_s][n]["sign"] == sign_type]


def define_func(node_list, sign_type, id_node, func_type, parent_weight):
    """
    Define target functions

    :param node_list: list, each element is a numeric id of a node
    :param sign_type: str, 'Inhibitor' or 'Activator'
    :param id_node: dict, key: node name, value: node id
    :param func_type: str, type of target func
    :param parent_weight: False or dict. key: parent nodes, value: edge weight

    :return: str, target func

    """

    id_list = [id_node[i] for i in node_list]

    if parent_weight:  # weighted avg
        weight_list = [parent_weight[i] for i in node_list]
    else:
        weight_list = False

    # no target function if a node has no parents
    if len(id_list) == 0:
        return ""

    # avg_func and max_func for convenience
    if len(id_list) == 1 and func_type != "prod" and func_type != "weighted_default":
        func = "var(%d)" % id_list[0]
    else:
        func = get_avg(
            id_list, weight_list
        )  # length > 1 | func_type = prod | func_type = weighted_default

        if "max" in func_type:
            if "avg" not in func_type:
                func = nested_max(id_list)

        elif func_type == "prod":
            func = get_prod(id_list, 1)

        elif func_type == "sum_avg":
            func = get_sum(id_list)

    # add minus (-) or divide (/) to inhibitor part
    if sign_type == "Inhibitor":
        if "max" in func_type:
            func = get_max(func, 1)

        if "max" in func_type or func_type == "prod":
            func = "/%s" % func
        else:
            func = "-%s" % func
            if func_type == "weighted_default":
                func = func.replace("+", "-")  # remove brackets

    return func


def get_func(n, id_node, max_range, G, func_type, edge_weight, const_dic):
    """
    Return target functions at each node by function type

    default: avg(p) - avg(n)
    sum_avg: (sum(p) - sum(n)) / n_p
    max: max(p) / max(1, n)
    floor_max: floor(max function)
    avg_max: avg(p) / max(1, avg(n))
    prod: product(1 + p) / product(1 + n)
    default_1: 1 + default

    where p = activators, and n = inhibitors

    :param n: str, node name
    :param id_node: dict, key: node name, value: node id
    :param max_range: int, maximum range of a node
    :param G: networkx.DiGraph
    :param func_type: str, type of target function
    :param edge_weight: False or nested dict (defaultdict(dict)).
           1st order key: a node, 2nd order key: parent node of 1st order key, value: edge weight
    :param const_dic: dict, key:

    :return: str, target function in BMA processable format

    """

    if const_dic is not None and n in const_dic:
        return str(const_dic[n])

    # activators
    node_pos = extract_node(n, G, "Activator")
    n_pos = len(node_pos)  # number of activators

    # inhibitors
    node_neg = extract_node(n, G, "Inhibitor")
    n_neg = len(node_neg)

    # return empty string '' if a node has no incoming edges
    if n_pos + n_neg == 0:  # length are non-negative, so if sum is 0 then both is 0
        return ""

    # parent node
    if edge_weight:  # weighted avg
        parent_node = edge_weight[n]
    else:
        parent_node = False

    # get target functions
    func_act = define_func(
        node_pos, "Activator", id_node, func_type, parent_node
    )  # activators
    func_inh = define_func(
        node_neg, "Inhibitor", id_node, func_type, parent_node
    )  # inhibitors

    if func_act == "":
        func_act = (
            "%d" % max_range
        )  # replace activator func with max_range when all parent nodes are inhibitors

    func = "%s%s" % (func_act, func_inh)

    # process function by type
    if func_type == "sum_avg":
        func = "(%s)/%d" % (func, n_pos)

    elif func_type == "floor_max":
        func = "floor(%s)" % func

    elif func_type == "default_1":
        func = "1+%s" % func

    if func_type == "default" and n_pos > 0:
        func = ""

    # replace /0 or /1 in denominator
    func = func.replace("/0", "")
    if func_type != "weighted_default":
        func = func.replace("/1", "")

    return func


def create_graph(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    sign_col: str,
) -> nx.MultiDiGraph:
    """
    Create networkx.MultiDiGraph for further use

    :param df: pandas.DataFrame
    :param source_col: str, column name of source node
    :param target_col: str, column name of target node
    :param sign_col: str, column name of sign, default: 'sign'.
           note that sign column takes value from {'Activator', 'Inhibitor', 'Both', 'Neither'},
           'Neither' signs will be converted to 'Activator'

    :return: networkx.MultiDiGraph constructed from input df
    """

    # process ambiguous direction/sign

    # replace 'Both' with activator and inhibitor
    df.loc[df[sign_col] == "Both", sign_col] = "Activator, Inhibitor"
    # df[sign_col] = df[sign_col].str.split(', ')
    df[sign_col] = df[sign_col].apply(
        lambda x: x.split(", ") if "," in x else x
    )  # Fix: if there is only one sign,
    # then the split would return NaN and ths break produced JSON files
    df = df.explode(sign_col)

    # replace 'neither' with activator
    df.loc[df[sign_col] == "Neither", sign_col] = "Activator"

    # replace 'undirected' with ??
    # currently use remove undirected version

    G = nx.MultiDiGraph()
    G.add_weighted_edges_from(
        zip(df[source_col], df[target_col], df[sign_col]), weight="sign"
    )

    return G


def _gen_json(
    G: nx.DiGraph | nx.MultiDiGraph,
    model_name: str,
    scale: float,
    min_range: int,
    max_range: int,
    func_type: str | None,
    gene_sets: dict | None,
    colour_dic: dict | None,
    pos: dict | None,
    const_dic: dict | None,
    range_dic: dict | None,
) -> dict:
    """
    Generate a .json file for BMA from a graph

    :param G: networkx.DiGraph or networkx.MultiDiGraph
    :param model_name: str. model name
    :param scale: float, default 2. the scale to zoom in/out the distances between nodes in the generated network
    :param min_range: int, default 0. minimum range of a node
    :param max_range: int, default 2. maximum range of a node
    :param func_type: weighted_default or None. Former simply transfers weights from G with weighting handled elsewhere, whereas None does not set edge_weights.
    :param gene_sets: dict, key: str, gene category (mut/deg/pheno), value: set of genes under corresponding category
    :param colour_dic: dict, key: str, gene categoty (same as gene_sets), value: str, colour.
           coloring scheme of genes so that genes in the same category have the same colour
    :param pos: dict, key: str, node name, value: tuple, (coordinate x, coordinate y) in BMA
           (equivalent to (['PositionX'], ['PositionY']) in a .json file).
           predefined node coordinates in BMA, can be extracted from existing .json file with function
           get_pos_from_json(.) below
    """

    # get graph attribute
    if not pos:
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(G)  # coordinates
            pos_alg = "pygrahviz"
        except (ValueError, ImportError):
            pos = dict(
                nx.kamada_kawai_layout(G)
            )  # skip pygrahviz layout when graphviz/pygrahviz is not installed
            scale *= 200
            pos_alg = "networkX.kamada_kawai_layout"

        print("node position optimised by: %s" % pos_alg)

    else:
        pos_alg = "pgv"

    # assign id to node and edge
    id_node = dict(zip(G.nodes(), range(1, G.number_of_nodes() + 1)))

    edge_attr = nx.get_edge_attributes(G, "sign")
    edge_attr = [tuple(list(k[:2]) + [v]) for k, v in edge_attr.items()]
    id_edge = dict(zip(edge_attr, range(1, len(edge_attr) + 1)))

    # get edge weight
    if func_type == "weighted_default":
        # edge_weight = nx.get_edge_attributes(G, 'edge_weight')

        # 1st order key: node, 2nd order key: parent node of a node
        edge_weight = defaultdict(dict)
        for v in G.nodes:
            for u in G.predecessors(v):
                edge_weight[v][u] = G[u][v]["edge_weight"]

    else:
        edge_weight = False

    # format for bma

    # Model, Name
    bma = defaultdict(dict)
    bma["Model"]["Name"] = model_name

    # Model, Variables
    if range_dic is None:
        bma["Model"]["Variables"] = [
            {
                "Name": n,
                "Id": id_node[n],
                "RangeFrom": min_range,
                "RangeTo": max_range,
                "Formula": get_func(
                    n, id_node, max_range, G, func_type, edge_weight, const_dic
                ),
            }
            for n in G.nodes()
        ]
    else:
        bma["Model"]["Variables"] = [
            {
                "Name": n,
                "Id": id_node[n],
                "RangeFrom": range_dic[n][0],
                "RangeTo": range_dic[n][1],
                "Formula": get_func(
                    n, id_node, max_range, G, func_type, edge_weight, const_dic
                ),
            }
            for n in G.nodes()
        ]

    # Model, Relationships
    if isinstance(G, nx.MultiDiGraph):
        bma["Model"]["Relationships"] = [
            {
                "Id": id_edge[(u, v, G[u][v][i]["sign"])],
                "FromVariable": id_node[u],
                "ToVariable": id_node[v],
                "Type": G[u][v][i]["sign"],
            }
            for u, v in set(G.edges())
            for i in G[u][v]
        ]
    elif isinstance(G, nx.DiGraph):
        bma["Model"]["Relationships"] = [
            {
                "Id": id_edge[(u, v, G[u][v]["sign"])],
                "FromVariable": id_node[u],
                "ToVariable": id_node[v],
                "Type": G[u][v]["sign"],
            }
            for u, v in set(G.edges())
        ]

    # Layout, Variables
    bma["Layout"]["Variables"] = [
        {
            "Id": id_node[n],
            "Name": n,
            "Type": "Default",
            "ContainerId": 1,
            "PositionX": pos[n][0] * scale,  # type: ignore
            "PositionY": pos[n][1] * scale,  # type: ignore
            "CellX": None,
            "CellY": None,
            "Angle": 0,
            "Description": "",
        }
        for n in G.nodes()
    ]

    # add colour
    if gene_sets:
        gene_sets = {vv: k for k, v in gene_sets.items() for vv in v}
        node_colour = {
            n: "BMA_%s" % colour_dic[gene_sets[n]].capitalize()  # type: ignore
            for n in G.nodes()
            if n in gene_sets
        }  # type: ignore
        for ele in bma["Layout"]["Variables"]:
            if ele["Name"] in node_colour:
                ele["Fill"] = node_colour[ele["Name"]]

    # Layout, Containers
    if pos_alg == "pgv":
        bma["Layout"]["Containers"] = [
            {"Id": 1, "Name": "C0", "Size": 3.8, "PositionX": -0.1, "PositionY": -0.5}
        ]
    else:
        bma["Layout"]["Containers"] = [
            {"Id": 1, "Name": "C0", "Size": 4, "PositionX": -2, "PositionY": -2}
        ]

    # Layout, AnnotatedGridCells
    bma["Layout"]["AnnotatedGridCells"] = []

    # Layout, Description
    bma["Layout"]["Description"] = ""

    # ltl, states
    bma["ltl"]["states"] = []

    # ltl, operations
    bma["ltl"]["operations"] = []

    return bma


def gen_json(
    G: nx.DiGraph | nx.MultiDiGraph,
    path: str | Path,
    model_name: str,
    scale: float = 2,
    min_range: int = 0,
    max_range: int = 2,
    func_type: str | None = "weighted_default",
    gene_sets: dict | None = None,
    colour_dic: dict | None = None,
    pos: dict | None = None,
    source_col: str = "from (shortest path)",
    target_col: str = "to (shortest path)",
    sign_col: str = "sign",
    const_dic: dict | None = None,
    range_dic: dict | None = None,
):
    """
    Generate a .json file for BMA from a graph or a table

    :param G: networkx.DiGraph or networkx.MultiDiGraph or pandas.DataFrame. network to convert to .json
    :param path: str. output directory
    :param model_name: str. model name
    :param scale: float, default 2. the scale to zoom in/out the distances between nodes in the generated network
    :param min_range: int, default 0. minimum range of a node
    :param max_range: int, default 2. maximum range of a node

    :param func_type: str. type of target function (uniform across all nodes)
           default: avg(p) - avg(n)
           sum_avg: (sum(p) - sum(n)) / n_p
           max: max(p) / max(1, n)
           floor_max: floor(max function)
           avg_max: avg(p) / max(1, avg(n))
           prod: product(1 + p) / product(1 + n)
           default_1: 1 + default
           weighted_default: same as default, but use weighted sum.
           weights are pre-defined float by edge e.g. G[u][v]['edge_weight'] = float
           where p = activators, and n = inhibitors

    :param gene_sets: dict, key: str, gene category (mut/deg/pheno), value: set of genes under corresponding category
    :param colour_dic: dict, key: str, gene categoty (same as gene_sets), value: str, colour.
           coloring scheme of genes so that genes in the same category have the same colour

    :param pos: dict, key: str, node name, value: tuple, (coordinate x, coordinate y) in BMA
           (equivalent to (['PositionX'], ['PositionY']) in a .json file).
           predefined node coordinates in BMA, can be extracted from existing .json file with function
           get_pos_from_json(.) below

    :param source_col: str, column name of source node
    :param target_col: str, column name of target node
    :param sign_col: str, column name of sign, default: 'sign'.
           note that sign column takes value from {'Activator', 'Inhibitor', 'Both', 'Neither'},
           'Neither' signs will be converted to 'Activator'

    :param const_dic: dict, key: str, node name, value: int, constant nodes in BMA
           equivalent to (constant) 'Formula' in a .json file

    :param range_dic: dict, key: str, node name, value: tuple (int: lower bound, int: upper bound) in BMA
           equivalent to ('RangeFrom', 'RangeTo') in a .json file

    """

    if isinstance(G, pd.DataFrame):
        raise NotImplementedError("DataFrame input not yet implemented")
        G = create_graph(G, source_col, target_col, sign_col)

    if not any((isinstance(G, nx.MultiDiGraph), isinstance(G, nx.DiGraph))):
        raise TypeError("G must be a directed graph, %s passed" % type(G))

    if gene_sets and not colour_dic:
        colour_dic = {"mut": "Green", "pheno": "Purple", "deg": "Orange"}

    if func_type not in ("weighted_default", "default", None):
        raise NotImplementedError(
            "func_type must be 'weighted_default', 'default', or None for now, %s passed"
            % func_type
        )

    bma = _gen_json(
        G,
        model_name,
        scale,
        min_range,
        max_range,
        func_type,
        gene_sets,
        colour_dic,
        pos,
        const_dic,
        range_dic,
    )

    # save result
    save_json(os.path.join(path, model_name), bma)
    print(os.path.join(path, model_name))


def get_pos_from_json(js: str | Path) -> dict:
    """
    Obtain pre-defined node positions from a .json file

    :param js: json object OR directory of .json file
    :return: dict of positions from a .json file

    """

    if js is None:
        raise ValueError("js is None")

    if isinstance(js, str):
        js_out: dict = read_json(js)
    elif isinstance(js, Path):
        js_out: dict = read_json(js)

    return {
        ele["Name"]: (ele["PositionX"], ele["PositionY"])
        for ele in js_out["Layout"]["Variables"]
    }


def json_to_graph(json_path: str | Path) -> nx.DiGraph:
    """
    Convert json to a networkX directed graph

    :param js: json object OR directory of .json file
    :return: networkX.DiGraph, a directed graph constructed from a .json file/json object

    """

    if isinstance(json_path, str | Path):
        js: dict = read_json(json_path)

    # node and edge dict from json file
    var_dic: dict = {ele["Id"]: ele["Name"] for ele in js["Model"]["Variables"]}
    # check for duplicates in var_dic
    duplicates = [
        node
        for node in list(var_dic.values())
        if list(var_dic.values()).count(node) > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate nodes found in json file: {duplicates}")
    edge_dic: dict = {
        (var_dic[ele["FromVariable"]], var_dic[ele["ToVariable"]]): {
            "sign": ele["Type"]
        }
        for ele in js["Model"]["Relationships"]
    }

    # add edge and attributes to graph
    G: nx.DiGraph = nx.DiGraph()
    G.add_edges_from(edge_dic.keys())
    nx.set_edge_attributes(G, values=edge_dic)

    return G


def get_range_from_json(js):
    """
    Obtain node value ranges from json object or .json file

    :param js: json object OR directory of .json file
    :return: dict, key: node name, value: tuple (lower bound, upper bound)

    """

    if isinstance(js, str):
        js = read_json(js)

    return {
        ele["Name"]: (ele["RangeFrom"], ele["RangeTo"])
        for ele in js["Model"]["Variables"]
    }


def get_const_from_json(json_path: str | Path) -> dict:
    """
    Obtain constant nodes from json object or .json file

    :param js: json object OR directory of .json file
    :return: dict, key: node name, value: int (value if a node is a constant in an existing network)

    """

    if isinstance(json_path, str | Path):
        js: dict = read_json(json_path)

    const_dic: dict = {ele["Name"]: ele["Formula"] for ele in js["Model"]["Variables"]}
    for k, v in const_dic.items():
        try:
            const_dic[k] = int(v)
        except ValueError:  # Catches failed int conversions
            continue
    const_dic = {k: v for k, v in const_dic.items() if isinstance(v, int)}

    return const_dic
