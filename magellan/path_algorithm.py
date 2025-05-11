import re
from ast import literal_eval
from itertools import chain, product, zip_longest

import numpy as np
import pandas as pd

from magellan.graph_algorithm import get_path


def get_direction(df):
    """
    Assign a single column of direction type: Undirected, Both, Neither, Activator, Inhibitor

    :param df: pandas.DataFrame, Omnipath database
    :return df: pandas.DataFrame, Omnipath database with an additional 'sign' column of direction types
    """

    # # convert index to multi index
    # df.index = pd.MultiIndex.from_tuples(df.index)

    # convert index to numbers
    df.index = range(df.shape[0])

    df["sign"] = ""

    for idx in df.index:
        if not df.at[idx, "is_directed"]:
            direct = "Undirected"
        elif df.at[idx, "is_stimulation"] and df.at[idx, "is_inhibition"]:
            direct = "Both"
        elif (not df.at[idx, "is_stimulation"]) and (not df.at[idx, "is_inhibition"]):
            direct = "Neither"
        elif df.at[idx, "is_stimulation"]:
            direct = "Activator"
        elif df.at[idx, "is_inhibition"]:
            direct = "Inhibitor"

        df.at[idx, "sign"] = direct

    return df


def get_comb(gene_sets, to_combine):
    """
    Gene pairs from gene combinations

    :param gene_sets: dict, key: gene set types, value: set of genes of the corresponding type
    :param to_combine: list or str. Gene types to combine.
             When list, combine gene types in the list. When str, combine genes from the same type

    :return gene_comb: set of gene pair tuples. Genes in each pair will be the start and end of a shortest path

    """

    if isinstance(to_combine, list):
        if not isinstance(to_combine[0], list):
            to_combine = [to_combine]

        gene_comb = [
            list(product(gene_sets[ele[0]], gene_sets[ele[1]])) for ele in to_combine
        ]
        gene_comb = set(chain(*gene_comb))

    elif isinstance(to_combine, str):
        gene_comb = product(gene_sets[to_combine], gene_sets[to_combine])

    else:
        raise TypeError("to_combine must be either a list or a string")

    gene_comb = {ele for ele in gene_comb if ele[0] != ele[1]}

    return gene_comb


#################
# style
def _get_edge_style(sign):
    dic_style = {
        "Activator": "-->",
        "Inhibitor": "--|",
        "Both": "-->|",
        "Neither": "==>",
        "Undirected": "--",
    }

    return [dic_style[ele] for ele in sign]


def _get_path_style(path, sign):
    style = _get_edge_style(sign)
    return "".join(list(chain(*zip_longest(path, style, fillvalue=""))))


def _get_style(path_list, sign_list):
    path_with_sign = ""
    if isinstance(path_list, list):
        path_with_sign = []
        for counter in range(len(path_list)):
            path_with_sign.append(
                _get_path_style(path_list[counter], sign_list[counter])
            )

        path_with_sign = ", ".join(path_with_sign)

    return path_with_sign


def get_sign_from_grpah(G, path):
    sign = []
    for counter in range(len(path) - 1):
        sign.append(G[path[counter]][path[counter + 1]]["sign"])

    return sign


def get_path_style_from_graph(G, path_list):
    sign_list = []
    for p in path_list:
        sign = get_sign_from_grpah(G, p)
        sign_list.append(sign)

    return _get_style(path_list, sign_list)


def literal_convert(x):
    try:
        return literal_eval(x)
    except (SyntaxError, ValueError):
        return x


def get_style(df):
    df[["shortest path", "sign"]] = df[["shortest path", "sign"]].fillna(value="")
    df["sign"] = df["sign"].apply(literal_convert)
    df["shortest path"] = df["shortest path"].apply(literal_convert)

    df["shortest path style"] = ""
    for idx in df.index:
        path_list = df.at[idx, "shortest path"]
        sign_list = df.at[idx, "sign"]
        path_with_style = _get_style(path_list, sign_list)

        df.at[idx, "shortest path style"] = path_with_style

    return df


#############
# shortest path


def assign_path(df, G, gene_comb, weighted_edge=None, filter_gene=None, thre=np.inf):
    for u, v in gene_comb:
        # list of shortest paths from u to v
        path_dic = get_path(G, u, v, weighted_edge, thre)

        if isinstance(path_dic, dict):
            # filter by selected genes:
            # if there exists paths from u to v that consists of only selected genes
            # then discard other paths
            if filter_gene:  # should return True whenever filter_gene is NOT False/None
                path_dic_filter = {
                    k: v
                    for k, v in path_dic.items()
                    if len(set(k) - set(filter_gene)) == 0
                }

                if len(path_dic_filter) > 0:
                    path_dic = path_dic_filter
                    del path_dic_filter

            path_list, sign_list = list(path_dic.keys()), list(path_dic.values())

        else:
            path_list, sign_list = path_dic

        # assign results to df
        df.loc[u, v][["from", "to"]] = [u, v]

        if len(path_list) == 0:  # no paths found
            n_path, avg_path = 0, ""
        else:
            n_path, avg_path = len(path_list), np.mean([len(k) for k in path_list])

        df.loc[u, v][["# path", "avg len"]] = [n_path, avg_path]
        df.at[(u, v), "shortest path"] = path_list

        df.at[(u, v), "sign"] = sign_list

    return df


def find_path(G, gene_sets, to_combine, weighted_edge=None, filter_gene=None, thre=2):
    """
    Find shortest paths between gene combinations on a graph

    :param G: BMATool.graph.Graph, node: gene, edge: interaction between genes in Omnipath
    :param gene_sets: dict, key: gene set types, value: set of genes of the corresponding type

    :return df: pandas.DataFrame, shortest paths results

    """

    # gene combinations
    # inter-type

    gene_comb_inter = get_comb(gene_sets, to_combine)

    # intra-type
    gene_comb_intra = get_comb(gene_sets, "mut")
    gene_comb_intra = gene_comb_intra.union(get_comb(gene_sets, "pheno"))
    gene_comb_intra -= gene_comb_inter

    # create dataframe
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(gene_comb_inter.union(gene_comb_intra)),
        columns=["from", "to", "shortest path", "sign", "# path", "avg len"],
    )

    df = assign_path(df, G, gene_comb_inter, weighted_edge, filter_gene, thre=np.inf)
    df = assign_path(
        df, G, gene_comb_intra, weighted_edge, filter_gene, thre=thre
    )  # only allow at most (thre - 2) gene between same-type genes

    return df


def _expand_single_path(e):
    """
    Expand shortest paths to edge pairs
    e.g. given shortest path [a, b, c], expand to [(a, b), (b, c)]

    :param e: list, nodes on a shortest list. Nodes are sorted by their orders on the path
    :return: list of edge pairs. If no paths are found, return edge pairs of empty strings

    """

    if len(e) > 0:
        return [(e[counter], e[counter + 1]) for counter in range(len(e) - 1)]
    else:
        return [
            ("", "")
        ]  # ! do not use return '' as if the 1st entry is empty, tolist() in expand_path() won't work


def _expand_single_sign(e):
    """
    This function is created for grammar reasons
    :param e: list, signs of edges along shortest paths

    """

    # process for grammar reasons
    # otherwise df.explode(mult col) will impose error if no connections
    # as shortest path col: [('', '')]
    # but shortest path sign col: ['', '']

    if isinstance(e, list) or isinstance(e, tuple):
        return e
    else:
        return [
            ("", "")
        ]  # ! do not use return '' as if the 1st entry is empty, tolist() in expand_path() won't work


def expand_path(df):
    """
    Expand shortest paths to rows of single edges
    e.g. given shortest path [a, b, c], create new rows with edges (a, b) and (b, c)

    :param df: pandas.DataFrame, shortest path results
    :return df_expand: pandas.DataFrame, expanded df

    """

    df_expand = df.copy()

    df_expand = df_expand.explode(["shortest path", "sign"])

    df_expand["shortest path expand"] = df_expand["shortest path"].apply(
        _expand_single_path
    )
    df_expand["sign expand"] = df_expand["sign"].apply(_expand_single_sign)

    df_expand = df_expand.explode(["shortest path expand", "sign expand"])

    df_expand[["from (shortest path)", "to (shortest path)"]] = pd.DataFrame(
        df_expand["shortest path expand"].tolist(), index=df_expand.index
    )
    df_expand.fillna("", inplace=True)

    df_expand.loc[
        df_expand["from (shortest path)"] == "", ["shortest path expand", "sign expand"]
    ] = ""
    df_expand.rename(
        columns={"sign": "sign concentrate", "sign expand": "sign"}, inplace=True
    )

    return df_expand


def remove_dup(df):
    """
    Remove duplicated entries from expanded shortest paths

    :param df: pandas.DataFrame, table of shortest path results, incl. expanded shortest paths
    :return df_new: cleaned df by removing duplicated edges from shortest paths

    """

    # remove duplicated entries
    df_new = (
        df.groupby(["from (shortest path)", "to (shortest path)", "sign"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    # remove empty rows (no paths between two nodes)
    df_new = df_new[df_new["from (shortest path)"] != ""]

    # re-index
    df_new.index = zip(df_new["from (shortest path)"], df_new["to (shortest path)"])

    return df_new


def cal_weight(G, path, weighted_edge="reciprocal_n_references"):
    """
    Calculate weight given a path on G

    :param G: networkx graph or BMATool.graph.Graph
    :param path: str, path of interest
    :param weighted_edge: naime of weight attribute

    :return: weight: scalar, calculated weight

    """

    path = re.split("[^a-zA-Z0-9_]", path)
    path = [ele for ele in path if ((ele != "") and (not ele.isspace()))]

    weight = 0
    for i in range(len(path) - 1):
        weight += G[path[i]][path[i + 1]][weighted_edge]

    return weight


def get_df_by_node(
    df,
    node_set,
    source_col="source_genesymbol",
    target_col="target_genesymbol",
    node_excl=True,
):
    """
    Extract rows with source and target nodes in a pre-defined set of nodes

    :param df: pandas.DataFrame, table of interactions
    :param node_set: set of str, each element is a node of interest
    :param source_col: str, column name of source node
    :param target_col: str, column name of target node
    :param node_excl: Boolean.
        if True, look for interactions from/to nodes only in node_set
        if False, look for interactions between nodes in node_set and all nodes

    :return: pandas.DataFrame, extracted data frame
    """

    if node_excl:
        return df[df[source_col].isin(node_set) & df[target_col].isin(node_set)]
    else:
        return df[df[source_col].isin(node_set) | df[target_col].isin(node_set)]


def get_df_by_edge(
    df, edge_set, source_col="source_genesymbol", target_col="target_genesymbol"
):
    """
    Extract rows corresponding to a pre-defined set of edges (interactions)

    :param df: pandas.DataFrame, table of interactions
    :param edge_set: set (or list/tuple) of tuple, each element is an edge/interaction of interest
    :param source_col: str, column name of source node
    :param target_col: str, column name of target node

    :return: pandas.DataFrame, extracted data frame
    """

    if not isinstance(list(df.index)[0], tuple):
        df.index = zip(df[source_col], df[target_col])

    edge_set = set(edge_set).intersection(df.index)

    return df.loc[sorted(list(edge_set))]


def flatten_path(p):
    """
    Flatten path list of edges

    :param p: path list of tuples, each element is an edge tuple
    :return: list, each element is a node string

    e.g.
    p = [(1, 2), (2, 3)]
    return: [1, 2, 3]

    """

    return [p[0][0]] + [ele[1] for ele in p]
