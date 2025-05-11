# examine and return shortest path
# also check style

import re
from itertools import product, zip_longest

import networkx as nx
import numpy as np
import omnipath as op
import pandas as pd


def get_graph(edge_list):
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from(edge_list, weight="type")

    return G


def get_direction(df):
    for idx in df.index:
        if not df.at[idx, "is_directed"]:
            direct = "undirected"
        elif df.at[idx, "is_stimulation"] and df.at[idx, "is_inhibition"]:
            direct = "both"
        elif (not df.at[idx, "is_stimulation"]) and (not df.at[idx, "is_inhibition"]):
            direct = "neither"
        elif df.at[idx, "is_stimulation"]:
            direct = "stimulation"
        elif df.at[idx, "is_inhibition"]:
            direct = "inhibition"
        df.at[idx, "direction_type"] = direct

    return df


def get_style(edge_type):
    dic_style = {
        "stimulation": "-->",
        "inhibition": "--|",
        "both": "-->|",
        "neither": "==>",
        "undirected": "--",
        "Activator": "-->",
        "Inhibitor": "--|",
    }

    return dic_style[edge_type]


def edge_style(u, v, G):
    dic_order = dict(
        zip(
            [
                "stimulation",
                "inhibition",
                "both",
                "neither",
                "undirected",
                "Activator",
                "Inhibitor",
            ],
            range(7),
        )
    )

    direct = [G[u][v][k]["type"] for k in G[u][v]]
    direct = sorted(direct, key=lambda t: dic_order[t])

    return get_style(direct[0])


def path_style(e, G):
    style = []
    for counter in range(len(e) - 1):
        style.append(edge_style(e[counter], e[counter + 1], G))

    style.append("")

    return "".join("".join(x) for x in zip_longest(e, style))


def get_comb(gene_1, gene_2=None):
    if gene_2 is None:
        gene_2 = gene_1

    gene_comb = list(product(gene_1, gene_2))
    gene_comb = [ele for ele in gene_comb if ele[0] != ele[1]]

    return gene_comb


def get_op(remove_undir=False):
    int_op = op.interactions.OmniPath.get(genesymbols=True, directed=False)  # type: ignore

    int_op = int_op.drop_duplicates(
        subset=[
            "source",
            "target",
            "source_genesymbol",
            "target_genesymbol",
            "is_directed",
            "is_stimulation",
            "is_inhibition",
        ]
    )
    int_op = int_op[int_op["n_primary_sources"] > 3]  # filter entries

    if remove_undir:
        int_op = int_op[int_op["is_directed"]]  # remove undirected edges
    else:
        int_undir = int_op[~int_op["is_directed"]]
        int_undir = int_undir.rename(
            columns={
                "source": "target",
                "target": "source",
                "source_genesymbol": "target_genesymbol",
                "target_genesymbol": "source_genesymbol",
            }
        )
        int_op = pd.concat([int_op, int_undir], axis=0, ignore_index=True)

    int_op = get_direction(int_op)

    G = get_graph(
        zip(
            int_op["source_genesymbol"],
            int_op["target_genesymbol"],
            int_op["direction_type"],
        )
    )

    return int_op, G


def assign_path(
    gene_comb,
    G,
    edge_to_compare=None,
    path_type="shortest_path",
    filter_gene=False,
    gene_to_filter=None,
):
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(gene_comb),
        columns=[
            "from gene",
            "to gene",
            "shortest path",
            "# path",
            "avg len",
            "bma pair",
            "bma path",
        ],
    )

    for u, v in gene_comb:
        edge = gene_pair()
        edge.find_path(u, v, G, edge_to_compare, path_type, filter_gene, gene_to_filter)

        df.loc[u, v][["from gene", "to gene"]] = [u, v]
        df.loc[u, v][["# path", "avg len"]] = [edge.n_path, edge.avg_path]

        df.at[(u, v), "bma pair"] = edge.bma_pair
        df.at[(u, v), "bma path"] = edge.bma_path

        df.at[(u, v), "shortest path"] = edge.shortest_path

    return df


def _remove_space(str_list):
    return [ele for ele in str_list if ((ele != "") and (not ele.isspace()))]


def _remove_dir(_path):
    if isinstance(_path, str):
        _path = re.split("[^a-zA-Z0-9]", _path)
        _path = _remove_space(_path)
    else:
        _path = ""

    return _path


def _replace_sign(_sign):
    inv_dic = {
        "-->": "Activator",
        "--|": "Inhibitor",
        "-->|": "Both",
        "==>": "Neither",
        "--": "Undirected",
    }

    return [inv_dic[ele] for ele in _sign]


def _get_dir(_path):
    if isinstance(_path, str):
        _sign = re.split("[a-zA-Z0-9]", _path)
        _sign = _remove_space(_sign)
        _sign = _replace_sign(_sign)

    else:
        _sign = "", ""

    return _sign


def _expand_path(e):
    if isinstance(e, list):
        return [(e[counter], e[counter + 1]) for counter in range(len(e) - 1)]
    else:
        return [
            ("", "")
        ]  # ! do not use return '' as if the 1st entry is empty, tolist() in expand_path() won't work


def _expand_sign(e):
    # process for grammar reasons
    # otherwise df.explode(mult col) will impose error if no connections
    # as shortest path col: [('', '')]
    # but shortest path sign col: ['', '']

    if isinstance(e, list):
        return e
    else:
        return [
            ("", "")
        ]  # ! do not use return '' as if the 1st entry is empty, tolist() in expand_path() won't work


def expand_path(df):
    df_expand = df.copy()

    # split multiple paths by comma and explode
    df_expand["shortest path expand"] = df_expand["shortest path"].str.split(",")
    df_expand["shortest path sign expand"] = df_expand["shortest path"].str.split(",")

    df_expand = df_expand.explode(["shortest path expand", "shortest path sign expand"])

    # split each path into genes (by non-alphabetic/non-numeric characters)
    df_expand["shortest path expand"] = df_expand["shortest path expand"].apply(
        _remove_dir
    )
    df_expand["shortest path expand"] = df_expand["shortest path expand"].apply(
        _expand_path
    )

    # split each path into signs (by alphabetic/numeric characters)
    df_expand["shortest path sign expand"] = df_expand[
        "shortest path sign expand"
    ].apply(_get_dir)
    df_expand["shortest path sign expand"] = df_expand[
        "shortest path sign expand"
    ].apply(_expand_sign)

    df_expand = df_expand.explode(["shortest path expand", "shortest path sign expand"])
    df_expand[["from (shortest path)", "to (shortest path)"]] = pd.DataFrame(
        df_expand["shortest path expand"].tolist(), index=df_expand.index
    )
    df_expand.fillna("", inplace=True)

    return df_expand


class gene_pair:
    def __init__(self):
        self.n_path = 0
        self.avg_path = 0

        self.bma_pair = False
        self.bma_path = ""

        self.path_list = []
        self.path_with_style = []

        self.shortest_path = ""

    def _compare_edge(self, u, v, edge_to_compare):
        if (u, v) in edge_to_compare:
            self.bma_pair = True
            self.bma_path = get_style(edge_to_compare[(u, v)][0]).join([u, v])

    def _get_path(
        self, u, v, G, path_type="shortest_path", filter_gene=False, gene_to_filter=None
    ):
        if u in G.nodes() and v in G.nodes():
            try:
                if path_type == "shortest_path":
                    self.path_list = list(nx.algorithms.all_shortest_paths(G, u, v))
                elif path_type == "simple_path":
                    shortest_path = nx.shortest_path_length(G, u, v)
                    self.path_list = list(
                        nx.algorithms.all_simple_paths(
                            G, u, v, cutoff=shortest_path + 2
                        )
                    )
                else:
                    raise ValueError("path_type: %s not supported yet" % path_type)

                if filter_gene:
                    if gene_to_filter is None:
                        raise ValueError(
                            "must pass gene_to_filter if filter_gene is True"
                        )

                    if not isinstance(gene_to_filter, set):
                        gene_to_filter = set(gene_to_filter)

                    # only include paths with genes in a given list
                    self.path_list = [
                        ele
                        for ele in self.path_list
                        if len(set(ele) - gene_to_filter) == 0
                    ]

                if len(self.path_list) == 0:
                    self.path_list = []
                    self.shortest_path = "unconnected via bma gene"
                else:
                    self.n_path = len(self.path_list)
                    self.avg_path = np.mean([len(ele) - 1 for ele in self.path_list])

            except Exception:
                self.shortest_path = "unconnected"

        else:
            self.shortest_path = "%s not found in Omnipath" % ({u, v} - set(G.nodes()))

    def _get_style(self, G):
        if isinstance(self.path_list, list):
            path_list_with_style = []
            for e in self.path_list:
                path_list_with_style.append(path_style(e, G))

            self.path_with_style = path_list_with_style
            self.shortest_path = ", ".join(self.path_with_style)

    def find_path(
        self,
        u,
        v,
        G,
        edge_to_compare=None,
        path_type="shortest_path",
        filter_gene=False,
        gene_to_filter=None,
    ):
        if edge_to_compare is not None:
            self._compare_edge(u, v, edge_to_compare)

        self._get_path(u, v, G, path_type, filter_gene, gene_to_filter)
        self._get_style(G)
