import networkx as nx
import numpy as np
import pandas as pd


def _assign_empty_path():
    return "", ""


def _get_sign(G, path_list):
    return [
        tuple(G[ele[i]][ele[i + 1]]["sign"] for i in range(len(ele) - 1))
        for ele in path_list
    ]


def _get_weighted_path(G, u, v, weighted_edge, thre):
    """
    Look for the shortest path with lowest weight. The path must be shorter than thre, otherwise return empty results

    :param G: networkx DiGraph
    :param u: source node
    :param v: target node
    :param weighted_edge: boolean or str. when str, add attributes to edges equal to 1 / value of weighted_edge
    :param thre: int. Upper bound of # nodes in a path

    :return path_list: a list of a single shortest path with lowest weight from node u to node v
    :return sign_list: a list of signs on the returned path from node u to node v

    """

    path_list = list(nx.algorithms.all_shortest_paths(G, u, v))

    if len(path_list[0]) > thre:
        return _assign_empty_path()

    else:
        weight = np.inf
        for p in path_list:
            w = np.sum([G[p[i]][p[i + 1]][weighted_edge] for i in range(len(p) - 1)])
            if w < weight:
                weight = w
                path = p

        try:
            isinstance(path, list)
        except Exception:
            print(path_list)

        path_list = [tuple(path)]
        sign_list = _get_sign(G, path_list)

        return path_list, sign_list


def get_path(G, u, v, weighted_edge=None, thre=np.inf):
    """
    Return paths from node u to node v on graph G

    :param G: networkx DiGraph
    :param u: source node
    :param v: target node
    :param weighted_edge: boolean or str. when str, add attributes to edges equal to 1 / value of weighted_edge
    :param thre: int. Upper bound of # nodes in a path

    :return path_list: a list of shortest paths from node u to node v
    :return sign_list: a list of signs on the shortest paths from node u to node v

    """

    if u in G.nodes() and v in G.nodes():
        # shortest path between two nodes
        if not weighted_edge or isinstance(
            weighted_edge, tuple
        ):  # non-weighted paths or hybrid approach
            try:
                path_list = list(nx.algorithms.all_shortest_paths(G, u, v))
                path_list = [tuple(ele) for ele in path_list]
                sign_list = _get_sign(G, path_list)

                if isinstance(weighted_edge, tuple) and weighted_edge[0] == "hybrid":
                    weight = np.inf
                    weighted_edge = weighted_edge[1]

                    for p_counter, p in enumerate(path_list):
                        w = np.sum(
                            G[p[i]][p[i + 1]][weighted_edge] for i in range(len(p) - 1)
                        )
                        if w < weight:
                            weight = w
                            path_counter = p_counter

                    path_list = [path_list[path_counter]]
                    sign_list = [sign_list[path_counter]]

            except nx.exception.NetworkXNoPath:
                path_list, sign_list = _assign_empty_path()

        # weighted shortest path, weight by 1 / weighted_edge
        # note that this approach does not find ALL weighted shortest paths
        elif isinstance(weighted_edge, str):
            try:
                path_list = [tuple(nx.dijkstra_path(G, u, v, weight=weighted_edge))]
                sign_list = _get_sign(G, path_list)

                if len(path_list[0]) > thre:
                    path_list, sign_list = _get_weighted_path(
                        G, u, v, weighted_edge, thre
                    )

            except nx.exception.NetworkXNoPath:
                path_list, sign_list = _assign_empty_path()

        else:
            raise TypeError("weighted_edge must be boolean or str")

    else:
        path_list, sign_list = _assign_empty_path()

    if len(path_list) > 0 and len(path_list[0]) <= thre:
        return dict(zip(path_list, sign_list))
    else:
        path_list, sign_list = _assign_empty_path()
        return [path_list, sign_list]


def get_path_weight(G, path, weighted_edge):
    path_weight = []
    for i in range(len(path) - 1):
        path_weight.append(G[path[i]][path[i + 1]][weighted_edge])

    return sum(path_weight), path_weight


def graph_to_df(G):
    dic = nx.get_edge_attributes(G, "sign")
    df = pd.DataFrame.from_dict(dic, orient="index")
    df[["from", "to"]] = pd.DataFrame(df.index.tolist(), index=df.index)

    return df


# def gen_adj(G):

#     node_list = sorted(G.nodes, reverse=False)
#     n_node = len(node_list)

#     # adjacency mask matrix
#     A = pd.DataFrame(0, index=node_list, columns=node_list)
#     for u, v in G.edges:
#         if G[u][v]['sign'] == 'Activator':
#             A.at[v, u] = 1
#         else:
#             A.at[v, u] = -1

#     # all-inhibitor nodes
#     A_inh = (A <= 0).all(axis=1) & (~(A == 0).all(axis=1))

#     # replace -1/1 with activator/inhibitot average
#     # A_mult = pd.DataFrame(avg_mask(A, 1) + avg_mask(A, -1), index=node_list, columns=node_list)

#     T = np.zeros([n_node, n_exp])
#     T[np.asarray(A_inh), :] = 2
