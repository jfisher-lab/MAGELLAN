from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd


def get_sorted_node_list(G: nx.DiGraph) -> list:
    return sorted(G.nodes)


def get_node_index(node_name: str, node_dic: dict) -> int:
    """Get node index from node dictionary, with error handling."""
    if node_name not in node_dic:
        raise KeyError(f"Node '{node_name}' not found in node dictionary")
    return node_dic[node_name]


def enforce_pert_dic_order(pert_dic: OrderedDict) -> OrderedDict:
    return OrderedDict(sorted(pert_dic.items(), key=lambda t: t[0]))


def get_data(
    pert_dic_all: OrderedDict, G: nx.DiGraph, y_replace_missing_with_zero: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract data (X and y) from spec where:

    - X contains perturbation values (which nodes are perturbed in each experiment)
    - y contains expected values (what values nodes should reach)

    :param pert_dic_all: spec dict. see below for example
        pert_dic_all = {experiment:
                        {'pert': {node_1: value_1, node_2: value_2, ...},
                         'exp': {node_x, value_x, node_y: value_y}}
                         }
    :param node_list: list of node names in ascending order
    :param y_replace_missing_with_zero: if True, replace missing values in y with 0

    :return X: pandas.DataFrame, index: node_list, columns: experiments in ascending order
        perturbation data, values: perturbed values for perturbed nodes under each experiment, 0 otherwise
    :return y: pandas.DataFrame, index: node_list, columns: experiments in ascending order
        expectation data, values: expected values for expected nodes under each experiment, 0 otherwise


    """
    pert_dic_all = enforce_pert_dic_order(pert_dic_all)
    node_list = get_sorted_node_list(G)
    if node_list is None or len(node_list) == 0:
        raise ValueError("node_list cannot be None or empty")

    # Modified check for empty dictionary
    if not pert_dic_all:
        raise ValueError("pert_dic_all cannot be empty")

    # New check for empty inner dictionaries
    for exp_name, exp_data in pert_dic_all.items():
        if not exp_data.get("pert") and not exp_data.get("exp"):
            raise ValueError("pert_dic_all contains empty experiment data")

    exp_list = sorted(list(pert_dic_all.keys()))

    if y_replace_missing_with_zero:
        y = pd.DataFrame(data=0.0, index=node_list, columns=exp_list)
    else:
        y = pd.DataFrame(data=np.nan, index=node_list, columns=exp_list)
    X = pd.DataFrame(data=0.0, index=node_list, columns=exp_list)

    for experiment_name, experiment_data in pert_dic_all.items():
        pert, exp = experiment_data["pert"], experiment_data["exp"]

        X.loc[pert.keys(), experiment_name] = list(pert.values())
        y.loc[exp.keys(), experiment_name] = list(exp.values())

    return X, y


def update_function_mask(
    A: pd.DataFrame, val: int, method: str = "avg"
) -> pd.DataFrame:
    """
    Apply mask to model update function in BMA. Currently two methods are supported.

    - "avg": Average the mask matrix A, e.g. node X has 2 parents, |A_{X, .}| = 1/2
    - "sum": Sum the mask matrix A, e.g. node X has 2 parents, |A_{X, .}| = 1

    :param A: array-like, (shape n_node, n_node).
        mask adjacency matrix. A_{ji} = 1 if i-->j, -1 if i--|j
    :param val: 1 or -1
    :param method: "avg" or "sum"
    :return: array-like, (shape n_node, n_node).
        averaged mask adjacency matrix

    """

    A_arr = np.asarray(A.copy())
    A_arr[A_arr != val] = 0
    if method == "avg":
        row_sums = np.sum(A_arr == val, axis=1)[:, None]
        A_arr = np.divide(
            A_arr, row_sums, out=np.zeros(A_arr.shape), where=row_sums != 0
        )
    elif method == "sum":
        A_arr = A_arr
    else:
        raise ValueError(
            f"Invalid method: {method}, currently supported are 'avg' and 'sum'"
        )

    # A_arr[np.isnan(A_arr)] = 0

    return pd.DataFrame(A_arr, index=A.index, columns=A.columns)


def get_adj_single(
    A_single: pd.DataFrame, inh: list, pert_dic: dict, G: nx.DiGraph
) -> pd.DataFrame:
    """
    Get the adjacency matrix for a single experiment, adding dummy nodes for all inhibitor nodes, and replacing parents with dummy nodes when perturbed.

    :param A_single: pd.DataFrame, the adjacency matrix for a single experiment
    :param inh: list, the inhibitor nodes
    :param pert_dic: dict, the perturbation dictionary
    :param node_list: list, the node list
    """
    node_list = get_sorted_node_list(G)
    A_single = A_single.copy()
    A_single = pd.DataFrame(data=A_single, index=node_list, columns=node_list)

    # add dummy node for all inhibitor nodes
    # note that the normalising constant in A (by average is not changed after dummy node is added)
    # e.g. 1/2 will not change to 1/3
    for node in inh:
        dummy = "dummy_%s" % node
        A_single.at[dummy, dummy] = 1  # self loop for dummy nodes
        A_single.at[node, dummy] = (
            1  # activation from dummy node to all-inhibitor node, dummy-->n
        )

    # the following code was before the above code (for n in inh ... A_single.at[n. dummy] = 1)
    # it is moved here because the dummy parents need to be removed if a node is perturbed
    # remove parent node, add self loop for perturbed node
    if pert_dic is not None:
        for node in pert_dic:
            A_single.loc[node] = 0  # remove parents
            A_single.at[node, node] = 1  # add self loop

    A_single = A_single.fillna(0)

    # HERE! #
    # modify code to make w for dummy-->n stay at 1 (so dummy stays constant)

    return A_single


def base_adj(G: nx.DiGraph) -> pd.DataFrame:
    """
    Get the base adjacency matrix

    :param G: networkx.DiGraph, the graph
    :param node_list: list, the node list

    :return: pd.DataFrame, the base adjacency matrix as a pandas dataframe
    """
    if G.number_of_nodes() == 0:
        raise ValueError("The graph G is empty.")

    if G.number_of_edges() == 0:
        raise ValueError("The graph G has no edges.")

    node_list = get_sorted_node_list(G)
    A = pd.DataFrame(0, index=node_list, columns=node_list)

    if set(node_list) != set(G.nodes):
        raise ValueError("The node_list does not match the nodes in the graph G.")

    for u, v in G.edges:
        if G[u][v]["sign"] == "Activator":
            A.at[v, u] = 1
        elif G[u][v]["sign"] == "Inhibitor":
            A.at[v, u] = -1
        else:
            raise ValueError(f"Invalid edge sign: {G[u][v]['sign']}")

    return A


def get_adj(
    G: nx.DiGraph,
    pert_dic_all: dict,
    node_list: list,
    method: str = "avg",
) -> tuple[np.ndarray, list]:
    """
    Get the adjacency matrix of the graph with the update function applied.

    :param G: networkx.DiGraph
    :param pert_dic_all: dict, the perturbation dictionary
    :param node_list: list, the node list
    :param method: str, the method to use for the update function
    """
    # if node_list is None:
    #     node_list = list(G.nodes)
    raise NotImplementedError("This function is deprecated.")

    if set(node_list) != set(G.nodes):
        raise ValueError("The node_list does not match the nodes in the graph G.")

    if node_list is None or len(node_list) == 0:
        raise ValueError("node_list cannot be None or empty")

    if G.number_of_nodes() == 0:
        raise ValueError("The graph G is empty.")

    node_list = sorted(node_list)

    A_base = base_adj(G, node_list)

    # all-inhibitor nodes
    A_inh = (A_base <= 0).all(axis=1) & (~(A_base == 0).all(axis=1))
    inh = sorted(list(A_inh[A_inh].index))

    # replace -1/1 with activator/inhibitor average
    A_mult = pd.DataFrame(
        update_function_mask(A_base, 1, method)
        + update_function_mask(A_base, -1, method),
        index=node_list,
        columns=node_list,
    )

    A = []
    for pert_dic in pert_dic_all.values():
        pert_dic = pert_dic["pert"]
        A.append(get_adj_single(A_mult, inh, pert_dic, node_list))

    A = np.array(A)

    return A, inh


# def pred_y_single(X_col, W, A, eye_correct, zero_correct, idx, t=10):
#     """

#     :param X_col: array-like
#     :param W:
#     :param A:
#     :param idx: list of integers, the indices for perturbation nodes
#     :param t: positive integer, # timesteps
#     :return:

#     """

#     W = eye_correct.dot(W) + zero_correct
#     pred_single = np.linalg.matrix_power(np.multiply(A, W), t).dot(X_col)

#     return pred_single[idx]


# def pred_mat(X, W, A, eye_correct, zero_correct, t=10):
#     W_new = (
#         np.matmul(eye_correct, W) + zero_correct
#     )  # self-loops for perturbed nodes is weighted as 1
#     # W_new = np.abs(W_new)
#     pred_stack = np.matmul(
#         np.linalg.matrix_power(np.multiply(A, W_new), t), np.asarray(X)
#     )

#     return pred_stack


# def pred_all(X, y, W, A, pert_dic_all, time_step):
#     # n_node = X.shape[0]
#     node_list = list(X.index)

#     idx_dic = dict(zip(node_list, range(len(node_list))))

#     E = np.zeros_like(A, dtype=float)
#     idx = range(len(node_list))
#     E[:, idx, idx] = 1
#     Identity = E.copy()

#     for counter, (k, v) in enumerate(pert_dic_all.items()):
#         # E = pd.DataFrame(data=np.eye(n_node), index=node_list, columns=node_list)  # eye_correct

#         for n in v["pert"]:
#             E[counter, idx_dic[n], idx_dic[n]] = 0

#         # Z = np.eye(n_node) - E  # zero_correct
#         #
#         # pred[idx_expect[idx_experiment], idx_experiment] = pred_y_single(X[:, idx_experiment], W, A_copy, E, Z, idx_expect[idx_experiment], time_step)

#     # obtain indices for dummy nodes and their child node
#     idx = [
#         (idx_dic[k.replace("dummy_", "")], idx_dic[k]) for k in idx_dic if "dummy" in k
#     ]
#     idx = tuple(zip(*idx))

#     Z = Identity - E
#     Z[:, idx[0], idx[1]] = 1

#     pred_stack = pred_mat(
#         X, W, A, E, Z, time_step
#     )  # pred for ALL experiments with 3-D matrix. 1st dimension: experiment

#     # extract predictions in each experiment
#     pred = np.zeros_like(y, dtype=float)
#     for counter, v in enumerate(pert_dic_all.values()):
#         idx = [idx_dic[n] for n in v["exp"]]
#         pred[idx, counter] = pred_stack[counter, idx, counter]

#     return pred


def round_minmax(X: np.ndarray, min_val: int = 0, max_val: int = 2) -> np.ndarray:
    """
    Round values between min_val and max_val

    Args:
        X: Input array
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Array with values rounded and clamped between min_val and max_val

    Raises:
        ValueError: If min_val is greater than max_val
    """
    if min_val > max_val:
        raise ValueError(
            f"min_val ({min_val}) must be less than or equal to max_val ({max_val})"
        )

    X = X.copy()  # Avoid modifying input array
    X[X < min_val] = min_val
    X[X > max_val] = max_val

    return np.floor(X + 0.5).astype(X.dtype)


def pred_mat_bound_single(
    X: pd.DataFrame,
    W: np.ndarray,
    Adjacency_per_experiment: list[pd.DataFrame],
    eye_correct: np.ndarray,
    zero_correct: np.ndarray,
    t: int = 10,
    min_val: int = 0,
    max_val: int = 2,
):
    """
    Iterate through time steps to predict the values of the nodes in the network.
    1. round to the nearest integer
    2. negative values change to 0
    3. set > max_val values to max_val
    4. calculate difference between time step t and t - 1

    :param X:
    :param W:
    :param Adjacency_per_experiment:
    :param eye_correct:
    :param zero_correct:
    # :param idx_dummy: tuple, indices of dummy nodes corresponding to 2-array X (row: node, column: experiment)
    :param t:
    :param min_val:
    :param max_val:

    :return: predicted values

    """

    if isinstance(t, float) and t.is_integer():
        t = int(t)
    elif not isinstance(t, int):
        raise ValueError("t must be an integer")

    W_new = np.asarray(W)
    W_new = (
        np.matmul(eye_correct, W_new) + zero_correct
    )  # self-loops for perturbed nodes is weighted as 1

    prev = np.asarray(X.copy())
    # prev[X != 0] = 1  # set starting perturbation to be 1

    for _ in range(t):
        curr = np.matmul(np.multiply(Adjacency_per_experiment, W_new), prev)
        curr = round_minmax(curr, min_val, max_val)

        # curr[X != 0] = X[X != 0]  # - 1  # set perturbed values to actual perturbation - 1, if changed by more than 1, the following code will cut the diff to at most 1
        # curr[idx_dummy] += 1  # dummy nodes start from perturbation, this allows all-inhibitor nodes to sync with others

        diff = curr - prev
        idx = np.where(np.abs(diff) > 1)
        diff[idx] = np.sign(diff[idx])
        curr = prev + diff
        curr = round_minmax(curr, min_val, max_val)

        prev = curr.copy()

    return curr


def pred_all_bound_single(
    X: pd.DataFrame,
    y: pd.DataFrame,
    W: pd.DataFrame,
    Adjacency_per_experiment: list[pd.DataFrame],
    G: nx.DiGraph,
    pert_dic_all: OrderedDict,
    time_step: int,
    min_val: int = 0,
    max_val: int = 2,
    extract_exp: bool = True,
):
    """
    Predict the values of the nodes in the network.

    Note pred_bound will predict zero for perturbed nodes, due to different settings in scipy opt results

    :param X: pd.DataFrame, the perturbation data
    :param y: pd.DataFrame, the expectation data
    :param W: np.ndarray | pd.DataFrame, the weight matrix
    :param A: np.ndarray, the adjacency matrix
    :param pert_dic_all: dict, the perturbation dictionary
    :param time_step: int, the number of time steps
    :param min_val: int, the minimum value of nodes in the BMA network
    :param max_val: int, the maximum value of nodes in the BMA network
    :param extract_exp: bool, whether to extract the predictions for the expected nodes
    """
    pert_dic_all = enforce_pert_dic_order(pert_dic_all)
    if time_step == 0:
        raise ValueError("time_step cannot be 0")

    # check that all rows and columns of all elements of Adjacency_per_experiment are the same order as all rows and columns of W
    for a in Adjacency_per_experiment:
        if not np.array_equal(a.index, W.index) or not np.array_equal(
            a.columns, W.columns
        ):
            raise ValueError(
                "Adjacency_per_experiment does not have the same order as W"
            )

    if isinstance(W, pd.DataFrame):
        W_array = np.asarray(W)

    # raise error if X, y, W, A or pert_dic_all is empty
    if (
        len(X) == 0
        or len(y) == 0
        or len(W_array) == 0
        or len(Adjacency_per_experiment) == 0
        or len(pert_dic_all) == 0
    ):
        raise ValueError("X, y, W, A or pert_dic_all cannot be empty")

    # raise error if X, y, W, A or pert_dic_all are incorrect shape and print the shape of each
    if (
        X.shape[0] != y.shape[0]  # Same number of nodes
        or X.shape[1] != y.shape[1]  # Same number of experiments
        or X.shape[0] != W_array.shape[0]  # Nodes match weight matrix rows
        or W_array.shape[0] != W_array.shape[1]  # Weight matrix is square
        or any(
            a.shape[0] != W_array.shape[0] for a in Adjacency_per_experiment
        )  # A's node dimensions match W
        or any(
            a.shape[1] != W_array.shape[0] for a in Adjacency_per_experiment
        )  # A's node dimensions match W
        or len(pert_dic_all)
        != len(Adjacency_per_experiment)  # Number of experiments matches
    ):
        raise ValueError(
            f"Shape mismatch: X:{X.shape}, y:{y.shape}, W:{W_array.shape}, "
            f"Adjacency_per_experiment:{[a.shape for a in Adjacency_per_experiment]}, pert_dic_all:{len(pert_dic_all)}"
        )

    node_list = get_sorted_node_list(G)
    idx_dic = dict(zip(node_list, range(len(node_list))))

    eye_correct = np.zeros_like(Adjacency_per_experiment, dtype=float)
    idx = range(len(node_list))
    eye_correct[:, idx, idx] = 1
    Identity = eye_correct.copy()

    for counter, (_, v) in enumerate(pert_dic_all.items()):
        for n in v["pert"]:
            eye_correct[counter, idx_dic[n], idx_dic[n]] = 0

    zero_correct = Identity - eye_correct

    # Handle dummy nodes if they exist
    dummy_nodes = [
        (idx_dic[k.replace("dummy_", "")], idx_dic[k]) for k in idx_dic if "dummy" in k
    ]
    if dummy_nodes:
        idx_child, idx_dummy = zip(*dummy_nodes)
        zero_correct[:, idx_child, idx_dummy] = 1

    pred_stack = pred_mat_bound_single(
        X,
        W_array,
        Adjacency_per_experiment,
        eye_correct,
        zero_correct,
        time_step,
        min_val,
        max_val,
    )
    pred = np.zeros_like(y, dtype=float)

    if not extract_exp:  # return all node predictions
        for counter in range(len(pert_dic_all)):
            pred[:, counter] = pred_stack[counter, :, counter]

    # return only expected nodes predictions
    # extract predictions in each experiment
    else:
        for counter, v in enumerate(pert_dic_all.values()):
            idx = [idx_dic[n] for n in v["exp"]]
            pred[idx, counter] = pred_stack[counter, idx, counter]

    return pred


def add_perturbed_nodes(
    pred_bound_zero_perturbed: np.ndarray | pd.DataFrame,
    pert_dic_small: OrderedDict,
    y: pd.DataFrame,
) -> pd.DataFrame:
    """
    Manually sets perturbed nodes to their specified values in the prediction DataFrame.

    Args:
        pred_bound_zero_perturbed: DataFrame containing predictions with perturbed nodes set to zero
        pert_dic_small: Dictionary containing perturbation information
        y: Target DataFrame with same structure as pred_bound_zero_perturbed

    Returns:
        DataFrame with perturbed nodes corrected to their specified values

    Raises:
        KeyError: If a node in pert_dic_small doesn't exist in the DataFrame
    """

    # # raise error if pert_dic_small is empty
    # if not pert_dic_small:
    #     raise ValueError("pert_dic_small cannot be empty")
    pert_dic_small = enforce_pert_dic_order(pert_dic_small)
    # raise error when when perturbation dictionary contains nodes not in DataFrame
    if not set(pert_dic_small.keys()).issubset(set(y.columns)):
        raise KeyError("pert_dic_small contains nodes not in y")

    # convert pred_bound_zero_perturbed to DataFrame
    pred_bound_zero_perturbed = pd.DataFrame(
        pred_bound_zero_perturbed.copy(), index=y.index, columns=y.columns
    )

    # Validate all nodes exist before making any changes
    existing_nodes = set(pred_bound_zero_perturbed.index)
    for exp_data in pert_dic_small.values():
        pert_nodes = set(exp_data["pert"].keys())
        missing_nodes = pert_nodes - existing_nodes
        if missing_nodes:
            raise KeyError(
                f"Node(s) in perturbation dictionary not found in DataFrame: {missing_nodes}"
            )

    pred_bound_perts_corrected = pd.DataFrame(
        pred_bound_zero_perturbed.copy(), index=y.index, columns=y.columns
    )
    for k, v in pert_dic_small.items():
        for kk, vv in v["pert"].items():
            pred_bound_perts_corrected.loc[kk, k] = vv
    return pred_bound_perts_corrected


def pred_bound_perts(
    X: pd.DataFrame,
    y: pd.DataFrame,
    W: pd.DataFrame,
    Adjacency_per_experiment: list[pd.DataFrame],
    G: nx.DiGraph,
    pert_dic_all: OrderedDict,
    time_step: int = 10,
    min_val: int = 0,
    max_val: int = 2,
    extract_exp: bool = True,
) -> pd.DataFrame:
    """
    Predict the values of the nodes in the network.

    :param X: pd.DataFrame, the perturbation data
    :param y: pd.DataFrame, the expectation data
    :param W: np.ndarray | pd.DataFrame, the weight matrix
    :param A: list[pd.DataFrame], the adjacency matrix
    :param pert_dic_all: dict, the perturbation dictionary
    :param time_step: int, the number of time steps
    :param min_val: int, the minimum value of nodes in the BMA network
    :param max_val: int, the maximum value of nodes in the BMA network
    :param extract_exp: bool, whether to extract the predictions for the expected nodes
    """
    pert_dic_all = enforce_pert_dic_order(pert_dic_all)
    pred_bound_zero_perturbed = pred_all_bound_single(
        X=X,
        y=y,
        W=W,
        Adjacency_per_experiment=Adjacency_per_experiment,
        G=G,
        pert_dic_all=pert_dic_all,
        time_step=time_step,
        min_val=min_val,
        max_val=max_val,
        extract_exp=extract_exp,
    )
    pred_bound_perts_corrected = add_perturbed_nodes(
        pred_bound_zero_perturbed=pred_bound_zero_perturbed,
        pert_dic_small=pert_dic_all,
        y=y,
    )
    return pred_bound_perts_corrected


# def pred_mat_bound(X, W, A, eye_correct, zero_correct, t=10, min_val=0, max_val=2):
#     W_new = (
#         np.matmul(eye_correct, W) + zero_correct
#     )  # self-loops for perturbed nodes is weighted as 1
#     pred = np.matmul(np.multiply(A, W_new), np.asarray(X))  # 1st iter
#     pred = round_minmax(pred, min_val, max_val)

#     counter = 0
#     while counter < t:
#         pred = np.matmul(np.multiply(A, W_new), pred)
#         pred = round_minmax(pred, min_val, max_val)

#         counter += 1

#     return pred


# def pred_all_bound(X, y, W, A, pert_dic_all, time_step, min_val=0, max_val=2):
#     node_list = list(X.index)

#     idx_dic = dict(zip(node_list, range(len(node_list))))

#     E = np.zeros_like(A, dtype=float)
#     idx = range(len(node_list))
#     E[:, idx, idx] = 1
#     Identity = E.copy()

#     for counter, (k, v) in enumerate(pert_dic_all.items()):
#         for n in v["pert"]:
#             E[counter, idx_dic[n], idx_dic[n]] = 0

#     # obtain indices for dummy nodes and their child node
#     idx = [
#         (idx_dic[k.replace("dummy_", "")], idx_dic[k]) for k in idx_dic if "dummy" in k
#     ]
#     idx = tuple(zip(*idx))

#     Z = Identity - E
#     Z[:, idx[0], idx[1]] = 1

#     pred_stack = pred_mat_bound(
#         X, W, A, E, Z, time_step, min_val, max_val
#     )  # pred for ALL experiments with 3-D matrix. 1st dimension: experiment

#     # extract predictions in each experiment
#     pred = np.zeros_like(y, dtype=float)
#     for counter, v in enumerate(pert_dic_all.values()):
#         idx = [idx_dic[n] for n in v["exp"]]

#         pred[idx, counter] = pred_stack[counter, idx, counter]

#     return pred


# def get_diff_prev(idx_expect, y, pred):
#     y_copy = pred.copy()
#     for idx in range(y.shape[1]):
#         y_copy[idx_expect[idx], idx] = y[idx_expect[idx], idx]

#     return y_copy - pred


def get_diff(
    y: list | pd.DataFrame | np.ndarray, pred: list | pd.DataFrame | np.ndarray
) -> np.ndarray:
    # raise value error if shape mismatch
    y = np.asarray(y)
    pred = np.asarray(pred)

    if y.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y:{y.shape}, pred:{pred.shape}")

    return y - pred


# def get_err(diff, err):
#     if err == "mse":
#         err_val = np.sum(np.power(diff, 2))
#     elif err == "mae":
#         err_val = np.sum(np.abs(diff))
#     # elif err == 'cross_entropy':
#     #     err_val =
#     else:
#         raise KeyError("%s not implemented" % err)

#     return err_val


def get_loss(
    diff: np.ndarray, W: np.ndarray, lambd: float, err: str = "mse", reg: int | None = 1
) -> float:
    if err == "mse":
        err_val = np.sum(np.power(diff, 2))
    elif err == "mae":
        err_val = np.sum(np.abs(diff))
    else:
        raise KeyError("err must be mse or mae")

    if reg is None:
        reg_val = 0
    else:
        reg_val = lambd * np.linalg.norm(W, reg)

    return err_val + reg_val


def loss_func(
    W: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    A_base: np.ndarray,
    pert_dic_all: OrderedDict,
    lambd: float,
    err: str,
    reg: int | None,
    time_step: int,
    min_val: int = 0,
    max_val: int = 2,
) -> float:
    """
    Used in scipy optimisation method, which is no longer used.

    Calculate loss ||y - pred||^2 + lambda * ||w||_1 (error and reg may change)

    :param X: array-like, shape (n_node, n_exp)
    :param y: array-like, shape (n_node, n_exp)

    :param W: array-like, shape (n_node * n_node). Initial weight matrix.
    :param A: array-like, shape (n_exp, n_node, n_node).
        Weighted adjacency matrix. A_.ji = 1/|Pa(j)| (i—>j, negative if i—|j).
    :param A_base: array-like, shape (n_node, n_node).
        Adjacency matrix. A_ji = 1 if i-->j and -1 if i--|j

    :param pert_dic_all: dict, spec.  {experiment: {'pert': {node: value}, 'exp': {node: value}}}

    :param idx_pert: dict, key: exp name, value: list of integers. indices of perturbed node under each experiment
    :param idx_expect: dict, key: exp name, value: list of integers. indices of perturbed node under each experiment

    :param lambd: float, regularisation parameter
    :param err: str, error function. 'mse': mean squared error, 'mae': mean absolute error.
    :param reg: int or None. If int, reg = 1 or 2, l1 or l2 regularisation

    :param time_step: int, number of time steps

    :param min_val: int, minimum value, default 0 (no activity)
    :param max_val: int, maximum value, default 2 (1: mid, 2: high activity)

    :return loss: float, loss function result

    """

    W = np.reshape(W, A[0, :, :].shape)
    pred = pred_all_bound_single(X, y, W, A, pert_dic_all, time_step, min_val, max_val)  # type: ignore

    # comment for now
    # pred = pred_all(X, y, W, A, pert_dic_all, time_step)
    #
    # # round to integers between 0 and 4 (lower loss if use real values)
    # pred = np.floor(pred + 0.5)
    # pred[pred > 4] = 4
    # pred[pred < 0] = 0

    n_node = A_base.shape[0]
    W = np.multiply(np.abs(A_base), W[:n_node, :n_node])

    diff = get_diff(y, pred)[:n_node, :]
    # loss = np.sum(np.power(diff, 2)) + lambd * np.linalg.norm(W, 1)
    # err_val = get_err(diff, err)
    loss = get_loss(diff, W, lambd, err, reg)

    print(
        "current loss: %.4f, # absolute err: %.3f, l1 reg: %.3f"
        % (loss, np.sum(np.abs(diff)), lambd * np.linalg.norm(W, 1))
    )

    # print(W)
    # print(np.sum(np.power(diff, 2)), lambd * np.linalg.norm(W, 1))
    # print('diff, loss', diff, print(loss))

    return loss


# def callback(
#     W, X, y, A, A_base, pert_dic_all, lambd, err, reg, time_step, min_val, max_val
# ):
#     W = np.reshape(W, A[0, :, :].shape)
#     pred = pred_all_bound_single(X, y, W, A, pert_dic_all, time_step, min_val, max_val)

#     n_node = A_base.shape[0]
#     diff = get_diff(y, pred)[:n_node, :]

#     print("call back: ", np.sum(np.abs(diff)), np.sum(np.abs(diff)) == 0)
#     if np.sum(np.abs(diff)) == 0:
#         return W
#         raise ValueError("call back reached")

#     if np.sum(np.abs(diff)) == 0.0:
#         return True  # Return True to stop the optimization
