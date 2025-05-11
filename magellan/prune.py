"""Note that all matrices are indexed [child, parent]"""

from collections import ChainMap, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from torch_geometric.data import Data

from magellan.gnn_model import Net
from magellan.graph import Graph
from magellan.sci_opt import (
    base_adj,
    enforce_pert_dic_order,
    get_adj_single,
    get_data,
    get_sorted_node_list,
    update_function_mask,
)


def get_spec(
    input_df: pd.DataFrame | str | Path,
    spec_size: str,
    const_dic: dict | None = None,
    remove_unstable: bool = True,
    remove_duplicate: bool = True,
) -> OrderedDict:
    """
    Extract fake spec from BMA outputs

    :param df: pd.DataFrame or str.
        if str, df should be the full path of BMA output file results.csv or results_short.csv
    :param spec_size: str, 'full' or 'part'.
        if 'full', BMA output for all nodes will be extracted.
        else, BMA output for randomly selected nodes will be extracted (nodes selected in gen_synthetic)
    :param const_dic: dict, key: node name, value: int. predefined constant nodes
    :param remove_unstable: bool, whether or not to remove unstable output
    :param remove_duplicate: bool, whether or not to remove experiments with duplicated perturbations.
        if True, both the experiments (as a identifier) and expectations will be merged

    :return:
        pert_dic_all: dict, key: str, experiment name,
            value: None or dict.
            If dict, {'pert': {perturbed node name: perturbed value}, 'exp': {expected node name: expected value}}

    """

    if isinstance(input_df, str | Path):
        df: pd.DataFrame = pd.read_csv(input_df)
    else:
        df: pd.DataFrame = input_df

    if const_dic is None:
        const_dic = {}  # initialise with empty dict

    # remove unstable results
    if remove_unstable:
        df["lo_equalTo_hi"] = df["lo"] == df["hi"]
        df = df[df["lo_equalTo_hi"]]

    # create spec dic
    spec_dic = dict(
        zip(
            zip(
                zip(df["cell_line"], df["source"], df["experiment_particular"]),
                df["gene"],
            ),
            zip(df["mean_result"], df["perturbation"], df["expectation_bma"]),
        )
    )

    # create pert & exp dic
    pert_dic_all = {}
    for (experiment, gene), (val, perturb, expect) in spec_dic.items():
        experiment = "_".join([str(ele) for ele in experiment])

        if experiment not in pert_dic_all:
            pert_dic_all[experiment] = {"pert": const_dic.copy(), "exp": {}}

        if not pd.isna(perturb):  # perturbation node
            pert_dic_all[experiment]["pert"][gene] = int(
                val
            )  # cast this to int to avoid bugs when passing to BMA command line

        # expectation node
        # note: const_dic and perturbed nodes are also considered as expectation node if spec_size = 'full'
        if spec_size == "full":
            pert_dic_all[experiment]["exp"][gene] = val
        else:
            if not pd.isna(expect):
                pert_dic_all[experiment]["exp"][gene] = val

    # remove entire experiment if no expectation nodes are stable (thus result an empty exp dict in pert_dic_all)
    to_del = []
    for experiment, experiment_dic in pert_dic_all.items():
        if len(experiment_dic["exp"]) == 0:
            to_del.append(experiment)

    for experiment in to_del:
        del pert_dic_all[experiment]

    # remove and merge duplicated perturbations
    # dup_pert structure: dict: key: perturbed nodes, values: {experiment: exp}
    if remove_duplicate:
        dup_pert = defaultdict(dict)
        for experiment, dic in pert_dic_all.items():
            dup_pert[tuple(sorted(dic["pert"].items(), key=lambda t: t[0]))][
                experiment
            ] = pert_dic_all[experiment]["exp"]

        for pert, dic in dup_pert.items():
            merged_experiment = "|".join(sorted(dic.keys()))
            merged_exp = dict(ChainMap(*dic.values()))

            pert_dic_all[merged_experiment] = {"pert": dict(pert), "exp": merged_exp}

            if len(dic) > 1:
                for experiment in dic:
                    del pert_dic_all[experiment]

    return OrderedDict(pert_dic_all)


def filter_specification(
    pert_dic_small: OrderedDict, graph_nodes: set[str], verbose: bool = False
) -> OrderedDict:
    """
    Filter experimental specifications to only include nodes that exist in the graph.

    Args:
        pert_dic_small: Dictionary of experimental specifications
        graph_nodes: Set of valid node names in the graph
        verbose: Whether to print information about filtered nodes

    Returns:
        Dictionary containing only valid experiments with existing nodes

    Raises:
        ValueError: If input dictionary is malformed
    """
    filtered_spec = OrderedDict()
    removed_experiments = set()

    for experiment, specs in pert_dic_small.items():
        if not isinstance(specs, dict) or "pert" not in specs or "exp" not in specs:
            raise ValueError(f"Malformed experiment specification for {experiment}")

        # Check if all perturbation nodes exist in graph
        pert_dict = specs["pert"]
        if all(pert in graph_nodes for pert in pert_dict):
            # Filter expected outcomes to valid nodes
            filtered_exp = {
                key: value for key, value in specs["exp"].items() if key in graph_nodes
            }

            # Only keep experiment if it has valid outcomes
            if filtered_exp:
                filtered_spec[experiment] = {"pert": pert_dict, "exp": filtered_exp}
            else:
                removed_experiments.add(experiment)

    if verbose and removed_experiments:
        print(
            f"Removed {len(removed_experiments)} invalid experiments: {removed_experiments}"
        )

    return enforce_pert_dic_order(filtered_spec)


def check_paths_in_graph(G: nx.DiGraph, pert_dic_small: OrderedDict) -> None:
    """
    Checks if there are paths in the directed graph G from each 'pert' node to each 'exp' node in pert_dic_small.
    If a path does not exist, it prints a warning.

    Parameters:
        G (networkx.DiGraph): The directed graph representing the network.
        pert_dic_small (dict): The dictionary containing experiments, with 'pert' and 'exp' nodes.
    """
    for experiment, specs in pert_dic_small.items():
        pert_nodes = specs["pert"].keys()
        exp_nodes = specs["exp"].keys()

        for pert in pert_nodes:
            for exp in exp_nodes:
                if not nx.has_path(G, pert, exp):
                    print(
                        f"Warning: No path from '{pert}' to '{exp}' in experiment '{experiment}'."
                    )



def calc_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray | None = None,
    mask_val: str = "",
) -> dict:
    """
    Compute weighted F1-score, weighted Cohen's Kappa, MCC, and output a confusion matrix.

    Args:
        y_pred (array-like): Predicted labels.
        y_true (array-like): True labels.
        mask (array-like, optional): Mask to filter valid elements. Default is None.
        mask_val: Value in the mask to include. Default is ''.

    Returns:
        dict: Metrics including F1-score, Cohen's Kappa, MCC, and the confusion matrix.
    """
    if mask is not None:
        # Apply the mask to filter out invalid elements
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_pred[mask != mask_val] = np.nan
        y_true[mask != mask_val] = np.nan

    # Remove NaNs
    valid_indices = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]

    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Return results
    return {
        "weighted_f1": f1,
        "weighted_kappa": kappa,
        "mcc": mcc,
        "confusion_matrix": cm,
    }


def round_df(df: pd.DataFrame) -> pd.DataFrame:
    # round pandas.DataFrame

    """
    Round the values of a pandas.DataFrame to the nearest integer.
    :param df:
    :return:
    """

    df_round = df.copy() + 0.5
    df_round = df_round.apply(np.floor)

    return df_round


def calc_err(
    y_pred: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.DataFrame,
    mask: np.ndarray | pd.DataFrame | None = None,
    mask_val: str = "",
    return_nan_if_all_masked: bool = False,
) -> tuple[float, float]:
    """
    Calculate the sum and mean of the absolute error between two arrays.

    Args:
        y_pred: Predicted values as numpy array or pandas DataFrame
        y_true: True values as numpy array or pandas DataFrame
        mask: Optional mask array/DataFrame to filter values
        mask_val: Value in mask to keep (other values will be set to nan)
        return_nan_if_all_masked: If True, returns (nan, nan) when all values are masked.
            If False, returns (0, nan) when all values are masked.

    Returns:
        tuple containing:
            - float: Sum of absolute errors (ignoring nans). Returns nan or 0 if all masked
            - float: Mean absolute error (ignoring nans). Returns nan if all masked
    """

    # throw error is y_pred or y_true are empty
    if len(y_pred) == 0 or len(y_true) == 0:
        raise ValueError("y_pred or y_true cannot be empty")

    y_pred_masked = y_pred.copy()  # Create copy to avoid modifying input
    if mask is not None:
        y_pred_masked[mask != mask_val] = np.nan

    err = np.abs(np.asarray(y_pred_masked - y_true))

    # Check if all values are nan
    if np.all(np.isnan(err)):
        return (np.nan, np.nan) if return_nan_if_all_masked else (0.0, np.nan)

    return float(np.nansum(err)), float(np.nanmean(err))


def make_subplots(
    to_plot: list[tuple[Any, Any]],
    to_save: str,
    cmap: str = "RdBu_r",
    center: int = 0,
    vmin: int = -1,
    vmax: int = 1,
    figsize: tuple[int, int] = (18, 6),
    xticklabels: str = "auto",
    yticklabels: str = "auto",
    annot: Any = None,
    fmt: str = "",
):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=figsize, sharey=True)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.02, 0.4])  # type: ignore

    for i, pred in enumerate(to_plot):
        sns.heatmap(
            pred[0],
            cmap=cmap,
            center=center,
            ax=axs[i],
            vmin=vmin,
            vmax=vmax,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar=i == 0,
            cbar_ax=None if i else cbar_ax,
            annot=annot,
            fmt=fmt,
        )
        axs[i].set_title(pred[1])

    fig.tight_layout(rect=[0, 0, 0.9, 1])  # type: ignore
    plt.savefig(to_save, dpi=200)


def get_pert_dic(file_path: str | Path, const_dic: dict, spec_size: str) -> OrderedDict:
    """
    This function is used to extract the perturbation dictionary from a given results (spec_size = 'full')
    or specification (spec_size = 'non full') file.

    Parameters:
    path_data (str): The path to the directory where the result or spec file is located.
    result_or_spec_file (str): The name of the file from which to extract the perturbation dictionary.
    const_dic: constant dictionary generated from get_const_from_json
    spec_size (str): The size of the spec. It should be either 'full' or 'non full'.
    'full' means that the spec file should be a bma_results file and includes nodes without pre-defined expectation.
    'non full' means that the spec file does not include nodes without pre-defined expectation.

    Returns:
    output_pert_dic (OrderedDict): The extracted perturbation dictionary.
    The dictionary is sorted by keys to avoid different experiment orders in A and X/y.

    Raises:
    ValueError: If spec_size is not 'full' or 'non full'.
    """

    match spec_size:
        case "full":
            # extract spec (incl. nodes without pre-defined expectation)
            # note in this case the spec file should be a bma_results file
            output_pert_dic = get_spec(
                file_path,
                spec_size="full",
                const_dic=const_dic,
                remove_duplicate=True,
            )
            output_pert_dic = enforce_pert_dic_order(output_pert_dic)
            # sort pert dic keys to avoid different exp orders in A and X/y

        case "non full":
            # print(join(path_data, result_or_spec_file))
            output_pert_dic = get_spec(
                file_path,
                spec_size="non full",
                const_dic=const_dic,
                remove_unstable=False,
                remove_duplicate=True,
            )
            output_pert_dic = enforce_pert_dic_order(output_pert_dic)

        case _:
            raise ValueError('spec_size should be either "full" or "non full"')
    return output_pert_dic


# check if pert_dic contains values outside of min_val and max_val
def check_pert_dic_values(
    pert_dic: OrderedDict, min_val: float, max_val: float
) -> None:
    for k, v in pert_dic.items():
        for kk, vv in v["pert"].items():
            if vv < min_val or vv > max_val:
                raise ValueError(
                    f"Value {vv} for {kk} in {k} is outside of min_val and max_val for the BMA network."
                )


def add_dummy_nodes_and_generate_A_inh(
    G: nx.DiGraph | Graph,
    pert_dic: OrderedDict,
    max_range: int,
    tf_method: str = "avg",
) -> tuple[nx.DiGraph, list[pd.DataFrame], list, dict]:
    """
    Add dummy nodes to the graph and generate the adjacency matrix and inhibitor list.

    Note that this function modifies G and pert_dic in place

    Parameters:
    G (networkx.DiGraph): The directed graph representing the network.
    pert_dic (dict): The dictionary containing experiments, with 'pert' and 'exp' nodes.
    max_range (int): The maximum range for nodes.
    tf_method (str): The method to use for the update function, either "avg" or "sum".
    Returns:
    tuple: A tuple containing the updated graph, adjacency matrix, node list, inhibitor list, and inhibitor dictionary.
    """
    pert_dic = enforce_pert_dic_order(pert_dic)
    # create a dummy node to lock index 0
    G.add_node("0A_node00")
    # A, inh = get_adj(G, pert_dic, node_list, method=tf_method)
    A_mult = get_adjacency_matrix_mult(G, method=tf_method)
    inh = get_inh(G)
    Adjacency_per_experiment = get_experiment_array(A_mult, inh, pert_dic, G)

    # add dummy node to G to allow control of inhibitor-only nodes
    for ele in inh:
        dummy_node = "dummy_%s" % ele
        G.add_edge(dummy_node, ele, sign="Activator")
        G.add_edge(dummy_node, dummy_node, sign="Activator")

    inh = ["dummy_%s" % ele for ele in inh]
    # node_list = node_list + inh  # add dummy node to node list
    inh_dic = {ele: max_range for ele in inh}

    # sort A rows and columns to match sorted(G.nodes()) This MUST occur AFTER the dummy nodes are added
    node_list = get_sorted_node_list(G)
    # Ensure A matrices have consistent row/column order with node_list
    Adjacency_per_experiment = [
        pd.DataFrame(a, index=node_list, columns=node_list)
        for a in Adjacency_per_experiment
    ]
    Adjacency_per_experiment = [
        a.loc[node_list, node_list] for a in Adjacency_per_experiment
    ]  # Reorder both rows and columns

    for v in pert_dic.values():
        v["pert"].update(
            inh_dic
        )  # add dummy parents to all-inhibitor nodes as perturbations
    return G, Adjacency_per_experiment, inh, inh_dic


def dummy_setup(
    G: nx.DiGraph | Graph,
    pert_dic_small: OrderedDict,
    train_pert_dic: OrderedDict,
    test_pert_dic: OrderedDict | None,
    max_range: int,
    tf_method: str,
):
    """
    Set up G, inh with dummy nodes for test and train
    Get Adjacency matrix for train and test

    Note that this function modifies G and pert_dic in place

    :param G: networkx graph
    :param pert_dic_small: dictionary of perturbations
    :param max_range: maximum range of the network
    :param tf_method: method to use for the adjacency matrix
    :return: G, Adjacency_per_experiment_train, Adjacency_per_experiment_test
    """

    G_no_dummy_1 = G.copy()
    G_no_dummy_2 = G.copy()
    G_no_dummy_3 = G.copy()
    # Set up G, inh with dummy nodes for test and train
    G, _, inh, _ = add_dummy_nodes_and_generate_A_inh(
        G_no_dummy_1,  # type: ignore
        pert_dic_small,
        max_range,
        tf_method,
    )

    # Get Adjacency matrix for train and test
    _, Adjacency_per_experiment_train, _, _ = add_dummy_nodes_and_generate_A_inh(
        G_no_dummy_2,  # type: ignore
        train_pert_dic,
        max_range,
        tf_method,
    )
    if test_pert_dic is not None:
        _, Adjacency_per_experiment_test, _, _ = add_dummy_nodes_and_generate_A_inh(
            G_no_dummy_3,  # type: ignore
            test_pert_dic,
            max_range,
            tf_method,
        )
    else:
        Adjacency_per_experiment_test = None

    return G, inh, Adjacency_per_experiment_train, Adjacency_per_experiment_test


def check_dummy_nodes(G: nx.DiGraph | Graph) -> None:
    if not isinstance(G, (nx.DiGraph, Graph)):
        raise TypeError("G must be a networkx.DiGraph or BMATool.Graph")

    node_list = get_sorted_node_list(G)

    if not all(isinstance(node, str) for node in node_list):
        raise TypeError("All elements in node_list must be strings")

    if any("dummy" in node.lower() for node in node_list):
        dummy_nodes = [node for node in node_list if "dummy" in node.lower()]
        raise ValueError(
            f"Nodes containing the word 'dummy' are not allowed in the node list. The following nodes are dummy nodes: {dummy_nodes}"
        )

    if any("0A_node00".lower() in node.lower() for node in node_list):
        raise ValueError("Node '0A_node00' is not allowed in the node list")


def has_self_loop(G: nx.Graph, node):
    try:
        if G[node][node] is not None:
            return True
    except Exception:
        return False


def identify_self_loops(G: nx.DiGraph) -> list[str]:
    return [node for node in G.nodes if has_self_loop(G, node)]


def check_no_self_loops(G: nx.DiGraph) -> None:
    if any(has_self_loop(G, node) for node in G.nodes):
        raise ValueError(
            f"The input graph contains self-loops. These are not currently supported. The following nodes have self-loops: {identify_self_loops(G)}"
        )


def graph_node_list_checks(G: nx.DiGraph) -> None:
    if G.number_of_nodes() == 0:
        raise ValueError("The graph G is empty.")
    node_list = get_sorted_node_list(G)
    if set(node_list) != set(G.nodes):
        raise ValueError("The node_list does not match the nodes in the graph G.")

    if node_list is None or len(node_list) == 0:
        raise ValueError("node_list cannot be None or empty")

    return None


def get_inh(G: nx.DiGraph) -> list:
    A_base = base_adj(G)
    A_inh = (A_base <= 0).all(axis=1) & (~(A_base == 0).all(axis=1))
    inh = sorted(list(A_inh[A_inh].index))
    return inh


def get_adjacency_matrix_mult(G: nx.DiGraph, method: str = "avg") -> pd.DataFrame:
    """
    Get the adjacency matrix of the graph with the update function applied.

    :param G: networkx.DiGraph
    :param node_list: list of nodes in the graph
    :param method: "avg" or "sum"
    :return: pd.DataFrame
    """
    graph_node_list_checks(G)
    node_list = get_sorted_node_list(G)

    A_base = base_adj(G)
    A_mult = pd.DataFrame(
        update_function_mask(A_base, 1, method)
        + update_function_mask(A_base, -1, method),
        index=node_list,
        columns=node_list,
    )
    return A_mult


def get_experiment_array(
    A_mult: pd.DataFrame, inh: list, pert_dic_all: dict, G: nx.DiGraph
) -> list[pd.DataFrame]:
    """
    Get the experiment array for each experiment from the adjacency matrix and the perturbation dictionary by adding dummy nodes for all inhibitor nodes, and replacing parents with dummy nodes when perturbed.

    :param A_mult: pd.DataFrame, the adjacency matrix
    :param inh: list, the inhibitor nodes
    :param pert_dic_all: dict, the perturbation dictionary
    :param node_list: list, the node list
    """

    # node_list = get_sorted_node_list(G)
    A: list[pd.DataFrame] = []
    for pert_dic in pert_dic_all.values():
        pert_dic = pert_dic["pert"]
        A.append(get_adj_single(A_mult, inh, pert_dic, G))

    # A = np.array(A)
    return A


def get_data_and_update_y(
    pert_dic: OrderedDict, G: nx.DiGraph, replace_missing_with_zero: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get data and create perturbation and expectation matrices where the perturbations are also included as expected values.

    pert_dic: dictionary containing perturbation information
    node_list: list of nodes in the graph

    Returns:
    X: perturbation matrix
    y: expectation matrix
    """
    pert_dic = enforce_pert_dic_order(pert_dic)
    X, y = get_data(pert_dic, G, replace_missing_with_zero)
    for experiment_name, experiment_data in pert_dic.items():
        for perturbed_node, perturbation_value in experiment_data["pert"].items():
            X.at[perturbed_node, experiment_name] = perturbation_value
    return X, y


def get_pert_list(pert_dic: OrderedDict, inh: list) -> list:
    """get perturbed nodes list from pert_dic"""
    pert_dic = enforce_pert_dic_order(pert_dic)
    pert_list = set()
    for v in pert_dic.values():
        pert_list.update(set(v["pert"].keys()) - set(inh))
    return sorted(list(pert_list))


def make_node_dic(G: nx.DiGraph) -> dict:
    """
    Create a dictionary mapping node names to indices.

    Args:
        node_list: List of node names

    Returns:
        dict: Mapping from node names to indices

    Raises:
        ValueError: If duplicate nodes are found in the list
    """
    node_list = get_sorted_node_list(G)
    # Removed, as nx.DiGraph() does not allow duplicate nodes, checked in json_to_graph() instead
    # duplicates = [node for node in list(node_list) if node_list.count(node) > 1]
    # if duplicates:
    #    raise ValueError(f"Duplicate nodes found in node list: {duplicates}")

    return dict(zip(node_list, range(len(node_list))))


def make_pert_idx(pert_list: list, node_dic: dict) -> list:
    """make perturbation index from perturbed nodes list and node dictionary"""
    return [node_dic[ele] for ele in pert_list]


def make_edge_idx(matrix: pd.DataFrame, pert_idx: list) -> torch.Tensor:
    """
    This function extracts the edge index from the adjacency matrix of a graph.

    Parameters:
    matrix (numpy.ndarray): The adjacency matrix of the graph.
    pert_idx (list): List of indices of the perturbed nodes in the graph.

    Returns:
    torch.Tensor: A tensor representing the edge index of the graph. The edge index is a 2D tensor where each column
    represents an edge. The first row of the column represents the source node and the second row represents the
    destination node. For example, if the edge index is [[0, 1, 2], [2, 1, 0]], there are three edges in the graph:
    0 --> 2, 1 --> 1, and 2 --> 0.

    The function works as follows:
    1. It first finds the indices where the adjacency matrix is not zero. These indices represent the edges in the graph.
    2. It then stacks these indices vertically to create a 2D array.
    3. It adds the indices of the perturbed nodes to this 2D array. This is done because in the graph, each perturbed
       node has a self-loop.
    4. Finally, it converts this 2D array into a PyTorch tensor and returns it.
    """

    edge_idx = np.where(np.asarray(matrix.T) != 0)
    edge_idx = np.vstack(edge_idx)
    edge_idx = np.hstack((edge_idx, np.vstack((pert_idx, pert_idx))))
    edge_idx = torch.tensor(edge_idx).to(torch.int64)
    return edge_idx


def create_pyg_data_object(
    X: pd.DataFrame, y: pd.DataFrame, edge_idx: torch.Tensor
) -> Data:
    """
    Create PyG data object using input features X, target labels y, and edge index edge_idx.

    Parameters:
    - X (numpy.ndarray): Input features.
    - y (numpy.ndarray): Target labels.
    - edge_idx (numpy.ndarray): Edge index.

    Returns:
    - data (torch_geometric.data.Data): PyG data object.
    """
    data = Data(
        x=torch.tensor(np.asarray(X), dtype=torch.float32),
        y=torch.tensor(np.asarray(y), dtype=torch.float32),
        edge_index=edge_idx.clone().detach(),
    )
    return data


def create_edge_scale(A_mult: pd.DataFrame, pert_idx: list) -> torch.Tensor:
    """
    Create edge scaling tensor by combining adjacency matrix values with perturbation indices.

    This function:
    1. Extracts non-zero values from the transposed adjacency matrix
    2. Appends ones for each perturbation index
    3. Converts the result to a PyTorch tensor

    Parameters:
    -----------
    A_mult : pd.DataFrame
        Adjacency matrix where each entry represents edge weights
    pert_idx : list
        List of indices for perturbed nodes

    Returns:
    --------
    torch.Tensor
        Edge scaling tensor containing:
        - Original non-zero edge weights from adjacency matrix
        - Ones for each perturbation index

    Example:
    --------
    If A_mult has non-zero values [0.5, -0.3] and pert_idx has length 2,
    the result would be tensor([0.5, -0.3, 1.0, 1.0])
    """

    # Extract non-zero values from transposed adjacency matrix
    A_trans = np.asarray(A_mult.T)
    non_zero_edges = A_trans[np.where(A_trans != 0)]

    # Create array of ones for perturbation indices
    pert_ones = np.ones(len(pert_idx))

    # Combine edge weights with perturbation ones
    edge_scale = np.hstack((non_zero_edges, pert_ones))

    # Convert to PyTorch tensor
    return torch.tensor(edge_scale, dtype=torch.float32)


def construct_mask_dic(
    pert_dic: OrderedDict,
    node_dic: dict,
    edge_idx: torch.Tensor,
    mask_debug: bool = False,
) -> dict:
    """
    Construct a dictionary of masks for each experiment index.

    Parameters:
    - pert_dic: dictionary where keys are experiment indices and values contain perturbation information
    - node_dic: dictionary that maps node names to indices
    - edge_idx: 2D array containing edge indices

    Returns:
    - mask_dic: dictionary where keys are experiment indices and values are binary lists representing masks
    1: self-loops for perturbed nodes & (actual edges - parents of perturbed nodes)
    0: parents of perturbed nodes & self loops for non-perturbed nodes
    """
    pert_dic = enforce_pert_dic_order(pert_dic)
    if mask_debug:
        node_names = {v: k for k, v in node_dic.items()}

    mask_dic = {}
    for experiment_name, experiment_data in pert_dic.items():
        mask_dic[experiment_name] = torch.ones(edge_idx.shape[1])
        pert_idx = [node_dic[ele] for ele in experiment_data["pert"]]

        if mask_debug:
            print(f"Processing perturbation: {experiment_name}")
            print("Edges being masked:")

        # set to zero edges to parents of perturbed nodes (excl. self loops)
        for idx in pert_idx:
            mask_dic[experiment_name][
                torch.where(
                    (edge_idx[1, :] == idx) & (edge_idx[0, :] != edge_idx[1, :])
                )[0]
            ] = 0

            # Print masked edges with node names
            if mask_debug:
                masked_edges = edge_idx[
                    :, (edge_idx[1, :] == idx) & (edge_idx[0, :] != idx)
                ]
                for edge in masked_edges.t().tolist():
                    source, target = edge
                    edge_name = (
                        f"Edge from '{node_names[source]}' to '{node_names[target]}'"
                    )
                    print(edge_name)

        # set to zero self loops for non-perturbed nodes
        mask_dic[experiment_name][
            torch.where(
                (edge_idx[0, :] == edge_idx[1, :])
                & (~torch.isin(edge_idx[0, :], torch.tensor(pert_idx)))
            )[0]
        ] = 0

        # Print self-loop edges that are masked
        if mask_debug:
            non_perturbed_masked_edges = edge_idx[
                :,
                (edge_idx[0, :] == edge_idx[1, :])
                & (~torch.isin(edge_idx[0, :], torch.tensor(pert_idx))),
            ]
            for edge in non_perturbed_masked_edges.t().tolist():
                source, target = edge
                edge_name = f"Self-loop on '{node_names[source]}'"
                print(edge_name)
            print()  # New line for better readability between perturbation outputs

    return mask_dic


def create_pert_mask(edge_idx: torch.Tensor, node_dic: dict) -> torch.Tensor:
    """
    Create a perturbation mask based on self-loops and relationships between dummy nodes and children nodes.

    Parameters:
    - edge_idx (torch.Tensor): Edge indices of the graph.
    - node_dic (dict): Dictionary mapping node names to their corresponding indices.

    Returns:
    - pert_mask (torch.Tensor): Perturbation mask with self-loops and connections from dummy nodes to children nodes set to 1.
    """

    # Get the set of valid node indices
    valid_indices = set(node_dic.values())

    # Check if all indices in edge_idx are valid
    all_indices = torch.unique(edge_idx)
    invalid_indices = [
        idx.item() for idx in all_indices if idx.item() not in valid_indices
    ]

    if invalid_indices:
        raise ValueError(
            f"Edge index contains invalid node indices {invalid_indices}. "
            f"Valid indices are {sorted(list(valid_indices))}"
        )

    pert_mask = torch.zeros(edge_idx.shape[1]).to(torch.int64)
    pert_mask[edge_idx[0, :] == edge_idx[1, :]] = 1  # Set self-loops to 1

    dummy_idx = [v for k, v in node_dic.items() if "dummy" in k]
    pert_mask[torch.isin(edge_idx[0, :], torch.tensor(dummy_idx))] = 1

    return pert_mask


# def initialize_model(
#     edge_idx: torch.Tensor,
#     min_range: int,
#     max_range: int,
#     max_update: bool,
#     round_val: bool,
#     learning_rate: float,
# ) -> tuple[Net, nn.Module, optim.Optimizer, torch.Tensor]:
#     """
#     Initializes a neural network model for optimization.

#     Parameters:
#     - edge_idx (torch.Tensor): Tensor containing edge indices
#     - min_range (float): Minimum value for range initialization
#     - max_range (float): Maximum value for range initialization
#     - max_update (bool): Whether to use maximum update
#     - round_val (bool): Whether to round values
#     - learning_rate (float): Learning rate for optimization

#     Returns:
#     - model (Net): Initialized neural network model
#     - loss_func (torch.nn.Module): Loss function
#     - opt (torch.optim.Optimizer): Optimizer
#     - edge_idx_copy (torch.Tensor): Copy of edge indices
#     """

#     edge_idx_copy: torch.Tensor = edge_idx.detach().clone()
#     edge_weight: torch.Tensor = torch.ones(edge_idx.shape[1]) * 0.5

#     init_model: Net = Net(
#         edge_weight=edge_weight,
#         min_val=min_range,
#         max_val=max_range,
#         n_iter=15,
#         max_update=max_update,
#         round_val=round_val,
#     )
#     loss_func: nn.Module = nn.MSELoss()
#     opt: optim.Optimizer = optim.SGD(init_model.parameters(), lr=learning_rate)

#     return init_model, loss_func, opt, edge_idx_copy


# def train_model(
#     training_length: int,
#     pert_dic: dict,
#     node_dic: dict,
#     edge_idx_copy: torch.Tensor,
#     model: nn.Module,
#     data: torch.Tensor,
#     mask_dic: dict,
#     edge_scale: torch.Tensor,
#     pert_mask: torch.Tensor,
#     loss_func: nn.Module,
#     opt: optim.Optimizer,
# ) -> tuple[list, list]:
#     """
#     Train the model for a specified number of iterations.

#     Args:
#         training_length (int): Number of training iterations.
#         pert_dic (dict): Dictionary of perturbations.
#         node_dic (dict): Dictionary of nodes.
#         edge_idx_copy (torch.Tensor): Copy of edge indices.
#         model (torch.nn.Module): Neural network model.
#         data (torch.Tensor): Input data.
#         mask_dic (dict): Dictionary of masks.
#         edge_scale (float): Scale of the edges.
#         pert_mask (torch.Tensor): Perturbation mask.
#         loss_func: Loss function.
#         opt: Optimizer.

#     Returns:
#         total_loss (list): List of losses during training.
#         sum_grad (list): List of sum of gradients during training.
#     """
#     total_loss = []
#     sum_grad = []
#     for i in range(training_length):  # usually I use 500
#         for j, k in enumerate(pert_dic):
#             idx_exp = [node_dic[ele] for ele in pert_dic[k]["exp"]]

#             edge_idx = edge_idx_copy.detach().clone()
#             pred = model(
#                 data.x[:, j].reshape([-1, 1]),
#                 edge_index=edge_idx,
#                 edge_mask=mask_dic[k].to(torch.int64),
#                 edge_scale=edge_scale,
#                 pert_mask=pert_mask,
#             )

#             opt.zero_grad()
#             loss = loss_func(pred[idx_exp], data.y[idx_exp, j].reshape([-1, 1]))
#             loss.backward()
#             opt.step()

#             total_loss.append(loss.detach().numpy())
#             sum_grad.append(model.edge_weight.grad.sum().numpy())

#             # print(i, j, loss)
#             print(f"Iteration: {i}, Experiment: {j}, Loss: {loss.item()}")

#     return model, total_loss, sum_grad


def predict_nn(
    model: Net,
    y: pd.DataFrame,
    data: Data,
    pert_dic_small: OrderedDict,
    A_mult: pd.DataFrame,
    pert_idx: list,
    mask_dic: dict,
    edge_scale: torch.Tensor,
    pert_mask: torch.Tensor,
) -> pd.DataFrame:
    """
    Predictions of the neural network model using the estimated parameters.

    Args:
        model (Net): The neural network model.
        y (pd.DataFrame): The true experimental values.
        data (Data): The PyG data object.
        pert_dic_small (dict): The perturbation dictionary.
        A_mult (pd.DataFrame): The adjacency matrix.
        pert_idx (list): The perturbation indices.
        mask_dic (dict): The mask dictionary.
        edge_scale (torch.Tensor): The edge scaling factors.
        pert_mask (torch.Tensor): The perturbation mask.
    """
    if data is None:
        raise ValueError("data is None")
    if data.x is None:
        raise ValueError("data.x is None")
    if data.y is None:
        raise ValueError("data.y is None")
    if model is None:
        raise ValueError("model is None")
    if y is None:
        raise ValueError("y is None")
    if pert_idx is None:
        raise ValueError("pert_idx is None")
    if mask_dic is None:
        raise ValueError("mask_dic is None")
    if edge_scale is None:
        raise ValueError("edge_scale is None")
    if pert_mask is None:
        raise ValueError("pert_mask is None")
    if pert_dic_small is None:
        raise ValueError("pert_dic_small is None")

    pert_dic_small = enforce_pert_dic_order(pert_dic_small)
    if data.x is not None:
        pred_nn = pd.DataFrame(index=y.index, columns=y.columns)
        for j, k in enumerate(pert_dic_small):
            pred_nn[k] = (
                model(
                    data.x[:, j].reshape([-1, 1]),
                    edge_index=make_edge_idx(A_mult, pert_idx).detach().clone(),
                    edge_mask=mask_dic[k].to(torch.int64),
                    edge_scale=edge_scale,
                    pert_mask=pert_mask,
                )
                .detach()
                .numpy()
            )
        return pred_nn
    else:
        raise ValueError("data.x is None")


def get_real_indices(y: pd.DataFrame) -> list[str]:
    """Get indices excluding dummy nodes and node00."""
    return [ele for ele in y.index if (("dummy" not in ele) and ("node00" not in ele))]


def calculate_errors_by_type(
    predictions: list[tuple[pd.DataFrame, str]],
    y: pd.DataFrame,
    annot: pd.DataFrame,
    mask_val: str,
    idx_real: list[str],
) -> dict[str, tuple[float, float]]:
    """Calculate errors for each prediction type against true values.

    Args:
        predictions: List of (prediction DataFrame, name) tuples
        y: True values DataFrame
        annot: Annotation matrix
        mask_val: Value in annotation matrix to filter by
        idx_real: List of indices to use

    Returns:
        Dict mapping prediction name to (sum_error, mean_error) tuple
    """

    if len(predictions) == 0 or predictions is None:
        raise ValueError("predictions is empty")

    errors = {}
    for pred, name in predictions:
        errors[name] = calc_err(
            pred.loc[idx_real],
            y.loc[idx_real],
            mask=annot.loc[idx_real],
            mask_val=mask_val,
        )
    return errors


def calculate_error_by_gene(
    y: pd.DataFrame,
    pred: pd.DataFrame,
    idx_real: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Calculate RMSE and MAE per gene.

    Args:
        y: True values DataFrame
        pred: Predicted values DataFrame
        idx_real: List of valid indices

    Returns:
        Tuple of (RMSE Series, MAE Series) with gene names as index
    """

    # Check for empty DataFrames
    if y.empty or pred.empty:
        raise ValueError("Input DataFrames cannot be empty")

    # Check for DataFrames with no actual data
    if len(y.columns) == 0 or len(pred.columns) == 0:
        raise ValueError("Input DataFrames must contain at least one column of data")

    # Check if all values are NaN
    if y.loc[idx_real].isna().all().all() or pred.loc[idx_real].isna().all().all():
        raise ValueError(
            "Input DataFrames contain no valid data for the specified indices"
        )

    if not y.index.equals(pred.index):
        raise ValueError("y and pred indices do not match")

    if not y.columns.equals(pred.columns):
        raise ValueError("y and pred columns do not match")

    err = y.loc[idx_real] - pred.loc[idx_real]
    # rmse = pd.Series(np.sqrt((err**2).mean(axis=1)), index=y.index)
    # mae = pd.Series(err.abs().mean(axis=1), index=y.index)
    rmse = pd.Series(
        np.sqrt((err**2).mean(axis=1)), index=idx_real
    )  # Only include idx_real
    mae = pd.Series(err.abs().mean(axis=1), index=idx_real)  # Only include idx_real
    return rmse, mae


def get_trained_network(
    G_original: nx.DiGraph,
    W: pd.DataFrame,
) -> nx.DiGraph:
    """Get trained network from original graph and weights.

    :param G_original: networkx.DiGraph. original graph
    :param W: pd.DataFrame. trained weights
    :return: networkx.DiGraph. trained graph
    """

    G_trained = nx.DiGraph(G_original.copy())

    # Assign trained weights to original graph
    for u, v in G_trained.edges:
        G_trained[u][v]["edge_weight"] = W.at[v, u]
    return G_trained


def filter_spec_by_parents(
    pert_dic: dict, G: nx.DiGraph, verbose: bool = False
) -> dict:
    """Filter the specification to remove expectation nodes with no parents in the graph.
    These nodes cannot vary and so cannot be reliable fit to.

    Args:
        pert_dic: Dict with experiment specifications
        G: Directed graph representing the network
        verbose: Whether to print information about removed nodes

    Returns:
        Filtered perturbation dictionary with invalid expectation nodes removed
    """
    # Find nodes with no parents in the graph
    nodes_with_no_parents = {
        node for node in G.nodes() if len(list(G.predecessors(node))) == 0
    }

    filtered_pert_dic = {}
    removed_nodes = set()
    removed_experiments = set()

    for experiment, specs in pert_dic.items():
        # Make copies to avoid modifying original data
        filtered_exp = {
            k: v for k, v in specs["exp"].items() if k not in nodes_with_no_parents
        }

        # If there are no expectation nodes left, skip this experiment
        if len(filtered_exp) == 0:
            removed_experiments.add(experiment)
            removed_nodes.update(specs["exp"].keys())
            continue

        # Otherwise keep the experiment with filtered expectation nodes
        filtered_pert_dic[experiment] = {
            "pert": specs["pert"].copy(),
            "exp": filtered_exp,
        }

        # Track removed nodes
        removed_nodes.update(set(specs["exp"].keys()) - set(filtered_exp.keys()))

    if verbose and removed_nodes:
        print(
            f"Removed {len(removed_nodes)} expectation nodes with no parents: {removed_nodes}"
        )
        if removed_experiments:
            print(
                f"Removed {len(removed_experiments)} experiments with no valid expectation nodes"
            )

    return filtered_pert_dic


def filter_spec_by_children(
    pert_dic: dict, G: nx.DiGraph, aggressive: bool = False, verbose: bool = False
) -> dict:
    """Filter the specification to remove perturbed nodes with no children in the graph.
    These nodes cannot change any downstream nodes and so cannot be a measurable perturbation.

    Args:
        pert_dic: Dict with experiment specifications
        G: Directed graph representing the network
        aggressive: If True, remove experiment if ANY perturbation has no children;
                   if False, remove experiment only if ALL perturbations have no children
        verbose: Whether to print information about removed nodes

    Returns:
        Filtered perturbation dictionary with invalid perturbation nodes removed
    """
    # Find nodes with no children in the graph
    nodes_with_no_children = {
        node for node in G.nodes() if len(list(G.successors(node))) == 0
    }

    filtered_pert_dic = {}
    removed_nodes = set()
    removed_experiments = set()

    for experiment, specs in pert_dic.items():
        # Make copies to avoid modifying original data
        filtered_pert = {
            k: v for k, v in specs["pert"].items() if k not in nodes_with_no_children
        }

        # Track removed nodes
        removed_in_this_exp = set(specs["pert"].keys()) - set(filtered_pert.keys())
        removed_nodes.update(removed_in_this_exp)

        # Apply filtering strategy
        if (aggressive and len(removed_in_this_exp) > 0) or len(filtered_pert) == 0:
            # In aggressive mode, remove experiment if ANY perturbation is invalid
            # In non-aggressive mode, remove only if ALL perturbations are invalid
            removed_experiments.add(experiment)
            continue

        # Otherwise keep the experiment with filtered perturbation nodes
        filtered_pert_dic[experiment] = {
            "pert": filtered_pert,
            "exp": specs["exp"].copy(),
        }

    if verbose and removed_nodes:
        removal_mode = "aggressive" if aggressive else "non-aggressive"
        print(f"Using {removal_mode} filtering mode")
        print(
            f"Removed {len(removed_nodes)} perturbation nodes with no children: {removed_nodes}"
        )
        if removed_experiments:
            print(
                f"Removed {len(removed_experiments)} experiments with invalid perturbation nodes"
            )

    return filtered_pert_dic


def filter_spec_invalid_experiments(
    pert_dic: OrderedDict,
    G: nx.DiGraph,
    filters: list[Callable] | None = None,
    aggressive: bool = False,
    verbose: bool = False,
) -> OrderedDict:
    """Filter the specification to remove invalid experiments that cannot be fit to the graph.
    An experiment is considered invalid if:
    - Its experimental nodes have no parents in the graph (cannot vary)
    - Its perturbation nodes have no children in the graph (cannot affect anything)

    Args:
        pert_dic: Dict with experiment specifications
        G: Directed graph representing the network
        filters: List of filtering functions to apply
        aggressive: If True, remove experiment if ANY perturbation has no children
        verbose: Whether to print information about removed nodes

    Returns:
        Filtered perturbation dictionary
    """
    if filters is None:
        filters = [filter_spec_by_parents, filter_spec_by_children]

    filtered_dic = pert_dic
    for filter_fn in filters:
        if filter_fn == filter_spec_by_children:
            filtered_dic = filter_fn(
                filtered_dic, G, aggressive=aggressive, verbose=verbose
            )
        else:
            filtered_dic = filter_fn(filtered_dic, G, verbose=verbose)

    return enforce_pert_dic_order(filtered_dic)
