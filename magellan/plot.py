import os
from collections import OrderedDict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import colormaps
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from magellan.json_io import (
    gen_json,
    get_const_from_json,
    get_pos_from_json,
    json_to_graph,
)
from magellan.prune import (
    calc_err,
    get_trained_network,
)
from magellan.pydot_io import graph_to_pydot
from magellan.sci_opt import enforce_pert_dic_order

try:
    import pygraphviz as pgv
except ImportError:
    pgv = None

os.environ["PATH"] += os.pathsep + "/usr/local/bin"

# Terminal colors
ANSI_COLORS = {"red": "\033[91m", "yellow": "\033[93m", "reset": "\033[0m"}


def _dot_from_edge(edge_set: set[tuple[str, str]]) -> str:
    return ";\n".join(["%s->%s" % (u, v) for u, v in edge_set])


def _get_edge_set(node_set: set[str], G: nx.DiGraph, thre: int) -> set[tuple[str, str]]:
    node_comb = product(node_set, node_set)
    node_comb = {(u, v) for u, v in node_comb if u != v}

    edge_set = set()
    for u, v in node_comb:
        if u in G.nodes and v in G.nodes:
            try:
                path = nx.algorithms.shortest_path(G, u, v)
                if len(path) <= thre:
                    edge_set.add((u, v))
            except nx.exception.NetworkXNoPath:
                continue

    return edge_set


def _pgv_subgraph(
    node_set: set[str], G: nx.DiGraph, cluster_name: str = "cluster", thre: int = 3
) -> tuple[str, set[tuple[str, str]]]:
    """
    Create subgraph in DOT

    :param node_set: set of node
    :param G: networkx DiGraph
    :param cluster_name: str, name of cluster

    :return G_sub_dot: str, DOT subgraph containing nodes in node_set

    """

    edge_set = _get_edge_set(node_set, G, thre)
    sub_dot = _dot_from_edge(edge_set)

    G_sub_dot = "subgraph cluster_%s {%s}\n" % (cluster_name, sub_dot)

    return G_sub_dot, edge_set


def pgv_dot(gene_sets: dict[str, set[str]], G: nx.DiGraph, thre: int = 3) -> str:
    """
    Create DOT graph, with subgraphs from genes of the same type (e.g. mut, pheno)

    :param gene_dic: dict, key: gene type, value: gene set
    :param G: networkx DiGraph

    :return: G_dot: str, DOT graph for layered G

    """

    G_dot = ""
    edge_same_type = set()

    # add subgraph constructed from single-type genes
    for k, v in gene_sets.items():
        if k == "deg":  # no separate subgraph for DEGs
            continue

        G_sub_dot, edge_set = _pgv_subgraph(v, G, cluster_name=k, thre=thre)

        G_dot += G_sub_dot
        edge_same_type = edge_same_type.union(edge_set)

    # add all other edges
    edge_diff_type = set(G.edges) - edge_same_type
    G_dot += _dot_from_edge(edge_diff_type)

    G_dot = 'strict digraph ""{orientation=90; ratio=compress;rankdir="TB";%s}' % G_dot

    return G_dot


def plot_pgv(
    G: nx.DiGraph,
    gene_sets: dict[str, set[str]],
    G_dot: str | None = None,
    thre: int = 3,
    path: str = "",
    file_name: str = "test pgv",
) -> None:
    """
    Generate and plot graph visualization, using pygraphviz if available,
    otherwise falling back to networkx drawing.

    Args:
        G_dot: DOT graph string representation
        G: NetworkX directed graph
        gene_sets: Dictionary mapping gene types to sets of genes
        thre: Threshold for edge inclusion
        path: Directory for saved figure
        file_name: Name of saved figure
    """
    if path:
        os.makedirs(path, exist_ok=True)

    if pgv is not None:
        # Use pygraphviz if available
        if not G_dot:
            G_dot = pgv_dot(gene_sets, G, thre)

        G_pgv = pgv.agraph.AGraph(G_dot)
        G_pgv.draw(f"{path}/{file_name}.png", prog="dot")
    else:
        print("pygraphviz is not installed, falling back to networkx drawing")
        # Fallback to networkx drawing
        plt.figure(figsize=(12, 8))

        # Create a layout that separates different gene types
        pos = nx.spring_layout(G)

        # Draw nodes for each gene type with different colors
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for i, (gene_type, nodes) in enumerate(gene_sets.items()):
            if gene_type == "deg":  # Skip DEGs as per original function
                continue

            # Get subset of nodes that exist in the graph
            valid_nodes = [n for n in nodes if n in G.nodes]
            if valid_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=valid_nodes,
                    node_color=colors[i % len(colors)],
                    node_size=500,
                    alpha=0.6,
                    label=gene_type,
                )

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True)

        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Network Visualization")
        plt.legend()
        plt.axis("off")
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{path}/{file_name}.png", dpi=300, bbox_inches="tight")
        plt.close()


# Pruning


def plot_comparison_heatmaps(
    to_plot: list[tuple[pd.DataFrame, str]],
    to_save: str | Path,
    cmap: str = "RdBu_r",
    center: float = 0,
    vmin: float = -1,
    vmax: float = 1,
    figsize: tuple[int, int] = (18, 6),
    xticklabels: str = "auto",
    yticklabels: str = "auto",
    annot: pd.DataFrame | None = None,
    fmt: str = "",
):
    """
    Plot a heatmap of the predicted values and True values.

    Args:
        to_plot: List of tuples containing (predicted values, title)
        to_save: Path to save the output figure, including filename
        cmap: Colormap for the heatmap. Defaults to "RdBu_r"
        center: Center value for the heatmap. Defaults to 0
        vmin: Minimum value for the heatmap. Defaults to -1
        vmax: Maximum value for the heatmap. Defaults to 1
        figsize: Figure dimensions (width, height) in inches. Defaults to (18, 6)
        annot: Annotation matrix to use for the heatmap. Defaults to None
        fmt: Format string for the annotation. Defaults to ""
    """

    if to_plot is None or len(to_plot) == 0:
        raise ValueError("to_plot is empty")

    if len(to_plot) != 3:
        raise ValueError("to_plot must contain 3 elements")

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


def plot_training_metrics(
    total_loss: list[float],
    sum_grad: list[float],
    n_experiments: int,
    output_path: str | Path,
    dpi: int = 200,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Plot training metrics (loss and gradient) over epochs.

    Args:
        total_loss: List of loss values for each iteration
        sum_grad: List of gradient sum values for each iteration
        n_experiments: Number of experiments per epoch (length of pert_dic)
        output_path: Path to save the output figure, including filename
        dpi: DPI for saved figure. Defaults to 150
        figsize: Figure dimensions (width, height) in inches. Defaults to (12, 6)
    """
    # Calculate epoch-wise sums
    sum_loss = [
        np.sum(total_loss[i : i + n_experiments])
        for i in range(0, len(total_loss), n_experiments)
    ]
    sum_sum_grad = [
        np.sum(sum_grad[i : i + n_experiments])
        for i in range(0, len(sum_grad), n_experiments)
    ]

    # Create subplot figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Plot loss
    ax1.scatter(range(len(sum_loss)), sum_loss, color="blue", s=15)
    ax1.set_title("Total Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")

    # Plot gradient
    ax2.scatter(range(len(sum_sum_grad)), sum_sum_grad, color="green", s=15)
    ax2.set_title("Gradient Sum Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gradient Sum")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()  # Close figure to free memory


def create_annotation_matrix(
    base_df: pd.DataFrame,
    perturbation_dict: OrderedDict,
    annotation_symbols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create an annotation matrix from a perturbation dictionary. Symbols are used to annotate output heatmaps.

    Args:
        base_df: DataFrame to base the annotation matrix dimensions on
        perturbation_dict: Nested dictionary with structure {condition: {"pert": {...}, "exp": {...}}}
        annotation_symbols: Dictionary mapping annotation types to symbols.
            Defaults to {"pert": "•", "exp": "-", "tst": ""}

    Returns:
        DataFrame with same dimensions as base_df containing annotation symbols
    """
    perturbation_dict = enforce_pert_dic_order(perturbation_dict)
    if annotation_symbols is None:
        annotation_symbols = {"pert": "•", "exp": "-", "tst": ""}

    # Check if all indices in perturbation_dict are present in base_df
    for condition, type_dict in perturbation_dict.items():
        # Check for missing conditions
        if condition not in base_df.columns:
            raise KeyError(f"Missing condition in base_df: {condition}")

        for annot_type in ("pert", "exp"):
            if annot_type in type_dict:
                missing_indices = set(type_dict[annot_type].keys()) - set(base_df.index)
                if missing_indices:
                    raise KeyError(
                        f"Missing indices in perturbation_dict for {annot_type}: {missing_indices}"
                    )

    # Initialize empty annotation matrix
    annot = pd.DataFrame(index=base_df.index, columns=base_df.columns)

    # Fill annotations based on perturbation dictionary
    for condition, type_dict in perturbation_dict.items():
        for annot_type in ("pert", "exp"):
            if annot_type in type_dict:
                annot.loc[type_dict[annot_type].keys(), condition] = annotation_symbols[  # type: ignore
                    annot_type
                ]  # type: ignore

    # Fill remaining values with test annotation
    return annot.fillna(annotation_symbols["tst"])


def save_trained_network_and_visualise(
    input_json: str | Path,
    W: pd.DataFrame,
    path_data: str | Path,
    file_name: str,
    min_range: int,
    max_range: int,
) -> None:
    """Save network analysis results and visualizations.

    Args:
        input_json: Path to input JSON file
        W: DataFrame of trained weights
        path_data: Path to save data
        file_name: Name of file
        min_range: Minimum value for colorbar
        max_range: Maximum value for colorbar
    """
    # Load base graph and attributes
    G_original: nx.DiGraph = json_to_graph(input_json)
    pos = get_pos_from_json(input_json)
    const_dic = get_const_from_json(input_json)

    # Analyze edge changes
    analyze_deleted_edges(G_original, W)
    analyze_sign_flips(G_original, W)

    # Assign trained weights to original graph
    G_trained: nx.DiGraph = get_trained_network(G_original, W)

    # Generate JSON with trained weights
    gen_json(
        G_trained,
        path_data,
        f"{file_name}_synthetic_weight_est_realSpec",
        min_range=min_range,
        max_range=max_range,
        func_type="weighted_default",
        pos=pos,
        const_dic=const_dic,
        scale=1,
    )

    # Create colored visualization of trained network
    G_trained_visual = annotate_graph(G_trained, W)
    graph_to_pydot(
        G_trained_visual,
        os.path.join(path_data, "network_after_training"),
        format="png",
    )


def annotate_graph(G: nx.DiGraph, W: pd.DataFrame) -> nx.DiGraph:
    """Create new graph with trained weights, edge colors based on weight values."""
    G_weighted = get_trained_network(G, W)

    for u, v in G_weighted.edges:
        weight = G_weighted[u][v]["edge_weight"]
        G_weighted[u][v]["color"] = (
            "red" if weight == 0 else "orange" if weight < 0 else "black"
        )
    return G_weighted


def get_filtered_indices(annot: pd.DataFrame, idx_real: list[str]) -> pd.Index:
    """Get indices after filtering empty annotation rows."""
    return annot.loc[idx_real][~(annot.loc[idx_real] == "").all(axis=1)].index


def analyze_sign_flips(
    G: nx.DiGraph, W: pd.DataFrame, colors: dict[str, str] = ANSI_COLORS
) -> list[tuple[str, str, dict, float]]:
    """Find edges where sign was flipped (negative weight) during training."""
    flipped_edges = []
    for u, v in G.edges:
        edge_weight = W.at[v, u]
        if edge_weight < 0:
            flipped_edges.append((u, v, G[u][v], edge_weight))

    if flipped_edges:
        print(
            "Here are the edges where the sign has been flipped (weight is negative):"
        )
        for u, v, edge_info, weight in flipped_edges:
            sign = edge_info["sign"]
            if sign == "Activator":
                print(
                    f"Edge: '{u}' ---> '{v}' : {colors['red']}SIGN FLIPPED to Inhibitor{colors['reset']} (Weight: {weight})"
                )
            elif sign == "Inhibitor":
                print(
                    f"Edge: '{u}' ---| '{v}' : {colors['yellow']}SIGN FLIPPED to Activator{colors['reset']} (Weight: {weight})"
                )
    else:
        print("No edges had their sign flipped (no negative weights detected).")

    return flipped_edges


def analyze_deleted_edges(
    G: nx.DiGraph, W: pd.DataFrame, colors: dict[str, str] = ANSI_COLORS
) -> list[tuple[str, str, dict]]:
    """Find edges that were deleted (weight=0) during training."""
    deleted_edges = []
    for u, v in G.edges:
        if W.at[v, u] == 0:
            deleted_edges.append((u, v, G[u][v]))

    if deleted_edges:
        print("Here are the edges that are deleted (set to 0) by the MPNN model:")
        for u, v, edge_info in deleted_edges:
            sign = edge_info["sign"]
            if sign == "Activator":
                print(
                    f"Edge: '{u}' ---> '{v}' : {colors['red']}DELETED{colors['reset']}"
                )
            elif sign == "Inhibitor":
                print(
                    f"Edge: '{u}' ---| '{v}' : {colors['yellow']}DELETED{colors['reset']}"
                )
    else:
        print("No edges were deleted (set to 0) by the MPNN model.")

    return deleted_edges


def save_node_error_metrics(
    G: nx.DiGraph,
    rmse_values: pd.Series,
    mae_values: pd.Series,
    output_path: str | Path,
) -> None:
    """Save node-wise error metrics to CSV.

    Args:
        G: NetworkX graph
        rmse_values: RMSE values Series
        mae_values: MAE values Series
        output_path: Path to save CSV
    """
    node_data = {
        "Node": list(G.nodes),
        "RMSE": [rmse_values.get(n, None) for n in G.nodes],
        "MAE": [mae_values.get(n, None) for n in G.nodes],
    }
    df_node_data = pd.DataFrame(node_data)
    df_node_data.to_csv(Path(output_path) / Path("node_errors.csv"), index=False)


def visualize_error_by_node(
    G: nx.DiGraph,
    error_values: dict[str, pd.Series],
    pos: dict,
    const_dic: dict,
    min_range: int,
    max_range: int,
    output_path: str | Path,
) -> None:
    """Visualize error metrics on network nodes.

    Args:
        G: NetworkX graph
        error_values: Dict mapping error type to error values Series
        pos: Node position dictionary
        const_dic: Dictionary of constant nodes
        max_range: Maximum value for colorbar
        output_path: Path to save visualizations
    """
    pos_invert = {k: (v[0], -v[1]) for k, v in pos.items()}

    for err_type, values in error_values.items():
        plt.figure(figsize=(15, 10))
        nx.set_node_attributes(G, values.to_dict(), err_type)

        node_plot = nx.draw_networkx_nodes(
            G,
            pos_invert,
            node_color=[G.nodes[n][err_type] for n in G.nodes],  # type: ignore
            vmin=min_range,
            vmax=max_range,
            cmap=colormaps.get_cmap("Greens"),
        )
        nx.draw_networkx_edges(G, pos_invert)
        nx.draw_networkx_labels(G, pos_invert, font_size=7, font_color="dimgray")
        nx.draw_networkx_labels(
            G, pos_invert, {n: n for n in const_dic}, font_size=7, font_color="black"
        )

        plt.colorbar(node_plot)
        plt.title(f"Average {err_type.upper()} across experiments")
        plt.tight_layout()
        plt.savefig(Path(output_path) / Path(f"{err_type}_by_node.png"), dpi=200)
        plt.close()


def plot_difference_heatmaps(
    predictions: list[tuple[pd.DataFrame, str]],
    y: pd.DataFrame,
    filtered_idx: pd.Index,
    annot: pd.DataFrame,
    annotation_symbols: dict[str, str],
    max_range: int,
    output_path: str | Path,
    shared_colorbar: bool = True,
) -> None:
    """Plot heatmaps showing differences between predictions and true values.

    Args:
        predictions: List of (prediction DataFrame, name) tuples
        y: True values DataFrame
        filtered_idx: Filtered indices to plot
        annot: Annotation matrix
        annotation_symbols: Dict mapping annotation types to symbols
        max_range: Maximum value for colorbar range
        output_path: Path to save plot
        shared_colorbar: Whether to use shared colorbar across subplots
    """
    if shared_colorbar:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
        cbar_ax = fig.add_axes([0.91, 0.3, 0.02, 0.4])  # type: ignore

    for i, (pred, title) in enumerate(predictions):
        if not shared_colorbar:
            fig, ax = plt.subplots(figsize=(14, 7))
            current_ax = ax
        else:
            current_ax = axs[i]

        diff = (pred - y).loc[filtered_idx]

        sns.heatmap(
            diff if shared_colorbar else diff.T,
            cmap="RdBu_r",
            center=0,
            ax=current_ax,
            vmin=-max_range,
            vmax=max_range,
            cbar=i == 0 if shared_colorbar else True,
            cbar_ax=None if i or not shared_colorbar else cbar_ax,
            yticklabels=True,
            annot=annot.loc[filtered_idx]
            if shared_colorbar
            else annot.loc[filtered_idx].T,
            fmt="",
        )

        mae_exp = calc_err(
            pred.loc[filtered_idx],
            y.loc[filtered_idx],
            mask=annot,
            mask_val=annotation_symbols["exp"],
        )[1]
        mae_tst = calc_err(
            pred.loc[filtered_idx],
            y.loc[filtered_idx],
            mask=annot,
            mask_val=annotation_symbols["tst"],
        )[1]

        current_ax.set_title(
            f"{title}\nMAE: {mae_exp:.2f} (expect), {mae_tst:.2f} (other)"
        )

        if not shared_colorbar:
            plt.tight_layout()
            plt.savefig(
                Path(output_path)
                / Path(f"synthetic_{title.lower().replace(' ', '_')}.png"),
                dpi=200,
            )
            plt.close(fig)

    if shared_colorbar:
        fig.tight_layout(rect=[0, 0, 0.9, 1])  # type: ignore
        plt.savefig(Path(output_path) / Path("synthetic_nn_comp.png"), dpi=200)
        plt.close(fig)


def plot_node_errors(error_dict: dict[str, float] | pd.Series, output_path: str | Path):
    """
    Create an interactive bar chart of mean absolute errors for each node.

    Args:
        error_dict (dict): Dictionary mapping node names to their error values
        output_path (str): Path to save the plot
    """
    import plotly.graph_objects as go

    # Sort the dictionary by error value in descending order
    sorted_items = sorted(error_dict.items(), key=lambda x: x[1], reverse=True)
    nodes = [item[0] for item in sorted_items]
    errors = [item[1] for item in sorted_items]

    # Create the bar chart
    fig = go.Figure(data=[go.Bar(x=nodes, y=errors, marker_color="rgb(55, 83, 109)")])

    # Update layout
    fig.update_layout(
        title="Mean Absolute Error by Node",
        xaxis_title="Node",
        yaxis_title="Mean Absolute Error",
        xaxis_tickangle=-45,
        template="plotly_white",
        showlegend=False,
        height=600,
    )

    # Save the plot
    # fig.write_html(Path(output_path) / Path("node_errors.html"))
    fig.write_image(Path(output_path) / Path("node_errors.png"))


# def calculate_quadratic_weighted_kappa(
#     confusion_matrix: np.ndarray,
# ) -> float:
#     """Calculate quadratic weighted kappa (QWK) score from a confusion matrix.

#     QWK measures agreement between two raters, taking into account how far apart
#     the ratings are, not just whether they exactly match. Disagreements are
#     weighted by their squared distance.

#     Args:
#         confusion_matrix: Square numpy array of confusion matrix

#     Returns:
#         float: QWK score between -1 and 1
#             1: Perfect agreement
#             0: Agreement equivalent to chance
#             -1: Complete disagreement
#     """
#     n_classes = confusion_matrix.shape[0]

#     # Calculate weights matrix (quadratic weights)
#     weights = np.zeros((n_classes, n_classes))
#     for i in range(n_classes):
#         for j in range(n_classes):
#             weights[i, j] = (i - j) ** 2

#     # Calculate expected matrix
#     row_sum = np.sum(confusion_matrix, axis=1)
#     col_sum = np.sum(confusion_matrix, axis=0)
#     expected = np.outer(row_sum, col_sum) / np.sum(confusion_matrix)

#     # Calculate weighted sums
#     w_observed = np.sum(weights * confusion_matrix)
#     w_expected = np.sum(weights * expected)

#     # Calculate kappa
#     if w_expected == 0:
#         return 1.0
#     else:
#         return 1 - (w_observed / w_expected)


# %%
def analyze_prediction_errors(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    pert_dic: OrderedDict,
    threshold: float = 0.1,
    binary_mode: bool = False,
) -> dict:
    """
    Analyze the fraction of times each node's predictions are too high or too low.

    Args:
        y_true (pd.DataFrame): True values
        y_pred (pd.DataFrame): Predicted values
        pert_dic (dict): Dictionary containing perturbation information
        threshold (float): Threshold for considering a prediction as different (default: 0.1)
        binary_mode (bool): If True, considers predictions correct if they match the binary state
                          (0 vs >0) of true values after rounding

    Returns:
        dict: Dictionary with fractions of 'over' and 'under' predictions for each node
    """
    pert_dic = enforce_pert_dic_order(pert_dic)
    if binary_mode:
        # Convert to binary and handle missing values
        valid_mask = ~y_true.isna()
        y_pred_binary = pd.DataFrame(np.nan, index=y_true.index, columns=y_true.columns)
        y_true_binary = pd.DataFrame(np.nan, index=y_true.index, columns=y_true.columns)

        y_pred_binary[valid_mask] = (y_pred[valid_mask].round() > 0).astype(int)
        y_true_binary[valid_mask] = (y_true[valid_mask] > 0).astype(int)
        differences = y_pred_binary - y_true_binary
    else:
        differences = y_pred - y_true

    # Count experiments per node from pert_dic
    node_experiments = {}
    for exp_data in pert_dic.values():
        for node in exp_data["exp"].keys():
            node_experiments[node] = node_experiments.get(node, 0) + 1

    error_fractions = {}
    for node in y_true.index:
        if node not in node_experiments:
            continue  # Skip nodes that were never measured

        # Consider only experiments where this node was measured
        valid_mask = differences.loc[node].notna()
        valid_differences = differences.loc[node][valid_mask]
        total_measurements = node_experiments[node]

        if total_measurements > 0:
            if binary_mode:
                # For binary mode, over = 1 when should be 0, under = 0 when should be 1
                over_count = (valid_differences > 0).sum()
                under_count = (valid_differences < 0).sum()
                correct_count = (valid_differences == 0).sum()
            else:
                # Calculate counts using threshold for continuous values
                over_count = (valid_differences > threshold).sum()
                under_count = (valid_differences < -threshold).sum()
                correct_count = total_measurements - over_count - under_count

            error_fractions[node] = {
                "over": over_count / total_measurements,
                "under": under_count / total_measurements,
                "correct": correct_count / total_measurements,
                "total_measurements": total_measurements,
            }

    return error_fractions


def plot_prediction_bias(error_fractions: dict[str, dict], output_path: str | Path):
    """
    Create an interactive stacked bar chart showing fractions of over/under predictions for each node.

    Args:
        error_fractions (dict): Dictionary with error fractions from analyze_prediction_errors
        output_path (str): Path to save the plot
    """
    import plotly.graph_objects as go

    # Prepare data
    nodes = list(error_fractions.keys())
    over_fracs = [error_fractions[node]["over"] for node in nodes]
    under_fracs = [error_fractions[node]["under"] for node in nodes]
    correct_fracs = [error_fractions[node]["correct"] for node in nodes]
    measurements = [error_fractions[node]["total_measurements"] for node in nodes]

    # Sort nodes by total error fraction
    total_error_fracs = [over + under for over, under in zip(over_fracs, under_fracs)]
    sorted_indices = sorted(
        range(len(nodes)), key=lambda k: total_error_fracs[k], reverse=True
    )

    nodes = [nodes[i] for i in sorted_indices]
    over_fracs = [over_fracs[i] for i in sorted_indices]
    under_fracs = [under_fracs[i] for i in sorted_indices]
    correct_fracs = [correct_fracs[i] for i in sorted_indices]
    measurements = [measurements[i] for i in sorted_indices]

    # Create hover text with number of measurements
    over_text = [
        f"Over-predictions: {frac:.1%}<br>({int(frac * meas)} / {meas} measurements)"
        for frac, meas in zip(over_fracs, measurements)
    ]
    under_text = [
        f"Under-predictions: {frac:.1%}<br>({int(frac * meas)} / {meas} measurements)"
        for frac, meas in zip(under_fracs, measurements)
    ]
    correct_text = [
        f"Correct predictions: {frac:.1%}<br>({int(frac * meas)} / {meas} measurements)"
        for frac, meas in zip(correct_fracs, measurements)
    ]

    # Create the stacked bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                name="Over-predictions",
                x=nodes,
                y=over_fracs,
                text=[f"{x:.1%}" for x in over_fracs],
                textposition="auto",
                hovertext=over_text,
                marker_color="rgba(219, 64, 82, 0.7)",
            ),
            go.Bar(
                name="Under-predictions",
                x=nodes,
                y=under_fracs,
                text=[f"{x:.1%}" for x in under_fracs],
                textposition="auto",
                hovertext=under_text,
                marker_color="rgba(55, 128, 191, 0.7)",
            ),
            go.Bar(
                name="Correct predictions",
                x=nodes,
                y=correct_fracs,
                text=[f"{x:.1%}" for x in correct_fracs],
                textposition="auto",
                hovertext=correct_text,
                marker_color="rgba(128, 128, 128, 0.7)",
            ),
        ]
    )

    # Update layout
    fig.update_layout(
        title="Prediction Bias by Node (as fraction of measurements)",
        xaxis_title="Node",
        yaxis_title="Fraction of Predictions",
        yaxis=dict(tickformat=",.0%", range=[0, 1]),
        xaxis_tickangle=-45,
        template="plotly_white",
        barmode="stack",
        height=600,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    # Save the plot
    # fig.show()
    # fig.write_html(output_path)
    output_path = (
        str(output_path)
        if str(output_path).endswith(".png")
        else str(output_path) + ".png"
    )
    fig.write_image(output_path)


# def plot_confusion_matrix(
#     conf_matrix: np.ndarray, title: str, output_path: str
# ) -> None:
#     # Calculate percentages for annotations
#     total = conf_matrix.sum()
#     percentages = conf_matrix / total * 100

#     # Create annotation text with counts and percentages
#     annotations = [
#         [f"{val}\n({pct:.1f}%)" for val, pct in zip(row, pct_row)]
#         for row, pct_row in zip(conf_matrix, percentages)
#     ]

#     # Create heatmap
#     fig = go.Figure(
#         data=go.Heatmap(
#             z=conf_matrix,
#             x=["Predicted 0", "Predicted >0"],
#             y=["Actual 0", "Actual >0"],
#             text=annotations,
#             texttemplate="%{text}",
#             textfont={"size": 14},
#             colorscale="RdYlBu",
#         )
#     )

#     fig.update_layout(
#         title=title,
#         xaxis_title="Predicted Value",
#         yaxis_title="Actual Value",
#         template="plotly_white",
#         width=600,
#         height=600,
#     )

#     # Save plots
#     fig.write_html(output_path + ".html")
#     fig.write_image(output_path + ".png")


def plot_multiclass_confusion_matrix(
    conf_matrix: np.ndarray,
    title: str,
    output_path: str | Path,
    class_labels: list[str] | None = None,
    binary_mode: bool = False,
) -> None:
    """Plot a confusion matrix for multiclass classification using plotly.

    The plot shows both raw counts and percentages normalized by true class totals.
    The color scheme indicates both magnitude (intensity) and whether the prediction
    is correct (blue) or incorrect (red).

    Args:
        conf_matrix: Confusion matrix as numpy array
        title: Title for the plot
        output_path: Path to save the plot (without extension)
        class_labels: Optional list of class labels. If None, uses 0 to n-1
        binary_mode: If True, use binary labels ["0", "> 0"]
    """
    n_classes = conf_matrix.shape[0]

    if binary_mode:
        class_labels = ["0", "> 0"]
    elif class_labels is None:
        class_labels = [str(i) for i in range(n_classes)]

    if len(class_labels) != n_classes:
        raise ValueError(
            f"Number of class labels ({len(class_labels)}) must match matrix dimensions ({n_classes})"
        )

    # Calculate row-wise percentages (normalize by true class counts)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(conf_matrix, row_sums, where=row_sums != 0) * 100

    # Create annotation text with counts and percentages
    annotations = []
    for i, (row, pct_row) in enumerate(zip(conf_matrix, percentages)):
        row_annotations = []
        for j, (val, pct) in enumerate(zip(row, pct_row)):
            total_class = row_sums[i][0]
            row_annotations.append(f"{int(val)}/{int(total_class)}\n({pct:.1f}%)")
        annotations.append(row_annotations)

    # Create custom colorscale that distinguishes diagonal vs off-diagonal
    # Blue for correct predictions (diagonal), red for incorrect (off-diagonal)
    colorscale = [
        [0, "rgb(255,255,255)"],  # White for zero
        [0.001, "rgb(255,220,220)"],  # Very light red for small errors
        [0.5, "rgb(255,0,0)"],  # Bright red for large errors
        [0.501, "rgb(220,220,255)"],  # Very light blue for small correct
        [1, "rgb(0,0,255)"],  # Bright blue for large correct
    ]

    # Create mask for diagonal vs off-diagonal elements
    mask = np.eye(n_classes)
    # Only shift diagonal values for non-zero rows
    z_matrix = np.where(
        np.logical_and(mask, row_sums != 0), percentages + 100, percentages
    )

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[f"Predicted {label}" for label in class_labels],
            y=[f"Actual {label}" for label in class_labels],
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="% of true class",
                ticktext=["0%", "25%", "50%", "75%", "100%"],
                tickvals=[0, 25, 50, 75, 100],
            ),
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Value",
        yaxis_title="Actual Value",
        template="plotly_white",
        width=800,
        height=800,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            tickfont=dict(size=12),
        ),
    )

    # Save plots
    # fig.write_html(Path(output_path).with_suffix(".html"))
    fig.write_image(Path(output_path).with_suffix(".png"))


def extract_valid_preds(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, binary_mode: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Create mask from non-NaN values in y_true
    valid_mask = ~y_true.isna()

    # Convert to binary only for valid values
    y_pred_nan = pd.DataFrame(np.nan, index=y_true.index, columns=y_true.columns)
    y_true_nan = pd.DataFrame(np.nan, index=y_true.index, columns=y_true.columns)

    if binary_mode:
        y_pred_nan[valid_mask] = (y_pred[valid_mask].round() > 0).astype(int)
        y_true_nan[valid_mask] = (y_true[valid_mask] > 0).astype(int)
    else:
        y_pred_nan[valid_mask] = y_pred[valid_mask].round()
        y_true_nan[valid_mask] = y_true[valid_mask]

    return y_true_nan, y_pred_nan, valid_mask


# Overall confusion matrix (only for valid values)
def calculate_overall_cm_inputs(
    y_true_nan: pd.DataFrame, y_pred_nan: pd.DataFrame, valid_mask: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    y_true_flat = y_true_nan[valid_mask].values.flatten()
    y_pred_flat = y_pred_nan[valid_mask].values.flatten()
    valid_indices = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_true_flat = y_true_flat[valid_indices]
    y_pred_flat = y_pred_flat[valid_indices]
    return y_true_flat, y_pred_flat


def check_metrics_inputs(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    if y_true.isna().all().all() or y_pred.isna().all().all():
        raise ValueError("No valid predictions found")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.index.name != y_pred.index.name:
        raise ValueError("y_true and y_pred must have the same index")

    if y_true.columns.name != y_pred.columns.name:
        raise ValueError("y_true and y_pred must have the same columns")

    if not y_true.columns.equals(y_pred.columns):
        raise ValueError("y_true and y_pred must have matching column names")

    # raise error if y_pred contains nans
    if y_pred.isna().any().any():
        raise ValueError("y_pred contains NaNs")


def analyze_node_level_errors(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    binary_mode: bool = False,
    error_threshold: float = 0.5,
) -> tuple[float, dict[str, float], list[str]]:
    """Analyze node-level prediction errors across experiments.

    Args:
        y_true: True experimental values
        y_pred: Model predictions
        binary_mode: If True, convert predictions to binary (0 vs >0)
        error_threshold: Threshold for considering a node consistently incorrect
                        (fraction of experiments where node is incorrect)

    Returns:
        tuple containing:
        - mean_incorrect_nodes: Average number of incorrect nodes per experiment
        - node_error_rates: Dictionary mapping node names to their error rates
        - consistently_wrong_nodes: List of nodes that are frequently incorrect
    """
    # Extract valid predictions
    y_true_nan, y_pred_nan, valid_mask = extract_valid_preds(
        y_true, y_pred, binary_mode
    )

    # Calculate incorrect predictions per experiment
    incorrect_mask = (y_true_nan != y_pred_nan) & valid_mask
    incorrect_per_experiment = incorrect_mask.sum(axis=0)
    # measured_per_experiment = valid_mask.sum(axis=0)

    # Calculate mean number of incorrect nodes per experiment
    mean_incorrect_nodes = incorrect_per_experiment.mean()

    # Calculate error rate per node
    total_measurements = valid_mask.sum(axis=1)
    total_errors = incorrect_mask.sum(axis=1)
    node_error_rates = (total_errors / total_measurements).fillna(0).to_dict()

    # Identify consistently wrong nodes (error rate above threshold)
    consistently_wrong_nodes = [
        node
        for node, rate in node_error_rates.items()
        if rate >= error_threshold and total_measurements[node] > 0
    ]

    return mean_incorrect_nodes, node_error_rates, consistently_wrong_nodes


def save_annotated_specification(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    pert_dic: dict,
    output_path: str | Path,
) -> None:
    """Save an annotated version of the specification with predictions and differences.

    Args:
        y_true: True experimental values DataFrame
        y_pred: Model predictions DataFrame
        pert_dic: Dictionary of perturbation experiments
        output_path: Path to save the annotated specification
    """
    # Create a list to store all rows
    rows = []

    # Filter out dummy nodes and 0A_node00
    filtered_nodes = [
        node for node in y_true.index if "dummy" not in node and "0A_node00" not in node
    ]
    y_true = y_true.loc[filtered_nodes]
    y_pred = y_pred.loc[filtered_nodes]

    # Replace any values in y_pred that are NaN in y_true with NaN
    y_pred = y_pred.mask(y_true.isna())

    # Process each experiment
    for exp_name, exp_data in pert_dic.items():
        # Process perturbations
        for node, value in exp_data["pert"].items():
            if "dummy" not in node and "0A_node00" not in node:
                rows.append(
                    {
                        "Experiment": exp_name,
                        "Node": node,
                        "Type": "perturbation",
                        "Expected": value,
                        "Predicted": value,  # Perturbations are fixed
                        "Difference": 0.0,  # No difference for perturbations
                    }
                )

        # Process experimental measurements
        for node, expected in exp_data["exp"].items():
            if "dummy" not in node and "0A_node00" not in node:
                predicted = y_pred.at[node, exp_name]
                difference = predicted - expected
            rows.append(
                {
                    "Experiment": exp_name,
                    "Node": node,
                    "Type": "experimental",
                    "Expected": expected,
                    "Predicted": predicted,
                    "Difference": difference,
                }
            )

    # Create DataFrame and sort it
    df = pd.DataFrame(rows)
    df = df.sort_values(["Experiment", "Type", "Node"])

    # Save to CSV
    df.to_csv(output_path, index=False)

    return None


def calculate_quadratic_weighted_kappa(
    confusion_matrix: np.ndarray,
) -> float:
    """Calculate quadratic weighted kappa (QWK) score from a confusion matrix.

    QWK measures agreement between two raters, taking into account how far apart
    the ratings are, not just whether they exactly match. Disagreements are
    weighted by their squared distance.

    Args:
        confusion_matrix: Square numpy array of confusion matrix

    Returns:
        float: QWK score between -1 and 1
            1: Perfect agreement
            0: Agreement equivalent to chance
            -1: Complete disagreement
    """
    raise NotImplementedError(
        "Replaced with cohen_kappa_score with quadratic weights from sklearn"
    )

    n_classes = confusion_matrix.shape[0]

    # Calculate weights matrix (quadratic weights)
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = (i - j) ** 2

    # Calculate expected matrix
    row_sum = np.sum(confusion_matrix, axis=1)
    col_sum = np.sum(confusion_matrix, axis=0)
    expected = np.outer(row_sum, col_sum) / np.sum(confusion_matrix)

    # Calculate weighted sums
    w_observed = np.sum(weights * confusion_matrix)
    w_expected = np.sum(weights * expected)

    # Calculate kappa
    if w_expected == 0:
        return 1.0
    else:
        return 1 - (w_observed / w_expected)


def analyze_predictions(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    pert_dic: OrderedDict | None = None,
    save_figs: bool = True,
    save_csv: bool = True,
    path_data: str | Path | None = None,
    plot_all_nodes: bool = True,
    config: dict | None = None,
    binary_mode: bool = False,
) -> dict:
    """
    Create confusion matrix plots for binary predictions (0 vs >0), only comparing values
    that are not NaN in y_true.

    Note that when plotting per-node confusion matrices, we only plot nodes that have at least 2 unique classes in both true and predicted values.

    Note also that this function rounds the predictions of the GNN to the nearest integer to simulate BioModelAnalyzer.

    Args:
        y_true: True experimental values
        y_pred: Model predictions
        save_figs: Whether to save figures
        save_csv: Whether to save metrics to CSV
        path_data: Path to save output plots
        plot_all_nodes: Whether to create plots for individual nodes
        config: Configuration parameters from TOML file
        binary_mode: Whether to treat predictions as binary (0 vs >0)
        pert_dic: Optional dictionary of perturbation experiments for saving annotated specification

    Returns:
        Dictionary containing classification metrics
    """
    if pert_dic is not None:
        pert_dic = enforce_pert_dic_order(pert_dic)
    if path_data is not None:
        os.makedirs(path_data, exist_ok=True)

    check_metrics_inputs(y_true, y_pred)

    max_range = int(max(y_true.max().max(), y_pred.max().max()))
    conf_matrix_labels = [0, 1] if binary_mode else [i for i in range(max_range + 1)]
    plot_labels = ["0", ">0"] if binary_mode else [str(i) for i in conf_matrix_labels]

    y_true_nan, y_pred_nan, valid_mask = extract_valid_preds(
        y_true, y_pred, binary_mode=binary_mode
    )

    y_true_flat, y_pred_flat = calculate_overall_cm_inputs(
        y_true_nan, y_pred_nan, valid_mask
    )

    # Get actual class labels based on the data

    overall_cm = confusion_matrix(y_true_flat, y_pred_flat, labels=conf_matrix_labels)

    if binary_mode:
        tn, fp, _, _ = overall_cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = "NA"

    f1 = (
        f1_score(y_true_flat, y_pred_flat)
        if binary_mode
        else f1_score(y_true_flat, y_pred_flat, average="weighted")
    )
    precision = (
        precision_score(y_true_flat, y_pred_flat)
        if binary_mode
        else precision_score(y_true_flat, y_pred_flat, average="weighted")
    )
    recall = (
        recall_score(y_true_flat, y_pred_flat)
        if binary_mode
        else recall_score(y_true_flat, y_pred_flat, average="weighted")
    )
    roc_auc = roc_auc_score(y_true_flat, y_pred_flat) if binary_mode else "NA"
    average_precision = (
        average_precision_score(y_true_flat, y_pred_flat) if binary_mode else "NA"
    )

    metrics = {
        "accuracy": accuracy_score(y_true_flat, y_pred_flat),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": matthews_corrcoef(y_true_flat, y_pred_flat),
        # "qwk": calculate_quadratic_weighted_kappa(overall_cm),
        "qwk": cohen_kappa_score(y_true_flat, y_pred_flat, weights="quadratic"),
        "roc_auc": roc_auc,
        "balanced_accuracy": balanced_accuracy_score(y_true_flat, y_pred_flat),
        "average_precision": average_precision,
        "total_predictions": len(y_true_flat),
    }

    # Add node-level error analysis
    mean_incorrect_nodes, node_error_rates, consistently_wrong_nodes = (
        analyze_node_level_errors(y_true, y_pred, binary_mode=binary_mode)
    )

    # Add node-level metrics to the metrics dictionary
    metrics.update(
        {
            "mean_incorrect_nodes": mean_incorrect_nodes,
            "consistently_wrong_nodes": consistently_wrong_nodes,
            "node_error_rates": node_error_rates,
        }
    )

    # Add timestamp and relevant config details to metrics
    metrics["timestamp"] = pd.Timestamp.now()
    if config:
        # Extract relevant config sections
        relevant_sections = ["model_params", "training", "paths"]
        for section in relevant_sections:
            if section in config:
                metrics.update(
                    {
                        f"{section}_{k}": v
                        for k, v in config[section].items()
                        if not isinstance(v, dict)  # Skip nested dictionaries
                    }
                )

    if save_figs:
        if path_data is None:
            raise ValueError("path_data must be provided if save_figs is True")
        plot_multiclass_confusion_matrix(
            overall_cm,
            "Overall Prediction Confusion Matrix",
            os.path.join(path_data, "confusion_matrix_overall"),
            class_labels=plot_labels,
            binary_mode=binary_mode,
        )

    if plot_all_nodes and save_figs:
        if path_data is None:
            raise ValueError("path_data must be provided if save_figs is True")
        # Per-node confusion matrices (only for nodes that have any valid values)
        measured_nodes = y_true.index[y_true.notna().any(axis=1)]

        for node in measured_nodes:
            # Get valid measurements for this node
            node_mask = valid_mask.loc[node]

            if node_mask.sum() > 0:  # Only create plot if we have measurements
                node_true = y_true_nan.loc[node][node_mask]
                node_pred = y_pred_nan.loc[node][node_mask]

                # Check if we have at least 2 unique classes in both true and predicted values
                # n_classes_true = len(np.unique(node_true))
                # n_classes_pred = len(np.unique(node_pred))

                # if n_classes_true >= 2:
                node_cm = confusion_matrix(
                    node_true, node_pred, labels=conf_matrix_labels
                )

                plot_multiclass_confusion_matrix(
                    node_cm,
                    f"Prediction Confusion Matrix - {node}",
                    Path(path_data) / Path(f"confusion_matrix_{node}"),
                    class_labels=plot_labels,
                    binary_mode=binary_mode,
                )

    # Save metrics to CSV
    if save_csv:
        if path_data is None:
            raise ValueError("path_data must be provided if save_csv is True")

        # Save main metrics as before
        metrics_file = os.path.join(path_data, "prediction_metrics.csv")
        # Create a copy of metrics without the node-specific data
        metrics_for_csv = {
            k: v
            for k, v in metrics.items()
            if k not in ["consistently_wrong_nodes", "node_error_rates"]
        }
        metrics_df = pd.DataFrame([metrics_for_csv])

        try:
            if os.path.exists(metrics_file):
                metrics_df.to_csv(metrics_file, mode="a", header=False, index=False)
            else:
                metrics_df.to_csv(metrics_file, index=False)

            # Save node-level error rates to a separate CSV
            node_errors_df = pd.DataFrame(
                {
                    "node": list(node_error_rates.keys()),
                    "error_rate": list(node_error_rates.values()),
                    "consistently_wrong": [
                        node in consistently_wrong_nodes
                        for node in node_error_rates.keys()
                    ],
                }
            )
            node_errors_df.to_csv(
                os.path.join(path_data, "node_error_rates.csv"), index=False
            )
            if pert_dic is None:
                raise ValueError("pert_dic must be provided if save_csv is True")
                # Save annotated specification
            save_annotated_specification(
                y_true=y_true,
                y_pred=y_pred,
                pert_dic=pert_dic,
                output_path=Path(path_data) / Path("annotated_specification.csv"),
            )

        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
            raise

    return metrics


def plot_classification_curves(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    pert_dic_small: OrderedDict,
    path_data: str | Path,
    plot_all_nodes: bool = True,
):
    """
    Create ROC and Precision-Recall curves for the model's binary predictions.

    Args:
        y_true (pd.DataFrame): True experimental values
        y_pred (pd.DataFrame): Model predictions (continuous values)
        pert_dic_small (dict): Dictionary of perturbation experiments
        path_data (str): Path to save output plots
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )

    pert_dic_small = enforce_pert_dic_order(pert_dic_small)
    # Convert to binary ground truth (0 vs >0)
    y_true_binary = (y_true > 0).astype(int)

    # Calculate curves for overall performance
    fpr, tpr, _ = roc_curve(y_true_binary.values.flatten(), y_pred.values.flatten())
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(
        y_true_binary.values.flatten(), y_pred.values.flatten()
    )
    avg_precision = average_precision_score(
        y_true_binary.values.flatten(), y_pred.values.flatten()
    )

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall Curve")
    )

    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, name=f"ROC (AUC = {roc_auc:.3f})", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], name="Random", line=dict(color="red", dash="dash")
        ),
        row=1,
        col=1,
    )

    # Add Precision-Recall curve
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            name=f"PR (AP = {avg_precision:.3f})",
            line=dict(color="green"),
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title="Model Performance Curves",
        template="plotly_white",
        height=500,
        width=1000,
        showlegend=True,
    )

    # Update axes
    fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)

    # Save overall curves
    # fig.write_html(os.path.join(path_data, "performance_curves_overall.html"))
    fig.write_image(os.path.join(path_data, "performance_curves_overall.png"))

    if plot_all_nodes:
        # Calculate and plot per-node curves
        measured_nodes = set()
        for exp_data in pert_dic_small.values():
            measured_nodes.update(exp_data["exp"].keys())

        # Create a summary DataFrame for node-specific metrics
        node_metrics = []

        for node in measured_nodes:
            valid_mask = ~y_true.loc[node].isna()
            if valid_mask.sum() > 0:  # Only create plot if we have measurements
                node_true = y_true_binary.loc[node][valid_mask]
                node_pred = y_pred.loc[node][valid_mask]

                # Calculate node-specific curves
                fpr, tpr, _ = roc_curve(node_true, node_pred)
                roc_auc = auc(fpr, tpr)
                avg_precision = average_precision_score(node_true, node_pred)

                # Store metrics
                node_metrics.append(
                    {
                        "node": node,
                        "roc_auc": roc_auc,
                        "average_precision": avg_precision,
                        "n_measurements": valid_mask.sum(),
                    }
                )

                # Create node-specific plot
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=(
                        f"{node} - ROC Curve",
                        f"{node} - Precision-Recall Curve",
                    ),
                )

                # Add ROC curve
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        name=f"ROC (AUC = {roc_auc:.3f})",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name="Random",
                        line=dict(color="red", dash="dash"),
                    ),
                    row=1,
                    col=1,
                )

                # Add Precision-Recall curve
                precision, recall, _ = precision_recall_curve(node_true, node_pred)
                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        name=f"PR (AP = {avg_precision:.3f})",
                        line=dict(color="green"),
                    ),
                    row=1,
                    col=2,
                )

                # Update layout
                fig.update_layout(
                    title=f"Model Performance Curves - {node}",
                    template="plotly_white",
                    height=500,
                    width=1000,
                    showlegend=True,
                )

                # Update axes
                fig.update_xaxes(
                    title_text="False Positive Rate", range=[0, 1], row=1, col=1
                )
                fig.update_yaxes(
                    title_text="True Positive Rate", range=[0, 1], row=1, col=1
                )
                fig.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
                fig.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)

                # Save node-specific plots
                # fig.write_html(
                #     Path(path_data) / Path(f"performance_curves_{node}.html")
                # )
                fig.write_image(
                    Path(path_data) / Path(f"performance_curves_{node}.png")
                )

        # Save node metrics summary
        pd.DataFrame(node_metrics).to_csv(
            Path(path_data) / Path("node_performance_metrics.csv"), index=False
        )

    return 0


def plot_node_weights(node_weight_df: pd.DataFrame, out_dir: str | Path) -> int:
    """Plot node weights as a bar chart.

    Args:
        node_weight_df: DataFrame containing node weights with 'node' column
        out_dir: Output directory for saving the plot

    Returns:
        0 on successful completion
    """
    plt.figure(figsize=(12, 6))

    # Create bar plot
    ax = plt.gca()
    node_weight_df.plot(kind="bar", ax=ax)

    # Set x-axis tick labels to node names
    ax.set_xticklabels(node_weight_df["node"], rotation=45, ha="right")

    plt.title("Class Weights by Node")
    plt.xlabel("Node")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / Path("node_class_weights.png"))
    plt.close()
    return 0


def write_metrics_to_txt(
    binary_metrics: dict, nonbinary_metrics: dict, path: Path
) -> None:
    """Write both binary and non-binary metrics to a text file in a formatted way.

    Args:
        binary_metrics: Dictionary containing binary classification metrics
        nonbinary_metrics: Dictionary containing non-binary classification metrics
        path: Path to output file
    """
    with open(path, "w") as f:
        f.write("Classification Metrics\n")
        f.write("=====================\n\n")
        f.write("Binary Classification:\n")
        f.write(f"  F1 Score: {binary_metrics['f1']:.4f}\n")
        f.write(f"  MCC:      {binary_metrics['mcc']:.4f}\n\n")
        f.write("Non-binary Classification:\n")
        f.write(f"  F1 Score: {nonbinary_metrics['f1']:.4f}\n")
        f.write(f"  MCC:      {nonbinary_metrics['mcc']:.4f}\n")
        f.write(f"  QWK:      {nonbinary_metrics['qwk']:.4f}\n")


def plot_loss_vs_validation_loss(
    epoch_losses: list[float], test_losses: list[float], out_dir: Path | str
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=epoch_losses, name="Training Loss"))
    fig.add_trace(go.Scatter(y=test_losses, name="Test Loss"))
    fig.update_layout(
        title="Training and Test Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
    )
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    # fig.write_html(out_dir / Path("loss_vs_validation_loss.html"))
    # print(
    #     f"Saved loss vs validation loss plot to {out_dir / Path('loss_vs_validation_loss.html')}"
    # )
    fig.write_image(out_dir / Path("loss_vs_validation_loss.png"))


def plot_metrics_vs_bias(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    x_axis_metric: str = "override_specification_generation.n_specifications",
    y_axis_metrics: list[str] = [
        "test_nonbinary_f1",
        "test_nonbinary_qwk",
        "test_nonbinary_mcc",
    ],
    neat_x_axis_label: str = "Specification Size",
    neat_y_axis_label: str = "Metric Value",
    figsize: tuple = (4, 3),
    y_min: float = 0,
    y_max: float = 1,
    show_title: bool = False,
    legend_loc: str = "best",
) -> None:
    """
    Plot multiple metrics vs x_axis_metric with error bars on a single plot.

    Args:
        df: DataFrame with columns for metrics and x_axis_metric
        output_path: Path to save the figure (if None, displays the plot)
        x_axis_metric: Column name to use for x-axis
        y_axis_metrics: List of metric column names to plot
        neat_x_axis_label: Display label for x-axis
        figsize: Size of the figure (width, height)
        y_min: Minimum value for y-axis limits
        y_max: Maximum value for y-axis limits
        show_title: Whether to display the title
        legend_loc: Location for legend placement
    """
    plt.style.use("seaborn-v0_8-paper")
    # Get the full colorblind palette (it has 10 colors)
    palette = sns.color_palette("colorblind", n_colors=10)

    # Define fixed colors for specific metrics using palette indices
    fixed_metric_colors = {
        "test_nonbinary_f1": palette[0],  # Blue
        "test_nonbinary_qwk": palette[1],  # Orange
        "test_nonbinary_mcc": palette[2],  # Green
    }

    # Assign colors to the metrics to be plotted, prioritizing fixed colors
    metric_colors = {}
    used_palette_indices = set()
    next_palette_index = 0

    for metric in y_axis_metrics:
        if metric in fixed_metric_colors:
            metric_colors[metric] = fixed_metric_colors[metric]
            # Track which palette index was used by the fixed color
            # (this part is a bit tricky as we don't know the exact index of the fixed color easily)
            # A simpler approach is to assign other colors sequentially from *unused* indices.
            pass  # We'll handle assignment below

    # Assign colors. First fixed, then others from remaining palette colors.
    # available_palette = list(range(len(palette)))
    assigned_colors = {}

    # Assign fixed colors first
    for metric in y_axis_metrics:
        if metric in fixed_metric_colors:
            assigned_colors[metric] = fixed_metric_colors[metric]
            # Optionally remove the index from available_palette if you know it

    # Assign colors to remaining metrics from available palette
    for metric in y_axis_metrics:
        if metric not in assigned_colors:
            # Find the next available color index
            while (
                next_palette_index in used_palette_indices
                and next_palette_index < len(palette)
            ):
                next_palette_index += 1
            if next_palette_index < len(palette):
                assigned_colors[metric] = palette[next_palette_index]
                used_palette_indices.add(next_palette_index)
            else:
                # Fallback if we run out of colors
                print(
                    f"Warning: Not enough colors in palette for metric {metric}. Using default."
                )
                assigned_colors[metric] = "gray"  # Or some default color

    # Use the assigned_colors dictionary for plotting
    metric_colors = assigned_colors

    # Create a copy of the dataframe to avoid modification warnings
    df = df.copy()

    # Ensure x_axis_metric is numeric
    if not pd.api.types.is_numeric_dtype(df[x_axis_metric]):
        try:
            df[x_axis_metric] = pd.to_numeric(df[x_axis_metric])
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert {x_axis_metric} to numeric type: {e}")

    # Sort the DataFrame by the x-axis metric for proper line connections
    df = df.sort_values(by=x_axis_metric)

    # Get unique x values in sorted order
    x_values = sorted(df[x_axis_metric].unique())

    # Create neat labels for metrics
    neat_metric_labels = {
        metric: " ".join(word.capitalize() for word in metric.replace("_", " ").split())
        .replace("Qwk", "QWK")
        .replace("Mcc", "MCC")
        .replace("Nonbinary", "")
        for metric in y_axis_metrics
    }

    # Create plot with adjusted size
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each metric
    for metric in y_axis_metrics:
        # Use the color from the newly created metric_colors dictionary
        color = metric_colors.get(
            metric, "gray"
        )  # Default to gray if somehow not assigned
        neat_label = neat_metric_labels[metric]

        # Calculate mean and confidence interval for each x value
        means = []
        cis = []

        for x_val in x_values:
            subset = df[df[x_axis_metric] == x_val]
            mean = subset[metric].mean()
            # Fix: Calculate SEM on the specific metric column
            sem = stats.sem(subset[metric]) if len(subset) > 1 else 0
            means.append(mean)
            cis.append(sem)

        # Line plot connecting means with compact styling
        ax.plot(
            x_values, means, color=color, label=neat_label, linewidth=1.5, alpha=0.8
        )

        # Add points with error bars
        ax.errorbar(
            x=x_values,
            y=means,
            yerr=cis,
            fmt="o",
            capsize=3,
            capthick=1,
            elinewidth=1,
            markersize=5,
            color=color,
        )

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)

    # Add labels and grid
    ax.set_xlabel(neat_x_axis_label, fontsize=11)
    ax.set_ylabel(neat_y_axis_label, fontsize=11)

    if show_title:
        ax.set_title(f"Performance Metrics vs {neat_x_axis_label}", fontsize=10)

    # Add lighter grid
    ax.grid(True, linestyle="--", alpha=0.4)

    # Add compact legend
    ax.legend(
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        loc=legend_loc,
        handlelength=1.5,
    )

    # Use actual x values for ticks with reduced number if many values
    if len(x_values) > 5:
        step = max(1, len(x_values) // 5)
        ax.set_xticks(x_values[::step])
    else:
        ax.set_xticks(x_values)

    # Tighter layout to reduce white space
    fig.tight_layout(pad=0.8)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
