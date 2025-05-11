import math
import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, NamedTuple

# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# import seaborn as sns
import sklearn.metrics
import toml
import torch
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

from magellan.gnn_model import (
    Net,
    extract_edge_signs,
    get_edge_weight_matrix,
    train_model,
)
from magellan.graph import Graph
from magellan.json_io import get_const_from_json, json_to_graph
from magellan.network_growth import grow_network
from magellan.plot import (
    analyze_predictions,
    create_annotation_matrix,
    get_filtered_indices,
    plot_loss_vs_validation_loss,
)
from magellan.prune import (
    check_dummy_nodes,
    check_no_self_loops,
    construct_mask_dic,
    create_edge_scale,
    create_pert_mask,
    create_pyg_data_object,
    dummy_setup,
    filter_spec_invalid_experiments,
    filter_specification,
    get_adjacency_matrix_mult,
    get_data_and_update_y,
    get_pert_dic,
    get_pert_list,
    get_real_indices,
    make_edge_idx,
    make_node_dic,
    make_pert_idx,
)
from magellan.prune_opt import (
    WarmupScheduler,
    calculate_node_class_weights,
)
from magellan.sci_opt import get_data, get_sorted_node_list, pred_bound_perts


def _gen_weight(G, dist=np.random.uniform, param=(0, 2)):
    for u, v in G.edges:
        G[u][v]["edge_weight"] = dist(*param)

    return G


def _gen_sign(G, idx_inh):
    for counter, (u, v) in enumerate(G.edges):
        if counter in idx_inh:
            G[u][v]["sign"] = "Inhibitor"
        else:
            G[u][v]["sign"] = "Activator"

    return G


def gen_synthetic(n_node, n_exp, prob=0.2, p_inh=0.2, weight_dist=(0, 2)):
    """
    Generate synthetic network from Erdos-Renyi graphs. Perturbed nodes are set to 'max'

    :param n_node: int, number of nodes
    :param n_exp: int, number of experiments
    :param prob: float, connection probability in the ER graph
    :param p_inh: float, proportion of inhibition
    :param weight_dist: tuple, (val1, val2), lower and upper bound for uniformly distributed weights

    :return:
        df: pd.DataFrame, spec table
        G: nx.DiGraph, attributed ER graph. attributes: sign, edge_weight

    """

    # generate ER graph
    G = nx.erdos_renyi_graph(n=n_node, p=prob, directed=True)
    G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])

    # change node name as BMA cannot handle numeric names (even if converted to string in json)
    G = nx.relabel_nodes(G, {k: "node%d" % k for k in G.nodes})

    # randomly select p_inh inhibited interactions
    n_egde = G.number_of_edges()
    node_list = get_sorted_node_list(G)

    idx_random = np.random.permutation(range(n_egde))[: int(n_egde * p_inh)]

    # assign sign and weight randomly
    G = _gen_sign(G, idx_random)
    G = _gen_weight(G, dist=np.random.uniform, *weight_dist)

    # generate random pert & exp node pairs
    # pert: two random nodes, exp: one random node
    comb = [np.random.choice(node_list, 3, replace=False) for _ in range(n_exp)]

    df = pd.DataFrame(
        columns=[
            "source",
            "paper_title",
            "experiment_overview",
            "cell_line",
            "experiment_particular",
            "gene",
            "perturbation",
            "expected_result_bma",
            "Notes",
        ]
    )

    df["gene"] = comb
    df["experiment_particular"] = ["%s_%s" % (ele[0], ele[1]) for ele in comb]

    for col in ["source", "paper_title", "experiment_overview", "cell_line"]:
        df[col] = range(1, len(comb) + 1)

    df = df.explode("gene")

    idx = np.asarray(range(n_exp * 3))
    idx_exp = idx[range(2, n_exp * 3, 3)]

    df["perturbation"] = "max"
    df.iloc[idx_exp, -2] = "max"
    df.iloc[idx_exp, -3] = np.nan

    return df, G


# V2 of synthetic data generation


@dataclass
class PruningTestConfig:
    """Configuration for pruning test"""

    # Network generation parameters
    generate_base_graph: bool = True
    grow_base_graph: bool = False
    growth_method: str = "duplication_divergence"
    # growth_params: dict = field(
    #     default_factory=lambda: {
    #         "p_forward": 0.3,  # Probability for forward connections
    #         "p_backward": 0.1,  # Probability for backward connections
    #         "p_inhibition": 0.3,  # Fraction of inhibitory edges
    #         "min_edge_weight": 0.1,  # Minimum edge weight
    #         "max_edge_weight": 1.0,  # Maximum edge weight
    #         "preserve_original": True,  # Preserve original network structure
    #         "seed": 42,  # Random seed for reproducibility
    #     },
    # )
    growth_params = {
        "divergence_prob": 0.3,  # Probability of connection divergence
        "edge_deletion_prob": 0.4,  # Probability of edge deletion during divergence
        "p_inhibition": 0.3,  # Fraction of inhibitory edges
        "min_edge_weight": 0.1,  # Minimum edge weight
        "max_edge_weight": 1.0,  # Maximum edge weight
        "preserve_original": True,  # Preserve original network structure
        "seed": 42,  # Random seed for reproducibility
    }
    base_graph_import: Path | None = None
    generate_specifications: bool = True
    spec_file: Path | None = None
    filter_spec: bool | None = None
    n_nodes: int = 100
    forward_edge_probability: float = 0.3
    backward_edge_probability: float = 0.1
    skip_layer_probability: float = 0.1
    n_spurious_edges: int = 30
    n_specifications: int = 50
    inhibition_fraction: float = 0.3
    min_edge_weight: float = 0.1
    max_edge_weight: float = 1.0
    terminal_node_bias: float | None = None
    expected_phenotype_nodes: list = field(
        default_factory=lambda: ["Proliferation", "Apoptosis"]
    )
    min_spec_inputs: int = 1
    max_spec_inputs: int = 10
    min_spec_outputs: int = 1
    max_spec_outputs: int = 5
    source_input_bias: float | None = 10.0
    input_degree_bias: float | None = 5.0
    input_degree_type: str = "total"
    measured_node_coverage: int | None = None
    class_imbalance: float = 0.0
    tf_method: str = "sum"
    out_dir: Path | str = Path(".")

    # Training parameters
    epochs: int = 1000
    learning_rate: float = 0.005
    min_range: int = 0
    max_range: int = 2
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    edge_weight_init: float = 0.5
    n_iter: int = 50
    max_update: bool = True
    round_val: bool = False
    allow_sign_flip: bool = False
    weight_threshold: float = 0.01
    category_curriculum: bool = False
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 100
    warmup_steps: int = 100
    warmup_initial_lr_factor: float = 0.1
    use_hybrid_loss: bool = True
    hybrid_loss_alpha: float = 1.0
    use_class_weights: bool = False
    use_node_class_weights: bool = True
    class_weight_method: str = "inverse_freq"
    noise_fraction: float = 0.0
    noise_std: float = 1.0
    tf_method: str = "avg"
    test_size: float = 0.0
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0
    weight_decay: float = 0.00
    model_save_dir: Path | str = Path(".")

    # Simulation parameters
    max_simulation_steps: int = 100
    step_size: float = 1.0

    # Evaluation parameters
    seed: int = 42
    save_confusion_plots: bool = False
    save_binary_metrics_csv: bool = False

    # Add new density parameter
    network_density: float = 1.0  # Multiplier for connection probabilities (0.1 to 2.0)

    @classmethod
    def from_toml(cls, path: str | Path) -> "PruningTestConfig":
        """Create config from TOML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        config_dict = toml.load(path)

        # Convert paths from strings
        if "graph_generation" in config_dict:
            if "base_graph_import" in config_dict["graph_generation"]:
                config_dict["graph_generation"]["base_graph_import"] = Path(
                    config_dict["graph_generation"]["base_graph_import"]
                )
            if "spec_file" in config_dict["graph_generation"]:
                config_dict["graph_generation"]["spec_file"] = Path(
                    config_dict["graph_generation"]["spec_file"]
                )

        # Flatten nested sections into main dict
        sections = [
            "graph_generation",
            "specification_generation",
            "training",
            "simulation",
            "evaluation",
        ]

        flattened_dict = {}
        for section in sections:
            if section in config_dict:
                flattened_dict.update(config_dict[section])

        # Filter out unknown fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in flattened_dict.items() if k in valid_fields}

        return cls(**filtered_dict)

    def copy(self) -> "PruningTestConfig":
        """Create a deep copy of the config"""
        return deepcopy(self)


class EdgeStructureStats(NamedTuple):
    """Statistics about edge structure"""

    spurious_removed: float  # Fraction of spurious edges removed
    true_removed: float  # Fraction of true edges removed
    edge_accuracy: float  # Accuracy of edge structure recovery
    edge_structure_f1: float  # F1 score of edge structure recovery
    edge_structure_mcc: float  # MCC of edge structure recovery
    edge_structure_qwk: float  # QWK of edge structure recovery
    edge_precision: float  # Precision of edge structure recovery
    edge_recall: float  # Recall of edge structure recovery


def generate_base_graph(config: PruningTestConfig, verbose: bool = True) -> Graph:
    """Generate base directed graph with dynamic layered structure"""
    if config.network_density < 0.1:
        raise ValueError("network_density must be greater than 0.1")

    if config.n_nodes <= 0:
        raise ValueError("n_nodes must be greater than 0")

    if (
        config.forward_edge_probability
        or config.backward_edge_probability
        or config.skip_layer_probability
    ) > 1.0:
        raise ValueError("probabilities must be less than 1.0")

    if (
        config.forward_edge_probability
        or config.backward_edge_probability
        or config.skip_layer_probability
    ) < 0.0:
        raise ValueError("probabilities must be greater than 0.0")

    if config.min_edge_weight < 0.0:
        raise ValueError("min_edge_weight must be greater than 0.0")

    if config.max_edge_weight < 0.0:
        raise ValueError("max_edge_weight must be greater than 0.0")

    if config.max_edge_weight < config.min_edge_weight:
        raise ValueError("max_edge_weight must be greater than min_edge_weight")

    # Scale probabilities while ensuring they stay in [0,1]
    forward_prob = min(1.0, config.forward_edge_probability * config.network_density)
    backward_prob = min(1.0, config.backward_edge_probability * config.network_density)
    skip_prob = min(1.0, config.skip_layer_probability * config.network_density)

    G = Graph(remove_cycle=True, remove_sign=True, remove_neither=True)

    # Create layered structure
    n_layers = 10  # Number of network layers
    nodes_per_layer = config.n_nodes // n_layers
    remaining = config.n_nodes % n_layers

    # Distribute nodes across layers
    layer_sizes = [
        nodes_per_layer + (1 if i < remaining else 0) for i in range(n_layers)
    ]
    layers = []
    node_counter = 0

    # Create nodes in each layer
    for size in layer_sizes:
        layer = [str(i) for i in range(node_counter, node_counter + size)]
        layers.append(layer)
        node_counter += size

    # Add edges between and within layers with higher forward connection probability
    edges = []

    # set forward edge weight to be higher than min_edge_weight but lower than max_edge_weight by 0.2 non randomly
    forward_edge_weight = min(config.max_edge_weight, config.min_edge_weight + 0.2)
    backward_edge_weight = max(config.min_edge_weight + 0.1, config.max_edge_weight)

    for i, layer in enumerate(layers):
        # Connect to next layer with high probability
        if i < len(layers) - 1:
            next_layer = layers[i + 1]
            for u in layer:
                for v in next_layer:
                    if np.random.random() < forward_prob:  # Use scaled probability
                        weight = np.random.uniform(
                            forward_edge_weight, config.max_edge_weight
                        )
                        edges.append(
                            (
                                u,
                                v,
                                {
                                    "edge_weight": weight,
                                    "sign": "Activator",
                                    "n_references": weight,
                                },
                            )
                        )

        # Add some within-layer connections
        for u in layer:
            for v in layer:
                if (
                    u != v and np.random.random() < backward_prob
                ):  # Use scaled probability
                    weight = np.random.uniform(
                        backward_edge_weight, config.max_edge_weight
                    )
                    edges.append(
                        (
                            u,
                            v,
                            {
                                "edge_weight": weight,
                                "sign": "Activator",
                                "n_references": weight,
                            },
                        )
                    )

        # Add some skip connections to layers 2+ steps ahead
        if i < len(layers) - 2:
            for u in layer:
                for future_layer in layers[i + 2 :]:
                    for v in future_layer:
                        if np.random.random() < skip_prob:  # Use scaled probability
                            weight = np.random.uniform(
                                forward_edge_weight, config.max_edge_weight
                            )
                            edges.append(
                                (
                                    u,
                                    v,
                                    {
                                        "edge_weight": weight,
                                        "sign": "Activator",
                                        "n_references": weight,
                                    },
                                )
                            )

    # Add edges to graph
    G.add_edges_from(edges)

    # Convert fraction of edges to inhibitory, focusing on within-layer and backwards edges
    n_inhibitory = int(len(G.edges()) * config.inhibition_fraction)

    # Prioritize within-layer and backwards edges for inhibition
    edge_scores = []
    for u, v in G.edges():
        u_layer = next(i for i, layer in enumerate(layers) if u in layer)
        v_layer = next(i for i, layer in enumerate(layers) if v in layer)

        # Score edges for inhibition probability
        # Higher scores for within-layer and backwards connections
        if u_layer == v_layer:
            score = 0.8  # High chance for within-layer inhibition
        elif u_layer > v_layer:
            score = 0.7  # Higher chance for backwards inhibition
        else:
            score = 0.4  # Lower chance for forward inhibition

        edge_scores.append((score, (u, v)))

    # Sort by score and convert top fraction to inhibitory
    edge_scores.sort(reverse=True)
    for _, (u, v) in edge_scores[:n_inhibitory]:
        G[u][v]["sign"] = "Inhibitor"
        # Increase inhibitory weights slightly
        G[u][v]["edge_weight"] *= 1.2
        G[u][v]["edge_weight"] = min(G[u][v]["edge_weight"], config.max_edge_weight)

    if verbose:
        print(
            f"Generated base graph with {len(G.edges())} edges and {len(G.nodes())} nodes"
        )

    return G


def add_spurious_edges(
    G: Graph | nx.DiGraph, config: PruningTestConfig
) -> tuple[Graph, set]:
    """Add spurious edges while maintaining realism and matching inhibition fraction.

    Optimized for sparse graphs where |E| << |V|^2.

    Args:
        G: Input graph
        config: Configuration parameters

    Returns:
        tuple containing:
        - Graph with added spurious edges
        - Set of spurious edge tuples

    Raises:
        ValueError: If not enough valid edges available to add
    """
    # Convert to BMATool Graph once
    G_noisy = Graph(remove_cycle=True, remove_sign=True, remove_neither=True)
    G_noisy.add_edges_from(G.edges(data=True))

    # Get nodes and existing edges efficiently
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_indices = {node: i for i, node in enumerate(nodes)}
    existing_edges = {(node_indices[u], node_indices[v]) for u, v in G.edges()}
    reverse_edges = {(v, u) for u, v in existing_edges}

    # Pre-calculate number of edges needed
    n_spurious_inhibitory = int(config.n_spurious_edges * config.inhibition_fraction)
    n_spurious_activator = config.n_spurious_edges - n_spurious_inhibitory

    # Function to generate random valid edges
    def generate_valid_edges(n_needed: int) -> set[tuple[int, int]]:
        valid_edges = set()
        attempts = 0
        max_attempts = n_needed * 10  # Avoid infinite loops

        while len(valid_edges) < n_needed and attempts < max_attempts:
            # Generate random edge indices
            u = np.random.randint(n_nodes)
            v = np.random.randint(n_nodes)
            edge = (u, v)

            # Check validity efficiently
            if (
                u != v  # Not self-loop
                and edge not in existing_edges  # Not existing
                and edge not in valid_edges  # Not already selected
                and edge not in reverse_edges
            ):  # Won't create cycle
                valid_edges.add(edge)

            attempts += 1

        return valid_edges

    # Generate edges
    activator_edges = generate_valid_edges(n_spurious_activator)
    remaining_edges = generate_valid_edges(n_spurious_inhibitory)

    if (
        len(activator_edges) < n_spurious_activator
        or len(remaining_edges) < n_spurious_inhibitory
    ):
        raise ValueError(
            f"Could not find enough valid edges. Found {len(activator_edges)} activator "
            f"and {len(remaining_edges)} inhibitor edges."
        )

    # Add edges to graph
    spurious_edges = set()

    # Helper to add edges of a specific type
    def add_edges_of_type(edges: set[tuple[int, int]], sign: str) -> None:
        for u_idx, v_idx in edges:
            u, v = nodes[u_idx], nodes[v_idx]
            weight = np.random.uniform(config.min_edge_weight, config.max_edge_weight)
            G_noisy.add_edge(
                u,
                v,
                weight=weight,
                sign=sign,
                n_references=config.min_edge_weight,
            )
            spurious_edges.add((u, v))

    # Add both types of edges
    with tqdm(total=2, desc="Adding spurious edges", leave=False) as pbar:
        add_edges_of_type(activator_edges, "Activator")
        pbar.update(1)
        add_edges_of_type(remaining_edges, "Inhibitor")
        pbar.update(1)

    return G_noisy, spurious_edges


def simulate_node_values(
    G: nx.DiGraph, input_nodes: set, input_values: dict, config: PruningTestConfig
) -> dict:
    """Simulate network dynamics using weighted average activation or weighted sum activation

    Args:
        G: Networkx graph
        input_nodes: Set of input nodes
        input_values: Dictionary of input node values
        config: Configuration parameters

    Returns:
        Dictionary of node values
    """
    # Initialize values
    values = {n: 0 for n in G.nodes()}
    for node in input_nodes:
        values[node] = input_values[node]

    # Track convergence
    history = [values.copy()]
    convergence_threshold = 0.1

    for _ in tqdm(
        range(config.max_simulation_steps),
        desc="Simulating network dynamics",
        leave=False,
    ):
        new_values = values.copy()
        max_change = 0

        for node in G.nodes():
            if node not in input_nodes:
                # Get all incoming edges
                activators = []
                activator_weights = []
                inhibitors = []
                inhibitor_weights = []

                for pred in G.predecessors(node):
                    weight = G[pred][node]["edge_weight"]
                    if G[pred][node]["sign"] == "Activator":
                        activators.append(values[pred])
                        activator_weights.append(weight)
                    else:
                        inhibitors.append(values[pred])
                        inhibitor_weights.append(weight)

                # Calculate weighted averages or sums
                if activators:
                    activator_sum = sum(
                        v * w for v, w in zip(activators, activator_weights)
                    )
                    total_activator_weight = sum(activator_weights)
                    if config.tf_method == "avg":
                        activation = activator_sum / total_activator_weight
                    elif config.tf_method == "sum":
                        activation = activator_sum
                    else:
                        raise ValueError(f"Invalid tf_method: {config.tf_method}")
                else:
                    activation = config.max_range if inhibitors else 0

                if inhibitors:
                    inhibitor_sum = sum(
                        v * w for v, w in zip(inhibitors, inhibitor_weights)
                    )
                    total_inhibitor_weight = sum(inhibitor_weights)
                    if config.tf_method == "avg":
                        inhibition = inhibitor_sum / total_inhibitor_weight
                    elif config.tf_method == "sum":
                        inhibition = inhibitor_sum
                    else:
                        raise ValueError(f"Invalid tf_method: {config.tf_method}")
                else:
                    inhibition = 0

                # Value update
                new_val = activation - inhibition
                new_val = max(config.min_range, min(config.max_range, new_val))

                # Track maximum change
                max_change = max(max_change, abs(new_val - values[node]))
                new_values[node] = int(round(new_val))

        values = new_values
        history.append(values.copy())

        # Check for convergence
        if max_change < convergence_threshold:
            break

        # Check for oscillation
        if len(history) > 4:
            recent_states = [str(state) for state in history[-4:]]
            if (
                recent_states[-1] == recent_states[-3]
                and recent_states[-2] == recent_states[-4]
            ):
                # Take average of oscillating states
                final_values = {}
                for node in values:
                    final_values[node] = (history[-1][node] + history[-2][node]) / 2
                values = final_values
                break

    # Round final values
    return {node: int(round(val)) for node, val in values.items()}


def split_specifications(
    specifications: dict | OrderedDict, config: PruningTestConfig
) -> tuple[dict, dict]:
    """Split specifications dictionary into train and test sets.

    Args:
        specifications: Dictionary of specifications to split
        config: Configuration parameters containing test_size value

    Returns:
        tuple[dict, dict]: Train and test specification dictionaries
    """
    test_size = config.test_size
    random_state = config.seed

    # If test_size is 0, return all data as training and empty dict as test
    if test_size == 0.0:
        return specifications, {}

    # Get list of specification keys
    spec_keys = list(specifications.keys())

    # Split the keys
    train_keys, test_keys = train_test_split(
        spec_keys, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Create train and test dictionaries
    train_specs = {k: specifications[k] for k in train_keys}
    test_specs = {k: specifications[k] for k in test_keys}

    return train_specs, test_specs


def generate_diverse_specifications(
    G: nx.DiGraph,
    config: PruningTestConfig,
    verbose: bool = True,
) -> tuple[dict, dict, dict]:
    """Generate diverse and informative specifications with optional noise and node-specific class imbalance

    Args:
        G: Input network
        config: Configuration parameters
        verbose: Whether to print verbose output

    Returns:
        tuple[dict, dict, dict]: Generated train and test specifications, and a dictionary of statistics
    """
    # Validate configuration parameters
    validate_config(G, config)

    # Initialize data structures
    nodes = list(G.nodes())
    specifications = {}
    node_coverage = {node: {"input": 0, "output": 0} for node in nodes}
    node_preferred_values = initialize_node_preferences(
        nodes, config.min_range, config.max_range
    )

    # Pre-compute node sets for efficient lookup
    node_sets = precompute_node_sets(G, nodes, config.terminal_node_bias)

    # Generate specifications
    specifications = generate_specifications(
        G, config, nodes, node_coverage, node_preferred_values, node_sets
    )

    # Calculate statistics if requested
    stats = {}
    if verbose:
        stats = calculate_statistics(
            specifications, nodes, node_preferred_values, node_sets["terminal_nodes"]
        )

    # Split specifications into train and test sets
    train_specs, test_specs = split_specifications(specifications, config)

    return train_specs, test_specs, stats


def validate_config(G: nx.DiGraph, config: PruningTestConfig) -> None:
    """Validate configuration parameters.

    Args:
        G: Input network
        config: Configuration parameters

    Raises:
        ValueError: If any configuration parameter is invalid
    """
    nodes = list(G.nodes())

    # Validate measured node coverage
    if config.measured_node_coverage:
        if (
            config.measured_node_coverage < 5
            or config.measured_node_coverage > config.n_nodes
        ):
            raise ValueError(
                f"Measured node coverage ({config.measured_node_coverage}) must be between 5 and "
                f"the number of nodes ({config.n_nodes})"
            )

    # Validate input and output ranges
    if config.min_spec_inputs > len(nodes) - 1:
        raise ValueError(
            f"Minimum number of inputs ({config.min_spec_inputs}) is greater than the "
            f"number of nodes minus one ({len(nodes) - 1})"
        )

    if config.min_spec_outputs > len(nodes) - 1:
        raise ValueError(
            f"Minimum number of outputs ({config.min_spec_outputs}) is greater than the "
            f"number of nodes minus one ({len(nodes) - 1})"
        )

    if config.min_spec_inputs >= config.max_spec_inputs:
        raise ValueError(
            f"Minimum number of inputs ({config.min_spec_inputs}) is greater than or equal to "
            f"maximum number of inputs ({config.max_spec_inputs})"
        )

    if config.min_spec_outputs >= config.max_spec_outputs:
        raise ValueError(
            f"Minimum number of outputs ({config.min_spec_outputs}) is greater than or equal to "
            f"maximum number of outputs ({config.max_spec_outputs})"
        )

    # Validate node connections
    nodes_with_children = {n for n in nodes if G.out_degree(n) > 0}
    nodes_with_parents = {n for n in nodes if G.in_degree(n) > 0}

    if len(nodes_with_children) == 0:
        raise ValueError("No nodes with children found in the graph")
    if len(nodes_with_parents) == 0:
        raise ValueError("No nodes with parents found in the graph")


def initialize_node_preferences(nodes: list, min_range: int, max_range: int) -> dict:
    """Initialize preferred values for each node.

    Args:
        nodes: List of nodes
        min_range: Minimum value range
        max_range: Maximum value range

    Returns:
        dict: Dictionary mapping nodes to their preferred values
    """
    return {node: np.random.randint(min_range, max_range + 1) for node in nodes}


def precompute_node_sets(
    G: nx.DiGraph, nodes: list, terminal_node_bias: float | None = None
) -> dict:
    """Precompute sets of nodes for efficient lookup.

    Args:
        G: Input network
        nodes: List of nodes
        terminal_node_bias: Bias for terminal nodes

    Returns:
        dict: Dictionary containing sets of nodes
    """
    terminal_nodes = (
        {n for n in nodes if G.out_degree(n) == 0} if terminal_node_bias else set()
    )
    nodes_with_children = {n for n in nodes if G.out_degree(n) > 0}
    nodes_with_parents = {n for n in nodes if G.in_degree(n) > 0}

    return {
        "terminal_nodes": terminal_nodes,
        "nodes_with_children": nodes_with_children,
        "nodes_with_parents": nodes_with_parents,
    }


def select_input_nodes(
    valid_nodes: list[str],  # Added type hint
    node_coverage: dict,
    min_inputs: int,
    max_inputs: int,
    node_sets: dict[str, set[str]],  # Need access to node sets (specifically sources)
    G: nx.DiGraph,  # Need access to the graph to get degrees
    config: PruningTestConfig,  # Need access to config for bias parameters
) -> set[str]:  # Added type hint
    """Select input nodes with preference for under-represented nodes, sources, and high-degree nodes."""

    input_weights = []
    # Pre-calculate degrees once
    # Add explicit type hints for dictionary keys and values
    in_degrees: dict[str, int] = dict(G.in_degree())
    out_degrees: dict[str, int] = dict(G.out_degree())
    total_degrees: dict[str, int] = dict(G.degree())  # type: ignore

    # Find max degree for normalization
    max_in_degree = max(in_degrees.values()) if in_degrees else 1
    max_out_degree = max(out_degrees.values()) if out_degrees else 1
    max_total_degree = max(total_degrees.values()) if total_degrees else 1

    for n in valid_nodes:
        # Base weight from coverage
        weight = 1.0 / (1 + node_coverage[n]["input"])

        # Apply source node bias
        if (
            config.source_input_bias is not None
            and n not in node_sets["nodes_with_parents"]
        ):
            weight *= config.source_input_bias

        # Apply degree bias
        if config.input_degree_bias is not None:
            degree_value = 0
            max_degree_value = 1
            if config.input_degree_type == "in":
                degree_value = in_degrees.get(n, 0)
                max_degree_value = max_in_degree
            elif config.input_degree_type == "out":
                degree_value = out_degrees.get(n, 0)
                max_degree_value = max_out_degree
            elif config.input_degree_type == "total":
                degree_value = total_degrees.get(n, 0)
                max_degree_value = max_total_degree
            else:
                raise ValueError(
                    f"Invalid input_degree_type: {config.input_degree_type}"
                )

            normalized_degree = (
                degree_value / max_degree_value if max_degree_value > 0 else 0
            )

            # Simple multiplicative bias based on normalized degree
            # weight *= (1 + config.input_degree_bias * normalized_degree)
            # Alternative: Exponential bias to strongly favor high degree nodes
            weight *= math.exp(config.input_degree_bias * normalized_degree)

        input_weights.append(weight)

    # Ensure we have valid weights and select nodes
    if sum(input_weights) <= 0:
        input_weights = [1.0] * len(
            valid_nodes
        )  # Fallback to uniform if all weights are zero

    effective_min_inputs = min(min_inputs, len(valid_nodes))
    effective_max_inputs = min(max_inputs, len(valid_nodes))

    if effective_min_inputs > effective_max_inputs:
        effective_min_inputs = effective_max_inputs  # Or handle as an error/warning

    # Ensure size is not more than available nodes
    n_inputs_to_select = np.random.randint(
        effective_min_inputs, effective_max_inputs + 1
    )
    n_inputs_to_select = min(n_inputs_to_select, len(valid_nodes))

    input_nodes = set(
        np.random.choice(
            valid_nodes,
            size=n_inputs_to_select,
            p=np.array(input_weights) / sum(input_weights),
            replace=False,
        )
    )

    return input_nodes


def select_output_nodes(
    available_outputs: list,
    node_coverage: dict,
    output_values: dict,
    node_preferred_values: dict,
    terminal_nodes: set,
    min_outputs: int,
    max_outputs: int,
    config,
) -> set:
    """Select output nodes with preference for under-represented nodes.

    Args:
        available_outputs: List of available output nodes
        node_coverage: Dictionary tracking node coverage
        output_values: Dictionary of simulated output values
        node_preferred_values: Dictionary of preferred values for each node
        terminal_nodes: Set of terminal nodes
        min_outputs: Minimum number of outputs
        max_outputs: Maximum number of outputs
        config: Configuration parameters

    Returns:
        set: Selected output nodes
    """
    if not available_outputs:
        return set()

    output_weights = []
    for n in available_outputs:
        # Base weight from coverage
        weight = 1.0 / (1 + node_coverage[n]["output"])

        # Increase weight for terminal nodes
        if config.terminal_node_bias and n in terminal_nodes:
            weight *= config.terminal_node_bias

        # Apply class imbalance bias: prefer nodes when their value matches their preferred value
        if config.class_imbalance > 0 and n in output_values:
            simulated_value = output_values[n]
            preferred_value = node_preferred_values[n]

            if simulated_value == preferred_value:
                # Boost weight if the simulated value matches the preferred value
                weight *= 1 + config.class_imbalance * 5
            else:
                # Reduce weight if it doesn't match
                weight *= 1 - config.class_imbalance * 0.5
                weight = max(0.01, weight)  # Ensure weight doesn't go to zero

        output_weights.append(weight)

    # Ensure we have valid weights
    if sum(output_weights) <= 0:
        output_weights = [1.0] * len(available_outputs)

    effective_min_outputs = min(min_outputs, len(available_outputs))
    effective_max_outputs = min(max_outputs, len(available_outputs))

    if effective_min_outputs >= effective_max_outputs:
        effective_min_outputs = max(1, effective_max_outputs - 1)

    n_outputs = np.random.randint(effective_min_outputs, effective_max_outputs + 1)

    output_nodes = set(
        np.random.choice(
            available_outputs,
            size=min(n_outputs, len(available_outputs)),
            p=np.array(output_weights) / sum(output_weights),
            replace=False,
        )
    )

    return output_nodes


def add_noise_to_outputs(
    output_nodes: set,
    output_values: dict,
    noise_fraction: float,
    noise_std: float,
    min_range: int,
    max_range: int,
) -> dict:
    """Add noise to randomly selected output nodes.

    Args:
        output_nodes: Set of output nodes
        output_values: Dictionary of output values
        noise_fraction: Fraction of output nodes to add noise to
        noise_std: Standard deviation of noise
        min_range: Minimum value range
        max_range: Maximum value range

    Returns:
        dict: Dictionary of output values with noise added
    """
    noisy_output_values = output_values.copy()

    n_noisy = int(len(output_nodes) * noise_fraction)
    if n_noisy > 0:
        noisy_nodes = np.random.choice(list(output_nodes), size=n_noisy, replace=False)
        for node in noisy_nodes:
            # Add Gaussian noise and clip to valid range
            noise = math.ceil(np.random.normal(0, noise_std))
            noisy_value = output_values[node] + noise
            noisy_output_values[node] = int(
                round(max(min_range, min(max_range, noisy_value)))
            )

    return noisy_output_values


def generate_specifications(
    G: nx.DiGraph,
    config: PruningTestConfig,
    nodes: list,
    node_coverage: dict,
    node_preferred_values: dict,
    node_sets: dict,
) -> dict:
    """Generate specifications based on configuration.

    Args:
        G: Input network
        config: Configuration parameters
        nodes: List of nodes
        node_coverage: Dictionary tracking node coverage
        node_preferred_values: Dictionary of preferred values for each node
        node_sets: Dictionary containing sets of nodes

    Returns:
        dict: Generated specifications
    """
    np.random.seed(config.seed)
    specifications = {}

    # Calculate effective output range
    if config.measured_node_coverage:
        min_spec_outputs = config.measured_node_coverage - 5
        max_spec_outputs = config.measured_node_coverage
    else:
        min_spec_outputs = config.min_spec_outputs
        max_spec_outputs = config.max_spec_outputs

    for i in tqdm(
        range(config.n_specifications),
        desc="Generating specifications",
        leave=False,
    ):
        # Select input nodes
        valid_input_nodes = [n for n in nodes if n in node_sets["nodes_with_children"]]
        input_nodes = select_input_nodes(
            valid_input_nodes,
            node_coverage,
            config.min_spec_inputs,
            config.max_spec_inputs,
            node_sets,
            G,
            config,
        )

        # Generate input values
        input_values = {
            node: np.random.randint(config.min_range, config.max_range + 1)
            for node in input_nodes
        }

        # Simulate to get output values
        output_values = simulate_node_values(G, input_nodes, input_values, config)

        # Select output nodes
        available_outputs = [
            n
            for n in nodes
            if n in node_sets["nodes_with_parents"] and n not in input_nodes
        ]

        output_nodes = select_output_nodes(
            available_outputs,
            node_coverage,
            output_values,
            node_preferred_values,
            node_sets["terminal_nodes"],
            min_spec_outputs,
            max_spec_outputs,
            config,
        )

        if not output_nodes:
            continue  # Skip this specification if no valid output nodes

        # Add noise to outputs
        output_values = add_noise_to_outputs(
            output_nodes,
            output_values,
            config.noise_fraction,
            config.noise_std,
            config.min_range,
            config.max_range,
        )

        # Update coverage
        for node in input_nodes:
            node_coverage[node]["input"] += 1
        for node in output_nodes:
            node_coverage[node]["output"] += 1

        specifications[str(i)] = {
            "pert": {str(n): v for n, v in input_values.items()},
            "exp": {str(n): output_values[n] for n in output_nodes},
        }

    return specifications


def calculate_statistics(
    specifications: dict, nodes: list, node_preferred_values: dict, terminal_nodes: set
) -> dict:
    """Calculate statistics for generated specifications.

    Args:
        specifications: Generated specifications
        nodes: List of nodes
        node_preferred_values: Dictionary of preferred values for each node
        terminal_nodes: Set of terminal nodes

    Returns:
        dict: Dictionary of statistics
    """
    stats = {}

    # Calculate value distribution per node
    node_class_counts = {node: {} for node in nodes}
    for spec in specifications.values():
        for node, value in spec["exp"].items():
            if value not in node_class_counts[node]:
                node_class_counts[node][value] = 0
            node_class_counts[node][value] += 1

    # Store basic spec statistics
    stats["n_specifications"] = len(specifications)

    if specifications:
        stats["mean_perturbations"] = np.mean(
            [len(spec["pert"]) for spec in specifications.values()]
        )
        stats["min_perturbations"] = np.min(
            [len(spec["pert"]) for spec in specifications.values()]
        )
        stats["max_perturbations"] = np.max(
            [len(spec["pert"]) for spec in specifications.values()]
        )
        stats["mean_outputs"] = np.mean(
            [len(spec["exp"]) for spec in specifications.values()]
        )
        stats["min_outputs"] = np.min(
            [len(spec["exp"]) for spec in specifications.values()]
        )
        stats["max_outputs"] = np.max(
            [len(spec["exp"]) for spec in specifications.values()]
        )
    else:
        stats["mean_perturbations"] = 0
        stats["min_perturbations"] = 0
        stats["max_perturbations"] = 0
        stats["mean_outputs"] = 0
        stats["min_outputs"] = 0
        stats["max_outputs"] = 0

    # Store terminal node stats
    n_spec_with_terminal_nodes = sum(
        1
        for out in specifications.values()
        if any(n in terminal_nodes for n in out["exp"])
    )
    total_measurements = sum(len(spec["exp"]) for spec in specifications.values())
    n_terminal_measurements = sum(
        1
        for spec in specifications.values()
        for node in spec["exp"]
        if node in terminal_nodes
    )

    stats["n_spec_with_terminal_nodes"] = n_spec_with_terminal_nodes
    stats["total_measurements"] = total_measurements
    stats["n_terminal_measurements"] = n_terminal_measurements
    stats["terminal_fraction"] = (
        n_terminal_measurements / total_measurements if total_measurements > 0 else 0
    )
    print(f"Terminal fraction: {stats['terminal_fraction']}")

    # Store class imbalance stats
    all_imbalance_scores = []
    node_value_distributions = {}  # Store per-node distributions

    for node, counts in node_class_counts.items():
        if len(counts) > 0:
            total_count = sum(counts.values())
            distribution = {
                value: count / total_count for value, count in counts.items()
            }
            node_value_distributions[node] = distribution
            # Only consider nodes with enough measurements for overall score
            if total_count > 5:
                values = list(counts.keys())
                frequencies = list(distribution.values())
                # Simple imbalance score: difference between max frequency and uniform frequency
                if len(values) > 0:
                    uniform_freq = 1.0 / len(values)
                    imbalance = max(frequencies) - uniform_freq
                    all_imbalance_scores.append(imbalance)

    stats["node_value_distributions"] = (
        node_value_distributions  # Add node distributions
    )
    stats["node_preferred_values"] = node_preferred_values  # Add preferred values

    if all_imbalance_scores:
        stats["avg_class_imbalance_score"] = sum(all_imbalance_scores) / len(
            all_imbalance_scores
        )
        stats["min_class_imbalance_score"] = min(all_imbalance_scores)
        stats["max_class_imbalance_score"] = max(all_imbalance_scores)
    else:
        stats["avg_class_imbalance_score"] = 0
        stats["min_class_imbalance_score"] = 0
        stats["max_class_imbalance_score"] = 0

    return stats


def evaluate_network_edges(
    model: Net,
    edge_idx: torch.Tensor,
    G: nx.DiGraph,
    spurious_edges: set,
    original_edges: set,
    config: PruningTestConfig,
    verbose: bool = True,
) -> EdgeStructureStats:
    """Comprehensive evaluation of network pruning and prediction"""
    model.eval()
    # Extract final weights
    weights = {}
    edge_weights = model.edge_weight.detach().cpu().numpy()
    edge_idx_np = edge_idx.cpu().numpy()

    node_list = get_sorted_node_list(G)
    idx_to_node = dict(enumerate(node_list))
    for i in range(edge_idx_np.shape[1]):
        u_idx, v_idx = edge_idx_np[:, i]
        if u_idx != v_idx:  # Skip self-loops
            u_name = idx_to_node[int(u_idx)]
            v_name = idx_to_node[int(v_idx)]
            weights[(u_name, v_name)] = float(edge_weights[i])

    # Remove edges involving dummy nodes
    weights = {
        edge: weight
        for edge, weight in weights.items()
        if not ("dummy" in edge[0] or "dummy" in edge[1])
        if not ("0A_node00" in edge[0] or "0A_node00" in edge[1])
    }

    # Calculate edge structure metrics
    pruned_edges = {
        edge for edge, weight in weights.items() if weight < config.weight_threshold
    }

    predicted_edges = {
        edge for edge, weight in weights.items() if weight >= config.weight_threshold
    }

    # All possible edges that were considered
    all_evaluated_edges = original_edges | spurious_edges

    assert predicted_edges & pruned_edges == set()
    assert all_evaluated_edges == predicted_edges | pruned_edges

    if len(spurious_edges) > 0:
        spurious_removed = spurious_edges & pruned_edges
        n_spurious_removed = len(spurious_removed)
        proportion_spurious_removed = n_spurious_removed / len(spurious_edges)
    else:
        if verbose:
            print("No spurious edges")
        spurious_removed = set()
        n_spurious_removed = 0
        proportion_spurious_removed = 0
    if len(original_edges) > 0:
        true_removed = original_edges & pruned_edges
        n_true_removed = len(true_removed)
        proportion_true_removed = n_true_removed / len(original_edges)
    else:
        if verbose:
            print("No original edges")
        true_removed = set()
        n_true_removed = 0
        proportion_true_removed = 0

    # Calculate edge structure accuracy

    assert len(all_evaluated_edges) == len(original_edges) + len(spurious_edges)
    assert (
        len(predicted_edges)
        == len(original_edges)
        + len(spurious_edges)
        - n_spurious_removed
        - n_true_removed
    )
    assert all_evaluated_edges == predicted_edges | spurious_removed | true_removed

    # Calculate true positives, false positives, true negatives, false negatives
    true_positives = len(predicted_edges & original_edges)
    false_positives = len(predicted_edges & spurious_edges)
    true_negatives = len(spurious_edges - predicted_edges)
    false_negatives = len(original_edges - predicted_edges)

    # Calculate edge accuracy as (TP + TN) / (TP + TN + FP + FN)
    edge_accuracy = (true_positives + true_negatives) / len(all_evaluated_edges)
    edge_precision = true_positives / (true_positives + false_positives)
    edge_recall = true_positives / (true_positives + false_negatives)
    edge_structure_f1 = (
        (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)
        if (2 * true_positives + false_positives + false_negatives) > 0
        else 0
    )
    edge_structure_mcc = (
        (true_positives * true_negatives - false_positives * false_negatives)
        / np.sqrt(
            (true_positives + false_positives)
            * (true_positives + false_negatives)
            * (true_negatives + false_positives)
            * (true_negatives + false_negatives)
        )
        if (true_positives + false_positives)
        * (true_positives + false_negatives)
        * (true_negatives + false_positives)
        * (true_negatives + false_negatives)
        > 0
        else 0
    )

    # Calculate Cohen's weighted kappa
    # Convert edge predictions to binary arrays for all possible edges
    all_possible_edges = set((u, v) for u in node_list for v in node_list if u != v)

    # Create ordered lists of edges for consistent indexing
    edge_list = sorted(all_possible_edges)

    # Create binary arrays for true and predicted labels
    y_true = np.array([1 if edge in original_edges else 0 for edge in edge_list])
    y_pred = np.array([1 if edge in predicted_edges else 0 for edge in edge_list])

    # Calculate weighted kappa with quadratic weights
    edge_structure_qwk = sklearn.metrics.cohen_kappa_score(
        y_true, y_pred, weights="linear"
    )

    return EdgeStructureStats(
        spurious_removed=proportion_spurious_removed,
        true_removed=proportion_true_removed,
        edge_accuracy=edge_accuracy,
        edge_precision=edge_precision,
        edge_recall=edge_recall,
        edge_structure_f1=edge_structure_f1,
        edge_structure_mcc=edge_structure_mcc,
        edge_structure_qwk=edge_structure_qwk,
    )


def train_network(
    train_data,
    test_data,
    model: Net,
    train_pert_dic: OrderedDict,
    test_pert_dic: OrderedDict | None,
    node_dic: dict,
    edge_idx: torch.Tensor,
    train_mask_dic: dict,
    test_mask_dic: dict | None,
    edge_scale: torch.Tensor,
    pert_mask: torch.Tensor,
    node_class_weights_train: dict,
    node_class_weights_test: dict | None,
    config: PruningTestConfig,
    edge_signs: torch.Tensor,
    verbose: bool = True,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Train network using curriculum learning and advanced loss functions"""
    # Setup optimizer and schedulers
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=50,
    )
    # early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=1e-4)
    initial_lr = config.learning_rate * config.warmup_initial_lr_factor
    warmup_scheduler = WarmupScheduler(
        optimizer=opt,
        warmup_steps=config.warmup_steps,
        initial_lr=initial_lr,
        target_lr=config.learning_rate,
    )

    # Train model using the advanced training function
    total_loss, sum_grad, train_epoch_losses, test_epoch_losses = train_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        train_pert_dic=train_pert_dic,
        test_pert_dic=test_pert_dic,
        node_dic=node_dic,
        edge_idx_original=edge_idx,
        train_mask_dic=train_mask_dic,
        test_mask_dic=test_mask_dic,
        edge_scale=edge_scale,
        pert_mask=pert_mask,
        opt=opt,
        max_range=config.max_range,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        early_stopping_enabled=config.early_stopping_enabled,
        early_stopping_patience=config.early_stopping_patience,
        allow_sign_flip=config.allow_sign_flip,
        node_class_weights_train=node_class_weights_train,
        node_class_weights_test=node_class_weights_test,
        save_dir=config.model_save_dir,
        warmup_steps=config.warmup_steps,
        verbose=verbose,
        edge_signs=edge_signs,
        epochs=config.epochs,
    )

    return total_loss, sum_grad, train_epoch_losses, test_epoch_losses


class EvaluationResult:
    def __init__(
        self,
        edge_stats: EdgeStructureStats,
        train_binary_metrics: dict,
        train_nonbinary_metrics: dict,
        test_binary_metrics: dict | None,
        test_nonbinary_metrics: dict | None,
        history: pd.DataFrame,
        spec_stats: dict,
    ):
        self.edge_stats = edge_stats
        self.train_binary_metrics = train_binary_metrics
        self.train_nonbinary_metrics = train_nonbinary_metrics
        self.test_binary_metrics = test_binary_metrics
        self.test_nonbinary_metrics = test_nonbinary_metrics
        self.history = history
        self.spec_stats = spec_stats

    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError("Not implemented")
        """
        Convert evaluation results to a single-row pandas DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing aggregated results.
        """
        data = {}

        # Add edge structure statistics
        for f in fields(self.edge_stats):
            data[f"edge_stats_{f.name}"] = getattr(self.edge_stats, f.name)

        # Add train binary metrics
        for key, value in self.train_binary_metrics.items():
            data[f"train_binary_{key}"] = value

        # Add train non-binary metrics
        for key, value in self.train_nonbinary_metrics.items():
            data[f"train_nonbinary_{key}"] = value

        # Add test binary metrics
        for key, value in self.test_binary_metrics.items():
            data[f"test_binary_{key}"] = value

        # Add test non-binary metrics
        for key, value in self.test_nonbinary_metrics.items():
            data[f"test_nonbinary_{key}"] = value

        # Add specification statistics (handle potential nested structures or specific keys)
        # Assuming spec_stats is relatively flat based on generate_diverse_specifications
        for key, value in self.spec_stats.items():
            # Skip nested dictionaries or complex types if you only want simple stats
            if not isinstance(value, (dict, list, pd.DataFrame)):
                data[f"spec_stats_{key}"] = value
            # Optionally handle specific complex keys if needed, e.g., by serializing or summarizing

        # Create DataFrame from the single row of data
        df = pd.DataFrame([data])

        return df


def get_metrics(
    X,
    y,
    y_no_zero_no_pert_as_expectation,
    W,
    Adjacency_per_experiment,
    G,
    pert_dic_all,
    config,
):
    pred_bound_perturbations = pred_bound_perts(
        X=X,
        y=y,
        W=W,
        Adjacency_per_experiment=Adjacency_per_experiment,
        G=G,
        pert_dic_all=pert_dic_all,
        time_step=config.n_iter,
        min_val=config.min_range,
        max_val=config.max_range,
        extract_exp=True,
    )
    annotation_symbols = {"pert": "•", "exp": "-", "tst": ""}
    annot = create_annotation_matrix(
        base_df=y,
        perturbation_dict=pert_dic_all,
        annotation_symbols=annotation_symbols,
    )
    idx_real = get_real_indices(y)
    filtered_idx_real = get_filtered_indices(annot, idx_real)
    # if not config.generate_base_graph:
    binary_metrics = analyze_predictions(
        y_true=y_no_zero_no_pert_as_expectation.loc[filtered_idx_real],
        y_pred=pred_bound_perturbations.loc[filtered_idx_real],
        save_figs=config.save_confusion_plots,
        save_csv=config.save_binary_metrics_csv,
        path_data=None,
        binary_mode=True,
        pert_dic=pert_dic_all,
    )
    nonbinary_metrics = analyze_predictions(
        y_true=y_no_zero_no_pert_as_expectation.loc[filtered_idx_real],
        y_pred=pred_bound_perturbations.loc[filtered_idx_real],
        save_figs=config.save_confusion_plots,
        save_csv=config.save_binary_metrics_csv,
        path_data=None,
        binary_mode=False,
        pert_dic=pert_dic_all,
    )

    return binary_metrics, nonbinary_metrics


def run_pruning_benchmark(
    config: PruningTestConfig | None = None,
    verbose: bool = True,
    override_out_dir: Path | None = None,
) -> EvaluationResult:
    """
    Run complete pruning benchmark and return evaluation metrics

    Args:
        config: Configuration for the benchmark. If None, uses defaults.

    Returns:
        edge_stats: Final evaluation statistics
        binary_metrics: Binary metrics
        history: DataFrame of metrics over training
    """
    if config is None:
        config = PruningTestConfig()
        print("Using default config")
    local_path = Path(__file__).parent.parent
    out_dir = local_path / Path(config.out_dir)
    if override_out_dir is not None:
        out_dir = override_out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Out dir: {out_dir}")
    # Set all random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.min_edge_weight < config.weight_threshold:
        print(
            f"Min edge weight {config.min_edge_weight} is less than weight threshold {config.weight_threshold}. The specification cannot be met even by a perfect run, as we would remove some edges that still have an effect. This will result in a MCC of 0."
        )
    # Generate synthetic network
    if config.generate_base_graph:
        G_true = generate_base_graph(config, verbose)
        # print(f"Generated base graph with {len(G_true.edges())} edges")
    else:
        if config.base_graph_import is None or not config.base_graph_import.exists():
            raise ValueError(
                f"Base graph import file does not exist: {config.base_graph_import}"
            )
        if config.grow_base_graph and not config.generate_base_graph:
            G_true, _ = grow_network(
                base_graph_path=config.base_graph_import,
                target_nodes=config.n_nodes,
                method=config.growth_method,
                **config.growth_params,
            )
        elif config.grow_base_graph and config.generate_base_graph:
            warnings.warn(
                "Grow base graph and generate base graph are both true. Falling back to generate base graph."
            )
            G_true = generate_base_graph(config, verbose)
        else:
            G_true = json_to_graph(config.base_graph_import)
            check_dummy_nodes(G_true)
            check_no_self_loops(G_true)
            const_dic = get_const_from_json(config.base_graph_import)
        if verbose:
            print(
                f"Imported base graph with {len(G_true.edges())} edges and {len(G_true.nodes())} nodes"
            )
        # Assign random edge weights between min and max to all edges
        for u, v in G_true.edges():
            G_true[u][v]["edge_weight"] = np.random.uniform(
                config.min_edge_weight, config.max_edge_weight
            )
    original_edges = set(G_true.edges())

    # Add spurious edges
    if config.n_spurious_edges > 0:
        G_noisy, spurious_edges = add_spurious_edges(G_true, config)
    else:
        G_noisy = G_true
        spurious_edges = set()

    if verbose:
        print(
            f"Network with {len(original_edges)} true edges and {len(spurious_edges)} spurious edges"
        )

    # Generate specifications
    if config.generate_specifications:
        train_specs, test_specs, spec_stats = generate_diverse_specifications(
            G_true, config, verbose
        )
        train_spec_df = pd.DataFrame.from_dict(train_specs, orient="index")
        train_spec_df.to_csv(
            Path(out_dir, "train_specifications_output.csv"), index=True
        )
        test_spec_df = pd.DataFrame.from_dict(test_specs, orient="index")
        test_spec_df.to_csv(Path(out_dir, "test_specifications_output.csv"), index=True)
        train_dic_small = OrderedDict(sorted(train_specs.items(), key=lambda t: t[0]))
        test_pert_dic = OrderedDict(sorted(test_specs.items(), key=lambda t: t[0]))
        if verbose:
            print(f"Generated {len(train_specs)} train specifications")
            print(f"Generated {len(test_specs)} test specifications")

        def count_node_participation(specs: dict) -> pd.DataFrame:
            perturbed_counts = {}
            measured_counts = {}
            for spec in specs.values():
                for node in spec["pert"]:
                    perturbed_counts[node] = perturbed_counts.get(node, 0) + 1
                for node in spec["exp"]:
                    measured_counts[node] = measured_counts.get(node, 0) + 1
            all_nodes = set(perturbed_counts) | set(measured_counts)
            data = []
            for node in sorted(all_nodes):
                data.append(
                    {
                        "node": node,
                        "perturbed_count": perturbed_counts.get(node, 0),
                        "measured_count": measured_counts.get(node, 0),
                    }
                )
            return pd.DataFrame(data)

        train_node_counts = count_node_participation(train_specs)
        test_node_counts = count_node_participation(test_specs)
        train_node_counts.to_csv(
            Path(out_dir, "train_node_participation.csv"), index=False
        )
        test_node_counts.to_csv(
            Path(out_dir, "test_node_participation.csv"), index=False
        )

    else:
        if config.spec_file is None or not config.spec_file.exists():
            raise ValueError(f"Specification file does not exist: {config.spec_file}")
        if config.generate_base_graph is True:
            raise ValueError("Cannot import specifications for a random network")
        train_dic_small = get_pert_dic(
            file_path=config.spec_file,
            const_dic=const_dic,
            spec_size="non full",
        )
        train_dic_small = filter_spec_invalid_experiments(
            train_dic_small, G_true, aggressive=False, verbose=True
        )
        graph_nodes = set(G_true.nodes())  # Use set for faster lookups
        if config.filter_spec:
            train_dic_small = filter_specification(train_dic_small, graph_nodes)

            spec_stats = {
                "n_specs": len(train_dic_small),
                "mean_perturbations": np.mean(
                    [len(spec["pert"]) for spec in train_dic_small.values()]
                ),
                "mean_outputs": np.mean(
                    [len(spec["exp"]) for spec in train_dic_small.values()]
                ),
                "min_perturbations": np.min(
                    [len(spec["pert"]) for spec in train_dic_small.values()]
                ),
                "max_perturbations": np.max(
                    [len(spec["pert"]) for spec in train_dic_small.values()]
                ),
                "min_outputs": np.min(
                    [len(spec["exp"]) for spec in train_dic_small.values()]
                ),
                "max_outputs": np.max(
                    [len(spec["exp"]) for spec in train_dic_small.values()]
                ),
            }

            if verbose:
                print(
                    f"Imported {spec_stats['n_specs']} specifications with mean {spec_stats['mean_perturbations']} perturbations and {spec_stats['mean_outputs']} outputs"
                )
                print(
                    f"Range of number of perturbations: {spec_stats['min_perturbations']} to {spec_stats['max_perturbations']}"
                )
                print(
                    f"Range of number of outputs: {spec_stats['min_outputs']} to {spec_stats['max_outputs']}"
                )
        # Calculate number of times an expected phenotype node is an output
        n_phenotype_outputs = sum(
            1
            for spec in train_dic_small.values()
            if any(node in spec["exp"] for node in config.expected_phenotype_nodes)
        )
        total_measurements = sum(len(spec["exp"]) for spec in train_dic_small.values())
        n_phenotype_measurements = sum(
            1
            for spec in train_dic_small.values()
            for node in spec["exp"]
            if node
            in set(
                config.expected_phenotype_nodes
            )  # Convert list to set for faster lookups
        )
        phenotype_fraction = (
            n_phenotype_measurements / total_measurements
            if total_measurements > 0
            else 0
        )
        spec_stats.update(
            {
                "phenotype_fraction": phenotype_fraction,
                "n_phenotype_measurements": n_phenotype_measurements,
                "total_measurements": total_measurements,
                "n_phenotype_outputs": n_phenotype_outputs,
            }
        )

        if verbose:
            print(
                f"Phenotype nodes make up {spec_stats['phenotype_fraction']:.1%} ({spec_stats['n_phenotype_measurements']} out of {spec_stats['total_measurements']}) of all measurements"
            )
            print(
                f"Phenotype nodes are measured in {spec_stats['n_phenotype_outputs']} specifications"
            )
            # print(f"pert_dic_small[0:10]: {list(pert_dic_small.items())[0:10]}")
        train_specs, test_specs = split_specifications(train_dic_small, config)
        train_spec_df = pd.DataFrame.from_dict(train_specs, orient="index")
        train_spec_df.to_csv(
            Path(out_dir, "train_specifications_output_nongenerated.csv"), index=True
        )
        test_spec_df = pd.DataFrame.from_dict(test_specs, orient="index")
        test_spec_df.to_csv(
            Path(out_dir, "test_specifications_output_nongenerated.csv"), index=True
        )
        train_dic_small = OrderedDict(sorted(train_specs.items(), key=lambda t: t[0]))
        test_pert_dic = OrderedDict(sorted(test_specs.items(), key=lambda t: t[0]))
    # Prepare network for training
    # G_noisy_no_dummy_1 = G_noisy.copy()
    # G_noisy_no_dummy_2 = G_noisy.copy()
    # G_noisy_no_dummy_3 = G_noisy.copy()
    pert_dic_all = train_dic_small | test_pert_dic
    # G_noisy, _, inh, _ = add_dummy_nodes_and_generate_A_inh(
    #     G_noisy_no_dummy_1,
    #     pert_dic_all,
    #     config.max_range,
    #     config.tf_method,
    # )
    # _, Adjacency_per_experiment_train, _, _ = add_dummy_nodes_and_generate_A_inh(
    #     G_noisy_no_dummy_2, train_dic_small, config.max_range, config.tf_method
    # )
    # _, Adjacency_per_experiment_test, _, _ = add_dummy_nodes_and_generate_A_inh(
    #     G_noisy_no_dummy_3, test_pert_dic, config.max_range, config.tf_method
    # )
    G_noisy, inh, Adjacency_per_experiment_train, Adjacency_per_experiment_test = (
        dummy_setup(
            G_noisy,
            pert_dic_all,
            train_dic_small,
            test_pert_dic,
            config.max_range,
            config.tf_method,
        )
    )

    # Get adjacency matrix
    A_mult = get_adjacency_matrix_mult(G_noisy, method=config.tf_method)

    # Generate training data
    X_train, y_train = get_data_and_update_y(train_dic_small, G_noisy)
    if test_pert_dic:
        X_test, y_test = get_data_and_update_y(test_pert_dic, G_noisy)
    else:
        X_test, y_test = None, None
    _, y_no_zero_no_pert_as_expectation_train = get_data(
        train_dic_small, G_noisy, y_replace_missing_with_zero=False
    )
    if test_pert_dic:
        _, y_no_zero_no_pert_as_expectation_test = get_data(
            test_pert_dic, G_noisy, y_replace_missing_with_zero=False
        )
    else:
        y_no_zero_no_pert_as_expectation_test = None
    node_class_weights_train = calculate_node_class_weights(
        y_no_zero_no_pert_as_expectation_train,
        method=config.class_weight_method,
        min_range=config.min_range,
        max_range=config.max_range,
    )
    if test_pert_dic and y_no_zero_no_pert_as_expectation_test is not None:
        node_class_weights_test = calculate_node_class_weights(
            y_no_zero_no_pert_as_expectation_test,
            method=config.class_weight_method,
            min_range=config.min_range,
            max_range=config.max_range,
        )
    else:
        node_class_weights_test = None
    # Create edge attributes
    pert_list = get_pert_list(pert_dic_all, inh)
    node_dic = make_node_dic(G_noisy)
    pert_idx = make_pert_idx(pert_list, node_dic)
    edge_idx = make_edge_idx(A_mult, pert_idx)
    edge_idx_original = edge_idx.clone()
    # Create PyG data object
    train_data = create_pyg_data_object(X_train, y_train, edge_idx)
    if X_test is not None and y_test is not None:
        test_data = create_pyg_data_object(X_test, y_test, edge_idx)
    else:
        test_data = None

    # Prepare edge scaling and masks
    edge_scale = create_edge_scale(A_mult, pert_idx)
    train_mask_dic = construct_mask_dic(train_dic_small, node_dic, edge_idx)
    test_mask_dic = construct_mask_dic(test_pert_dic, node_dic, edge_idx)
    pert_mask = create_pert_mask(edge_idx, node_dic)

    # Initialize model
    edge_weight = torch.ones(edge_idx.shape[1]) * config.edge_weight_init
    model = Net(
        edge_weight=edge_weight,
        min_val=config.min_range,
        max_val=config.max_range,
        n_iter=config.n_iter,
        max_update=config.max_update,
        round_val=config.round_val,
    )
    edge_signs = extract_edge_signs(G_noisy, edge_idx)
    # Train model]
    total_train_loss, sum_grad, train_epoch_losses, test_epoch_losses = train_network(
        train_data=train_data,
        test_data=test_data,
        model=model,
        train_pert_dic=train_dic_small,
        test_pert_dic=test_pert_dic,
        node_dic=node_dic,
        edge_idx=edge_idx,
        train_mask_dic=train_mask_dic,
        test_mask_dic=test_mask_dic,
        edge_scale=edge_scale,
        pert_mask=pert_mask,
        node_class_weights_train=node_class_weights_train,
        node_class_weights_test=node_class_weights_test,
        config=config,
        verbose=verbose,
        edge_signs=edge_signs,
    )

    plot_loss_vs_validation_loss(
        epoch_losses=train_epoch_losses,
        test_losses=test_epoch_losses,
        out_dir=out_dir,
    )

    # Evaluate results
    edge_stats = evaluate_network_edges(
        model=model,
        edge_idx=edge_idx,
        G=G_noisy,
        spurious_edges=spurious_edges,
        original_edges=original_edges,
        config=config,
        verbose=verbose,
    )

    # Create history DataFrame
    history = pd.DataFrame(
        {
            "epoch": np.arange(len(total_train_loss)) // len(train_dic_small),
            "loss": total_train_loss,
            "gradient": sum_grad,
        }
    )

    # Classify match to specification
    W = get_edge_weight_matrix(
        model=model,
        edge_idx_original=edge_idx_original,
        G=G_noisy,
        remove_dummy_and_self_loops=True,
    )

    binary_metrics_train, nonbinary_metrics_train = get_metrics(
        X=X_train,
        y=y_train,
        y_no_zero_no_pert_as_expectation=y_no_zero_no_pert_as_expectation_train,
        W=W,
        Adjacency_per_experiment=Adjacency_per_experiment_train,
        G=G_noisy,
        pert_dic_all=train_dic_small,
        config=config,
    )
    if X_test is not None and y_test is not None:
        binary_metrics_test, nonbinary_metrics_test = get_metrics(
            X=X_test,
            y=y_test,
            y_no_zero_no_pert_as_expectation=y_no_zero_no_pert_as_expectation_test,
            W=W,
            Adjacency_per_experiment=Adjacency_per_experiment_test,
            G=G_noisy,
            pert_dic_all=test_pert_dic,
            config=config,
        )
    else:
        binary_metrics_test, nonbinary_metrics_test = None, None

    return EvaluationResult(
        edge_stats,
        binary_metrics_train,
        nonbinary_metrics_train,
        binary_metrics_test,
        nonbinary_metrics_test,
        history,
        spec_stats,
    )


def compare_configs(
    config1_path: str | Path,
    config2_path: str | Path,
    name1: str = "config1",
    name2: str = "config2",
) -> None:
    """Compare two TOML configuration files and print differences.

    Args:
        config1_path: Path to first TOML file
        config2_path: Path to second TOML file
        name1: Name for first config in output
        name2: Name for second config in output
    """

    def flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
        """Flatten nested dictionary with dot notation."""
        items: list = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Load configs
    config1 = toml.load(config1_path)
    config2 = toml.load(config2_path)

    # Flatten both configs
    flat_config1 = flatten_dict(config1)
    flat_config2 = flatten_dict(config2)

    # Find all unique keys
    all_keys = sorted(set(flat_config1.keys()) | set(flat_config2.keys()))

    # Print comparison
    print("\nConfiguration Comparison:")
    print("=" * 80)
    print(f"{'Parameter':<40} {name1:<20} {name2:<20}")
    print("-" * 80)

    # First print matching keys
    print("\nMatching Values:")
    for key in all_keys:
        if key in flat_config1 and key in flat_config2:
            if flat_config1[key] == flat_config2[key]:
                print(
                    f"{key:<40} {str(flat_config1[key]):<20} {str(flat_config2[key]):<20}"
                )

    # Then print differing keys
    print("\nDiffering Values:")
    for key in all_keys:
        if key in flat_config1 and key in flat_config2:
            if flat_config1[key] != flat_config2[key]:
                print(
                    f"{key:<40} {str(flat_config1[key]):<20} {str(flat_config2[key]):<20}"
                )

    # Print unique keys for config1
    print(f"\nUnique to {name1}:")
    for key in all_keys:
        if key in flat_config1 and key not in flat_config2:
            print(f"{key:<40} {str(flat_config1[key]):<20}")

    # Print unique keys for config2
    print(f"\nUnique to {name2}:")
    for key in all_keys:
        if key not in flat_config1 and key in flat_config2:
            print(f"{key:<40} {str(flat_config2[key]):<20}")
    print("=" * 80)


# @dataclass
# class ParameterRange:
#     """Defines a range of parameter values to sweep over."""

#     name: str
#     values: list[float | int | str | bool] | None
#     param_path: list[
#         str
#     ]  # Path to parameter in config, e.g. ['graph_generation', 'n_nodes']
#     description: str = ""

#     def apply_to_config(self, config: PruningTestConfig, value: Any) -> None:
#         """Apply a value directly to the config attribute.

#         Args:
#             config: PruningTestConfig instance to modify
#             value: Value to set for the parameter
#         """
#         setattr(config, self.param_path[-1], value)


# @dataclass
# class ParameterCombination:
#     """Defines a specific combination of parameters for an experiment."""

#     values: dict[str, Any]
#     description: str = ""
#     metadata: dict[str, Any] = field(default_factory=dict)

#     def apply_to_config(self, config: PruningTestConfig) -> None:
#         """Apply all parameter values to the config.

#         Args:
#             config: PruningTestConfig instance to modify
#         """
#         for param_name, value in self.values.items():
#             if not hasattr(config, param_name):
#                 raise ValueError(f"Config has no parameter named '{param_name}'")
#             setattr(config, param_name, value)

#     @classmethod
#     def from_dict(
#         cls,
#         values: dict[str, Any],
#         description: str = "",
#         metadata: dict[str, Any] | None = None,
#     ) -> "ParameterCombination":
#         """Create a ParameterCombination from a dictionary of values.

#         Args:
#             values: Dictionary of parameter names and their values
#             description: Optional description of this parameter combination
#             metadata: Optional metadata dictionary

#         Returns:
#             New ParameterCombination instance
#         """
#         return cls(values=values, description=description, metadata=metadata or {})

#     def __str__(self) -> str:
#         """Return a string representation of the parameter combination."""
#         params = ", ".join(f"{k}={v}" for k, v in self.values.items())
#         return f"ParameterCombination({params})"


# @dataclass
# class ExperimentResult:
#     """Stores results from a single experiment run."""

#     config_values: dict[str, Any]
#     edge_stats: EdgeStructureStats
#     train_binary_metrics: dict[str, float]
#     train_nonbinary_metrics: dict[str, float]
#     test_binary_metrics: dict[str, float]
#     test_nonbinary_metrics: dict[str, float]
#     metadata: dict[str, Any] = field(default_factory=dict)


# class ExperimentSweep:
#     """Manages parameter sweeps across multiple experiments."""

#     def __init__(
#         self,
#         base_config: PruningTestConfig,
#         parameter_combinations: list[dict[str, Any]] | None = None,
#         parameter_ranges: list[ParameterRange] | None = None,
#         n_repeats: int = 3,
#     ):
#         """
#         Initialize experiment sweep.

#         Args:
#             base_config: Base configuration to modify
#             parameter_combinations: List of parameter dictionaries to use directly
#             parameter_ranges: List of parameters to sweep over (used only if parameter_combinations is None)
#             n_repeats: Number of times to repeat each experiment
#         """
#         self.base_config = base_config
#         self.parameter_ranges = parameter_ranges
#         self.parameter_combinations = parameter_combinations
#         self.n_repeats = n_repeats
#         self.results: list[ExperimentResult] = []

#         # Use seed from base_config
#         if self.base_config.seed is None:
#             raise ValueError("Base config must have a seed")

#         if self.base_config.seed is not None:
#             np.random.seed(self.base_config.seed)

#         # Generate parameter combinations if not provided directly
#         if self.parameter_combinations is None and self.parameter_ranges is not None:
#             self._generate_default_combinations()
#         elif self.parameter_combinations is None and self.parameter_ranges is None:
#             raise ValueError(
#                 "Either parameter_combinations or parameter_ranges must be provided"
#             )

#     def _generate_default_combinations(self) -> None:
#         """Generate parameter combinations using the old product-based approach"""
#         from itertools import product

#         param_values = [range.values for range in self.parameter_ranges]  # type: ignore
#         self.parameter_combinations = []

#         for values in product(*param_values):  # type: ignore
#             combination = {}
#             for param_range, value in zip(self.parameter_ranges, values):  # type: ignore
#                 # Preserve the original type by checking against the first value in the range
#                 if param_range.values and len(param_range.values) > 0:
#                     first_value = param_range.values[0]
#                     # Only convert if types don't match and target type is int
#                     if isinstance(first_value, int) and not isinstance(value, int):
#                         value = int(value)
#                 combination[param_range.name] = value
#             self.parameter_combinations.append(combination)

#     def run(
#         self,
#         experiment_name: str | None = None,
#         permissive: bool = False,
#         verbose=False,
#     ) -> None:
#         """Run all experiments."""
#         if self.parameter_combinations is None:
#             raise ValueError("No parameter combinations to run")

#         total_runs = len(self.parameter_combinations) * self.n_repeats

#         with tqdm(
#             total=total_runs, desc=f"Running {experiment_name} experiments"
#         ) as pbar:
#             for param_dict in self.parameter_combinations:
#                 for repeat in range(self.n_repeats):
#                     # Create new config instance for this run
#                     config = self.base_config.copy()

#                     # Create and apply parameter combination
#                     param_combo = ParameterCombination(values=param_dict)
#                     param_combo.apply_to_config(config)

#                     # Set seed for this run based on base_config seed
#                     run_seed = config.seed + repeat if config.seed is not None else None
#                     if run_seed is not None:
#                         config.seed = run_seed

#                     # Run experiment
#                     try:
#                         evaluation_result = run_pruning_benchmark(config, verbose)

#                         # Store results
#                         result = ExperimentResult(
#                             config_values=param_dict,
#                             edge_stats=evaluation_result.edge_stats,
#                             train_binary_metrics=evaluation_result.train_binary_metrics,
#                             train_nonbinary_metrics=evaluation_result.train_nonbinary_metrics,
#                             test_binary_metrics=evaluation_result.test_binary_metrics,
#                             test_nonbinary_metrics=evaluation_result.test_nonbinary_metrics,
#                             metadata={
#                                 "repeat": repeat,
#                                 "experiment_name": experiment_name,
#                                 "run_seed": run_seed,
#                             },
#                         )
#                         self.results.append(result)

#                     except Exception as e:
#                         if permissive:
#                             print(
#                                 f"Error in run with params {param_dict}, repeat {repeat}: {str(e)}"
#                             )
#                         else:
#                             raise e
#                         continue

#                     pbar.update(1)

#     def get_results_df(self) -> pd.DataFrame:
#         """Convert results to a DataFrame."""
#         records = []

#         for result in self.results:
#             record = {
#                 # Parameter values
#                 **result.config_values,
#                 # Edge structure metrics
#                 "spurious_removed": result.edge_stats.spurious_removed,
#                 "true_removed": result.edge_stats.true_removed,
#                 "edge_accuracy": result.edge_stats.edge_accuracy,
#                 "edge_structure_f1": result.edge_stats.edge_structure_f1,
#                 "edge_structure_mcc": result.edge_stats.edge_structure_mcc,
#                 "edge_structure_qwk": result.edge_stats.edge_structure_qwk,
#                 # Binary prediction metrics
#                 **{f"binary_{k}": v for k, v in result.train_binary_metrics.items()},
#                 # Non-binary prediction metrics
#                 **{
#                     f"nonbinary_{k}": v
#                     for k, v in result.train_nonbinary_metrics.items()
#                 },
#                 **{f"binary_{k}": v for k, v in result.test_binary_metrics.items()},
#                 **{
#                     f"nonbinary_{k}": v
#                     for k, v in result.test_nonbinary_metrics.items()
#                 },
#                 # Metadata
#                 **result.metadata,
#             }
#             records.append(record)

#         return pd.DataFrame(records)

#     def save_results(self, path: str | Path) -> None:
#         """Save results to CSV."""
#         df = self.get_results_df()
#         df.to_csv(path, index=False)

#     @classmethod
#     def load_results(cls, path: str | Path) -> pd.DataFrame:
#         """Load results from CSV."""
#         return pd.read_csv(path)


# def plot_sweep_results(
#     df: pd.DataFrame,
#     x_col: str,
#     y_cols: list[str],
#     experiment_name: str,
#     output_dir: str | Path,
#     param_ranges: list[ParameterRange],
#     facet: bool = False,
#     clean: bool = False,
# ) -> None:
#     """Create publication quality square plot of sweep results with legend inside the plot.

#     Args:
#         df: DataFrame containing results
#         x_col: Column name for x-axis
#         y_cols: List of column names for y-axis metrics
#         experiment_name: Name of the experiment
#         output_dir: Directory to save plot
#         param_ranges: List of parameter ranges containing descriptions
#         facet: If True, create vertical facets for each metric
#         clean: If True, removes 'nonbinary' from metric names and capitalizes F1, MCC, QWK
#     """
#     # Set style for publication quality
#     plt.style.use("seaborn-v0_8-paper")

#     if facet:
#         # Create vertical facets
#         fig, axes = plt.subplots(
#             len(y_cols), 1, figsize=(8, 4 * len(y_cols)), dpi=300, sharex=True
#         )
#         if len(y_cols) == 1:
#             axes = [axes]
#     else:
#         # Create single square figure
#         fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
#         axes = [ax] * len(y_cols)

#     # Use a professional color palette
#     colors = sns.color_palette("colorblind", n_colors=len(y_cols) + 1)

#     # Get x-axis description
#     x_axis_description = next(
#         (param.description for param in param_ranges if param.name == x_col),
#         x_col.replace("_", " ").title(),
#     )

#     # Plot each metric with improved styling
#     for i, (y_col, ax) in enumerate(zip(y_cols, axes)):
#         # Always use the metric-specific color regardless of faceting
#         color = colors[i]
#         mean_df = df.groupby(x_col)[y_col].mean().reset_index()
#         std_df = df.groupby(x_col)[y_col].std().reset_index()

#         # Plot individual points with low alpha
#         sns.scatterplot(
#             data=df, x=x_col, y=y_col, color=color, alpha=0.15, s=30, ax=ax, label=None
#         )

#         # Get simplified label name

#         if "nonbinary_f1" in y_col.lower():
#             label = "F1"
#         elif "nonbinary_mcc" in y_col.lower():
#             label = "MCC"
#         elif "nonbinary_qwk" in y_col.lower():
#             label = "QWK"
#         else:
#             label = y_col.replace("_", " ").title()

#         # Plot mean line with confidence band
#         sns.lineplot(
#             data=mean_df,
#             x=x_col,
#             y=y_col,
#             color=color,
#             label=label if not facet else None,
#             linewidth=2.5,
#             ax=ax,
#         )

#         # Add error bands
#         ax.fill_between(
#             mean_df[x_col],
#             mean_df[y_col] - std_df[y_col],
#             mean_df[y_col] + std_df[y_col],
#             color=color,
#             alpha=0.2,
#         )

#         # Add mean points
#         ax.scatter(
#             mean_df[x_col], mean_df[y_col], color=color, s=100, zorder=5, label=None
#         )

#         # Improve grid and spines
#         ax.grid(True, which="major", color="gray", linestyle="--", alpha=0.3)
#         ax.grid(True, which="minor", color="gray", linestyle=":", alpha=0.2)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.spines["left"].set_linewidth(1.5)
#         ax.spines["bottom"].set_linewidth(1.5)

#         # Set y-axis limits with some padding
#         ax.set_ylim(-0.05, 1.05)

#         if facet:
#             # Add y-axis label for each facet
#             ax.set_ylabel(label, labelpad=10, fontsize=12)

#             # Only show x-label on bottom facet
#             if i == len(y_cols) - 1:
#                 ax.set_xlabel(x_axis_description, labelpad=10, fontsize=12)
#             else:
#                 ax.set_xlabel("")

#         ax.tick_params(axis="both", which="major", labelsize=10)

#     if not facet:
#         # Make plot square by setting aspect ratio
#         ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

#         # Add labels for single plot
#         axes[0].set_xlabel(x_axis_description, labelpad=10, fontsize=12)
#         axes[0].set_ylabel("Metric Value", labelpad=10, fontsize=12)

#         # Place legend inside the plot in the upper right corner
#         legend = axes[0].legend(
#             loc="best",
#             frameon=True,
#             framealpha=0.9,
#             edgecolor="none",
#             fontsize=10,
#             ncol=1,
#             bbox_to_anchor=(0.98, 0.98),
#         )
#         legend.get_frame().set_facecolor("white")

#     # Add overall title
#     fig.suptitle(
#         f"{experiment_name}: {x_axis_description}",
#         y=0.98 if not facet else 0.99,
#         fontsize=14,
#         fontweight="bold",
#     )

#     # Adjust layout and save
#     plt.tight_layout()
#     if not facet:
#         plt.subplots_adjust(top=0.95)
#     else:
#         plt.subplots_adjust(top=0.95, hspace=0.3)

#     # Save with high quality settings
#     suffix = "_facet" if facet else ""
#     plt.savefig(
#         Path(output_dir, f"{experiment_name.lower().replace(' ', '_')}{suffix}.png"),
#         bbox_inches="tight",
#         dpi=300,
#         facecolor="white",
#         edgecolor="none",
#     )
#     plt.close()
