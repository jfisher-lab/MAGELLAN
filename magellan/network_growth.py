from pathlib import Path

import networkx as nx
import numpy as np
from tqdm.autonotebook import tqdm

from magellan.graph import Graph
from magellan.json_io import json_to_graph


def grow_network_preferential_attachment(
    base_graph: Graph | nx.DiGraph,
    target_nodes: int,
    p_forward: float = 0.3,
    p_backward: float = 0.1,
    p_inhibition: float = 0.3,
    min_edge_weight: float = 0.1,
    max_edge_weight: float = 1.0,
    preserve_original: bool = True,
    seed: int = 42,
) -> tuple[Graph, set]:
    """
    Grow a network using preferential attachment to reach the target number of nodes.

    Args:
        base_graph: The original graph to expand
        target_nodes: Target number of nodes after expansion
        p_forward: Probability for forward connections (new -> existing)
        p_backward: Probability for backward connections (existing -> new)
        p_inhibition: Fraction of new edges that should be inhibitory
        min_edge_weight: Minimum edge weight
        max_edge_weight: Maximum edge weight
        preserve_original: If True, original edges will be preserved
        seed: Random seed for reproducibility

    Returns:
        tuple containing:
        - Expanded graph
        - Set of new edges added during expansion
    """
    np.random.seed(seed)

    # Create a new Graph object for the expanded network
    G_expanded = Graph(remove_cycle=True, remove_sign=True, remove_neither=True)

    # Add original nodes and edges to expanded graph
    original_nodes = list(base_graph.nodes())
    n_original = len(original_nodes)

    if target_nodes <= n_original:
        raise ValueError(
            f"Target nodes ({target_nodes}) must be greater than original nodes ({n_original})"
        )

    # Add original nodes and edges
    G_expanded.add_nodes_from(base_graph.nodes(data=True))
    if preserve_original:
        G_expanded.add_edges_from(base_graph.edges(data=True))

    # Track new edges
    new_edges = set()

    # Calculate how many nodes to add
    n_new_nodes = target_nodes - n_original

    # Generate new node names
    existing_names = set(original_nodes)
    new_node_prefix = "new_node_"
    new_nodes = []

    for i in range(n_new_nodes):
        node_name = f"{new_node_prefix}{i}"
        # Ensure no name conflicts
        while node_name in existing_names:
            node_name = f"{new_node_prefix}{i}_{np.random.randint(1000)}"
        new_nodes.append(node_name)
        existing_names.add(node_name)
        G_expanded.add_node(node_name)

    # Calculate degree distributions for preferential attachment
    if len(base_graph.edges()) > 0:  # Only if there are edges in original graph
        in_degrees = dict(base_graph.in_degree())
        out_degrees = dict(base_graph.out_degree())

        # Add 1 to avoid zero probability
        in_degree_weights = {
            node: in_degrees.get(node, 0) + 1 for node in original_nodes
        }
        out_degree_weights = {
            node: out_degrees.get(node, 0) + 1 for node in original_nodes
        }
    else:
        # If no edges, use uniform weights
        in_degree_weights = {node: 1 for node in original_nodes}
        out_degree_weights = {node: 1 for node in original_nodes}

    # Connect new nodes to the network using preferential attachment
    for new_node in tqdm(new_nodes, desc="Adding nodes via preferential attachment"):
        # Forward connections: new node -> existing nodes
        for existing_node in original_nodes:
            if np.random.random() < p_forward * (
                in_degree_weights[existing_node] / sum(in_degree_weights.values())
            ):
                weight = np.random.uniform(min_edge_weight, max_edge_weight)
                sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"
                G_expanded.add_edge(
                    new_node,
                    existing_node,
                    edge_weight=weight,
                    sign=sign,
                    n_references=weight,
                )
                new_edges.add((new_node, existing_node))

        # Backward connections: existing nodes -> new node
        for existing_node in original_nodes:
            if np.random.random() < p_backward * (
                out_degree_weights[existing_node] / sum(out_degree_weights.values())
            ):
                weight = np.random.uniform(min_edge_weight, max_edge_weight)
                sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"
                G_expanded.add_edge(
                    existing_node,
                    new_node,
                    edge_weight=weight,
                    sign=sign,
                    n_references=weight,
                )
                new_edges.add((existing_node, new_node))

    # Add connections between new nodes
    for i, new_node_i in enumerate(new_nodes):
        for new_node_j in new_nodes[i + 1 :]:  # Avoid self-loops and duplicate edges
            if (
                np.random.random() < p_forward / 2
            ):  # Lower probability for new-new connections
                weight = np.random.uniform(min_edge_weight, max_edge_weight)
                sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"
                G_expanded.add_edge(
                    new_node_i,
                    new_node_j,
                    edge_weight=weight,
                    sign=sign,
                    n_references=weight,
                )
                new_edges.add((new_node_i, new_node_j))

    return G_expanded, new_edges


def grow_network_duplication_divergence(
    base_graph: Graph | nx.DiGraph,
    target_nodes: int,
    divergence_prob: float = 0.3,
    edge_deletion_prob: float = 0.4,
    p_inhibition: float = 0.3,
    min_edge_weight: float = 0.1,
    max_edge_weight: float = 1.0,
    preserve_original: bool = True,
    seed: int = 42,
) -> tuple[Graph, set]:
    """
    Grow a network using duplication-divergence model to reach the target number of nodes.

    In this model, new nodes are created by duplicating existing nodes and their connections,
    then some connections are modified or removed to simulate divergence.

    Args:
        base_graph: The original graph to expand
        target_nodes: Target number of nodes after expansion
        divergence_prob: Probability of modifying a connection during divergence
        edge_deletion_prob: Probability of deleting an edge during divergence
        p_inhibition: Fraction of new edges that should be inhibitory
        min_edge_weight: Minimum edge weight
        max_edge_weight: Maximum edge weight
        preserve_original: If True, original edges will be preserved
        seed: Random seed for reproducibility

    Returns:
        tuple containing:
        - Expanded graph
        - Set of new edges added during expansion
    """
    np.random.seed(seed)

    # Create a new Graph object for the expanded network
    G_expanded = Graph(remove_cycle=True, remove_sign=True, remove_neither=True)

    # Add original nodes and edges to expanded graph
    original_nodes = list(base_graph.nodes())
    n_original = len(original_nodes)

    if target_nodes <= n_original:
        raise ValueError(
            f"Target nodes ({target_nodes}) must be greater than original nodes ({n_original})"
        )

    # Add original nodes and edges
    G_expanded.add_nodes_from(base_graph.nodes(data=True))
    if preserve_original:
        G_expanded.add_edges_from(base_graph.edges(data=True))

    # Track new edges
    new_edges = set()

    # Calculate how many nodes to add
    n_new_nodes = target_nodes - n_original

    # Generate new node names
    existing_names = set(original_nodes)
    new_nodes = []

    # Duplication-divergence process
    for i in tqdm(range(n_new_nodes), desc="Duplicating nodes"):
        # Select a random node to duplicate
        template_node = np.random.choice(original_nodes + new_nodes[:i])

        # Create new node name
        new_node_name = f"dup_{template_node}_{i}"
        while new_node_name in existing_names:
            new_node_name = f"dup_{template_node}_{i}_{np.random.randint(1000)}"

        new_nodes.append(new_node_name)
        existing_names.add(new_node_name)
        G_expanded.add_node(new_node_name)

        # Duplicate outgoing edges with potential divergence
        for successor in G_expanded.successors(template_node):
            # Skip self-loops
            if successor == new_node_name:
                continue

            if np.random.random() > divergence_prob:
                # Copy edge with same properties
                edge_data = base_graph.get_edge_data(template_node, successor)
                if edge_data is None:
                    edge_data = G_expanded.get_edge_data(template_node, successor)

                weight = edge_data.get(
                    "edge_weight", np.random.uniform(min_edge_weight, max_edge_weight)
                )
                sign = edge_data.get("sign", "Activator")

                G_expanded.add_edge(
                    new_node_name,
                    successor,
                    edge_weight=weight,
                    sign=sign,
                    n_references=weight,
                )
                new_edges.add((new_node_name, successor))
            elif np.random.random() > edge_deletion_prob:
                # Create diverged edge with new properties
                weight = np.random.uniform(min_edge_weight, max_edge_weight)
                sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"

                G_expanded.add_edge(
                    new_node_name,
                    successor,
                    edge_weight=weight,
                    sign=sign,
                    n_references=weight,
                )
                new_edges.add((new_node_name, successor))

        # Duplicate incoming edges with potential divergence
        for predecessor in G_expanded.predecessors(template_node):
            # Skip self-loops
            if predecessor == new_node_name:
                continue

            if np.random.random() > divergence_prob:
                # Copy edge with same properties
                edge_data = base_graph.get_edge_data(predecessor, template_node)
                if edge_data is None:
                    edge_data = G_expanded.get_edge_data(predecessor, template_node)

                weight = edge_data.get(
                    "edge_weight", np.random.uniform(min_edge_weight, max_edge_weight)
                )
                sign = edge_data.get("sign", "Activator")

                G_expanded.add_edge(
                    predecessor,
                    new_node_name,
                    edge_weight=weight,
                    sign=sign,
                    n_references=weight,
                )
                new_edges.add((predecessor, new_node_name))
            elif np.random.random() > edge_deletion_prob:
                # Create diverged edge with new properties
                weight = np.random.uniform(min_edge_weight, max_edge_weight)
                sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"

                G_expanded.add_edge(
                    predecessor,
                    new_node_name,
                    edge_weight=weight,
                    sign=sign,
                    n_references=weight,
                )
                new_edges.add((predecessor, new_node_name))

    return G_expanded, new_edges


def analyze_network_structure(G: Graph | nx.DiGraph) -> dict:
    """
    Analyze network structure and return key metrics.

    Args:
        G: The graph to analyze

    Returns:
        dict of network metrics
    """
    metrics = {}

    # Basic statistics
    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)

    # Degree statistics
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    metrics["mean_in_degree"] = np.mean(in_degrees) if in_degrees else 0
    metrics["mean_out_degree"] = np.mean(out_degrees) if out_degrees else 0
    metrics["max_in_degree"] = max(in_degrees) if in_degrees else 0
    metrics["max_out_degree"] = max(out_degrees) if out_degrees else 0

    # Edge attributes
    inhibitory_count = 0
    edge_weights = []

    for _, _, data in G.edges(data=True):
        if "sign" in data and data["sign"] == "Inhibitor":
            inhibitory_count += 1
        if "edge_weight" in data:
            edge_weights.append(data["edge_weight"])

    metrics["inhibitory_fraction"] = (
        inhibitory_count / G.number_of_edges() if G.number_of_edges() > 0 else 0
    )
    metrics["mean_edge_weight"] = np.mean(edge_weights) if edge_weights else 0

    # Try to compute some additional metrics for connected graphs
    try:
        # Convert to undirected for some metrics
        undirected = G.to_undirected()
        connected_components = list(nx.connected_components(undirected))
        metrics["connected_components"] = len(connected_components)

        largest_cc = max(connected_components, key=len)
        metrics["largest_cc_size"] = len(largest_cc)
        metrics["largest_cc_fraction"] = len(largest_cc) / G.number_of_nodes()

        # Calculate clustering on largest component to avoid errors
        largest_cc_subgraph = undirected.subgraph(largest_cc)
        metrics["avg_clustering"] = nx.average_clustering(largest_cc_subgraph)
    except Exception as e:
        metrics["connected_components"] = "Error"
        metrics["largest_cc_size"] = "Error"
        metrics["largest_cc_fraction"] = "Error"
        metrics["avg_clustering"] = "Error"
        print(f"Error computing connectivity metrics: {e}")

    return metrics


def compare_networks(
    original_G: Graph | nx.DiGraph, expanded_G: Graph | nx.DiGraph
) -> dict:
    """
    Compare original and expanded networks and return comparison metrics.

    Args:
        original_G: The original graph
        expanded_G: The expanded graph

    Returns:
        dict of comparison metrics
    """
    original_metrics = analyze_network_structure(original_G)
    expanded_metrics = analyze_network_structure(expanded_G)

    comparison = {}

    # Calculate ratios and differences
    for key in original_metrics:
        if isinstance(original_metrics[key], (int, float)) and isinstance(
            expanded_metrics[key], (int, float)
        ):
            comparison[f"{key}_ratio"] = (
                expanded_metrics[key] / original_metrics[key]
                if original_metrics[key] != 0
                else float("inf")
            )
            comparison[f"{key}_diff"] = expanded_metrics[key] - original_metrics[key]

    # Add raw metrics
    comparison["original"] = original_metrics
    comparison["expanded"] = expanded_metrics

    return comparison


def grow_network(
    base_graph_path: str | Path,
    target_nodes: int,
    method: str = "preferential_attachment",
    output_path: str | Path | None = None,
    **kwargs,
) -> tuple[Graph, dict]:
    """
    Main function to grow a network using selected method.

    Args:
        base_graph_path: Path to the base graph in JSON format
        target_nodes: Target number of nodes after expansion
        method: Growth method ('preferential_attachment' or 'duplication_divergence')
        output_path: Optional path to save the expanded graph
        **kwargs: Additional parameters for the growth method

    Returns:
        tuple containing:
        - Expanded graph
        - Comparison metrics dictionary
    """
    # Load base graph
    base_graph = json_to_graph(base_graph_path)

    # Select growth method
    if method == "preferential_attachment":
        expanded_graph, new_edges = grow_network_preferential_attachment(
            base_graph, target_nodes, **kwargs
        )
    elif method == "duplication_divergence":
        expanded_graph, new_edges = grow_network_duplication_divergence(
            base_graph, target_nodes, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'preferential_attachment' or 'duplication_divergence'"
        )

    # Analyze and compare networks
    comparison = compare_networks(base_graph, expanded_graph)

    # Save expanded graph if requested
    if output_path:
        # Save graph using networkx write methods
        nx.write_gml(expanded_graph, output_path)
    return expanded_graph, comparison
