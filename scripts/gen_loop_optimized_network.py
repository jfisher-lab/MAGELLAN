"""
Generate synthetic networks optimized for loops/cycles to test the effect of network topology 
on training iteration convergence behavior.

This script creates networks with intentional loop structures of various types:
- Ring cycles of specified lengths
- Nested/overlapping loops  
- Chains of connected loops

The goal is to test how loop characteristics affect the number of iterations needed
for GNN training convergence.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
from tqdm import tqdm

from magellan.graph import Graph
from magellan.json_io import gen_json, json_to_graph


def create_ring_cycle(
    cycle_length: int, 
    node_prefix: str = "cycle", 
    min_edge_weight: float = 0.1,
    max_edge_weight: float = 1.0,
    p_inhibition: float = 0.3,
    seed: int = 42
) -> Graph:
    """
    Create a simple ring cycle of specified length.
    
    Args:
        cycle_length: Number of nodes in the cycle
        node_prefix: Prefix for node names
        min_edge_weight: Minimum edge weight
        max_edge_weight: Maximum edge weight  
        p_inhibition: Probability of inhibitory edges
        seed: Random seed
        
    Returns:
        Graph containing a single ring cycle
    """
    np.random.seed(seed)
    
    # Create graph without cycle removal
    G = Graph(remove_cycle=False, remove_sign=True, remove_neither=True)
    
    # Create nodes
    nodes = [f"{node_prefix}_{i}" for i in range(cycle_length)]
    G.add_nodes_from(nodes)
    
    # Create cycle edges: 0->1->2->...->n-1->0
    for i in range(cycle_length):
        current_node = nodes[i]
        next_node = nodes[(i + 1) % cycle_length]
        
        weight = np.random.uniform(min_edge_weight, max_edge_weight)
        sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"
        
        G.add_edge(
            current_node,
            next_node,
            edge_weight=weight,
            sign=sign,
            n_references=weight,
        )
    
    return G


def create_nested_loops(
    cycle_lengths: List[int],
    overlap_nodes: int = 1,
    node_prefix: str = "nested",
    min_edge_weight: float = 0.1,
    max_edge_weight: float = 1.0,
    p_inhibition: float = 0.3,
    seed: int = 42
) -> Graph:
    """
    Create nested/overlapping loops that share some nodes.
    
    Args:
        cycle_lengths: List of cycle lengths to create
        overlap_nodes: Number of nodes shared between cycles
        node_prefix: Prefix for node names
        min_edge_weight: Minimum edge weight
        max_edge_weight: Maximum edge weight
        p_inhibition: Probability of inhibitory edges
        seed: Random seed
        
    Returns:
        Graph containing nested loops
    """
    np.random.seed(seed)
    
    G = Graph(remove_cycle=False, remove_sign=True, remove_neither=True)
    
    all_nodes = set()
    shared_nodes = []
    
    for cycle_idx, cycle_length in enumerate(cycle_lengths):
        cycle_nodes = []
        
        # Use shared nodes if this isn't the first cycle
        if cycle_idx > 0 and len(shared_nodes) >= overlap_nodes:
            # Take some shared nodes
            cycle_nodes.extend(shared_nodes[:overlap_nodes])
        
        # Add new nodes to complete the cycle
        nodes_needed = cycle_length - len(cycle_nodes)
        new_nodes = [f"{node_prefix}_{cycle_idx}_{i}" for i in range(nodes_needed)]
        
        # Ensure unique node names
        node_counter = 0
        while any(node in all_nodes for node in new_nodes):
            new_nodes = [f"{node_prefix}_{cycle_idx}_{i}_{node_counter}" 
                        for i in range(nodes_needed)]
            node_counter += 1
            
        cycle_nodes.extend(new_nodes)
        all_nodes.update(cycle_nodes)
        
        # Update shared nodes for next cycle (last few nodes of current cycle)
        shared_nodes = cycle_nodes[-overlap_nodes:] if overlap_nodes > 0 else []
        
        # Add nodes to graph
        G.add_nodes_from(cycle_nodes)
        
        # Create cycle edges
        for i in range(cycle_length):
            current_node = cycle_nodes[i]
            next_node = cycle_nodes[(i + 1) % cycle_length]
            
            weight = np.random.uniform(min_edge_weight, max_edge_weight)
            sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"
            
            G.add_edge(
                current_node,
                next_node,
                edge_weight=weight,
                sign=sign,
                n_references=weight,
            )
    
    return G


def create_loop_chain(
    cycle_lengths: List[int],
    connection_nodes: int = 1,
    node_prefix: str = "chain",
    min_edge_weight: float = 0.1,
    max_edge_weight: float = 1.0,
    p_inhibition: float = 0.3,
    seed: int = 42
) -> Graph:
    """
    Create a chain of loops connected by bridge edges.
    
    Args:
        cycle_lengths: List of cycle lengths to chain together
        connection_nodes: Number of nodes to connect between cycles
        node_prefix: Prefix for node names
        min_edge_weight: Minimum edge weight
        max_edge_weight: Maximum edge weight
        p_inhibition: Probability of inhibitory edges
        seed: Random seed
        
    Returns:
        Graph containing chained loops
    """
    np.random.seed(seed)
    
    G = Graph(remove_cycle=False, remove_sign=True, remove_neither=True)
    
    all_cycle_nodes = []
    
    # Create individual cycles
    for cycle_idx, cycle_length in enumerate(cycle_lengths):
        cycle_nodes = [f"{node_prefix}_c{cycle_idx}_{i}" for i in range(cycle_length)]
        all_cycle_nodes.append(cycle_nodes)
        
        # Add nodes to graph
        G.add_nodes_from(cycle_nodes)
        
        # Create cycle edges
        for i in range(cycle_length):
            current_node = cycle_nodes[i]
            next_node = cycle_nodes[(i + 1) % cycle_length]
            
            weight = np.random.uniform(min_edge_weight, max_edge_weight)
            sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"
            
            G.add_edge(
                current_node,
                next_node,
                edge_weight=weight,
                sign=sign,
                n_references=weight,
            )
    
    # Connect cycles with bridge edges
    for i in range(len(all_cycle_nodes) - 1):
        current_cycle = all_cycle_nodes[i]
        next_cycle = all_cycle_nodes[i + 1]
        
        # Connect random nodes from each cycle
        for _ in range(connection_nodes):
            source_node = np.random.choice(current_cycle)
            target_node = np.random.choice(next_cycle)
            
            weight = np.random.uniform(min_edge_weight, max_edge_weight)
            sign = "Inhibitor" if np.random.random() < p_inhibition else "Activator"
            
            G.add_edge(
                source_node,
                target_node,
                edge_weight=weight,
                sign=sign,
                n_references=weight,
            )
    
    return G


def analyze_loop_structure(G: Graph | nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze the loop structure of a graph.
    
    Args:
        G: Graph to analyze
        
    Returns:
        Dictionary of loop metrics
    """
    metrics = {}
    
    # Basic graph statistics
    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)
    
    # Find all simple cycles
    try:
        cycles = list(nx.simple_cycles(G))
        metrics["n_cycles"] = len(cycles)
        
        if cycles:
            cycle_lengths = [len(cycle) for cycle in cycles]
            metrics["cycle_lengths"] = cycle_lengths
            metrics["min_cycle_length"] = min(cycle_lengths)
            metrics["max_cycle_length"] = max(cycle_lengths)
            metrics["mean_cycle_length"] = np.mean(cycle_lengths)
            metrics["median_cycle_length"] = np.median(cycle_lengths)
            
            # Cycle length distribution
            unique_lengths, counts = np.unique(cycle_lengths, return_counts=True)
            metrics["cycle_length_distribution"] = dict(zip(unique_lengths.tolist(), counts.tolist()))
            
            # Loop density: cycles per node
            metrics["loop_density"] = len(cycles) / G.number_of_nodes()
            
            # Nodes participating in cycles
            nodes_in_cycles = set()
            for cycle in cycles:
                nodes_in_cycles.update(cycle)
            metrics["nodes_in_cycles"] = len(nodes_in_cycles)
            metrics["fraction_nodes_in_cycles"] = len(nodes_in_cycles) / G.number_of_nodes()
            
        else:
            metrics["cycle_lengths"] = []
            metrics["min_cycle_length"] = 0
            metrics["max_cycle_length"] = 0
            metrics["mean_cycle_length"] = 0
            metrics["median_cycle_length"] = 0
            metrics["cycle_length_distribution"] = {}
            metrics["loop_density"] = 0
            metrics["nodes_in_cycles"] = 0
            metrics["fraction_nodes_in_cycles"] = 0
            
    except Exception as e:
        print(f"Error analyzing cycles: {e}")
        metrics["error"] = str(e)
        metrics["n_cycles"] = "Error"
    
    # Additional connectivity metrics
    try:
        if nx.is_weakly_connected(G):
            metrics["weakly_connected"] = True
            metrics["strongly_connected"] = nx.is_strongly_connected(G)
            metrics["n_strongly_connected_components"] = nx.number_strongly_connected_components(G)
        else:
            metrics["weakly_connected"] = False
            metrics["strongly_connected"] = False
            metrics["n_strongly_connected_components"] = nx.number_strongly_connected_components(G)
            
    except Exception as e:
        print(f"Error analyzing connectivity: {e}")
        metrics["connectivity_error"] = str(e)
    
    return metrics


def generate_loop_network_configurations() -> List[Dict[str, Any]]:
    """
    Generate different loop network configurations for testing.
    
    Returns:
        List of configuration dictionaries
    """
    configurations = []
    
    # Single ring cycles of different lengths
    for cycle_length in [3, 4, 5, 6, 8, 10]:
        configurations.append({
            "type": "ring_cycle",
            "name": f"ring_{cycle_length}",
            "params": {
                "cycle_length": cycle_length,
                "node_prefix": f"ring{cycle_length}",
                "seed": 42
            }
        })
    
    # Nested loops
    nested_configs = [
        {"cycle_lengths": [3, 4], "overlap_nodes": 1, "name": "nested_3_4_overlap1"},
        {"cycle_lengths": [4, 5], "overlap_nodes": 2, "name": "nested_4_5_overlap2"},
        {"cycle_lengths": [3, 5, 4], "overlap_nodes": 1, "name": "nested_3_5_4_overlap1"},
        {"cycle_lengths": [5, 6, 7], "overlap_nodes": 2, "name": "nested_5_6_7_overlap2"},
    ]
    
    for config in nested_configs:
        configurations.append({
            "type": "nested_loops",
            "name": config["name"],
            "params": {
                "cycle_lengths": config["cycle_lengths"],
                "overlap_nodes": config["overlap_nodes"],
                "node_prefix": "nested",
                "seed": 42
            }
        })
    
    # Loop chains
    chain_configs = [
        {"cycle_lengths": [3, 3, 3], "connection_nodes": 1, "name": "chain_3x3_conn1"},
        {"cycle_lengths": [4, 4, 4], "connection_nodes": 2, "name": "chain_3x4_conn2"},
        {"cycle_lengths": [3, 5, 4], "connection_nodes": 1, "name": "chain_3_5_4_conn1"},
        {"cycle_lengths": [5, 6, 5, 4], "connection_nodes": 1, "name": "chain_5_6_5_4_conn1"},
    ]
    
    for config in chain_configs:
        configurations.append({
            "type": "loop_chain",
            "name": config["name"],
            "params": {
                "cycle_lengths": config["cycle_lengths"],
                "connection_nodes": config["connection_nodes"],
                "node_prefix": "chain",
                "seed": 42
            }
        })
    
    return configurations


def main():
    parser = argparse.ArgumentParser(description="Generate loop-optimized synthetic networks")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="loop_networks",
        help="Output directory for generated networks"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Generate only specified configuration (optional)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing networks, don't generate new ones"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configurations = generate_loop_network_configurations()
    
    if args.config_name:
        configurations = [c for c in configurations if c["name"] == args.config_name]
        if not configurations:
            print(f"Configuration '{args.config_name}' not found!")
            return
    
    results = {}
    
    for config in tqdm(configurations, desc="Generating loop networks"):
        config_name = config["name"]
        
        if not args.analyze_only:
            print(f"Generating network: {config_name}")
            
            # Generate network based on type
            if config["type"] == "ring_cycle":
                network = create_ring_cycle(**config["params"])
            elif config["type"] == "nested_loops":
                network = create_nested_loops(**config["params"])
            elif config["type"] == "loop_chain":
                network = create_loop_chain(**config["params"])
            else:
                print(f"Unknown network type: {config['type']}")
                continue
            
            # Save network as BMA JSON using gen_json
            output_file = output_dir / f"{config_name}.json"
            gen_json(network, str(output_dir), config_name)
            print(f"Saved network to: {output_file}")
        else:
            # Load existing network
            output_file = output_dir / f"{config_name}.json"
            if not output_file.exists():
                print(f"Network file not found: {output_file}")
                continue
            
            # Load network from BMA JSON using json_to_graph
            network = json_to_graph(output_file)
        
        # Analyze network
        print(f"Analyzing network: {config_name}")
        analysis = analyze_loop_structure(network)
        analysis["config"] = config
        results[config_name] = analysis
        
        # Print summary
        print(f"  Nodes: {analysis['n_nodes']}, Edges: {analysis['n_edges']}")
        print(f"  Cycles: {analysis['n_cycles']}, Mean cycle length: {analysis.get('mean_cycle_length', 0):.1f}")
        print(f"  Loop density: {analysis.get('loop_density', 0):.3f}")
        print()
    
    # Save analysis results
    analysis_file = output_dir / "loop_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Analysis results saved to: {analysis_file}")
    print(f"Generated {len(configurations)} loop-optimized networks in {output_dir}")


if __name__ == "__main__":
    main()