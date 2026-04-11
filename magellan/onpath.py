"""On-path weight floor: a one-sided loss penalty on signal-carrying edges.

An edge is "on-path" if it lies on at least one shortest path from a
perturbation source node to an observation (expectation) node in the training
data.  The on-path floor penalty pulls those edge weights up toward a per-edge
target ``onpath_floor_target / sqrt(in_degree(dst))``, so the network preserves
signal propagation to phenotype nodes without constraining edges that are
already strong.

This module is consumed via the ``loss_fn`` hook on
:func:`magellan.gnn_model.train_model`: build the wrapped loss with
:func:`_make_onpath_floor_wrapper` and pass it as ``loss_fn=``.

``compute_onpath_edge_mask`` is the primitive public API (takes graph + dicts).
``_compute_onpath_edge_mask`` is a thin convenience wrapper over a
:class:`TrainingBundle`.  The minimal :class:`TrainingBundle`/:class:`SplitData`
dataclasses defined here exist for that wrapper and for behavioural tests; the
benchmark pipeline carries its own richer equivalents.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import networkx as nx
import torch

if TYPE_CHECKING:
    from magellan.gnn_model import Net


@dataclass
class SplitData:
    """Minimal per-split container.

    Only ``pert_dic`` is read by the on-path mask; the remaining fields exist
    so a full bundle can be constructed without coupling to the benchmark's
    richer split structure.

    Args:
        pert_dic: Dictionary of perturbations
        X: Feature matrix
        y: Target matrix
        y_no_zero: Target matrix without zero values
        node_class_weights: Node class weights
        pyg_data: PyG data
        mask_dic: Dictionary of masks
        adjacency_per_experiment: Dictionary of adjacency matrices per experiment

    Returns:
        SplitData: Split data
    """

    pert_dic: OrderedDict
    X: Any = None
    y: Any = None
    y_no_zero: Any = None
    node_class_weights: Any = None
    pyg_data: Any = None
    mask_dic: Any = None
    adjacency_per_experiment: Any = None


@dataclass
class TrainingBundle:
    """Minimal training bundle for the on-path mask.

    The mask reads only ``G``, ``node_dic``, ``edge_idx`` and
    ``splits["train"].pert_dic``.  The other fields mirror the benchmark's
    bundle shape so existing call sites and tests can construct it uniformly.

    Args:
        G: Graph
        splits: Dictionary of splits
        inh: Inhibitory edges
        A_mult: Multiplicative adjacency matrix
        node_dic: Dictionary of node indices
        pert_idx: Dictionary of perturbation indices
        edge_idx: Edge indices
        edge_idx_original: Original edge indices
        edge_scale: Edge scaling factors
        pert_mask: Perturbation mask
        edge_signs: Edge signs
        const_dic: Dictionary of constants
        combined_pert_dic: Dictionary of combined perturbations

    Returns:
        TrainingBundle: Training bundle
    """

    G: nx.DiGraph
    splits: dict[str, SplitData]
    inh: Any = None
    A_mult: Any = None
    node_dic: dict[str, int] = field(default_factory=dict)
    pert_idx: Any = None
    edge_idx: torch.Tensor | None = None
    edge_idx_original: torch.Tensor | None = None
    edge_scale: torch.Tensor | None = None
    pert_mask: torch.Tensor | None = None
    edge_signs: torch.Tensor | None = None
    const_dic: Any = None
    combined_pert_dic: OrderedDict = field(default_factory=OrderedDict)


def compute_onpath_edge_mask(
    G: nx.DiGraph,
    node_dic: dict[str, int],
    edge_idx: torch.Tensor,
    train_pert_dic: OrderedDict,
) -> torch.Tensor:
    """Compute a boolean mask over ``edge_idx`` marking on-path edges.

    An edge is "on-path" if it lies on at least one shortest path from any
    perturbation source node to any observation (expectation) node in the
    training data.  This identifies edges that could carry perturbation
    signal to phenotype nodes and should be exempt from weight decay.

    Uses BFS-based predecessor tracking per source to enumerate all
    shortest-path edges efficiently (avoids the O(sources x observations x
    paths) cost of calling ``nx.all_shortest_paths`` per pair).

    Returns a boolean tensor of shape ``(n_edges,)`` aligned with
    ``model.edge_weight`` / ``edge_idx[0]``.

    Args:
        G: Graph
        node_dic: Dictionary of node indices
        edge_idx: Edge indices
        train_pert_dic: Dictionary of training perturbations

    Returns:
        torch.Tensor: Boolean mask of shape (n_edges,)
    """
    sources: set[str] = set()
    observations: set[str] = set()
    for exp in train_pert_dic.values():
        sources |= set(exp.get("pert", {}).keys())
        observations |= set(exp.get("exp", {}).keys())
    sources = {s for s in sources if s in node_dic and s in G}
    observations = {o for o in observations if o in node_dic and o in G}

    onpath_edge_pairs: set[tuple[int, int]] = set()

    for src in sources:
        pred_map = nx.predecessor(G, src)
        reachable_obs = observations & pred_map.keys()
        for obs in reachable_obs:
            if obs == src:
                continue
            _collect_shortest_path_edges(
                pred_map, obs, src, node_dic, onpath_edge_pairs
            )

    n_edges = edge_idx.shape[1]
    mask = torch.zeros(n_edges, dtype=torch.bool)
    edge_src = edge_idx[0].tolist()
    edge_dst = edge_idx[1].tolist()
    for ei in range(n_edges):
        if (edge_src[ei], edge_dst[ei]) in onpath_edge_pairs:
            mask[ei] = True

    return mask


def _compute_onpath_edge_mask(bundle: TrainingBundle) -> torch.Tensor:
    """Compute the on-path edge mask from a :class:`TrainingBundle`.

    Thin wrapper over :func:`compute_onpath_edge_mask` reading the four inputs
    it needs (``G``, ``node_dic``, ``edge_idx`` and the train split's
    ``pert_dic``) off the bundle.

    Args:
        bundle: Training bundle

    Returns:
        torch.Tensor: Boolean mask of shape (n_edges,)
    """
    return compute_onpath_edge_mask(
        bundle.G,
        bundle.node_dic,
        bundle.edge_idx,
        bundle.splits["train"].pert_dic,
    )


def _collect_shortest_path_edges(
    pred_map: dict[str, list[str]],
    target: str,
    source: str,
    node_dic: dict[str, int],
    out: set[tuple[int, int]],
) -> None:
    """Backtrack through ``pred_map`` from ``target`` to ``source``, adding
    every edge traversed to ``out`` (as index pairs via ``node_dic``).

    ``pred_map`` is the output of ``nx.predecessor(G, source)`` — a dict
    mapping each node to a list of its predecessors on shortest paths from
    source.

    Args:
        pred_map: Predecessor map
        target: Target node
        source: Source node
        node_dic: Dictionary of node indices
        out: Set of edge pairs
    """
    stack = [target]
    visited: set[str] = set()
    while stack:
        node = stack.pop()
        if node == source or node in visited:
            continue
        visited.add(node)
        for pred in pred_map.get(node, []):
            if pred in node_dic and node in node_dic:
                out.add((node_dic[pred], node_dic[node]))
            stack.append(pred)


def _make_onpath_floor_wrapper(
    original_loss,
    model: Net,
    onpath_mask: torch.Tensor,
    edge_idx: torch.Tensor,
    onpath_floor_lambda: float,
    onpath_floor_target: float,
):
    """Wrap the loss with a one-sided penalty pulling on-path edges toward a target.

    For each on-path edge u->v, computes a per-edge target scaled by the
    destination node's in-degree: ``target(v) = onpath_floor_target / sqrt(in_degree(v))``.
    Only penalises edges *below* this target (one-sided relu), so strong edges
    are unconstrained.  Uses Polyak-style scaling (penalty * base_loss.detach())
    so the lambda is interpreted as a fraction of the base loss.

    Args:
        original_loss: Original loss function
        model: Model
        onpath_mask: Boolean mask of shape (n_edges,)
        edge_idx: Edge indices
        onpath_floor_lambda: Lambda for the on-path floor penalty
        onpath_floor_target: Target for the on-path floor penalty

    Returns:
        wrapped: Wrapped loss function
    """
    dst_indices = edge_idx[1][onpath_mask]
    in_deg = torch.bincount(edge_idx[1]).clamp_min(1).to(torch.float32)
    per_edge_target = onpath_floor_target / torch.sqrt(in_deg[dst_indices])

    def wrapped(pred, target, *args, **kwargs):
        base = original_loss(pred, target, *args, **kwargs)
        onpath_weights = model.edge_weight[onpath_mask]
        shortfall = torch.relu(per_edge_target - onpath_weights)
        penalty = (shortfall**2).mean()
        return base + onpath_floor_lambda * penalty * base.detach()

    return wrapped
