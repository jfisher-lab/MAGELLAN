"""Strategies for selecting which DEG genes become phenotype nodes.

Each strategy takes the full set of DEG genes and returns the subset to
assign as phenotype nodes.  The remainder stay as DEG-only (intermediate
signalling layer).  This controls network topology via ``to_combine``:
paths are found ``mut -> deg`` and ``deg -> pheno``.

All functions return a frozen set for determinism.

``refine_pheno_from_strategy`` dispatches over the four ``select_pheno_*``
strategies (``downstream``/``frequency``/``magnitude``/``random``) plus a
``none`` pass-through.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def select_pheno_downstream(
    deg_genes: set[str],
    mut_genes: set[str],
    int_op: pd.DataFrame,
    fraction: float,
    min_primary_sources: int = 1,
) -> set[str]:
    """Select the most downstream DEG genes as phenotype nodes.

    Builds a directed graph from Omnipath and computes the shortest-path
    distance from any mut gene to each DEG gene.  Genes with the largest
    distance (most downstream) are selected as pheno.  Genes unreachable
    from any mut gene are included as pheno (maximally downstream).

    Args:
        deg_genes: Full set of DEG genes (candidates for pheno).
        mut_genes: Perturbation source genes.
        int_op: Preprocessed Omnipath interactions DataFrame.
        fraction: Fraction of DEG genes to place in pheno (0.0 to 1.0).
        min_primary_sources: Edge filter threshold for building the distance graph.

    Returns:
        set[str]: Set of phenotype genes
    """
    import networkx as nx

    if not deg_genes:
        logger.warning(
            "downstream strategy received zero DEG genes; returning empty pheno set"
        )
        return set()

    filtered = int_op[int_op["n_primary_sources"] > min_primary_sources]
    G = nx.DiGraph()
    G.add_edges_from(zip(filtered["source_genesymbol"], filtered["target_genesymbol"]))

    distances: dict[str, int] = {}
    max_dist = len(G) + 1

    for gene in deg_genes:
        if gene not in G:
            distances[gene] = max_dist
            continue
        min_dist = max_dist
        for mut_gene in mut_genes:
            if mut_gene not in G:
                continue
            try:
                d = nx.shortest_path_length(G, source=mut_gene, target=gene)
                min_dist = min(min_dist, d)
            except nx.NetworkXNoPath:
                continue
        distances[gene] = min_dist

    n_pheno = min(len(deg_genes), max(1, int(len(deg_genes) * fraction)))
    sorted_genes = sorted(distances.keys(), key=lambda g: distances[g], reverse=True)
    pheno = set(sorted_genes[:n_pheno])

    logger.info(
        "downstream strategy: %d/%d DEGs -> pheno (fraction=%.2f, "
        "min_dist=%d, max_dist=%d among selected)",
        len(pheno),
        len(deg_genes),
        fraction,
        min(distances[g] for g in pheno),
        max(distances[g] for g in pheno),
    )
    return pheno


def select_pheno_frequency(
    deg_genes: set[str],
    discretised_df: pd.DataFrame,
    split_dict: dict[str, list[str]],
    fraction: float,
) -> set[str]:
    """Select DEG genes that are significant in the most training experiments.

    Genes that recurrently show up as differentially expressed across many
    perturbations are likely terminal readout genes (phenotypic signature),
    while genes significant in only one or two experiments are more likely
    intermediate transients.

    Args:
        deg_genes: Full set of DEG genes.
        discretised_df: DataFrame with columns: treatment, gene, qual_effect.
        split_dict: Split assignments (uses "train" key).
        fraction: Fraction of DEG genes to place in pheno.

    Returns:
        set[str]: Set of phenotype genes
    """
    train_treatments = set(split_dict.get("train", []))
    train_df = discretised_df[discretised_df["treatment"].isin(list(train_treatments))]
    significant = train_df[train_df["qual_effect"].isin([0, 2])]

    gene_counts = significant.groupby("gene").size()
    gene_counts = gene_counts[gene_counts.index.isin(deg_genes)]

    n_pheno = min(len(deg_genes), max(1, int(len(deg_genes) * fraction)))
    top_genes = gene_counts.nlargest(n_pheno).index.tolist()
    pheno = set(top_genes)

    if len(pheno) < n_pheno:
        remaining = sorted(deg_genes - pheno)
        pheno.update(remaining[: n_pheno - len(pheno)])

    logger.info(
        "frequency strategy: %d/%d DEGs -> pheno (fraction=%.2f, "
        "min_count=%d, max_count=%d among selected)",
        len(pheno),
        len(deg_genes),
        fraction,
        int(gene_counts[gene_counts.index.isin(pheno)].min()) if len(pheno) > 0 else 0,
        int(gene_counts[gene_counts.index.isin(pheno)].max()) if len(pheno) > 0 else 0,
    )
    return pheno


def select_pheno_magnitude(
    deg_genes: set[str],
    discretised_df: pd.DataFrame,
    split_dict: dict[str, list[str]],
    fraction: float,
) -> set[int]:
    """Select DEG genes with the largest mean effect magnitude.

    Genes with consistently strong differential expression (mean |qual_effect - 1|
    across experiments where they are significant) are the strongest phenotypic
    signals.

    Args:
        deg_genes: Full set of DEG genes.
        discretised_df: DataFrame with columns: treatment, gene, qual_effect.
        split_dict: Split assignments (uses "train" key).
        fraction: Fraction of DEG genes to place in pheno.
        seed: Random seed for reproducibility.

    Returns:
        set[int]: Set of phenotype genes
    """
    train_treatments = set(split_dict.get("train", []))
    train_df = discretised_df[discretised_df["treatment"].isin(list(train_treatments))]
    significant = train_df[train_df["qual_effect"].isin([0, 2])]

    gene_magnitude = significant.groupby("gene")["qual_effect"].apply(
        lambda x: np.abs(x - 1).mean()
    )
    gene_magnitude = gene_magnitude[gene_magnitude.index.isin(deg_genes)]

    n_pheno = min(len(deg_genes), max(1, int(len(deg_genes) * fraction)))
    top_genes = gene_magnitude.nlargest(n_pheno).index.tolist()
    pheno = set(top_genes)

    if len(pheno) < n_pheno:
        remaining = sorted(deg_genes - pheno)
        pheno.update(remaining[: n_pheno - len(pheno)])

    logger.info(
        "magnitude strategy: %d/%d DEGs -> pheno (fraction=%.2f)",
        len(pheno),
        len(deg_genes),
        fraction,
    )
    return pheno


def select_pheno_random(
    deg_genes: set[str],
    fraction: float,
    seed: int = 42,
) -> set[str]:
    """Select a random subset of DEG genes as phenotype nodes.

    Control strategy to test whether network sparsity per se helps,
    independent of gene selection criteria.

    Args:
        deg_genes: Full set of DEG genes.
        fraction: Fraction of DEG genes to place in pheno.
        seed: Random seed for reproducibility.

    Returns:
        set[str]: Set of phenotype genes
    """
    rng = np.random.default_rng(seed)
    n_pheno = min(len(deg_genes), max(1, int(len(deg_genes) * fraction)))
    genes_sorted = sorted(deg_genes)
    chosen_idx = rng.choice(len(genes_sorted), size=n_pheno, replace=False)
    pheno = {genes_sorted[i] for i in chosen_idx}

    logger.info(
        "random strategy: %d/%d DEGs -> pheno (fraction=%.2f, seed=%d)",
        len(pheno),
        len(deg_genes),
        fraction,
        seed,
    )
    return pheno


def refine_pheno_from_strategy(
    gene_sets: dict[str, set[str]],
    int_op: pd.DataFrame,
    strategy: str = "downstream",
    fraction: float = 0.25,
    discretised_df: pd.DataFrame | None = None,
    split_dict: dict[str, list[str]] | None = None,
    seed: int = 42,
    min_primary_sources: int = 1,
) -> dict[str, set[str]]:
    """Refine the pheno gene set using a topology/data-driven strategy.

    By default, ``prioritise_genes()`` sets ``pheno = deg``.  This function
    post-processes the gene sets to select a smaller, more biologically
    meaningful pheno layer using one of several strategies.

    Args:
        gene_sets: Dictionary with "mut", "deg", "pheno" keys.
        int_op: Preprocessed Omnipath interactions DataFrame.
        strategy: One of "downstream", "frequency", "magnitude", "random", or "none".
        fraction: Fraction of DEG genes to assign to pheno (0.0 to 1.0).
        discretised_df: Required for "frequency" and "magnitude" strategies.
        split_dict: Required for "frequency" and "magnitude" strategies.
        seed: Random seed for the "random" strategy.
        min_primary_sources: Omnipath edge confidence threshold used by the "downstream" strategy.

    Returns:
        dict[str, set[str]]: New gene_sets dict with refined pheno set.  "deg" remains unchanged (all original DEG genes); only "pheno" is narrowed.
    """
    if strategy == "none":
        logger.info(
            "pheno_strategy='none': keeping pheno = deg (%d genes)",
            len(gene_sets["pheno"]),
        )
        return gene_sets

    deg_genes = gene_sets["deg"]
    mut_genes = gene_sets["mut"]

    if strategy == "downstream" and not deg_genes:
        logger.warning(
            "pheno_strategy='downstream' requested with zero DEG genes; "
            "falling back to pheno_strategy='none' (pheno=deg)"
        )
        return {
            "mut": mut_genes.copy(),
            "deg": deg_genes.copy(),
            "pheno": deg_genes.copy(),
        }

    if strategy == "downstream":
        pheno = select_pheno_downstream(
            deg_genes=deg_genes,
            mut_genes=mut_genes,
            int_op=int_op,
            fraction=fraction,
            min_primary_sources=min_primary_sources,
        )
    elif strategy == "frequency":
        if discretised_df is None or split_dict is None:
            raise ValueError(
                "frequency strategy requires discretised_df and split_dict"
            )
        pheno = select_pheno_frequency(
            deg_genes=deg_genes,
            discretised_df=discretised_df,
            split_dict=split_dict,
            fraction=fraction,
        )
    elif strategy == "magnitude":
        if discretised_df is None or split_dict is None:
            raise ValueError(
                "magnitude strategy requires discretised_df and split_dict"
            )
        pheno = select_pheno_magnitude(
            deg_genes=deg_genes,
            discretised_df=discretised_df,
            split_dict=split_dict,
            fraction=fraction,
        )
    elif strategy == "random":
        pheno = select_pheno_random(
            deg_genes=deg_genes,
            fraction=fraction,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown pheno_strategy: {strategy!r}. "
            "Must be one of: downstream, frequency, magnitude, random, none"
        )

    logger.info(
        "pheno_strategy=%r (fraction=%.2f): pheno narrowed from %d to %d genes",
        strategy,
        fraction,
        len(deg_genes),
        len(pheno),
    )
    return {"mut": mut_genes.copy(), "deg": deg_genes.copy(), "pheno": pheno}
