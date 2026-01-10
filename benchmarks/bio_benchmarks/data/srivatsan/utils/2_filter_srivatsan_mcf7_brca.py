#!/usr/bin/env python3
"""
Filter Srivatsan dataset for MCF7 cell line and BRCA pathway genes.

This script:
1. Loads S5.parquet and filters for MCF7 cell line
2. Extracts gene list from BRCA JSON pathway model
3. Filters for genes present in BRCA pathway
4. Joins with Supplementary_Table_3.txt to add compound names
5. Saves filtered dataset to parquet

Usage:
    uv run filter_srivatsan_mcf7_brca.py
"""

from pathlib import Path

import fire
import pandas as pd

from benchmarks.bio_benchmarks.data.srivatsan.utils.srivatsan_funcs import (
    classify_effect,
    extract_json_genes,
    load_srivatsan_drug_table,
)


def main(
    s5_path: str | Path = str(Path(__file__).parent.parent / "S5.parquet"),
    supp_table_path: str | Path = str(Path(__file__).parent.parent / "Supplementary_Table_3.txt"),
    json_path: str | Path = str(Path(__file__).parent.parent.parent / "BRCA" / "v1" / "json" / "kegg_shortest_path_with_phenotype.json"),
    joined_output_path: str | Path = str(Path(__file__).parent.parent / "mcf7_brca_filtered.parquet"),
    processed_output_path: str | Path = str(Path(__file__).parent.parent / "mcf7_brca_filtered_processed.csv"),
    cell_type: str = "MCF7"
) -> None:
    """
    Filter Srivatsan dataset for MCF7 cell line and BRCA pathway genes.

    Args:
        s5_path: Path to S5.parquet file
        supp_table_path: Path to Supplementary_Table_3.txt file
        json_path: Path to BRCA pathway JSON model
        joined_output_path: Path to output mcf7_brca_filtered.parquet file
        processed_output_path: Path to output mcf7_brca_filtered_processed.csv file
        cell_type: Cell type to filter for (default: MCF7)
    """
    s5_path = Path(s5_path)
    supp_table_path = Path(supp_table_path)
    json_path = Path(json_path)
    joined_output_path = Path(joined_output_path)
    processed_output_path = Path(processed_output_path)

    print("Loading S5.parquet...")
    s5_df = pd.read_parquet(s5_path)
    print(f"Original S5 dataset shape: {s5_df.shape}")
    print(f"Filtering for {cell_type} cell line...")
    mcf7_df = s5_df[s5_df['cell_type'] == cell_type].copy()
    if not isinstance(mcf7_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(mcf7_df)}")
    # Optionally validate it's not empty
    if mcf7_df.empty:
        raise ValueError(f"{cell_type} DataFrame is empty")
    print(f"{cell_type} filtered shape: {mcf7_df.shape}")

    print(f"Extracting JSON genes from {json_path.stem}...")
    json_genes = extract_json_genes(json_path)
    print(f"Found {len(json_genes)} genes in JSON")

    print("Filtering for JSON model genes...")
    brca_mcf7_df = mcf7_df[mcf7_df['gene_short_name'].isin(json_genes)].copy() 
    if not isinstance(brca_mcf7_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(brca_mcf7_df)}")
    if brca_mcf7_df.empty:
        raise ValueError("BRCA + MCF7 DataFrame is empty")
    print(f"MCF7 + BRCA genes shape: {brca_mcf7_df.shape}")

    print("Loading compound mapping...")
    compound_map = load_srivatsan_drug_table(supp_table_path)
    print(f"Compound mapping shape: {compound_map.shape}")

    print("Joining with compound information...")
    # Join on treatment column
    final_df = brca_mcf7_df.merge(
        compound_map,
        on='treatment',
        how='left'
    )
    print(f"Final dataset shape: {final_df.shape}")

    print(f"\nSaving filtered dataset to {joined_output_path}...")
    final_df.to_parquet(joined_output_path, index=False)
    print("Done!")
    
    # Select only what is needed for the benchmark
    short_df = final_df[final_df['term'] != '(Intercept)'].copy()
    if not isinstance(short_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(short_df)}")
    if short_df.empty:
        raise ValueError("Short DataFrame is empty")
    short_df = short_df[['gene_short_name', 'name', 'treatment', 'CAS.Number', 'normalized_effect', 'q_value']]
    # cast normalized_effect and q_value to float
    short_df['normalized_effect'] = short_df['normalized_effect'].astype(float)
    short_df['q_value'] = short_df['q_value'].astype(float)
    # Make column qual_effect: 2 if q_value < 0.05 and normalized_effect > 0, 0 if q_value < 0.05 and normalized_effect < 0, 1 if q_value >= 0.05
    short_df['qual_effect'] = short_df.apply(classify_effect, axis=1)
    short_df.to_csv(processed_output_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)