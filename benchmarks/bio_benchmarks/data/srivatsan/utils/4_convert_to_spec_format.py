"""
Convert Srivatsan MCF7 BRCA filtered data to specification format.

This script transforms the processed Srivatsan data into the same format as
spec_master.csv files used in other benchmarks. The key transformations are:
- For each drug, look up its targets from ChEMBL and create perturbation rows
- For measured genes with significant effects (qual_effect != 1), create observation rows
"""

from pathlib import Path

import fire
import pandas as pd

from benchmarks.bio_benchmarks.data.srivatsan.utils.srivatsan_funcs import (
    create_target_perturbation_rows,
    load_lookup_tables,
    load_mcf7_basal_perturbations,
)


def main(
    input_file: str = str(
        Path(__file__).parent.parent / "mcf7_brca_filtered_processed.csv"
    ),
    output_file: str = str(
        Path(__file__).parent.parent / "spec" / "srivatsan_spec_master.csv"
    ),
    cas_to_chembl_file: str = str(Path(__file__).parent.parent / "cas_to_chembl.json"),
    chembl_to_targets_file: str = str(
        Path(__file__).parent.parent / "chembl_to_targets.json"
    ),
    brca_json_file: str = str(
        Path(__file__).parent.parent.parent
        / "BRCA"
        / "v1"
        / "json"
        / "kegg_shortest_path_with_phenotype.json"
    ),
    literature_spec_file: str = str(
        Path(__file__).parent.parent.parent
        / "BRCA"
        / "v1"
        / "spec"
        / "literature_curated_specification.csv"
    ),
    brca_max_pert_level: int = 2,
) -> None:
    """
    Convert Srivatsan MCF7 BRCA filtered data to specification format.

    Args:
        input_file: Path to input mcf7_brca_filtered_processed.csv file
        output_file: Path to output srivatsan_spec_master.csv file
        cas_to_chembl_file: Path to CAS to ChEMBL mapping JSON file
        chembl_to_targets_file: Path to ChEMBL to targets mapping JSON file
        brca_json_file: Path to BRCA pathway JSON model
        literature_spec_file: Path to literature curated specification CSV
        brca_max_pert_level: Maximum perturbation level for BRCA network
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    cas_to_chembl_path = Path(cas_to_chembl_file)
    chembl_to_targets_path = Path(chembl_to_targets_file)
    brca_json_path = Path(brca_json_file)
    literature_spec_path = Path(literature_spec_file)

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows")

    # Filter out qual_effect == 1 (non-significant effects) for observation rows
    print("\nFiltering rows where qual_effect != 1 for observation rows...")
    df_filtered = df.loc[df["qual_effect"] != 1].copy()
    print(
        f"After filtering: {len(df_filtered):,} rows ({len(df_filtered) / len(df) * 100:.1f}%)"
    )

    # Count by qual_effect value
    qual_counts = df_filtered.loc[:, "qual_effect"].value_counts().sort_index()
    print("\nQual_effect distribution:")
    for val, count in qual_counts.items():
        print(f"  {val}: {count:,} rows")

    print("\nLoading lookup tables...")
    cas_to_chembl, chembl_to_targets, brca_genes = load_lookup_tables(
        cas_to_chembl_path, chembl_to_targets_path, brca_json_path
    )
    print(f"  CAS to ChEMBL mappings: {len(cas_to_chembl)}")
    print(f"  ChEMBL to targets mappings: {len(chembl_to_targets)}")
    print(f"  BRCA network genes: {len(brca_genes)}")

    # Restrict observation candidates to BRCA network genes
    obs_net = df_filtered[df_filtered["gene_short_name"].isin(brca_genes)].copy()
    print(f"  Network observations after filter: {len(obs_net):,}")

    # Load MCF7.basal perturbations
    print("\nLoading MCF7.basal perturbations...")
    mcf7_basal = load_mcf7_basal_perturbations(literature_spec_path)

    # Create target perturbation rows
    print("\nCreating drug target perturbation rows...")
    target_rows_df = create_target_perturbation_rows(
        cas_to_chembl, chembl_to_targets, brca_genes, df_filtered, brca_max_pert_level
    )

    # Get unique drugs
    unique_drugs = df_filtered[["CAS.Number", "name"]].drop_duplicates()
    print(f"\nProcessing {len(unique_drugs)} unique drugs...")

    # For each drug, combine basal + drug-specific + observations
    all_rows = []

    dropped_no_targets = 0
    dropped_no_obs = 0
    kept_drugs = 0

    for _, drug_row in unique_drugs.iterrows():
        cas_number = drug_row["CAS.Number"]
        drug_name = drug_row["name"]

        # Get drug-specific target perturbations (non-basal perturbations on BRCA genes)
        drug_targets = target_rows_df[target_rows_df["cas_number"] == cas_number].copy()
        if drug_targets.empty:
            dropped_no_targets += 1
            continue

        # Network observations for this drug (only BRCA genes)
        drug_obs_net = obs_net[obs_net["CAS.Number"] == cas_number]
        if drug_obs_net.empty:
            dropped_no_obs += 1
            continue

        kept_drugs += 1

        drug_target_genes = set(drug_targets.loc[:, "gene"].values)

        # Add basal perturbations (excluding genes targeted by drug so targets override basal)
        for _, basal_row in mcf7_basal.iterrows():
            if basal_row["gene"] not in drug_target_genes:
                all_rows.append(
                    {
                        "gene": basal_row["gene"],
                        "perturbation": basal_row["perturbation"],
                        "expectation_bma": pd.NA,
                        "drug_name": drug_name,
                        "cas_number": cas_number,
                        "chembl_id": "",
                        "action_types": "basal",
                    }
                )

        # Add drug-specific target perturbations
        for _, target_row in drug_targets.iterrows():
            all_rows.append(target_row.to_dict())

        # Add observation rows for this drug restricted to BRCA network genes
        for _, obs_row in drug_obs_net.iterrows():
            all_rows.append(
                {
                    "gene": obs_row["gene_short_name"],
                    "perturbation": pd.NA,
                    "expectation_bma": obs_row["qual_effect"],
                    "drug_name": drug_name,
                    "cas_number": cas_number,
                    "chembl_id": "",
                    "action_types": "",
                }
            )

    # Combine all rows
    print("\nCombining all rows...")
    combined_df = pd.DataFrame(all_rows)
    print(f"  Total combined rows: {len(combined_df):,}")

    # Count different row types
    basal_rows = len(combined_df[combined_df["action_types"] == "basal"])
    drug_target_rows = len(
        combined_df[
            (combined_df["perturbation"].notna())
            & (combined_df["action_types"] != "basal")
        ]
    )
    observation_rows = len(combined_df[combined_df["expectation_bma"].notna()])
    print(f"    Basal perturbation rows: {basal_rows:,}")
    print(f"    Drug target perturbation rows: {drug_target_rows:,}")
    print(f"    Observation rows: {observation_rows:,}")

    # Log drop/keep stats for experiments
    print("\nExperiment eligibility summary:")
    print(f"  Kept drugs (eligible): {kept_drugs}")
    print(f"  Dropped (no network targets): {dropped_no_targets}")
    print(f"  Dropped (no network observations): {dropped_no_obs}")

    # Create spec format dataframe
    print("\nConverting to spec format...")

    # Simpler approach: use action_types directly, but map it properly
    def get_mutation_type(row):
        if row["action_types"] == "basal":
            return "basal"
        elif pd.notna(row["perturbation"]) and row["action_types"] != "basal":
            return "Drug"
        else:
            return ""

    mutation_or_amplification = combined_df.apply(get_mutation_type, axis=1)

    spec_df = pd.DataFrame(
        {
            "": range(1, len(combined_df) + 1),  # Row index
            # "X": range(1, len(combined_df) + 1),  # X index
            "source": "Srivatsan",  # Data source
            "DOI": "10.1126/science.aax6234",  # Srivatsan et al. Cell 2019
            "paper_title": "Proteomics of human cell lines identifies drug targets and pathways",
            "experiment_overview": "MCF7 BRCA cell line drug perturbation screen",
            "cell_line": "MCF7",  # Cell line
            "experiment_particular": combined_df["drug_name"]
            + " ("
            + combined_df["cas_number"]
            + ")",
            "gene": combined_df["gene"].values,  # Gene (target or measured)
            "mutation.or.Amplification": mutation_or_amplification,  # Type of perturbation
            "perturbation": combined_df[
                "perturbation"
            ].values,  # Perturbation level (0, 2, or NA)
            "expectation_bma": combined_df[
                "expectation_bma"
            ].values,  # Observed effect (0, 2, or NA)
            "action_types": combined_df["action_types"].values,
            "Author": "Srivatsan",
            "DEPRECATED": "N",
            "paper_code": "Srivatsan2019",
            "Figure": "",
            "mean_result": combined_df["expectation_bma"]
            .fillna(combined_df["perturbation"])
            .values,
        }
    )

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    print(f"\nSaving to {output_path}...")
    spec_df.to_csv(output_path, index=False)

    print(f"\n✓ Successfully created spec file with {len(spec_df):,} rows")
    print(f"  Output: {output_path}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Basal perturbation rows: {basal_rows:,}")
    print(f"  Drug target perturbation rows: {drug_target_rows:,}")
    print(f"  Observation rows: {observation_rows:,}")
    print(f"  Unique drugs: {combined_df['cas_number'].nunique()}")
    print(f"  Unique genes (all): {combined_df['gene'].nunique()}")

    # Show sample rows
    print("\nSample basal perturbation rows (first 3):")
    sample = spec_df.loc[spec_df["mutation.or.Amplification"] == "basal"].head(3)[
        ["experiment_particular", "gene", "perturbation", "expectation_bma"]
    ]
    print(sample.to_string(index=False))

    if drug_target_rows > 0:
        print("\nSample drug target perturbation rows (first 3):")
        sample = spec_df.loc[
            (spec_df.loc[:, "perturbation"].notna())
            & (spec_df.loc[:, "mutation.or.Amplification"] == "Drug")
        ].head(3)[["experiment_particular", "gene", "perturbation", "expectation_bma"]]
        print(sample.to_string(index=False))

    print("\nSample observation rows (first 3):")
    sample = spec_df.loc[spec_df.loc[:, "expectation_bma"].notna()].head(3)[
        ["experiment_particular", "gene", "perturbation", "expectation_bma"]
    ]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    fire.Fire(main)
