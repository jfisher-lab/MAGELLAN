"""Extract cell types, drugs, measured genes, and perturbed genes from annotated specification."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


def extract_cell_type(experiment: str) -> str:
    """Extract cell type from experiment name.

    Args:
        experiment: Experiment name (e.g., "MCF7_Srivatsan_AMG-900 (945595-80-2)")

    Returns:
        Cell type (e.g., "MCF7")
    """
    # Cell type is before the first underscore
    return experiment.split("_")[0]


def extract_drugs(experiment: str) -> List[str]:
    """Extract drug names from experiment name.

    Args:
        experiment: Experiment name with pipe-separated drug combinations
                   (e.g., "MCF7_Srivatsan_AMG-900 (945595-80-2)|MCF7_Srivatsan_CYC116 (693228-63-6)")

    Returns:
        List of drug names (e.g., ["AMG-900", "CYC116"])
    """
    drugs = []

    # Split by pipe for combination experiments
    parts = experiment.split("|")

    for part in parts:
        # Pattern: after second underscore, capture text before parentheses
        # MCF7_Srivatsan_AMG-900 (945595-80-2) -> AMG-900
        match = re.search(r"_[^_]+_(.+?)\s*\(", part)
        if match:
            drug_name = match.group(1).strip()
            drugs.append(drug_name)

    return drugs


def analyze_specification(csv_path: Path) -> Dict[str, any]:
    """Analyze annotated specification CSV to extract experiment metadata.

    Args:
        csv_path: Path to annotated_specification.csv

    Returns:
        Dictionary containing:
            - cell_types: Set of unique cell types
            - drugs: Set of unique drugs
            - measured_genes: Set of genes with type="perturbation"
            - perturbed_genes: Set of genes that are perturbed (also type="perturbation")
            - experiments: List of all unique experiments
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Extract cell types from all experiments
    cell_types: Set[str] = set()
    for exp in df["Experiment"].unique():
        cell_types.add(extract_cell_type(exp))

    # Extract all drugs
    drugs: Set[str] = set()
    for exp in df["Experiment"].unique():
        drugs.update(extract_drugs(exp))

    # Measured genes and perturbed genes are nodes where Type == "perturbation"
    perturbation_df = df.query('Type == "perturbation"')
    measured_genes = set(perturbation_df["Node"].unique())
    perturbed_genes = measured_genes  # Same as measured genes in this context

    # Experimental genes (Type == "experimental")
    experimental_df = df.query('Type == "experimental"')
    experimental_genes = set(experimental_df["Node"].unique())

    return {
        "cell_types": cell_types,
        "drugs": drugs,
        "measured_genes": measured_genes,
        "perturbed_genes": perturbed_genes,
        "experimental_genes": experimental_genes,
        "experiments": list(df["Experiment"].unique()),
        "total_experiments": len(df["Experiment"].unique()),
        "total_rows": len(df),
    }


def main() -> None:
    """Main function to run the analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract cell types, drugs, and genes from annotated specification CSV"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/Users/matthew/Documents/Github/magellanv2/benchmarks/bio_benchmarks/results/BRCA/"
            "model_kegg_shortest_path_with_phenotype_v1_spec_literature_curated_specification_v1_and_srivatsan_spec_master/"
            "test/nonbinary_metrics"
        ),
        help="Directory containing annotated_specification.csv (default: original results directory)",
    )
    args = parser.parse_args()

    # Get path to CSV file
    data_dir = args.data_dir
    csv_path = data_dir / "annotated_specification.csv"

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return

    # Analyze
    results = analyze_specification(csv_path)

    # Print results
    print("=" * 80)
    print("EXPERIMENT METADATA EXTRACTION")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")

    print(f"\nTotal experiments: {results['total_experiments']}")
    print(f"Total rows: {results['total_rows']}")

    print(f"\n{'Cell Types':-^80}")
    for cell_type in sorted(results["cell_types"]):
        print(f"  - {cell_type}")

    print(f"\n{'Drugs':-^80}")
    for drug in sorted(results["drugs"]):
        print(f"  - {drug}")

    print(f"\n{'Measured Genes (Type=perturbation)':-^80}")
    print(f"Total: {len(results['measured_genes'])}")
    for gene in sorted(results["measured_genes"]):
        print(f"  - {gene}")

    print(f"\n{'Perturbed Genes (Type=perturbation)':-^80}")
    print(f"Total: {len(results['perturbed_genes'])}")
    for gene in sorted(results["perturbed_genes"]):
        print(f"  - {gene}")

    print(f"\n{'Experimental Genes (Type=experimental)':-^80}")
    print(f"Total: {len(results['experimental_genes'])}")
    for gene in sorted(results["experimental_genes"]):
        print(f"  - {gene}")

    # Save to CSV
    output_dir = data_dir

    # Save summary
    summary_df = pd.DataFrame({
        "Metric": [
            "Total Experiments",
            "Total Rows",
            "Unique Cell Types",
            "Unique Drugs",
            "Measured Genes (perturbation)",
            "Experimental Genes",
        ],
        "Count": [
            results["total_experiments"],
            results["total_rows"],
            len(results["cell_types"]),
            len(results["drugs"]),
            len(results["measured_genes"]),
            len(results["experimental_genes"]),
        ],
    })
    summary_df.to_csv(output_dir / "extraction_summary.csv", index=False)

    # Save detailed lists
    pd.DataFrame({"Cell_Type": sorted(results["cell_types"])}).to_csv(
        output_dir / "cell_types.csv", index=False
    )
    pd.DataFrame({"Drug": sorted(results["drugs"])}).to_csv(
        output_dir / "drugs.csv", index=False
    )
    pd.DataFrame({"Gene": sorted(results["measured_genes"])}).to_csv(
        output_dir / "measured_genes.csv", index=False
    )
    pd.DataFrame({"Gene": sorted(results["perturbed_genes"])}).to_csv(
        output_dir / "perturbed_genes.csv", index=False
    )
    pd.DataFrame({"Gene": sorted(results["experimental_genes"])}).to_csv(
        output_dir / "experimental_genes.csv", index=False
    )

    print(f"\n{'Output Files':-^80}")
    print(f"  - extraction_summary.csv")
    print(f"  - cell_types.csv")
    print(f"  - drugs.csv")
    print(f"  - measured_genes.csv")
    print(f"  - perturbed_genes.csv")
    print(f"  - experimental_genes.csv")
    print(f"\nOutput saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
