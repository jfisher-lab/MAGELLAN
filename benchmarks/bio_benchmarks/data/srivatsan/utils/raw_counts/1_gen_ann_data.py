import argparse
from pathlib import Path

import anndata as ad
import joblib
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Generate AnnData from sciPlex3 raw counts"
)
parser.add_argument(
    "--force-reload",
    action="store_true",
    help="Force reload data, bypassing checkpoints",
)
args = parser.parse_args()

srivatsan_dir = Path(__file__).parent.parent.parent
raw_counts_dir = srivatsan_dir / "raw_counts"
ann_data_dir = srivatsan_dir / "ann_data"
ann_data_dir.mkdir(parents=True, exist_ok=True)

# Create checkpoint directory
checkpoint_dir = ann_data_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = checkpoint_dir / "count_matrix_parsed.joblib"

gene_annotations_file = (
    raw_counts_dir / "GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt"
)
cell_annotations_file = (
    raw_counts_dir / "GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt"
)
count_matrix_file = (
    raw_counts_dir / "GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix"
)
pData_file = raw_counts_dir / "GSM4150378_sciPlex3_pData.txt"

# 1. Load annotations
print("Loading gene annotations...")
genes = pd.read_csv(gene_annotations_file, sep=r"\s+", engine="python")
print(f"✓ Loaded {len(genes)} genes")
print(f"  Columns: {genes.columns.tolist()}")

print("\nLoading cell annotations...")
cells = pd.read_csv(
    cell_annotations_file, sep="\t", header=None, names=["cell_barcode", "sample"]
)
print(f"✓ Loaded {len(cells)} cells")

# Load metadata (space-delimited with quoted values)
print("\nLoading metadata...")
pData = pd.read_csv(pData_file, sep=" ", quotechar='"')
print(f"✓ Loaded {len(pData)} metadata rows")

# Merge cell annotations with metadata
print("\nMerging cell annotations with metadata...")
cells = cells.merge(pData, left_on="cell_barcode", right_on="cell", how="left")
print(f"✓ Merged data has {len(cells)} rows")

# 2. Load sparse count matrix
if checkpoint_file.exists() and not args.force_reload:
    print(f"\n✓ Checkpoint found at {checkpoint_file}")
    print("Loading count matrix from checkpoint...")
    checkpoint_data = joblib.load(checkpoint_file)
    rows = checkpoint_data["rows"]
    cols = checkpoint_data["cols"]
    counts = checkpoint_data["counts"]
    n_genes_checkpoint = checkpoint_data["n_genes"]
    n_cells_checkpoint = checkpoint_data["n_cells"]
    print(f"✓ Loaded count matrix from checkpoint ({len(rows)} entries)")
else:
    if args.force_reload:
        print("\n--force-reload specified, bypassing checkpoint")
    print("\nParsing count matrix (this takes ~7 minutes)...")
    rows, cols, counts = [], [], []
    with open(count_matrix_file) as f:
        for line in tqdm(f, desc="Loading count matrix"):
            r, c, v = line.strip().split()
            rows.append(int(r) - 1)
            cols.append(int(c) - 1)
            counts.append(int(v))

    # Save checkpoint
    print(f"\nSaving checkpoint to {checkpoint_file}...")
    checkpoint_data = {
        "rows": rows,
        "cols": cols,
        "counts": counts,
        "n_genes": len(genes),
        "n_cells": len(cells),
    }
    joblib.dump(checkpoint_data, checkpoint_file)
    print("✓ Checkpoint saved")

print("\nConstructing sparse matrix...")
X = sp.coo_matrix((counts, (rows, cols)), shape=(len(genes), len(cells)), dtype=int)
X = X.tocsc()  # Convert to CSC format for efficient column slicing
print(f"✓ Sparse matrix shape: {X.shape[0]} genes × {X.shape[1]} cells")

# 3. Filter for MCF7 cells
print("\nFiltering for MCF7 cells...")
mcf7_cells = cells[cells["cell_type"].str.upper() == "MCF7"]
mcf7_idx = mcf7_cells.index.values
print(f"✓ Found {len(mcf7_cells)} MCF7 cells out of {len(cells)} total cells")

X_mcf7 = X[:, mcf7_idx]
print(f"✓ Filtered matrix shape: {X_mcf7.shape[0]} genes × {X_mcf7.shape[1]} cells")

# 4. Create AnnData object
print("\nCreating AnnData object...")
adata = ad.AnnData(X=X_mcf7.T)

# Verify gene columns before assignment
print(f"Available gene columns: {genes.columns.tolist()}")
if "id" not in genes.columns:
    raise ValueError(
        f"'id' column not found in genes DataFrame. Available columns: {genes.columns.tolist()}"
    )

adata.var["gene_id"] = genes["id"].values
adata.var["gene_short_name"] = genes["gene_short_name"].values
adata.obs = mcf7_cells.reset_index(drop=True)
print(f"✓ AnnData created: {adata.n_obs} cells × {adata.n_vars} genes")

# 5. Save
out_file = ann_data_dir / "sciPlex3_MCF7_raw_counts.h5ad"
print(f"\nSaving to {out_file}...")
adata.write(out_file)

print(
    f"\n✅ Successfully saved {adata.n_obs} MCF7 cells × {adata.n_vars} genes to {out_file}"
)
print("\nTo force reload from original data files, run with: --force-reload")
