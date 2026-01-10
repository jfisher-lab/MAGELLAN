from pathlib import Path

import fire
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from benchmarks.bio_benchmarks.data.srivatsan.utils.srivatsan_funcs import (
    combine_header_lines,
)


def main(
    input_file: str = str(Path(__file__).parent.parent / "Supplementary_Table_5.txt"),
    output_file: str = str(Path(__file__).parent.parent / "S5.parquet"),
    chunksize: int = 200_000
) -> None:
    """
    Clean and convert Srivatsan Supplementary Table 5 to Parquet format.

    Args:
        input_file: Path to input Supplementary_Table_5.txt file
        output_file: Path to output S5.parquet file
        chunksize: Number of rows to process per chunk
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    # Step 1: Read and combine header lines properly
    columns = combine_header_lines(input_path)
    print(f"Detected {len(columns)} columns: {columns}")

    # Step 2: Estimate total chunks (for progress bar only)
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8')) - 2
    total_chunks = total_lines // chunksize + 1
    print(f"Processing ~{total_lines:,} rows in {total_chunks} chunks")

    # Step 3: Create a generator for chunks
    # Using delim_whitespace handles multiple spaces/tabs as single delimiter
    # This prevents NA values encoded as spaces from creating extra columns
    chunks = pd.read_csv(
        input_path,
        sep=r"\s+",
        skiprows=2,
        names=columns,
        engine="python",
        chunksize=chunksize,
        dtype=str,                    # Read everything as strings to avoid casting errors
        on_bad_lines="skip",          # Skip malformed lines
        keep_default_na=True,         # Treat empty fields as NaN
        na_values=['', 'NA', 'na', 'N/A', 'n/a', 'NaN', 'nan'],  # Explicit NA values
    )

    # Step 4: Write safely with schema consistency
    first_chunk = True
    writer = None
    schema = None
    expected_columns = columns  # Keep original column order

    with tqdm(total=total_chunks, desc="Converting to Parquet", unit="chunk") as pbar:
        for chunk_idx, chunk in enumerate(chunks):
            # Check for column count mismatches
            if len(chunk.columns) != len(expected_columns):
                print(f"Chunk {chunk_idx}: Expected {len(expected_columns)} columns, got {len(chunk.columns)}")

                # Add missing columns
                for col in expected_columns:
                    if col not in chunk.columns:
                        chunk[col] = np.nan

                # Reorder to match expected columns (drop any extra columns)
                chunk = chunk[expected_columns]

            # Ensure all columns are present and in the correct order
            chunk = chunk.reindex(columns=expected_columns, fill_value=np.nan)

            if first_chunk:
                # Create schema from first chunk with all string types
                # This ensures consistent types even if some chunks have all-null columns
                schema = pa.schema([
                    (col, pa.string()) for col in expected_columns
                ])
                writer = pq.ParquetWriter(output_path, schema)
                first_chunk = False

            # Convert to Arrow table with the consistent schema
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)

            # Write the chunk
            if writer is not None:
                writer.write_table(table)
            pbar.update(1)

    if writer:
        writer.close()

    print(f"Finished converting to Parquet: {output_path}")
    print(f"Output file size: {output_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    fire.Fire(main)

