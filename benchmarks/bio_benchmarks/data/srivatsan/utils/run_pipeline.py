#!/usr/bin/env python3
import subprocess
import tomllib
from pathlib import Path

# ==================== SETUP ====================

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "pipeline_config.toml"

print("=" * 42)
print("Srivatsan Data Processing Pipeline")
print("=" * 42)
print()

with open(CONFIG_FILE, "rb") as f:
    config = tomllib.load(f)

skip_existing = bool(config["general"]["skip_existing"])
print(f"Configuration loaded from: {CONFIG_FILE}")
print(f"Skip existing files: {skip_existing}")
print()

# ==================== HELPERS ====================


def run_step(name, output_files, command_args):
    """
    Run one pipeline step unless skip_existing is True
    and all output files already exist.
    """
    print("=" * 42)
    print(f"STEP: {name}")
    print("=" * 42)

    if skip_existing and all(Path(f).is_file() for f in output_files):
        print(f"⏭️  Outputs exist, skipping: {', '.join(output_files)}")
        print()
        return

    print(f"Running command:\n{' '.join(command_args)}\n")
    subprocess.run(command_args, check=True)
    print(f"✅ {name} complete\n")


def rel_path(path_str):
    """Convert TOML-relative paths to absolute paths relative to utils directory."""
    return str((SCRIPT_DIR / path_str).resolve())


# ==================== STEP 1 ====================

s1 = config["step_1"]
run_step(
    "Clean Srivatsan Table 5",
    [rel_path(s1["output_file"])],
    [
        "python3",
        str(SCRIPT_DIR / "1_clean_srivatsan_table_5.py"),
        "--input_file",
        rel_path(s1["input_file"]),
        "--output_file",
        rel_path(s1["output_file"]),
        "--chunksize",
        str(s1["chunksize"]),
    ],
)

# ==================== STEP 2 ====================

s2 = config["step_2"]
run_step(
    "Filter for MCF7 and BRCA genes",
    [rel_path(s2["processed_output_path"])],
    [
        "python3",
        str(SCRIPT_DIR / "2_filter_srivatsan_mcf7_brca.py"),
        "--s5_path",
        rel_path(s2["s5_path"]),
        "--supp_table_path",
        rel_path(s2["supp_table_path"]),
        "--json_path",
        rel_path(s2["json_path"]),
        "--joined_output_path",
        rel_path(s2["joined_output_path"]),
        "--processed_output_path",
        rel_path(s2["processed_output_path"]),
        "--cell_type",
        str(s2["cell_type"]),
    ],
)

# ==================== STEP 3 ====================

s3 = config["step_3"]
run_step(
    "Annotate with ChEMBL IDs",
    [rel_path(s3["cas_to_chembl_file"]), rel_path(s3["chembl_to_targets_file"])],
    [
        "python3",
        str(SCRIPT_DIR / "3_annotate_chembl_ids.py"),
        "--csv_file",
        rel_path(s3["csv_file"]),
        "--cas_to_chembl_file",
        rel_path(s3["cas_to_chembl_file"]),
        "--chembl_to_targets_file",
        rel_path(s3["chembl_to_targets_file"]),
        "--sleep_time",
        str(s3["sleep_time"]),
    ],
)

# ==================== STEP 4 ====================

s4 = config["step_4"]
run_step(
    "Convert to specification format",
    [rel_path(s4["output_file"])],
    [
        "python3",
        str(SCRIPT_DIR / "4_convert_to_spec_format.py"),
        "--input_file",
        rel_path(s4["input_file"]),
        "--output_file",
        rel_path(s4["output_file"]),
        "--cas_to_chembl_file",
        rel_path(s4["cas_to_chembl_file"]),
        "--chembl_to_targets_file",
        rel_path(s4["chembl_to_targets_file"]),
        "--brca_json_file",
        rel_path(s4["brca_json_file"]),
        "--literature_spec_file",
        rel_path(s4["literature_spec_file"]),
        "--brca_max_pert_level",
        str(s4["brca_max_pert_level"]),
    ],
)

# ==================== DONE ====================

print("=" * 42)
print("✅ Pipeline complete!")
print("=" * 42)
