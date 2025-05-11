# sweep_runner.py
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import toml
from tqdm.autonotebook import tqdm


def deep_merge_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    Deeply merges dict2 into dict1.
    If a key exists in both and both values are dicts, they are merged recursively.
    Otherwise, the value from dict2 overwrites the value from dict1.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class SweepConfig:
    name: str
    description: str
    output_dir_suffix: str
    default_seeds: list[int]
    base_params: dict[str, Any]
    parameter_sets: list[dict[str, Any]]
    plotting_configs: list[dict[str, Any]]

    @classmethod
    def from_toml(cls, config_path: Path) -> "SweepConfig":
        """Load sweep configuration from TOML file"""
        with open(config_path, "r") as f:
            data = toml.load(f)

        return cls(
            name=data["name"],
            description=data["description"],
            output_dir_suffix=data["output_dir_suffix"],
            default_seeds=data["default_seeds"],
            base_params=data.get("base_params", {}),
            parameter_sets=data["parameter_sets"],
            plotting_configs=data.get("plotting_configs", []),
        )


def run_parameter_sweep(
    script_path: str | Path,
    base_config: str | Path,
    sweep_config: SweepConfig,
    seeds: list[int] | None = None,
    base_output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run a parameter sweep based on a sweep configuration.
    Returns combined DataFrame of all results.
    """
    script_path = Path(script_path)
    base_config = Path(base_config)

    # Validate files exist
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not base_config.exists():
        raise FileNotFoundError(f"Config not found: {base_config}")

    # Setup output directory
    base_output_dir = base_output_dir or Path("sweep_results")
    output_dir = base_output_dir / f"{sweep_config.output_dir_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup model save directory.
    # This modification will affect sweep_config.base_params for all subsequent parameter sets.
    model_save_dir_path = output_dir / "models"
    model_save_dir_path.mkdir(exist_ok=True)

    # Ensure 'training' key exists and is a dictionary in base_params before updating
    if "training" not in sweep_config.base_params or not isinstance(
        sweep_config.base_params.get("training"), dict
    ):
        sweep_config.base_params["training"] = {}
    sweep_config.base_params["training"]["model_save_dir"] = str(
        model_save_dir_path.resolve()
    )
    print(
        f"Model save directory set in sweep_config.base_params: {sweep_config.base_params['training']['model_save_dir']}"
    )

    # Use provided seeds or default from config
    seeds = seeds or sweep_config.default_seeds

    all_results = []

    for param_idx, params_from_set in tqdm(
        enumerate(sweep_config.parameter_sets),
        total=len(sweep_config.parameter_sets),
        desc="Parameter Sets",
    ):
        for seed in tqdm(seeds, total=len(seeds), leave=False, desc="Seeds"):
            # Perform a deep merge of base_params and the current parameter set
            # The resolved model_save_dir from base_params will be used.
            full_params = deep_merge_dicts(sweep_config.base_params, params_from_set)

            # Create override arguments
            override_args = ["--override"]
            for key, value in full_params.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        override_args.extend([f"{key}.{subkey}", str(subvalue)])
                else:
                    override_args.extend([key, str(value)])

            # Add seed override
            override_args.extend(["simulation.seed", str(seed)])

            # Add output directory override
            override_args.extend(
                ["graph_generation.out_dir", str(output_dir.resolve())]
            )

            # Run the experiment
            cmd = [
                "uv",
                "run",
                str(script_path),
                "--config",
                str(base_config),
                *override_args,
            ]

            try:
                _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"\n{'=' * 80}")
                print(f"ERROR RUNNING PARAMETER SET {param_idx} WITH SEED {seed}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Exit code: {e.returncode}")
                print(f"\nSTDOUT:\n{e.stdout}")
                print(f"\nSTDERR:\n{e.stderr}")
                print("\nPython traceback:")
                traceback.print_exc(file=sys.stdout)
                print(f"{'=' * 80}\n")
                raise

            # Read results
            results_path = output_dir / "evaluation_results.csv"
            if results_path.exists():
                df = pd.read_csv(results_path)
                # Add metadata
                df["parameter_set"] = param_idx
                df["seed"] = seed
                df["sweep_name"] = sweep_config.name
                all_results.append(df)

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"sweep_results_{timestamp}.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        return combined_df

    return pd.DataFrame()


def list_sweeps(sweeps_dir: Path) -> None:
    """List all available parameter sweeps"""
    print("\nAvailable parameter sweeps:")
    print("-" * 40)

    for config_file in sorted(sweeps_dir.glob("*.toml")):
        with open(config_file, "r") as f:
            data = toml.load(f)
        print(f"\n{data['name']}:")
        print(f"  Description: {data['description']}")
        print(f"  Config file: {config_file.name}")
