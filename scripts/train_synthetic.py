import argparse
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import toml

from magellan.synthetic import (
    PruningTestConfig,
    # compare_configs,
    run_pruning_benchmark,
)

FLATTEN_SECTIONS = [
    "graph_generation",
    "specification_generation",
    "training",
    "simulation",
    "evaluation",
]


def get_valid_config_keys(config: dict[str, Any], prefix: str = "") -> set[str]:
    """Recursively get all valid configuration keys in dot notation."""
    valid_keys = set()
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key
        valid_keys.add(full_key)
        if isinstance(value, dict):
            nested_keys = get_valid_config_keys(value, f"{full_key}.")
            valid_keys.update(nested_keys)
    return valid_keys


def parse_overrides(arg_values: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """Parse key value pairs for overrides."""
    overrides = {}
    valid_keys = get_valid_config_keys(config)

    # Process args in pairs
    idx = 0
    while idx < len(arg_values):
        key = arg_values[idx]
        if idx + 1 >= len(arg_values):
            raise ValueError(f"Missing value for override key: {key}")

        # Validate that the override key exists in config
        if key not in valid_keys:
            raise ValueError(
                f"Invalid override key: '{key}'. Key not found in configuration."
            )

        value = arg_values[idx + 1]
        overrides[key] = value
        idx += 2

    return overrides


def set_nested_config(config: dict[str, Any], key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation."""
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Try to convert value to int/float/bool if possible
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
    d[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        set_nested_config(config, key, value)
        # If the key is sectioned (e.g., training.epochs), also set at top level if in FLATTEN_SECTIONS
        if "." in key:
            section, subkey = key.split(".", 1)
            if section in FLATTEN_SECTIONS:
                config[subkey] = config[section][subkey]


def flatten_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested override keys for storage."""
    return {f"override_{k}": v for k, v in overrides.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--override",
        nargs="+",
        help="Override config values: key value [key value ...]",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="baseline_noise_synthetic_config.toml",
        help="Path to the config file to use",
    )
    args = parser.parse_args()

    config_path = args.config
    config_dict = toml.load(config_path)

    if args.override:
        if len(args.override) % 2 != 0:
            raise ValueError("Overrides must be in key value pairs.")
        # Update parse_overrides call to include config_dict
        overrides = parse_overrides(args.override, config_dict)
        apply_overrides(config_dict, overrides)

    # Save the modified config dict to a temporary TOML file
    with tempfile.NamedTemporaryFile("w+", suffix=".toml", delete=False) as tmpfile:
        toml.dump(config_dict, tmpfile)
        tmpfile.flush()
        config = PruningTestConfig.from_toml(tmpfile.name)

    # compare to current main config
    main_config_path = Path(__file__).parent / "prune_config.toml"
    # compare_configs(config_path, main_config_path, "synthetic_config", "prune_config")

    evaluation_result = run_pruning_benchmark(config, verbose=True)

    # Print results
    print("\nFinal Results:")
    print(f"True edges removed: {evaluation_result.edge_stats.true_removed:.1%}")
    print(
        f"Spurious edges removed: {evaluation_result.edge_stats.spurious_removed:.1%}"
    )
    print(f"Train F1 Score: {evaluation_result.train_binary_metrics['f1']:.4f}")
    print(f"Train Accuracy: {evaluation_result.train_binary_metrics['accuracy']:.4f}")
    print(f"Train Precision: {evaluation_result.train_binary_metrics['precision']:.4f}")
    print(f"Train Recall: {evaluation_result.train_binary_metrics['recall']:.4f}")
    print(f"Train MCC: {evaluation_result.train_binary_metrics['mcc']:.4f}")
    print(
        f"Total Predictions: {evaluation_result.train_binary_metrics['total_predictions']}"
    )
    if config.n_spurious_edges > 0:
        print(
            f"Edge Structure Accuracy: {evaluation_result.edge_stats.edge_accuracy:.4f}"
        )
        print(
            f"Edge Structure Precision: {evaluation_result.edge_stats.edge_precision:.4f}"
        )
        print(f"Edge Structure Recall: {evaluation_result.edge_stats.edge_recall:.4f}")
        print(
            f"Edge Structure F1: {evaluation_result.edge_stats.edge_structure_f1:.4f}"
        )
        print(
            f"Edge Structure MCC: {evaluation_result.edge_stats.edge_structure_mcc:.4f}"
        )
        print(
            f"Edge Structure Kappa: {evaluation_result.edge_stats.edge_structure_qwk:.4f}"
        )
    print("=" * 100)
    print(f"Train Nonbinary F1: {evaluation_result.train_nonbinary_metrics['f1']:.4f}")
    print(
        f"Train Nonbinary MCC: {evaluation_result.train_nonbinary_metrics['mcc']:.4f}"
    )
    print(
        f"Train Nonbinary Kappa: {evaluation_result.train_nonbinary_metrics['qwk']:.4f}"
    )
    print("=" * 100)

    # Print test metrics only if they exist
    if (
        evaluation_result.test_nonbinary_metrics
        and evaluation_result.test_binary_metrics
    ):
        print("=" * 100)
        print(
            f"Test Nonbinary F1: {evaluation_result.test_nonbinary_metrics['f1']:.4f}"
        )
        print(
            f"Test Nonbinary MCC: {evaluation_result.test_nonbinary_metrics['mcc']:.4f}"
        )
        print(
            f"Test Nonbinary Kappa: {evaluation_result.test_nonbinary_metrics['qwk']:.4f}"
        )
        print("=" * 100)
        print(f"Test Binary F1: {evaluation_result.test_binary_metrics['f1']:.4f}")
        print(f"Test Binary MCC: {evaluation_result.test_binary_metrics['mcc']:.4f}")
        print(f"Test Binary Kappa: {evaluation_result.test_binary_metrics['qwk']:.4f}")

    # Save evaluation results to CSV
    # Add overrides to results_dict if present
    override_info = flatten_overrides(overrides) if args.override else {}

    results_dict = {
        # Edge structure metrics
        "spurious_edges_removed": evaluation_result.edge_stats.spurious_removed,
        "true_edges_removed": evaluation_result.edge_stats.true_removed,
        "edge_accuracy": evaluation_result.edge_stats.edge_accuracy,
        "edge_precision": evaluation_result.edge_stats.edge_precision,
        "edge_recall": evaluation_result.edge_stats.edge_recall,
        "edge_structure_f1": evaluation_result.edge_stats.edge_structure_f1,
        "edge_structure_mcc": evaluation_result.edge_stats.edge_structure_mcc,
        "edge_structure_qwk": evaluation_result.edge_stats.edge_structure_qwk,
        # Training binary metrics
        **{
            f"train_binary_{k}": v
            for k, v in evaluation_result.train_binary_metrics.items()
        },
        # Training nonbinary metrics
        **{
            f"train_nonbinary_{k}": v
            for k, v in evaluation_result.train_nonbinary_metrics.items()
        },
        # Test metrics - add only if present
        **(
            {
                f"test_binary_{k}": v
                for k, v in evaluation_result.test_binary_metrics.items()
            }
            if evaluation_result.test_binary_metrics
            else {}
        ),
        **(
            {
                f"test_nonbinary_{k}": v
                for k, v in evaluation_result.test_nonbinary_metrics.items()
            }
            if evaluation_result.test_nonbinary_metrics
            else {}
        ),
        # Add override info
        **override_info,
        # Specification statistics
        **{f"spec_stats_{k}": v for k, v in evaluation_result.spec_stats.items()},
    }

    # Convert to DataFrame and save
    results_df = pd.DataFrame([results_dict])
    local_path = Path(__file__).parent.parent
    out_path = local_path / Path(config.out_dir) / "evaluation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
