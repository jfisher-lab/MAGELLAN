# run_sweep.py
import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from magellan.benchmarking_utils import SweepConfig, list_sweeps, run_parameter_sweep
from magellan.plot import plot_metrics_vs_bias


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run parameter sweeps

    Run specific sweeps by name: python run_sweep.py --sweep terminal_node_bias_sweep
    Run multiple sweeps: python run_sweep.py --sweep sweep1 sweep2
    Run all sweeps: python run_sweep.py --all
    List available sweeps: python run_sweep.py --list
    Override seeds: python run_sweep.py --sweep sweep1 --seeds 1 2 3
    Override script/config paths: python run_sweep.py --sweep sweep1 --script path/to/script.py --config path/to/config.toml
    Plot sweep results: python run_sweep.py --plot-sweep sweep_name
    """
    default_output_dir = (
        Path(__file__).parent.parent
        / "benchmarks"
        / "synthetic_benchmarks"
        / "results"
        / "sweep_results"
    )
    Path(default_output_dir).mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run parameter sweeps")

    parser.add_argument(
        "--list", action="store_true", help="List available parameter sweeps"
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        help="Directory containing sweep configs",
        default=Path(__file__).parent.parent
        / "benchmarks"
        / "synthetic_benchmarks"
        / "configs"
        / "sweeps",
    )
    parser.add_argument("--sweep", nargs="+", help="Names of sweeps to run")

    parser.add_argument("--all", action="store_true", help="Run all available sweeps")

    parser.add_argument("--seeds", type=int, nargs="+", help="Override default seeds")

    parser.add_argument("--script", type=Path, help="Override path to main script")

    parser.add_argument("--config", type=Path, help="Override path to base config")

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base directory for sweep results",
        default=default_output_dir,
    )

    parser.add_argument(
        "--plot-sweep", type=str, help="Plot results for specified sweep"
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Plot all sweeps in the output directory",
    )

    args = parser.parse_args(argv)

    # Default paths
    default_script_path = (
        Path(__file__).parent.parent / "scripts" / "train_synthetic.py"
    )
    default_config_path = (
        Path(__file__).parent.parent
        / "scripts"
        / "example_configs"
        / "example_benchmark_config.toml"
    )

    script_path = args.script or default_script_path
    base_config = args.config or default_config_path

    sweeps_dir = args.sweep_dir

    if args.list:
        list_sweeps(sweeps_dir)
        return 0

    if args.plot_sweep:
        plot = plot_sweep_results(args.plot_sweep, sweeps_dir, args.output_dir)
        if plot is not None:
            return plot
        else:
            return 1

    if not args.sweep and not args.plot_all:
        parser.print_help()
        return 1

    # Get list of sweeps to run
    sweep_names = []
    if args.all or args.plot_all:
        sweep_names = [config_file.stem for config_file in sweeps_dir.glob("*.toml")]
        print(f"Identified sweeps: {sweep_names}")
    else:
        sweep_names = args.sweep

    if args.plot_all:
        print("Attempting to plot all sweeps...")
        if not sweep_names:
            print("No sweep configs found to plot.")
            return 1
        for sweep_name in sweep_names:
            print(f"Processing sweep for plotting: {sweep_name}")
            plot_sweep_results(sweep_name, sweeps_dir, args.output_dir)
        return 0

    # Run each requested sweep
    for sweep_name in sweep_names:
        config_path = sweeps_dir / f"{sweep_name}.toml"
        if not config_path.exists():
            print(f"Error: Sweep config not found: {config_path}")
            continue

        print(f"\nRunning sweep: {sweep_name}")
        sweep_config = SweepConfig.from_toml(config_path)

        try:
            run_parameter_sweep(
                script_path=script_path,
                base_config=base_config,
                sweep_config=sweep_config,
                seeds=args.seeds,
                base_output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"Error running sweep {sweep_name}: {e}")
            if not args.all:  # Only exit early if not running all sweeps
                return 1

    return 0


def plot_sweep_results(
    sweep_name: str, sweeps_dir: Path, output_dir: Path
) -> int | None:
    """
    Plot results for a specific sweep based on plotting configuration from the sweep's TOML file

    Args:
        sweep_name: Name of the sweep to plot
        sweeps_dir: Directory containing sweep TOML files
        output_dir: Base directory for sweep results and plots

    Returns:
        0 on success, 1 on error
    """
    # Load sweep config
    config_path = sweeps_dir / f"{sweep_name}.toml"
    if not config_path.exists():
        print(f"Error: Sweep config not found: {config_path}")
        return 1

    sweep_config = SweepConfig.from_toml(config_path)

    # Find the most recent results file
    sweep_result_dir = output_dir / sweep_config.output_dir_suffix
    results_files = list(sweep_result_dir.glob("sweep_results_*.csv"))
    if not results_files:
        print(f"Error: No results found for sweep {sweep_name}")
        return 1

    # Sort by modification time (newest first)
    results_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_results_file = results_files[0]

    print(f"Using results file: {latest_results_file}")

    # Load results
    df = pd.read_csv(latest_results_file)

    # Define the plotting directory
    plot_dir = sweep_result_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Check for plotting configuration in sweep config
    if hasattr(sweep_config, "plotting_configs"):
        # Multiple plot configurations
        if len(sweep_config.plotting_configs) == 0:
            raise ValueError(
                "No plotting configuration found in sweep config, using defaults."
            )
        plot_configs = sweep_config.plotting_configs
        for i, plot_config in enumerate(plot_configs):
            output_path = plot_dir / f"{sweep_name}_plot_{i + 1}.png"

            plot_metrics_vs_bias(
                df=df,
                output_path=output_path,
                x_axis_metric=plot_config.get(
                    "x_axis_metric",
                    "override_specification_generation.n_specifications",
                ),
                y_axis_metrics=plot_config.get(
                    "y_axis_metrics",
                    ["test_nonbinary_f1", "test_nonbinary_qwk", "test_nonbinary_mcc"],
                ),
                neat_x_axis_label=plot_config.get(
                    "neat_x_axis_label", "Specification Size"
                ),
                # figsize=tuple(plot_config.get("figsize", (10, 10))),
                y_min=plot_config.get("y_min", 0),
                y_max=plot_config.get("y_max", 1),
            )

            print(f"Plot saved to: {output_path}")
    else:
        # Use default configuration if no plotting config is found
        raise ValueError(
            "No plotting configuration found in sweep config, using defaults."
        )


if __name__ == "__main__":
    sys.exit(main())
