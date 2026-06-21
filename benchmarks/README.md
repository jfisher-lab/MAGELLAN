# Benchmarking

This directory contains the benchmarking code for MAGELLAN.

## Usage

### General Sweeps

To run general sweeps, run:

```bash
uv run scripts/benchmarking_sweeps.py --netw-config scripts/example_configs/example_benchmark_config.toml --sweep-configs-dir benchmarks/synthetic_benchmarks/configs/general_sweeps/ --all
```

To plot these results, run:

```bash
uv run scripts/benchmarking_sweeps.py --netw-config scripts/example_configs/example_benchmark_config.toml --sweep-configs-dir benchmarks/synthetic_benchmarks/configs/general_sweeps/ --plot-all
```

### Loop Sweeps

To generate synthetic networks with loops, run the `gen_loop_optimized_network.py` script.

```bash
uv run scripts/gen_loop_optimized_network.py
```

Then, to run loop sweepss, run:

```bash
uv run scripts/benchmarking_sweeps.py --netw-config scripts/example_configs/example_benchmark_config.toml --sweep-configs-dir benchmarks/synthetic_benchmarks/configs/loop_sweeps/ --all
```

To plot these results, run:

```bash
uv run scripts/benchmarking_sweeps.py --netw-config scripts/example_configs/example_benchmark_config.toml --sweep-configs-dir benchmarks/synthetic_benchmarks/configs/loop_sweeps/ --plot-all
```