# MAGELLAN Configuration Guide

This guide documents the TOML configuration files used to control MAGELLAN's three main scripts:

- **`gen_shortest_path_net.py`**: Generate biological networks from pathway databases (e.g.Omnipath)
- **`train_bio.py`**: Train graph neural network models on biological data
- **`train_synthetic.py`**: Run synthetic benchmarks for model validation


## Overview

MAGELLAN uses TOML configuration files to specify all parameters for network generation, model training, and evaluation. 

The typical workflow is:

1. **Network Generation** (`gen_shortest_path_net.py`): Extract biological pathways from databases and construct network models
2. **Model Training** (`train_bio.py` or `train_synthetic.py`): Train GNN models to fit experimental specifications
3. **Evaluation**: Analyze model performance through metrics and visualizations

All configuration files follow the TOML format with sections denoted by `[section_name]` headers. Parameters are specified as `key = value` pairs with type-appropriate values (strings in quotes, numbers unquoted, booleans as `true`/`false`).

---

## 1. train_bio.py Configuration

Train graph neural network models on biological experimental data.

**Usage**: `uv run scripts/train_bio.py --config path/to/config.toml`

### [paths]

Specifies file locations and versioning for inputs and outputs.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `root_path` | string | Yes | - | Root directory for benchmarks (e.g., "benchmarks/bio_benchmarks") |
| `data_dir` | string | Yes | - | Main data directory within root_path (e.g., "data/example/") |
| `model_version` | string | Yes | - | Version identifier for the model directory |
| `model_path` | string | Yes | - | Path to JSON file containing network structure |
| `spec_version` | string | Yes | - | Version identifier for specification files |
| `spec_path` | string | Yes | - | Path to CSV file with experimental specifications |
| `spec_path_2` | string | No | - | Optional second specification file for dual-spec mode |
| `spec_version_2` | string | No | `spec_version` | Version for second specification (defaults to spec_version) |
| `output_prefix` | string | No | "" | Prefix for output directory naming |

**Note**: All paths are relative to `root_path`. The full path to the model is constructed as: `root_path/data_dir/model_version/model_path`.

### [model_params]

Defines model architecture and optimization parameters.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `min_range` | integer | Required | ≥ 0 | Minimum output value for network predictions. Recommended to leave at 0. |
| `max_range` | integer | Required | ≥ min_range | Maximum output value for network predictions |
| `filter_spec` | boolean | `true` | - | Filter specification to remove nodes not in graph |
| `learning_rate` | float | Required | > 0 | Initial learning rate for AdamW optimizer |
| `n_iter` | integer | Required | ≥ 1 | Number of graph convolutional layers (network depth). Recommended to of the same order as the number of nodes in the graph. |
| `edge_weight_init` | float | Required | Any | Initial value for all edge weights (ignored if `use_random_weight_init=true`). Recommended to use `0.5 <= edge_weight_init < 1.0`. |
| `use_random_weight_init` | boolean | `false` | - | Initialize edge weights from distribution instead of constant |
| `random_weight_init_distribution` | string | `"uniform"` | "uniform" | Distribution for random weight initialization |
| `random_weight_init_lower` | float | `0.0` | Any | Lower bound for random weight initialization |
| `random_weight_init_upper` | float | `2.0` | Any | Upper bound for random weight initialization |
| `max_update` | boolean | `true` | - | Restrict edge weight updates to maximum of 1.0 per step |
| `round_val` | boolean | `false` | - | Round predicted values to nearest integer |
| `check_paths` | boolean | `false` | - | Verify paths exist between perturbed and measured nodes |
| `allow_sign_flip` | boolean | `false` | - | Allow edge weights to become negative during training |
| `tf_method` | string | `"sum"` | "sum" | Method for combining edge weights in graph propagation |

**Weight Initialization**: When `use_random_weight_init=true`, edges are initialized to random values in `[random_weight_init_lower, random_weight_init_upper]` using the specified distribution.

### [training]

Controls the training loop, optimization, and data handling.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `epochs` | integer | `10000` | ≥ 1 | Maximum number of training epochs |
| `early_stopping_enabled` | boolean | `true` | - | Enable early stopping |
| `early_stopping_patience` | integer | `20` | ≥ 1 | Epochs to wait before stopping if no improvement |
| `class_weight_method` | string | `"inverse_freq"` | See below | Method for calculating node class weights |
| `node_class_weights_extreme_boost` | float | `1.0` | ≥ 0 | Multiplier to boost weight of extreme values (`0` and `max_range`) |
| `use_warmup` | boolean | `true` | - | Enable learning rate warmup phase |
| `warmup_steps` | integer | `10` | ≥ 1 | Number of epochs for learning rate warmup |
| `warmup_initial_lr_factor` | float | `0.1` | 0 to 1 | Learning rate factor at warmup start (`initial_lr = learning_rate × factor`) |
| `seed` | integer | `42` | Any | Random seed for reproducibility |
| `test_size` | float | `0.0` | 0.0 to 1.0 | Fraction of data for test set (`0.0` = no test split) |
| `dual_spec_mode` | string | None | See below | Mode for handling two specification files |
| `grad_clip_max_norm` | float | None | ≥ 0 or None | Gradient clipping threshold (None = no clipping) |
| `onpath_floor_lambda` | float | `0.0` | ≥ 0 | Strength of the on-path weight floor penalty, as a fraction of the base loss. `0.0` disables the feature. See On-Path Weight Floor below. |
| `onpath_floor_target` | float | `1.0` | ≥ 0 | Base target weight for on-path edges. The per-edge target is `onpath_floor_target / sqrt(in_degree(dst))`. Only used when `onpath_floor_lambda > 0`. |

**Class Weight Methods**:
- `"inverse_freq"`: Weight values inversely proportional to frequency; unobserved values get weight = max_range + 1
- `"inverse_freq_sparsity_stable"`: Only weight observed values; unobserved values get zero weight
- `"soft_inverse_freq"`: Baseline weight (0.1) for unobserved values, inverse frequency for observed
- `"balanced"`: Equal weight for all observed values
- `"no_weighting"`: Unit weights for all values

### [debug]

Debugging and diagnostic options.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_debug` | boolean | `false` | Print debug information about edge masking during training |

### Advanced Features

#### Dual Specification Mode

When `spec_path_2` is provided, `dual_spec_mode` controls how the two specification files are used for training and testing:

**Mode: `"separate"`**
- Spec 1 → All used for training
- Spec 2 → All used for testing
- `test_size` is ignored
- **Use case**: Completely separate hold-out test set

**Mode: `"split_second"`**
- Spec 1 → All used for training
- Spec 2 → Split based on `test_size`
  - `test_size = 0.0`: All of Spec 2 → training
  - `test_size = 1.0`: All of Spec 2 → testing (zero-shot evaluation)
  - `0.0 < test_size < 1.0`: Fraction of Spec 2 → testing, rest → training
- **Use case**: Combine datasets with selective test set allocation

**Mode: `"split_first"`**
- Spec 1 → Split based on `test_size`
- Spec 2 → All used for testing
  - `test_size = 0.0`: All Spec 1 → training, all Spec 2 → testing
  - `0.0 < test_size < 1.0`: Fraction of Spec 1 → testing (combined with all of Spec 2)
- **Use case**: Dynamic split of primary dataset with additional test data

**Single Spec Mode** (when `spec_path_2` is omitted):
- Single spec file split by `test_size`
- `test_size = 0.0`: All data for training only
- `0.0 < test_size < 1.0`: Random train/test split

#### Class Weighting Strategies

The `class_weight_method` parameter controls how prediction errors are weighted during training. This is particularly important for handling class imbalance in experimental data:

- **inverse_freq**: Strongly penalizes errors on rare values; suitable for highly imbalanced data
- **inverse_freq_sparsity_stable**: Ignores unobserved values; use when many theoretical values are never observed
- **soft_inverse_freq**: Moderate penalization with baseline weight for unobserved values
- **balanced**: Equal importance to all observed values regardless of frequency
- **no_weighting**: Standard unweighted loss; use when classes are balanced

The `node_class_weights_extreme_boost` parameter applies additional weight to extreme values (0 and max_range) when using frequency-based methods.

#### Weight Initialization Strategies

Edge weights can be initialized in two ways:

1. **Constant Initialization** (`use_random_weight_init=false`):
   - All edges start with `edge_weight_init` value
   - Provides consistent starting point
   - Recommended for most applications

2. **Random Initialization** (`use_random_weight_init=true`):
   - Edges initialized from `[random_weight_init_lower, random_weight_init_upper]`
   - Can help avoid local minima
   - Useful for exploring weight space

#### On-Path Weight Floor

The on-path weight floor adds an optional one-sided penalty that keeps
signal-carrying edges from decaying to zero during training. It is disabled by
default (`onpath_floor_lambda = 0.0`).

An edge is considered 'on-path' if it lies on at least one shortest path from a
perturbation source node to an observation (measured) node in the training data.
These are the edges that propagate the perturbation signal to phenotype nodes.

When enabled (`onpath_floor_lambda > 0`):

- Each on-path edge `u → v` is given a per-edge target
  `onpath_floor_target / sqrt(in_degree(v))`, so edges feeding high-fan-in nodes
  get a proportionally lower floor.
- A one-sided ReLU penalty is applied: only edges below their target are
  penalised, so already-strong edges are left unconstrained.
- The penalty uses Polyak-style scaling (`penalty × base_loss.detach()`), so
  `onpath_floor_lambda` is interpreted as a fraction of the base loss
  (e.g. `1.0` ≈ up to a 100% additional penalty scaling).

This is an advanced feature for networks where signal fails to propagate through
to phenotype nodes; leave it off unless you observe that behaviour.

---

## 2. train_synthetic.py Configuration

Run synthetic benchmarks for model validation and parameter tuning.

**Usage**: `uv run scripts/train_synthetic.py --config path/to/config.toml`

### [graph_generation]

Controls synthetic graph structure and experimental data generation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_base_graph` | boolean | Required | Generate a new synthetic graph |
| `grow_base_graph` | boolean | Required | Grow graph using layered structure |
| `base_graph_import` | string | Required | Path to import existing graph (used if generate_base_graph=false) |
| `generate_specifications` | boolean | Required | Generate synthetic experimental specifications |
| `spec_file` | string | Required | Path to import specifications (used if generate_specifications=false) |
| `filter_spec` | boolean | Required | Filter specification to match graph nodes |
| `n_nodes` | integer | Required | Number of nodes in synthetic graph (if generating) |
| `forward_edge_probability` | float | Required | Probability of edges between adjacent layers |
| `backward_edge_probability` | float | Required | Probability of edges to previous layers (feedback) |
| `skip_layer_probability` | float | Required | Probability of edges skipping layers |
| `n_spurious_edges` | integer | Required | Number of spurious (noise) edges to add |
| `inhibition_fraction` | float | Required | Fraction of edges that are inhibitory |
| `min_edge_weight` | float | Required | Minimum edge weight value |
| `max_edge_weight` | float | Required | Maximum edge weight value |
| `out_dir` | string | Required | Output directory for results |

### [specification_generation]

Controls generation of synthetic experimental data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_specifications` | integer | Required | Number of experimental specifications to generate |
| `terminal_node_bias` | float | Required | Bias toward selecting terminal (output) nodes for measurement |
| `noise_fraction` | float | Required | Fraction of measurements to corrupt with noise |
| `noise_std` | float | Required | Standard deviation of Gaussian noise added to corrupted measurements |
| `min_spec_inputs` | integer | Required | Minimum number of perturbed nodes per experiment |
| `max_spec_inputs` | integer | Required | Maximum number of perturbed nodes per experiment |
| `min_spec_outputs` | integer | Required | Minimum number of measured nodes per experiment |
| `max_spec_outputs` | integer | Required | Maximum number of measured nodes per experiment |
| `class_imbalance` | float | Required | Degree of class imbalance in generated data (0.0 = balanced) |

### [training]

Training configuration for synthetic benchmarks. Inherits most parameters from train_bio.py with additions:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| All train_bio.py [training] parameters | - | - | See train_bio.py training section |
| `device` | string | `"cpu"` | Device for training ("cpu" or "cuda") |
| `l1_lambda` | float | `0.0` | L1 regularization strength |
| `l2_lambda` | float | `0.0` | L2 regularization strength |
| `weight_decay` | float | `0.0001` | Weight decay for AdamW optimizer |
| `model_save_dir` | string | Required | Directory to save trained model checkpoints |
| `use_hybrid_loss` | boolean | `true` | Use hybrid loss combining multiple objectives |
| `hybrid_loss_alpha` | float | `1.0` | Weight for hybrid loss component |
| `use_class_weights` | boolean | `false` | Enable class weighting in loss function |
| `use_node_class_weights` | boolean | `true` | Enable node-specific class weights |
| `onpath_floor_lambda` | float | `0.0` | Strength of the on-path weight floor penalty (`0.0` disables). Same behaviour as train_bio.py — see [On-Path Weight Floor](#on-path-weight-floor). |
| `onpath_floor_target` | float | `1.0` | Base target weight for on-path edges (`target / sqrt(in_degree(dst))`). Only used when `onpath_floor_lambda > 0`. |

### [simulation]

Parameters for model simulation after training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_simulation_steps` | integer | `100` | Maximum number of simulation steps |
| `step_size` | float | `1.0` | Step size for simulation integration |
| `seed` | integer | `42` | Random seed for simulation |

### [evaluation]

Controls evaluation outputs and metrics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_confusion_plots` | boolean | `false` | Save confusion matrix plots |
| `save_binary_metrics_csv` | boolean | `false` | Save binary classification metrics to CSV |
| `weight_threshold` | float | `0.01` | Threshold for considering edges as present |

### Command-Line Overrides

The `train_synthetic.py` script supports command-line parameter overrides:

```bash
uv run scripts/train_synthetic.py --config config.toml --override training.learning_rate 0.01 training.epochs 2000
```

Override keys use dot notation to specify nested parameters (e.g., `section.parameter`). This allows testing parameter variations without modifying config files.

---

## 3. gen_shortest_path_net.py Configuration

Generate biological networks from pathway databases using shortest path algorithms.

**Usage**: `uv run scripts/gen_shortest_path_net.py --config path/to/config.toml`

### [DEFAULT]

Specifies directory structure for inputs and outputs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipe_dir` | string | Required | Pipeline directory (typically "scripts") |
| `data_dir` | string | Required | Data directory name |
| `shortest_path_data_dir` | string | Required | Subdirectory for shortest path data |
| `simulation_data_dir` | string | Required | Subdirectory for simulation data |
| `results_dir` | string | Required | Subdirectory for results |

### [shortest_path_settings]

Controls shortest path algorithm and pathway filtering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thre` | integer | Required | Maximum path length (number of genes in mutation-mutation or phenotype-phenotype paths, inclusive) |
| `cut_off` | boolean/tuple | `false` | Filter interactions by curation effort (e.g., `("curation_effort", 15)` for ≥15 curation effort) |
| `weighted_edge` | string | Required | Edge weighting method for Dijkstra's algorithm (e.g., "reciprocal_n_references") |
| `consensus_sign` | boolean | Required | Use consensus sign when interaction is both activation AND inhibition |
| `to_combine` | string | Required | Pathway linking strategy (e.g., "mut_deg_pheno" to link mutations → DEGs → phenotypes) |
| `filter_source_prime` | string | None | Only use interactions from specific source database (e.g., "SIGNOR") |
| `filter_source` | list | None | Only use interactions from at least one of the specified databases (e.g., `["HPRD", "SPIKE"]`) |
| `replace_gene` | dict | Required | Replace proteins with same gene names (e.g., `{Q8N726 = "CDKN2A_arf", P42771 = "CDKN2A_p16"}`) |
| `gene_replace_dict` | dict | Required | Replace gene names for standardization (e.g., `{gene_name = {ERBB1 = "EGFR"}}`) |
| `pheno_strategy` | string | `"none"` | Strategy for selecting the phenotype layer from DEG genes. In-script values: `"downstream"`, `"random"`, `"none"`.|
| `pheno_fraction` | float | `0.25` | Fraction of DEG genes to assign to the phenotype layer (used by `downstream`/`random`). |
| `pheno_min_primary_sources` | integer | `1` | Omnipath edge confidence threshold for building the distance graph (used by `downstream`). |
| `pheno_seed` | integer | `42` | Random seed for the `random` strategy. |

**Pathway Strategy (`to_combine`)**: Specifies how to link different node types:
- `"mut_deg_pheno"`: Connect mutations → differentially expressed genes → phenotypes
- Other strategies depend on node types in `gene_node_table_file`

**Edge Weighting (`weighted_edge`)**: Controls path selection in Dijkstra's algorithm:
- `"reciprocal_n_references"`: Prefer well-documented interactions (1 / number_of_references)
- `false`: Unweighted (all edges have equal cost)

**Phenotype Layer Selection (`pheno_strategy`)**: By default the phenotype layer
is exactly the genes marked as phenotypes in the gene node table. Setting
`pheno_strategy` instead narrows the phenotype layer to a fraction
(`pheno_fraction`) of the DEG genes using one of:

- `"downstream"`: Select the most downstream DEG genes by shortest-path distance
  from the mutation genes (genes unreachable from any mutation count as maximally
  downstream). Uses `pheno_min_primary_sources` to filter the Omnipath graph.
- `"random"`: Select a random subset of DEG genes (a control for network
  sparsity, independent of any selection criterion). Uses `pheno_seed`.
- `"none"` (default): Keep the phenotype genes from the gene node table.

### [shortest_path_input]

Specifies input data files.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_node_table_file` | string | Required | CSV file specifying gene nodes and their types (mutation, DEG, phenotype) |

**Gene Node Table Format**: CSV with columns indicating node names and types (e.g., `mutant`, `phenotype`, `key` for DEGs).

### [shortest_path_output]

Specifies output file names.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shortest_path_out_file_name` | string | Required | Base name for output JSON network file |

### [simulation_settings]

Configuration for BioModelAnalyzer (BMA) simulation (optional).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_step` | integer | Required | Number of time steps per simulation |
| `max_time_step` | integer | Required | Maximum simulation time steps |
| `bma_console` | string | Required | Path to BMA console executable (Windows) |

**Note**: BMA simulation is primarily supported on Windows systems.

### [simulation_input]

Input files for BMA simulation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file` | string | Required | JSON network file for simulation |
| `spec_file` | string | Required | CSV specification file for simulation |

### [simulation_output]

Output files for BMA simulation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_file` | string | Required | CSV file for simulation results |

---

## Example Configurations

The `example_configs/` directory contains several example configurations for common use cases:

### train_bio.py Examples

- **`example_train_bio_config.toml`**
  - Basic biological training configuration
  - Single specification file, no test split
  - Standard parameters for initial training

### train_synthetic.py Examples

- **`example_benchmark_config.toml`**
  - Synthetic benchmark using imported biological network
  - Generates synthetic specifications from network structure
  - Includes 10% test split for evaluation

### gen_shortest_path_net.py Examples

- **`example_shortest_path_config.toml`**
  - Generate network from Omnipath pathway database
  - Connect mutations → DEGs → phenotypes
  - Uses reciprocal reference count weighting
  - Maximum path length of 3 genes

---

## Additional Notes

### Parameter Interdependencies

1. **min_range and max_range**: Must satisfy `min_range ≤ max_range`. All experimental data must fall within this range.

2. **Learning rate and warmup**: Initial learning rate during warmup is `learning_rate × warmup_initial_lr_factor`, linearly increasing to `learning_rate` over `warmup_steps` epochs.

3. **Dual spec mode and test_size**: Behavior of `test_size` depends on `dual_spec_mode` setting. See Dual Specification Mode section for details.

4. **Random weight initialization**: When `use_random_weight_init=true`, `edge_weight_init` is ignored and `random_weight_init_*` parameters are used instead.

5. **Class weighting boost**: `node_class_weights_extreme_boost` only affects frequency-based methods (`inverse_freq`, `inverse_freq_sparsity_stable`, `soft_inverse_freq`).

### Output Structure

Training scripts create organized output directories:

- **train_bio.py**: `results/{model_type}/{prefix}_model_{name}_spec_{name}/`
  - Training metrics, predictions, and visualizations
  - `test/` subdirectory for test set results (if `test_size > 0`)

- **train_synthetic.py**: Specified by `out_dir` parameter
  - Benchmark results, edge structure metrics, and evaluation CSV

- **gen_shortest_path_net.py**: Specified by `results_dir` parameter
  - Generated network JSON file
  - Network visualization PNG

### Common Issues

- **Missing nodes**: If specification contains nodes not in graph, set `filter_spec=true` to automatically filter them
- **Path connectivity**: Enable `check_paths=true` to verify paths exist between perturbed and measured nodes
- **Sign inconsistency**: For networks with conflicting edge signs, set `consensus_sign=true` in gen_shortest_path_net.py
- **Convergence**: If training doesn't converge, try enabling `use_warmup`, increasing `warmup_steps`, or adjusting `learning_rate`
