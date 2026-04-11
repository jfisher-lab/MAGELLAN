# %%
import argparse
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from magellan.gnn_model import (
    Net,
    extract_edge_signs,
    get_edge_weight_matrix,
    train_model,
)
from magellan.json_io import (
    get_const_from_json,
    get_pos_from_json,
    json_to_graph,
)
from magellan.onpath import _make_onpath_floor_wrapper, compute_onpath_edge_mask
from magellan.plot import (
    analyze_prediction_errors,
    analyze_predictions,
    create_annotation_matrix,
    get_filtered_indices,
    plot_classification_curves,
    plot_comparison_heatmaps,
    plot_difference_heatmaps,
    plot_loss_vs_validation_loss,
    plot_metrics_lollipop,
    plot_node_errors,
    plot_node_weights,
    plot_prediction_bias,
    plot_training_metrics,
    save_node_error_metrics,
    save_trained_network_and_visualise,
    visualize_error_by_node,
    write_metrics_to_txt,
)
from magellan.prune import (
    calculate_error_by_gene,
    # calculate_errors_by_type,
    check_dummy_nodes,
    check_no_self_loops,
    check_paths_in_graph,
    check_pert_dic_values,
    construct_mask_dic,
    create_edge_scale,
    create_pert_mask,
    create_pyg_data_object,
    dummy_setup,
    filter_spec_invalid_experiments,
    filter_specification,
    get_adjacency_matrix_mult,
    get_data,
    get_data_and_update_y,
    get_pert_dic,
    get_pert_list,
    get_real_indices,
    make_edge_idx,
    make_node_dic,
    make_pert_idx,
    predict_nn,
    round_df,
)
from magellan.prune_opt import (
    WarmupScheduler,
    calculate_node_class_weights,
    node_weight_dict_to_df,
    weighted_node_earth_mover_loss,
)
from magellan.pydot_io import graph_to_pydot
from magellan.sci_opt import pred_bound_perts

# %%

# Set up argument parser
parser = argparse.ArgumentParser(description="Run pruning with config file")
parser.add_argument(
    "--config",
    type=Path,
    default=Path(__file__).parent
    / Path("example_configs/example_train_bio_config.toml"),
    help="Path to config file (default: example_train_bio_config.toml in script directory)",
)
args = parser.parse_args()
config_path = args.config


def main(config_path: Path) -> tuple[dict, dict | None]:
    config = toml.load(config_path)

    local_path = Path(__file__).parent.parent

    # Set up paths
    root_path = local_path / Path(config["paths"]["root_path"])
    data_path = root_path / Path(config["paths"]["data_dir"])
    model_path = (
        data_path
        / Path(config["paths"]["model_version"])
        / Path(config["paths"]["model_path"])
    )
    spec_path = (
        data_path
        / Path(config["paths"]["spec_version"])
        / Path(config["paths"]["spec_path"])
    )

    # Check for second spec file
    spec_path_2 = None
    if "spec_path_2" in config["paths"]:
        spec_version_2 = config["paths"].get(
            "spec_version_2", config["paths"]["spec_version"]
        )
        spec_path_2 = (
            data_path / Path(spec_version_2) / Path(config["paths"]["spec_path_2"])
        )

    output_prefix = config["paths"].get("output_prefix", "")

    model_type = data_path.stem
    out_root_dir = root_path / "results" / Path(model_type)
    out_root_dir.mkdir(parents=True, exist_ok=True)

    # Input files
    input_json = model_path
    print("Loading model from:", input_json)
    spec_file = spec_path
    print("Loading spec from:", spec_file)

    spec_file_2 = None
    if spec_path_2 is not None:
        spec_file_2 = spec_path_2
        print("Loading second spec from:", spec_file_2)

    # Output directory
    prefix_str = f"{output_prefix}_" if output_prefix else ""
    if spec_file_2 is not None:
        out_dir = (
            out_root_dir
            / f"{prefix_str}model_{model_path.stem}_{config['paths']['model_version']}_spec_{spec_file.stem}_{config['paths']['spec_version']}_and_{spec_file_2.stem}"
        )
    else:
        out_dir = (
            out_root_dir
            / f"{prefix_str}model_{model_path.stem}_{config['paths']['model_version']}_spec_{spec_file.stem}_{config['paths']['spec_version']}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load parameters
    min_range = config["model_params"]["min_range"]
    max_range = config["model_params"]["max_range"]

    filter_spec = config["model_params"]["filter_spec"]
    learning_rate = config["model_params"]["learning_rate"]
    n_iter = config["model_params"]["n_iter"]
    edge_weight_init = config["model_params"]["edge_weight_init"]
    use_random_weight_init = config["model_params"].get("use_random_weight_init", False)
    random_weight_init_distribution = config["model_params"].get(
        "random_weight_init_distribution", "uniform"
    )
    random_weight_init_lower = config["model_params"].get(
        "random_weight_init_lower", 0.0
    )
    random_weight_init_upper = config["model_params"].get(
        "random_weight_init_upper", 2.0
    )
    max_update = config["model_params"]["max_update"]
    round_val = config["model_params"]["round_val"]
    check_paths = config["model_params"]["check_paths"]
    allow_sign_flip = config["model_params"]["allow_sign_flip"]

    # Training parameters
    tf_method = config["model_params"]["tf_method"]
    early_stopping_enabled = config["training"]["early_stopping_enabled"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    class_weight_method = config["training"]["class_weight_method"]
    node_class_weights_extreme_boost = config["training"].get(
        "node_class_weights_extreme_boost", 1.0
    )
    grad_clip_max_norm = config["training"].get("grad_clip_max_norm", None)
    weight_decay = config["training"].get("weight_decay", 0.01)
    # On-path weight floor (optional one-sided loss penalty). Disabled when
    # onpath_floor_lambda <= 0.
    onpath_floor_lambda = config["training"].get("onpath_floor_lambda", 0.0)
    onpath_floor_target = config["training"].get("onpath_floor_target", 1.0)
    # Debug options
    mask_debug = config["debug"]["mask_debug"]
    use_warmup = config["training"]["use_warmup"]
    warmup_steps = config["training"]["warmup_steps"]
    warmup_initial_lr_factor = config["training"]["warmup_initial_lr_factor"]
    seed = config["training"]["seed"]
    test_size = config["training"].get("test_size", 0.0)
    val_size = config["training"].get("val_size", 0.0)
    _y_missing_fill_value_raw = config["training"].get("y_missing_fill_value", 0.0)
    if _y_missing_fill_value_raw is None or (
        isinstance(_y_missing_fill_value_raw, float)
        and math.isnan(_y_missing_fill_value_raw)
    ):
        y_missing_fill_value: float | None = None
    else:
        y_missing_fill_value = float(_y_missing_fill_value_raw)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(out_dir)
    file_name = input_json.stem

    # %%

    G = json_to_graph(input_json)
    check_dummy_nodes(G)
    check_no_self_loops(G)
    const_dic = get_const_from_json(input_json)
    # node_list_original = sorted(G.nodes)

    # Save initial network visualization
    graph_to_pydot(G, out_dir / Path("network_before_training"), format="png")

    # Load first spec file
    pert_dic_small: OrderedDict = get_pert_dic(
        file_path=out_dir / Path(spec_file),
        const_dic=const_dic,
        spec_size="non full",
    )

    check_pert_dic_values(pert_dic_small, min_range, max_range)
    pert_dic_small = filter_spec_invalid_experiments(
        pert_dic_small, G, aggressive=False, verbose=True
    )

    # Load second spec file if provided
    pert_dic_small_2: OrderedDict | None = None
    if spec_file_2 is not None:
        pert_dic_small_2 = get_pert_dic(
            file_path=out_dir / Path(spec_file_2),
            const_dic=const_dic,
            spec_size="non full",
        )
        check_pert_dic_values(pert_dic_small_2, min_range, max_range)
        pert_dic_small_2 = filter_spec_invalid_experiments(
            pert_dic_small_2, G, aggressive=False, verbose=True
        )

    # check whether paths exist between perturbed nodes and experimental readings
    if check_paths:
        check_paths_in_graph(G, pert_dic_small)
        if pert_dic_small_2 is not None:
            check_paths_in_graph(G, pert_dic_small_2)

    graph_nodes = set(G.nodes())  # Use set for faster lookups
    if filter_spec:
        pert_dic_small = filter_specification(pert_dic_small, graph_nodes, verbose=True)
        if pert_dic_small_2 is not None:
            pert_dic_small_2 = filter_specification(
                pert_dic_small_2, graph_nodes, verbose=True
            )

    # Get dual spec mode from config
    dual_spec_mode = config["training"].get("dual_spec_mode", None)

    # Handle train/test split based on mode
    if dual_spec_mode is not None and pert_dic_small_2 is not None:
        # Dual spec file mode
        if dual_spec_mode == "separate":
            # spec_path = train, spec_path_2 = test (no splitting)
            train_pert_dic: OrderedDict = pert_dic_small
            val_pert_dic: OrderedDict | None = None
            test_pert_dic: OrderedDict | None = pert_dic_small_2
            print(
                f"Dual spec mode 'separate': {len(train_pert_dic)} train samples from spec 1, {len(test_pert_dic)} test samples from spec 2"
            )

        elif dual_spec_mode == "split_second":
            # spec_path = all train, split spec_path_2
            spec_keys_2 = list(pert_dic_small_2.keys())
            val_pert_dic = (
                None  # Not supported in this mode, use combined_split instead
            )
            if test_size == 1.0:
                # If test_size is 1.0, use all of spec 2 for testing (zero-shot)
                train_pert_dic = OrderedDict(pert_dic_small)
                test_pert_dic = OrderedDict(pert_dic_small_2)
                print(
                    f"Dual spec mode 'split_second' with test_size=1.0 (zero-shot): {len(train_pert_dic)} train samples from spec 1, {len(test_pert_dic)} test samples from spec 2"
                )
            elif test_size > 0.0:
                train_keys_2, test_keys_2 = train_test_split(
                    spec_keys_2, test_size=test_size, random_state=seed, shuffle=True
                )
                # Combine spec 1 (all) with train portion of spec 2
                train_pert_dic = OrderedDict(
                    {**pert_dic_small, **{k: pert_dic_small_2[k] for k in train_keys_2}}
                )
                test_pert_dic = OrderedDict(
                    {k: pert_dic_small_2[k] for k in test_keys_2}
                )
                print(
                    f"Dual spec mode 'split_second': {len(pert_dic_small)} from spec 1 + {len(train_keys_2)} from spec 2 = {len(train_pert_dic)} train samples, {len(test_pert_dic)} test samples"
                )
            else:
                # If test_size is 0, use all of spec 2 for training
                train_pert_dic = OrderedDict({**pert_dic_small, **pert_dic_small_2})
                test_pert_dic = None  # type: ignore
                print(
                    f"Dual spec mode 'split_second' with test_size=0: {len(train_pert_dic)} train samples, no test set"
                )

        elif dual_spec_mode == "split_first":
            # split spec_path, spec_path_2 = all test
            spec_keys_1 = list(pert_dic_small.keys())
            val_pert_dic = (
                None  # Not supported in this mode, use combined_split instead
            )
            if test_size > 0.0:
                train_keys_1, test_keys_1 = train_test_split(
                    spec_keys_1, test_size=test_size, random_state=seed, shuffle=True
                )
                train_pert_dic = OrderedDict(
                    {k: pert_dic_small[k] for k in train_keys_1}
                )
                # Combine test portion of spec 1 with all of spec 2
                test_pert_dic = OrderedDict(
                    {**{k: pert_dic_small[k] for k in test_keys_1}, **pert_dic_small_2}
                )
                print(
                    f"Dual spec mode 'split_first': {len(train_pert_dic)} train samples, {len(test_keys_1)} from spec 1 + {len(pert_dic_small_2)} from spec 2 = {len(test_pert_dic)} test samples"
                )
            else:
                # If test_size is 0, use all of spec 1 for training and spec 2 for testing
                train_pert_dic = pert_dic_small
                test_pert_dic = pert_dic_small_2
                print(
                    f"Dual spec mode 'split_first' with test_size=0: {len(train_pert_dic)} train samples, {len(test_pert_dic)} test samples"
                )

        elif dual_spec_mode == "combined_split":
            # Combine both specs, then split into train/val/test
            combined_spec = OrderedDict({**pert_dic_small, **pert_dic_small_2})
            combined_keys = list(combined_spec.keys())
            holdout_size = val_size + test_size

            if holdout_size > 0.0:
                # First split: train vs (val + test)
                train_keys, holdout_keys = train_test_split(
                    combined_keys,
                    test_size=holdout_size,
                    random_state=seed,
                    shuffle=True,
                )
                train_pert_dic = OrderedDict({k: combined_spec[k] for k in train_keys})

                # Second split: val vs test
                if val_size > 0.0 and test_size > 0.0:
                    test_proportion = test_size / holdout_size
                    val_keys, test_keys = train_test_split(
                        holdout_keys,
                        test_size=test_proportion,
                        random_state=seed,
                        shuffle=True,
                    )
                    val_pert_dic = OrderedDict({k: combined_spec[k] for k in val_keys})
                    test_pert_dic = OrderedDict(
                        {k: combined_spec[k] for k in test_keys}
                    )
                elif val_size > 0.0:
                    val_pert_dic = OrderedDict(
                        {k: combined_spec[k] for k in holdout_keys}
                    )
                    test_pert_dic = None  # type: ignore
                else:
                    val_pert_dic = None  # type: ignore
                    test_pert_dic = OrderedDict(
                        {k: combined_spec[k] for k in holdout_keys}
                    )

                print(
                    f"Dual spec mode 'combined_split': {len(train_pert_dic)} train, "
                    f"{len(val_pert_dic) if val_pert_dic else 0} val, "
                    f"{len(test_pert_dic) if test_pert_dic else 0} test samples"
                )
            else:
                train_pert_dic = combined_spec
                val_pert_dic = None  # type: ignore
                test_pert_dic = None  # type: ignore
                print(
                    f"Dual spec mode 'combined_split' with val_size=0 and test_size=0: {len(train_pert_dic)} train samples, no val/test set"
                )

        else:
            raise ValueError(
                f"Unknown dual_spec_mode: {dual_spec_mode}. Must be 'separate', 'split_second', 'split_first', or 'combined_split'"
            )

    else:
        # Single spec file mode (original behavior)
        spec_keys = list(pert_dic_small.keys())
        holdout_size = val_size + test_size
        if holdout_size > 0.0:
            # First split: train vs (val + test)
            train_keys, holdout_keys = train_test_split(
                spec_keys, test_size=holdout_size, random_state=seed, shuffle=True
            )

            # Second split: val vs test
            if val_size > 0.0 and test_size > 0.0:
                test_proportion = test_size / holdout_size
                val_keys, test_keys = train_test_split(
                    holdout_keys,
                    test_size=test_proportion,
                    random_state=seed,
                    shuffle=True,
                )
            elif val_size > 0.0:
                val_keys = holdout_keys
                test_keys = []
            else:
                val_keys = []
                test_keys = holdout_keys
            train_pert_dic = OrderedDict({k: pert_dic_small[k] for k in train_keys})
            val_pert_dic = (
                OrderedDict({k: pert_dic_small[k] for k in val_keys})
                if val_keys
                else None
            )
            test_pert_dic = (
                OrderedDict({k: pert_dic_small[k] for k in test_keys})
                if test_keys
                else None
            )
            print(
                f"Single spec mode: {len(train_pert_dic)} train, "
                f"{len(val_pert_dic) if val_pert_dic else 0} val, "
                f"{len(test_pert_dic) if test_pert_dic else 0} test samples"
            )
        else:
            train_pert_dic = pert_dic_small
            val_pert_dic = None  # type: ignore
            test_pert_dic = None  # type: ignore
            print(
                f"Single spec mode: {len(train_pert_dic)} train samples, no val/test set"
            )

    # Combine all spec dicts for unique node checking and dummy setup
    combined_pert_dic = dict(train_pert_dic)
    if val_pert_dic:
        combined_pert_dic.update(val_pert_dic)
    if test_pert_dic:
        combined_pert_dic.update(test_pert_dic)

    unique_nodes = set()
    for perturbation in combined_pert_dic.values():
        pert_nodes = perturbation[
            "pert"
        ].keys()  # Get the keys from the 'pert' dictionary
        unique_nodes.update(pert_nodes)  # Add keys to the set

    missing_nodes = unique_nodes - graph_nodes
    if missing_nodes:
        print(
            "Warning: The following nodes from the specification are not present in the graph G:"
        )
        for node in missing_nodes:
            print(f"- {node}")
    else:
        print("All nodes from the specification are present in the graph G.")

    G, inh, Adjacency_per_experiment_train, Adjacency_per_experiment_test = dummy_setup(
        G, combined_pert_dic, train_pert_dic, test_pert_dic, max_range, tf_method
    )

    # get adjacency matrix

    A_mult = get_adjacency_matrix_mult(G, method=tf_method)

    # Note that this function will fill missing values with `y_missing_fill_value` and include perturbations as expected values
    X_train, y_train = get_data_and_update_y(
        train_pert_dic, G, y_missing_fill_value=y_missing_fill_value
    )
    # Get data without filling missing values (NaN) and without including perturbations as expected values — required by class-weight computation
    _, y_no_zero_no_pert_as_expectation_train = get_data(
        train_pert_dic, G, y_missing_fill_value=None
    )

    # After loading X and y
    node_class_weights_train = calculate_node_class_weights(
        y_no_zero_no_pert_as_expectation_train,
        method=class_weight_method,
        min_range=min_range,
        max_range=max_range,
        extreme_boost=node_class_weights_extreme_boost,
    )
    if val_pert_dic:
        X_val, y_val = get_data_and_update_y(
            val_pert_dic, G, y_missing_fill_value=y_missing_fill_value
        )
        _, y_no_zero_no_pert_as_expectation_val = get_data(
            val_pert_dic, G, y_missing_fill_value=None
        )
        node_class_weights_val = calculate_node_class_weights(
            y_no_zero_no_pert_as_expectation_val,
            method=class_weight_method,
            min_range=min_range,
            max_range=max_range,
            extreme_boost=node_class_weights_extreme_boost,
        )
    else:
        X_val, y_val = None, None
        y_no_zero_no_pert_as_expectation_val = None
        node_class_weights_val = None

    if test_pert_dic:
        X_test, y_test = get_data_and_update_y(
            test_pert_dic, G, y_missing_fill_value=y_missing_fill_value
        )
        _, y_no_zero_no_pert_as_expectation_test = get_data(
            test_pert_dic, G, y_missing_fill_value=None
        )
        node_class_weights_test = calculate_node_class_weights(
            y_no_zero_no_pert_as_expectation_test,
            method=class_weight_method,
            min_range=min_range,
            max_range=max_range,
            extreme_boost=node_class_weights_extreme_boost,
        )
    else:
        X_test, y_test = None, None
        y_no_zero_no_pert_as_expectation_test = None
        node_class_weights_test = None

    ##########################
    # create edge attributes #
    ##########################

    pert_list = get_pert_list(pert_dic_small, inh)
    node_dic = make_node_dic(G)
    # node_dic_original = make_node_dic(node_list_original)
    pert_idx = make_pert_idx(pert_list, node_dic)
    edge_idx = make_edge_idx(A_mult, pert_idx)

    # create PyG data object using X and y and A_mult
    train_data = create_pyg_data_object(X_train, y_train, edge_idx)
    if X_val is not None and y_val is not None:
        val_data = create_pyg_data_object(X_val, y_val, edge_idx)
    else:
        val_data = None
    if X_test is not None and y_test is not None:
        test_data = create_pyg_data_object(X_test, y_test, edge_idx)
    else:
        test_data = None

    # extract edge scale
    edge_scale = create_edge_scale(A_mult, pert_idx)

    train_mask_dic = construct_mask_dic(train_pert_dic, node_dic, edge_idx, mask_debug)
    if val_pert_dic:
        val_mask_dic = construct_mask_dic(val_pert_dic, node_dic, edge_idx, mask_debug)
    else:
        val_mask_dic = None
    if test_pert_dic:
        test_mask_dic = construct_mask_dic(
            test_pert_dic, node_dic, edge_idx, mask_debug
        )
    else:
        test_mask_dic = None

    pert_mask = create_pert_mask(edge_idx, node_dic)

    edge_idx_original = (
        edge_idx.detach().clone()
    )  # this won't change edge_idx_original if edge_idx is changed

    # Initialize edge weights
    if use_random_weight_init:
        if random_weight_init_distribution == "uniform":
            edge_weight = (
                torch.rand(edge_idx.shape[1])
                * (random_weight_init_upper - random_weight_init_lower)
                + random_weight_init_lower
            )
        else:
            raise ValueError(
                f"Unsupported random weight initialization distribution: {random_weight_init_distribution}"
            )
    else:
        edge_weight = torch.ones(edge_idx.shape[1]) * edge_weight_init

    model = Net(
        edge_weight=edge_weight,
        min_val=min_range,
        max_val=max_range,
        n_iter=n_iter,
        max_update=max_update,
        round_val=round_val,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    warmup_scheduler = None
    if use_warmup:
        initial_lr = learning_rate * warmup_initial_lr_factor
        warmup_scheduler = WarmupScheduler(
            optimizer=opt,
            warmup_steps=warmup_steps,
            initial_lr=initial_lr,
            target_lr=learning_rate,
        )

    total_loss = []
    sum_grad = []

    edge_signs = extract_edge_signs(G, edge_idx)

    # Build the on-path weight floor loss wrapper when enabled. Pulls edges on
    # shortest paths from perturbations to observations up toward a per-edge
    # target onpath_floor_target / sqrt(in_degree(dst)).
    loss_fn = None
    if onpath_floor_lambda > 0.0:
        onpath_mask = compute_onpath_edge_mask(G, node_dic, edge_idx, train_pert_dic)
        loss_fn = _make_onpath_floor_wrapper(
            weighted_node_earth_mover_loss,
            model,
            onpath_mask,
            edge_idx,
            onpath_floor_lambda,
            onpath_floor_target,
        )
        print(
            f"On-path weight floor enabled: lambda={onpath_floor_lambda}, "
            f"target={onpath_floor_target} ({int(onpath_mask.sum())} on-path edges)"
        )

    total_loss, sum_grad, train_epoch_losses, val_epoch_losses, test_epoch_losses = (
        train_model(
            model=model,
            loss_fn=loss_fn,
            train_data=train_data,
            test_data=test_data,
            train_pert_dic=train_pert_dic,
            test_pert_dic=test_pert_dic,
            train_mask_dic=train_mask_dic,
            test_mask_dic=test_mask_dic,
            node_class_weights_train=node_class_weights_train,
            node_class_weights_test=node_class_weights_test,
            save_dir=out_dir,
            node_dic=node_dic,
            edge_idx_original=edge_idx_original,
            edge_scale=edge_scale,
            pert_mask=pert_mask,
            opt=opt,
            scheduler=scheduler,
            edge_signs=edge_signs,
            warmup_scheduler=warmup_scheduler,
            early_stopping_enabled=early_stopping_enabled,
            early_stopping_patience=early_stopping_patience,
            allow_sign_flip=allow_sign_flip,
            min_range=min_range,
            max_range=max_range,
            grad_clip_max_norm=grad_clip_max_norm,
            val_data=val_data,
            val_pert_dic=val_pert_dic,
            val_mask_dic=val_mask_dic,
            node_class_weights_val=node_class_weights_val,
        )
    )

    # Use validation losses for plotting if available, otherwise use test losses
    plot_loss_vs_validation_loss(
        epoch_losses=train_epoch_losses,
        test_losses=val_epoch_losses if val_epoch_losses else test_epoch_losses,
        out_dir=out_dir,
    )

    plot_training_metrics(
        total_loss=total_loss,
        sum_grad=sum_grad,
        n_experiments=len(pert_dic_small),
        output_path=out_dir / Path("loss_grad.png"),
    )

    W: pd.DataFrame = get_edge_weight_matrix(
        model=model,
        edge_idx_original=edge_idx_original,
        G=G,
        remove_dummy_and_self_loops=True,
    )

    pred_bound_perturbations_train = pred_bound_perts(
        X=X_train,
        y=y_train,
        W=W,
        Adjacency_per_experiment=Adjacency_per_experiment_train,
        G=G,
        pert_dic_all=train_pert_dic,
        time_step=n_iter,
        min_val=min_range,
        max_val=max_range,
        extract_exp=True,
    )

    pred_nn_train = predict_nn(
        model=model,
        y=y_train,
        data=train_data,
        pert_dic_small=train_pert_dic,
        A_mult=A_mult,
        pert_idx=pert_idx,
        mask_dic=train_mask_dic,
        edge_scale=edge_scale,
        pert_mask=pert_mask,
    )

    # round nn predictions to the nearest integers
    pred_nn_round_train = round_df(pred_nn_train)

    annotation_symbols = {"pert": "•", "exp": "-", "tst": ""}
    annot_train = create_annotation_matrix(
        base_df=y_train,
        perturbation_dict=train_pert_dic,
        annotation_symbols=annotation_symbols,
    )

    # Get indices excluding dummy nodes
    idx_real_train = get_real_indices(y_train)

    # Calculate errors for test and experimental nodes
    predictions_train = [
        (pred_nn_train, "GNN"),
        (pred_nn_round_train, "GNN (rounded)"),
        (pred_bound_perturbations_train, "BMA GNN"),
    ]

    pred_nn_train.to_csv(out_dir / Path("pred_nn_train.csv"))
    pred_nn_round_train.to_csv(out_dir / Path("pred_nn_round_train.csv"))
    pred_bound_perturbations_train.to_csv(
        out_dir / Path("pred_bound_perturbations_train.csv")
    )

    # Calculate per-gene errors
    rmse_by_gene_train, mae_by_gene_train = calculate_error_by_gene(
        y_train, pred_bound_perturbations_train, idx_real_train
    )

    # Filter indices
    filtered_idx_real_train = get_filtered_indices(annot_train, idx_real_train)

    # Plot difference heatmaps
    plot_difference_heatmaps(
        predictions=predictions_train,
        y=y_train,
        filtered_idx=filtered_idx_real_train,
        annot=annot_train,
        annotation_symbols=annotation_symbols,
        max_range=max_range,
        output_path=out_dir,
        shared_colorbar=True,
    )
    plot_difference_heatmaps(
        predictions=predictions_train,
        y=y_train,
        filtered_idx=filtered_idx_real_train,
        annot=annot_train,
        annotation_symbols=annotation_symbols,
        max_range=max_range,
        output_path=out_dir,
        shared_colorbar=False,
    )

    # Plot individual transposed heatmaps

    plot_comparison_heatmaps(
        [
            (pred_nn_round_train.loc[filtered_idx_real_train], "GNN (rounded)"),
            (pred_bound_perturbations_train.loc[filtered_idx_real_train], "BMA GNN"),
            (y_train.loc[filtered_idx_real_train], "True"),
        ],
        out_dir / Path("synthetic_non_full_spec_pred_vs_y.png"),
        cmap="Greens",
        vmin=min_range,
        vmax=max_range,
        center=None,  # type: ignore
        figsize=(15, 10),
        annot=annot_train.loc[filtered_idx_real_train],
        fmt="",
    )

    # Visualize errors on network
    G_visualise = json_to_graph(input_json)
    pos = get_pos_from_json(input_json)
    error_values_train = {"rmse": rmse_by_gene_train, "mae": mae_by_gene_train}
    visualize_error_by_node(
        G=G_visualise,
        error_values=error_values_train,
        pos=pos,
        const_dic=const_dic,
        min_range=min_range,
        max_range=max_range,
        output_path=out_dir,
    )

    save_node_error_metrics(G_visualise, rmse_by_gene_train, mae_by_gene_train, out_dir)

    save_trained_network_and_visualise(
        input_json=input_json,
        W=W,
        path_data=out_dir,
        file_name=file_name,
        min_range=min_range,
        max_range=max_range,
    )

    plot_node_errors(mae_by_gene_train, output_path=out_dir)

    error_fractions_continuous_train = analyze_prediction_errors(
        y_true=y_no_zero_no_pert_as_expectation_train.loc[filtered_idx_real_train],
        y_pred=pred_bound_perturbations_train.loc[filtered_idx_real_train],
        pert_dic=train_pert_dic,
        threshold=0.1,
        binary_mode=False,
    )

    error_fractions_binary_train = analyze_prediction_errors(
        y_true=y_no_zero_no_pert_as_expectation_train.loc[filtered_idx_real_train],
        y_pred=pred_bound_perturbations_train.loc[filtered_idx_real_train],
        pert_dic=train_pert_dic,
        threshold=0.1,
        binary_mode=True,
    )

    plot_prediction_bias(
        error_fractions_continuous_train,
        output_path=out_dir
        / Path("node_prediction_bias_fractions_continuous_train.png"),
    )
    plot_prediction_bias(
        error_fractions_binary_train,
        output_path=out_dir / Path("node_prediction_bias_fractions_binary_train.png"),
    )

    binary_metrics_train = analyze_predictions(
        y_true=y_no_zero_no_pert_as_expectation_train.loc[filtered_idx_real_train],
        y_pred=pred_bound_perturbations_train.loc[filtered_idx_real_train],
        path_data=out_dir / Path("binary_metrics"),
        save_figs=True,
        save_csv=True,
        plot_all_nodes=True,
        config=config,
        binary_mode=True,
        pert_dic=train_pert_dic,
    )

    print(f"binary f1: {binary_metrics_train['f1']}")
    print(f"binary mcc: {binary_metrics_train['mcc']}")

    nonbinary_metrics_train = analyze_predictions(
        y_true=y_no_zero_no_pert_as_expectation_train.loc[filtered_idx_real_train],
        y_pred=pred_bound_perturbations_train.loc[filtered_idx_real_train],
        path_data=out_dir / Path("nonbinary_metrics"),
        save_figs=True,
        save_csv=True,
        plot_all_nodes=True,
        config=config,
        binary_mode=False,
        pert_dic=train_pert_dic,
    )

    print(f"nonbinary f1: {nonbinary_metrics_train['f1']}")
    print(f"nonbinary mcc: {nonbinary_metrics_train['mcc']}")

    write_metrics_to_txt(
        binary_metrics_train,
        nonbinary_metrics_train,
        out_dir / Path("metrics_summary.txt"),
    )

    plot_metrics_lollipop(
        binary_metrics_train,
        out_dir / Path("binary_metrics_lollipop.png"),
        title="Binary Classification Metrics (Train)",
        metrics_to_plot=["accuracy", "f1", "mcc"],
    )

    plot_metrics_lollipop(
        nonbinary_metrics_train,
        out_dir / Path("nonbinary_metrics_lollipop.png"),
        title="Non-Binary Classification Metrics (Train)",
        metrics_to_plot=["accuracy", "f1", "mcc", "qwk"],
    )

    plot_classification_curves(
        y_true=y_no_zero_no_pert_as_expectation_train.loc[filtered_idx_real_train],
        y_pred=pred_bound_perturbations_train.loc[filtered_idx_real_train],
        pert_dic_small=train_pert_dic,
        path_data=out_dir / Path("nonbinary_metrics"),
        plot_all_nodes=False,
    )

    node_weight_df_train = node_weight_dict_to_df(node_class_weights_train)

    plot_node_weights(node_weight_df_train, out_dir)

    # Near the end of the function, before the final plotting
    metrics_train = {
        "train_binary_f1": binary_metrics_train["f1"],
        "train_binary_mcc": binary_metrics_train["mcc"],
        "train_nonbinary_f1": nonbinary_metrics_train["f1"],
        "train_nonbinary_mcc": nonbinary_metrics_train["mcc"],
        "train_nonbinary_qwk": nonbinary_metrics_train["qwk"],
    }

    if (
        test_pert_dic
        and X_test is not None
        and y_test is not None
        and test_data is not None
        and test_mask_dic is not None
        and node_class_weights_test is not None
        and y_no_zero_no_pert_as_expectation_test is not None
        and Adjacency_per_experiment_test is not None
    ):
        out_dir_test = out_dir / Path("test")
        Path(out_dir_test).mkdir(parents=True, exist_ok=True)
        pred_bound_perturbations_test = pred_bound_perts(
            X=X_test,
            y=y_test,
            W=W,
            Adjacency_per_experiment=Adjacency_per_experiment_test,
            G=G,
            pert_dic_all=test_pert_dic,
            time_step=n_iter,
            min_val=min_range,
            max_val=max_range,
            extract_exp=True,
        )
        pred_nn_test = predict_nn(
            model=model,
            y=y_test,
            data=test_data,
            pert_dic_small=test_pert_dic,
            A_mult=A_mult,
            pert_idx=pert_idx,
            mask_dic=test_mask_dic,
            edge_scale=edge_scale,
            pert_mask=pert_mask,
        )
        pred_nn_round_test = round_df(pred_nn_test)
        annot_test = create_annotation_matrix(
            base_df=y_test,
            perturbation_dict=test_pert_dic,
            annotation_symbols=annotation_symbols,
        )
        idx_real_test = get_real_indices(y_test)
        predictions_test = [
            (pred_nn_test, "GNN"),
            (pred_nn_round_test, "GNN (rounded)"),
            (pred_bound_perturbations_test, "BMA GNN"),
        ]
        pred_nn_test.to_csv(out_dir_test / Path("pred_nn_test.csv"))
        pred_nn_round_test.to_csv(out_dir_test / Path("pred_nn_round_test.csv"))
        pred_bound_perturbations_test.to_csv(
            out_dir_test / Path("pred_bound_perturbations_test.csv")
        )
        rmse_by_gene_test, mae_by_gene_test = calculate_error_by_gene(
            y_test, pred_bound_perturbations_test, idx_real_test
        )
        filtered_idx_real_test = get_filtered_indices(annot_test, idx_real_test)
        plot_difference_heatmaps(
            predictions=predictions_test,
            y=y_test,
            filtered_idx=filtered_idx_real_test,
            annot=annot_test,
            annotation_symbols=annotation_symbols,
            max_range=max_range,
            output_path=out_dir_test,
            shared_colorbar=True,
        )
        plot_difference_heatmaps(
            predictions=predictions_test,
            y=y_test,
            filtered_idx=filtered_idx_real_test,
            annot=annot_test,
            annotation_symbols=annotation_symbols,
            max_range=max_range,
            output_path=out_dir_test,
            shared_colorbar=False,
        )
        plot_comparison_heatmaps(
            [
                (pred_nn_round_test.loc[filtered_idx_real_test], "GNN (rounded)"),
                (pred_bound_perturbations_test.loc[filtered_idx_real_test], "BMA GNN"),
                (y_test.loc[filtered_idx_real_test], "True"),
            ],
            out_dir_test / Path("synthetic_non_full_spec_pred_vs_y_test.png"),
            cmap="Greens",
            vmin=min_range,
            vmax=max_range,
            center=None,  # type: ignore
            figsize=(15, 10),
            annot=annot_test.loc[filtered_idx_real_test],
            fmt="",
        )
        error_values_test = {"rmse": rmse_by_gene_test, "mae": mae_by_gene_test}
        visualize_error_by_node(
            G=G_visualise,
            error_values=error_values_test,
            pos=pos,
            const_dic=const_dic,
            min_range=min_range,
            max_range=max_range,
            output_path=out_dir_test,
        )
        save_node_error_metrics(
            G_visualise, rmse_by_gene_test, mae_by_gene_test, out_dir_test
        )
        plot_node_errors(mae_by_gene_test, output_path=out_dir_test)
        error_fractions_continuous_test = analyze_prediction_errors(
            y_true=y_no_zero_no_pert_as_expectation_test.loc[filtered_idx_real_test],
            y_pred=pred_bound_perturbations_test.loc[filtered_idx_real_test],
            pert_dic=test_pert_dic,
            threshold=0.1,
            binary_mode=False,
        )
        error_fractions_binary_test = analyze_prediction_errors(
            y_true=y_no_zero_no_pert_as_expectation_test.loc[filtered_idx_real_test],
            y_pred=pred_bound_perturbations_test.loc[filtered_idx_real_test],
            pert_dic=test_pert_dic,
            threshold=0.1,
            binary_mode=True,
        )
        plot_prediction_bias(
            error_fractions_continuous_test,
            output_path=out_dir_test
            / Path("node_prediction_bias_fractions_continuous_test.png"),
        )
        plot_prediction_bias(
            error_fractions_binary_test,
            output_path=out_dir_test
            / Path("node_prediction_bias_fractions_binary_test.png"),
        )
        binary_metrics_test = analyze_predictions(
            y_true=y_no_zero_no_pert_as_expectation_test.loc[filtered_idx_real_test],
            y_pred=pred_bound_perturbations_test.loc[filtered_idx_real_test],
            path_data=out_dir_test / Path("binary_metrics"),
            save_figs=True,
            save_csv=True,
            plot_all_nodes=True,
            config=config,
            binary_mode=True,
            pert_dic=test_pert_dic,
        )

        print(f"test binary f1: {binary_metrics_test['f1']}")
        print(f"test binary mcc: {binary_metrics_test['mcc']}")
        nonbinary_metrics_test = analyze_predictions(
            y_true=y_no_zero_no_pert_as_expectation_test.loc[filtered_idx_real_test],
            y_pred=pred_bound_perturbations_test.loc[filtered_idx_real_test],
            path_data=out_dir_test / Path("nonbinary_metrics"),
            save_figs=True,
            save_csv=True,
            plot_all_nodes=True,
            config=config,
            binary_mode=False,
            pert_dic=test_pert_dic,
        )

        print(f"test nonbinary f1: {nonbinary_metrics_test['f1']}")
        print(f"test nonbinary mcc: {nonbinary_metrics_test['mcc']}")
        write_metrics_to_txt(
            binary_metrics_test,
            nonbinary_metrics_test,
            out_dir_test / Path("metrics_summary_test.txt"),
        )
        plot_metrics_lollipop(
            binary_metrics_test,
            out_dir_test / Path("binary_metrics_lollipop.png"),
            title="Binary Classification Metrics (Test)",
            metrics_to_plot=["accuracy", "f1", "mcc"],
        )
        plot_metrics_lollipop(
            nonbinary_metrics_test,
            out_dir_test / Path("nonbinary_metrics_lollipop.png"),
            title="Non-Binary Classification Metrics (Test)",
            metrics_to_plot=["accuracy", "f1", "mcc", "qwk"],
        )
        plot_classification_curves(
            y_true=y_no_zero_no_pert_as_expectation_test.loc[filtered_idx_real_test],
            y_pred=pred_bound_perturbations_test.loc[filtered_idx_real_test],
            pert_dic_small=test_pert_dic,
            path_data=out_dir_test / Path("nonbinary_metrics"),
            plot_all_nodes=False,
        )
        node_weight_df_test = node_weight_dict_to_df(node_class_weights_test)
        plot_node_weights(node_weight_df_test, out_dir_test)
        metrics_test = {
            "test_binary_f1": binary_metrics_test["f1"],
            "test_binary_mcc": binary_metrics_test["mcc"],
            "test_nonbinary_f1": nonbinary_metrics_test["f1"],
            "test_nonbinary_mcc": nonbinary_metrics_test["mcc"],
            "test_nonbinary_qwk": nonbinary_metrics_test["qwk"],
        }
    else:
        predictions_test = None
        metrics_test = None

    # Create the 'predictions' subdirectory if it doesn't exist
    predictions_dir = os.path.join(out_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)  # exist_ok prevents error if dir exists

    # Save the `y` DataFrame as a CSV with row names (index)
    y_train.to_csv(
        os.path.join(predictions_dir, "real.csv"), index=True
    )  # save real.csv in main out_dir

    if y_test is not None:
        y_test.to_csv(
            os.path.join(predictions_dir, "real_test.csv"), index=True
        )  # save real.csv in main out_dir

    # Save the `predictions` list as CSV with row names (index)
    for _, (pred_df, name) in enumerate(predictions_train):
        filename = f"prediction_{name.lower().replace(' ', '_')}.csv"
        filepath = os.path.join(
            predictions_dir, filename
        )  # construct path inside predictions dir
        pred_df.to_csv(filepath, index=True)

    if predictions_test is not None:
        for _, (pred_df, name) in enumerate(predictions_test):
            filename = f"test_prediction_{name.lower().replace(' ', '_')}.csv"
            filepath = os.path.join(
                predictions_dir, filename
            )  # construct path inside predictions dir
            pred_df.to_csv(filepath, index=True)

    return metrics_train, metrics_test


if __name__ == "__main__":
    main(config_path)
