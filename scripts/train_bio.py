# %%
import argparse
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
from magellan.plot import (
    analyze_prediction_errors,
    analyze_predictions,
    create_annotation_matrix,
    get_filtered_indices,
    plot_classification_curves,
    plot_comparison_heatmaps,
    plot_difference_heatmaps,
    plot_loss_vs_validation_loss,
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
)
from magellan.pydot_io import graph_to_pydot
from magellan.sci_opt import pred_bound_perts

# %%

# Set up argument parser
parser = argparse.ArgumentParser(description="Run pruning with config file")
parser.add_argument(
    "--config",
    type=Path,
    default=Path(__file__).parent / Path("example_configs/example_train_bio_config.toml"),
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
    model_type = data_path.stem
    out_root_dir = root_path / "results" / Path(model_type)
    out_root_dir.mkdir(parents=True, exist_ok=True)

    # Input files
    input_json = model_path
    print("Loading model from:", input_json)
    spec_file = spec_path
    print("Loading spec from:", spec_file)

    # Output directory
    out_dir = (
        out_root_dir
        / f"model_{model_path.stem}_{config['paths']['model_version']}_spec_{spec_file.stem}_{config['paths']['spec_version']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load parameters
    min_range = config["model_params"]["min_range"]
    max_range = config["model_params"]["max_range"]

    filter_spec = config["model_params"]["filter_spec"]
    learning_rate = config["model_params"]["learning_rate"]
    n_iter = config["model_params"]["n_iter"]
    edge_weight_init = config["model_params"]["edge_weight_init"]
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
    # Debug options
    mask_debug = config["debug"]["mask_debug"]
    use_warmup = config["training"]["use_warmup"]
    warmup_steps = config["training"]["warmup_steps"]
    warmup_initial_lr_factor = config["training"]["warmup_initial_lr_factor"]
    seed = config["training"]["seed"]
    test_size = config["training"].get("test_size", 0.0)
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


    pert_dic_small: OrderedDict = get_pert_dic(
        file_path=out_dir / Path(spec_file),
        const_dic=const_dic,
        spec_size="non full",
    )

    check_pert_dic_values(pert_dic_small, min_range, max_range)
    pert_dic_small = filter_spec_invalid_experiments(
        pert_dic_small, G, aggressive=False, verbose=True
    )

    # check whether paths exist between perturbed nodes and experimental readings
    if check_paths:
        check_paths_in_graph(G, pert_dic_small)

    graph_nodes = set(G.nodes())  # Use set for faster lookups
    if filter_spec:
        pert_dic_small = filter_specification(pert_dic_small, graph_nodes, verbose=True)
    spec_keys = list(pert_dic_small.keys())
    if test_size > 0.0:
        train_keys, test_keys = train_test_split(
            spec_keys, test_size=test_size, random_state=seed, shuffle=True
        )
        train_pert_dic: OrderedDict = OrderedDict(
            {k: pert_dic_small[k] for k in train_keys}
        )
        test_pert_dic: OrderedDict = OrderedDict(
            {k: pert_dic_small[k] for k in test_keys}
        )
    else:
        train_pert_dic = pert_dic_small
        test_pert_dic = None  # type: ignore

    unique_nodes = set()
    for perturbation in pert_dic_small.values():
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
        G, pert_dic_small, train_pert_dic, test_pert_dic, max_range, tf_method
    )



    # get adjacency matrix

    A_mult = get_adjacency_matrix_mult(G, method=tf_method)

    # Note that this function will zero out missing values and include perturbations as expected values
    X_train, y_train = get_data_and_update_y(
        train_pert_dic, G, replace_missing_with_zero=True
    )
    # Get data without zeroing out missing values and without including perturbations as expected values
    _, y_no_zero_no_pert_as_expectation_train = get_data(
        train_pert_dic, G, y_replace_missing_with_zero=False
    )


    # After loading X and y
    node_class_weights_train = calculate_node_class_weights(
        y_no_zero_no_pert_as_expectation_train,
        method=class_weight_method,
        min_range=min_range,
        max_range=max_range,
        extreme_boost=node_class_weights_extreme_boost,
    )
    if test_pert_dic:
        X_test, y_test = get_data_and_update_y(
            test_pert_dic, G, replace_missing_with_zero=True
        )
        _, y_no_zero_no_pert_as_expectation_test = get_data(
            test_pert_dic, G, y_replace_missing_with_zero=False
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
    if X_test is not None and y_test is not None:
        test_data = create_pyg_data_object(X_test, y_test, edge_idx)
    else:
        test_data = None

    # extract edge scale
    edge_scale = create_edge_scale(A_mult, pert_idx)

    train_mask_dic = construct_mask_dic(train_pert_dic, node_dic, edge_idx, mask_debug)
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
    edge_weight = (
        torch.ones(edge_idx.shape[1]) * edge_weight_init
    )  


    model = Net(
        edge_weight=edge_weight,
        min_val=min_range,
        max_val=max_range,
        n_iter=n_iter,
        max_update=max_update,
        round_val=round_val,
    )
    
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
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

    total_loss, sum_grad, train_epoch_losses, test_epoch_losses = train_model(
        model=model,
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
    )

    plot_loss_vs_validation_loss(
        epoch_losses=train_epoch_losses,
        test_losses=test_epoch_losses,
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
