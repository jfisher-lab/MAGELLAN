from collections import OrderedDict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import SimpleConv
from tqdm.autonotebook import tqdm

from magellan.prune_opt import (
    EarlyStopping,
    ModelCheckpoint,
    WarmupScheduler,
    curriculum_weighted_loss_func,
    get_curric_node_weights,
    get_extreme_value_weights,
    hybrid_loss,
    weighted_hybrid_loss,
    weighted_node_earth_mover_loss,
)
from magellan.sci_opt import enforce_pert_dic_order, get_sorted_node_list


class Net(torch.nn.Module):
    """
    This class defines a neural network model for graph data. It inherits from the PyTorch's Module class.

    Attributes:
    edge_dim (int): The dimension of the edge features.
    edge_weight (torch.nn.Parameter): The weights of the edges in the graph.
    conv_layers (torch.nn.ModuleList): A list of convolutional layers.
    act (torch.nn.Hardtanh): The activation function used in the network.
    tanh (torch.nn.Hardtanh): The tanh function used for restricting the update.
    soft (torch.nn.Softmax): The softmax function used for classification.
    max_update (bool): A flag indicating whether to restrict the update to be max 1.
    round_val (bool): A flag indicating whether to round updated values to the nearest integer.
    """

    def __init__(
        self,
        edge_dim: int = 1,
        edge_weight: torch.Tensor | None = None,
        min_val: int = 0,
        max_val: int = 2,
        n_iter: int = 6,
        max_update: bool = False,
        round_val: bool = False,
    ):
        """
        The constructor for the Net class.

        Parameters:
        edge_dim (int, optional): The dimension of the edge features. Default is None.
        edge_weight (torch.nn.Parameter, optional): The weights of the edges in the graph. Default is None.
        min_val (int, optional): The minimum value for the Hardtanh activation function. Default is 0.
        max_val (int, optional): The maximum value for the Hardtanh activation function. Default is 2.
        n_iter (int, optional): The number of iterations (i.e., the number of convolutional layers). Default is 6.
        max_update (bool, optional): A flag indicating whether to restrict the update to be max 1. Default is False.
        round_val (bool, optional): A flag indicating whether to round updated values to the nearest integer. Default is False.
        """

        super(Net, self).__init__()

        if edge_weight is None:
            self.edge_weight = torch.nn.Parameter(torch.ones(edge_dim))
            self.edge_dim = edge_dim
        else:
            self.edge_weight = torch.nn.Parameter(edge_weight)
            self.edge_dim = self.edge_weight.shape[0]

        if not isinstance(n_iter, int) and isinstance(n_iter, float):
            if n_iter.is_integer():
                n_iter = int(n_iter)
            else:
                raise ValueError("n_iter must be an integer")

        self.conv_layers = torch.nn.ModuleList(
            [SimpleConv(aggr="sum") for i in range(1, n_iter)]
        )
        self.act = torch.nn.Hardtanh(min_val, max_val)
        self.tanh = torch.nn.Hardtanh(-1, 1)
        self.soft = torch.nn.Softmax()

        self.max_update = max_update
        self.round_val = round_val

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_scale: torch.Tensor,
        pert_mask: torch.Tensor,
        classify: bool = False,
    ):
        """
        The forward method for the Net class. This method defines the forward pass of the neural network.

        Parameters:
        x (torch.Tensor): The input features of the nodes in the graph.
        edge_index (torch.Tensor): The edge indices of the graph.
        edge_mask (torch.Tensor): The mask for the edges in the graph.
        edge_scale (torch.Tensor): The scale for the edges in the graph.
        pert_mask (torch.Tensor): The mask for the perturbed nodes in the graph.
        classify (bool, optional): A flag indicating whether to apply the softmax function for classification. Default is False.

        Returns:
        torch.Tensor: The output features of the nodes in the graph after the forward pass.
        """

        # mask edge index
        edge_index = edge_index.clone()  # Make local copy to avoid modifying inputs
        edge_mask = edge_mask.to(torch.int64)  # ensure correct type
        edge_index *= edge_mask  # NOTE: this will change all masked index to 0-->0, need to add a 0-indexed dummy node

        # set edge_weight for self loops to 1
        self.edge_weight.data = (
            self.edge_weight.data * (torch.ones(self.edge_dim) - pert_mask) + pert_mask
        )

        # scale edge with pre-defined edge_scale (to mimic avg)
        self.edge_weight.data = self.edge_weight.data * edge_scale

        # feed forward
        for conv in self.conv_layers:
            x_new = conv(x, edge_index, self.edge_weight)

            if self.max_update:  # restrict update to be max 1
                x_new = x + self.tanh(x_new - x)

            x = self.act(x_new)

            if self.round_val:  # round updated values to the nearest integer
                x = (x + 0.5).floor()

        if classify:
            # x = self.soft(x)
            raise NotImplementedError("Classification not implemented")

        self.edge_weight.data = self.edge_weight.data / edge_scale

        return x


def get_edge_weight_matrix(
    model: Net,
    edge_idx_original: torch.Tensor,
    G: nx.DiGraph,
    remove_dummy_and_self_loops: bool = False,
) -> pd.DataFrame:
    """Create weight matrix using fresh edge indices."""
    # initialize W as a zero matrix with the same shape as A_mult (adjacency matrix).
    # W = np.zeros_like(A_mult)
    node_list = get_sorted_node_list(G)
    W = np.zeros((len(node_list), len(node_list)), dtype=np.float64)
    edge_idx = edge_idx_original.detach().clone().cpu().numpy()

    for i in range(
        edge_idx.shape[1]
    ):  # Iterate over all the edges from the original edge_idx (as the one used in the training loop might be affected by edge_mask for a particular experiment).
        u, v = edge_idx[:, i]
        # if u != v: # note this already removes self loops
        W[v, u] = model.edge_weight.data[i].item()
    # For each edge, retrieve the source node u and destination node v. If u != v (i.e., the edge is not a self-loop), assign the corresponding learned weight to W[v, u]
    W = pd.DataFrame(W, index=node_list, columns=node_list)

    if remove_dummy_and_self_loops:
        for n in node_list:
            W.at[n, n] = 0  # set self loops to 0
            if "dummy" in n:
                W[n] = 0  # set dummy --> child nodes to 0
                W.loc[n] = 0  # set child nodes --> dummy to 0

    return W


def extract_edge_signs(G: nx.DiGraph, edge_idx: torch.Tensor) -> torch.Tensor:
    """
    Extract edge signs from the graph structure.

    Args:
        G: The directed graph
        edge_idx: Tensor of edge indices
        node_list: List of node names

    Returns:
        Tensor with values 1 for activator edges and -1 for inhibitor edges
    """
    node_list = get_sorted_node_list(G)
    edge_signs = torch.ones(edge_idx.shape[1], dtype=torch.float32)
    # node_idx_map = {node: idx for idx, node in enumerate(node_list)}

    edge_idx_np = edge_idx.cpu().numpy()

    for i in range(edge_idx.shape[1]):
        u_idx, v_idx = edge_idx_np[:, i]

        # Skip self-loops if they're not in the graph
        if u_idx == v_idx:
            continue

        # Convert indices to node names
        u = node_list[u_idx]
        v = node_list[v_idx]

        # If edge exists in graph, check its sign
        if G.has_edge(u, v):
            sign = G[u][v]["sign"]
            if sign == "Inhibitor":
                edge_signs[i] = -1

    return edge_signs


def train_model_flexible(
    model: Net,
    data: Data,
    pert_dic_small: dict,
    node_dic: dict,
    edge_idx_original: torch.Tensor,
    mask_dic: dict,
    edge_scale: torch.Tensor,
    pert_mask: torch.Tensor,
    curriculum_stages: list[dict],
    opt: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    edge_signs: torch.Tensor,
    max_range: int,
    min_range: int = 0,
    use_hybrid_loss: bool = False,
    boundary_penalty_alpha: float | None = None,
    warmup_scheduler: WarmupScheduler | None = None,
    hybrid_loss_alpha: float | None = None,
    use_class_weights: bool = False,
    use_node_class_weights: bool = False,
    class_weights: tuple[float, float] | None = None,
    node_class_weights: dict | None = None,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = 10,
    allow_sign_flip: bool = True,
    verbose: bool = True,
    grad_clip_max_norm: float | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Train the neural network model through curriculum stages.

    Args:
        model: Neural network model
        data: PyG Data object containing features and targets
        pert_dic_small: Dictionary of perturbations
        node_dic: Dictionary mapping node names to indices
        edge_idx_original: Original edge indices
        mask_dic: Dictionary of masks for each perturbation
        edge_scale: Edge scaling factors
        pert_mask: Perturbation mask
        curriculum_stages: List of curriculum learning stages
        opt: Optimizer
        scheduler: Learning rate scheduler
        warmup_scheduler: Optional warmup scheduler
        use_hybrid_loss: Whether to use hybrid loss
        hybrid_loss_alpha: Alpha parameter for hybrid loss
        use_class_weights: Whether to use class weights
        use_node_class_weights: Whether to use node-specific class weights
        class_weights: Class weights array
        node_class_weights: Node-specific class weights
        early_stopping_enabled: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        allow_sign_flip: Whether to allow negative weights

    Returns:
        Tuple containing:
        - List of total losses
        - List of gradient sums
        - List of epoch losses
    """
    raise NotImplementedError(
        "This function is was for comparing different methods and is deprecated"
    )

    total_loss = []
    sum_grad = []
    epoch_losses = []

    for stage in curriculum_stages:
        if verbose:
            print(f"\nStarting curriculum stage: {stage['name']}")
        node_weights = get_curric_node_weights(data.y, stage, node_dic)

        total_steps = stage["epochs"] * len(pert_dic_small)
        pert_items = list(
            enumerate(pert_dic_small.keys())
        )  # Keep index-key pairs together

        # Reset average loss for new stage
        avg_epoch_loss = float("inf")

        with tqdm(
            total=total_steps, desc=f"Training {stage['name']}", ncols=100, leave=False
        ) as pbar:
            # Create new early stopping instance for each stage
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=1e-4,
                enabled=early_stopping_enabled,
                pbar=pbar,
            )
            for i in range(stage["epochs"]):
                epoch_loss = []
                # Shuffle while maintaining index-key correspondence
                np.random.shuffle(pert_items)

                for (
                    j,
                    k,
                ) in pert_items:  # j maintains the original index for data.x/data.y
                    if warmup_scheduler is not None:
                        warmup_scheduler.step()

                    idx_exp = [node_dic[ele] for ele in pert_dic_small[k]["exp"]]
                    edge_idx_training = edge_idx_original.detach().clone()

                    if data.x is None:
                        raise ValueError("data.x is None")
                    input_features = data.x[:, j].reshape([-1, 1])

                    if not isinstance(data.y, torch.Tensor):
                        raise ValueError("data.y is not a tensor")
                    target = data.y[idx_exp, j].reshape([-1, 1])

                    pred = model(
                        x=input_features,
                        edge_index=edge_idx_training,
                        edge_mask=mask_dic[k],
                        edge_scale=edge_scale,
                        pert_mask=pert_mask,
                    )

                    opt.zero_grad()

                    if stage.get("focus_extreme_values", False):
                        # Get weights that emphasize extreme values
                        extreme_weights = get_extreme_value_weights(
                            target=target,
                            pred=pred[idx_exp],
                            min_range=min_range,
                            max_range=max_range,
                            middle_penalty=0.5,
                            extreme_boost=stage.get("extreme_values_weight", 3.0),
                        )

                        # Combine with existing weights
                        combined_weights = node_weights[idx_exp] * extreme_weights
                    else:
                        combined_weights = node_weights[idx_exp]

                    # Calculate loss based on configuration
                    if use_hybrid_loss:
                        if use_class_weights and not use_node_class_weights:
                            if class_weights is None:
                                raise ValueError(
                                    "class_weights is None but use_class_weights is True"
                                )
                            if hybrid_loss_alpha is None:
                                raise ValueError(
                                    "hybrid_loss_alpha is None but use_hybrid_loss is True"
                                )
                            loss = weighted_hybrid_loss(
                                pred=pred[idx_exp],
                                target=target,
                                weights=combined_weights,
                                alpha=hybrid_loss_alpha,
                                pos_weight=class_weights[1],
                                class_weights=class_weights,
                            )
                        elif use_node_class_weights:
                            if node_class_weights is None:
                                raise ValueError(
                                    "node_class_weights is None but use_node_class_weights is True"
                                )
                            exp_node_names = [
                                node for node, idx in node_dic.items() if idx in idx_exp
                            ]
                            loss = weighted_node_earth_mover_loss(
                                pred=pred[idx_exp],
                                target=target,
                                node_weights=node_class_weights,
                                node_names=exp_node_names,
                                min_range=min_range,
                                max_range=max_range,
                                # boundary_penalty_alpha=boundary_penalty_alpha,
                                # alpha=hybrid_loss_alpha,
                            )
                        else:
                            if hybrid_loss_alpha is None:
                                raise ValueError(
                                    "hybrid_loss_alpha is None but use_hybrid_loss is True"
                                )
                            loss = hybrid_loss(
                                pred=pred[idx_exp],
                                target=target,
                                alpha=hybrid_loss_alpha,
                            )
                    else:
                        loss = curriculum_weighted_loss_func(
                            pred=pred[idx_exp],
                            target=target,
                            weights=node_weights[idx_exp],
                        )

                    loss.backward()
                    # Apply gradient adjustments based on edge signs
                    for i, (sign) in enumerate(edge_signs):
                        if sign < 0:  # For inhibitory edges
                            if model.edge_weight.grad is not None:
                                # Invert the gradient direction for inhibitory edges
                                model.edge_weight.grad[i] = -model.edge_weight.grad[i]
                    if grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip_max_norm
                        )
                    opt.step()

                    if not allow_sign_flip:
                        with torch.no_grad():
                            for param in model.parameters():
                                param.data.clamp_(min=0)

                    loss_val = loss.detach().numpy()
                    total_loss.append(loss_val)
                    epoch_loss.append(loss_val)

                    if model.edge_weight.grad is None:
                        raise ValueError("model.edge_weight.grad is None")
                    sum_grad.append(model.edge_weight.grad.sum().numpy())

                    # Update progress bar with both loss and early stopping counter
                    postfix_dict = {
                        "loss": f"{loss_val:.6f}",
                        "early_stop": f"{early_stopping.counter + 1}/{early_stopping.patience}",
                    }
                    pbar.set_postfix(**postfix_dict)  # type: ignore
                    pbar.update(1)

                avg_epoch_loss = np.mean(epoch_loss)
                epoch_losses.append(avg_epoch_loss)
                print(f"Epoch {i + 1}, Average Loss: {avg_epoch_loss:.6f}")

                scheduler.step(avg_epoch_loss)

                early_stopping(float(avg_epoch_loss))
                if early_stopping.early_stop:
                    if verbose:
                        print(
                            f"\nEarly stopping triggered during {stage['name']} at epoch {i + 1}"
                        )
                    break

        if verbose:
            print("\nTraining complete.")
    return total_loss, sum_grad, epoch_losses


def train_model(
    model: Net,
    train_data: Data,
    test_data: Data | None,
    train_pert_dic: OrderedDict,
    test_pert_dic: OrderedDict | None,
    node_dic: dict,
    edge_idx_original: torch.Tensor,
    train_mask_dic: dict,
    test_mask_dic: dict | None,
    edge_scale: torch.Tensor,
    pert_mask: torch.Tensor,
    opt: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    edge_signs: torch.Tensor,
    max_range: int,
    node_class_weights_train: dict,
    node_class_weights_test: dict | None,
    save_dir: Path | str,
    min_range: int = 0,
    epochs: int = 10000,
    warmup_scheduler: WarmupScheduler | None = None,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = 10,
    allow_sign_flip: bool = True,
    verbose: bool = True,
    grad_clip_max_norm: float | None = None,
    warmup_steps: int = 100,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Train the neural network model through curriculum stages.

    Args:
        model: Neural network model
        data: PyG Data object containing features and targets
        pert_dic_small: Dictionary of perturbations
        node_dic: Dictionary mapping node names to indices
        edge_idx_original: Original edge indices
        train_mask_dic: Dictionary of masks for each perturbation
        test_mask_dic: Dictionary of masks for each perturbation
        edge_scale: Edge scaling factors
        pert_mask: Perturbation mask
        curriculum_stages: List of curriculum learning stages
        opt: Optimizer
        scheduler: Learning rate scheduler
        warmup_scheduler: Optional warmup scheduler
        use_hybrid_loss: Whether to use hybrid loss
        hybrid_loss_alpha: Alpha parameter for hybrid loss
        use_class_weights: Whether to use class weights
        use_node_class_weights: Whether to use node-specific class weights
        class_weights: Class weights array
        node_class_weights: Node-specific class weights
        early_stopping_enabled: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        allow_sign_flip: Whether to allow negative weights

    Returns:
        Tuple containing:
        - List of total losses
        - List of gradient sums
        - List of epoch losses
    """
    train_pert_dic_small = enforce_pert_dic_order(train_pert_dic)
    if test_pert_dic is not None:
        test_pert_dic_small = enforce_pert_dic_order(test_pert_dic)
    else:
        test_pert_dic_small = None  # type: ignore
    total_train_loss = []
    train_loss = []
    test_loss = []
    sum_grad = []
    train_epoch_losses = []
    test_epoch_losses = []
    # total_steps = epochs * len(train_pert_dic_small)
    train_pert_items = list(
        enumerate(train_pert_dic_small.keys())
    )  # Keep index-key pairs together
    if test_pert_dic_small is not None:
        test_pert_items = list(
            enumerate(test_pert_dic_small.keys())
        )  # Keep index-key pairs together
    else:
        test_pert_items = None  # type: ignore

    # Check if test set is empty
    no_test_set = test_pert_items is None or len(test_pert_items) == 0
    if verbose and no_test_set:
        print("No test set provided. Monitoring training loss for callbacks.")

    monitor_metric = "train_loss" if no_test_set else "val_loss"
    monitor_mode = "min"

    with tqdm(
        total=epochs,  # Change total to epochs as pbar updates once per epoch now
        desc="Training",
        ncols=100,
        leave=False,
        # unit_scale=False,
        # unit="",
        # bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]",
    ) as pbar:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-4,
            enabled=early_stopping_enabled,
            pbar=pbar,
        )
        model_checkpoint = ModelCheckpoint(
            model=model,
            save_dir=save_dir,
            monitor=monitor_metric,  # Use conditional monitor metric
            mode=monitor_mode,  # Use min mode for loss
            save_best_only=True,
        )

        for epoch in range(epochs):
            model.train()
            train_epoch_loss = []
            np.random.shuffle(train_pert_items)

            # Use a nested tqdm for batch progress within an epoch if needed
            # with tqdm(total=len(train_pert_items), desc=f"Epoch {epoch+1} Train", leave=False) as batch_pbar:
            if warmup_scheduler is not None:
                # Step warmup scheduler at the beginning of each epoch or step?
                # Stepping per epoch seems more common unless warmup_steps < len(train_pert_items)
                warmup_scheduler.step()  # Assuming step per epoch is desired

            for j, k in train_pert_items:
                idx_exp = [node_dic[ele] for ele in train_pert_dic_small[k]["exp"]]
                edge_idx_training = edge_idx_original.detach().clone()

                if train_data.x is None:
                    raise ValueError("data.x is None")
                input_features = train_data.x[:, j].reshape([-1, 1])

                if not isinstance(train_data.y, torch.Tensor):
                    raise ValueError("data.y is not a tensor")
                target = train_data.y[idx_exp, j].reshape([-1, 1])

                pred = model(
                    x=input_features,
                    edge_index=edge_idx_training,
                    edge_mask=train_mask_dic[k],
                    edge_scale=edge_scale,
                    pert_mask=pert_mask,
                )

                opt.zero_grad()

                exp_node_names = [
                    node for node, idx in node_dic.items() if idx in idx_exp
                ]
                train_loss = weighted_node_earth_mover_loss(
                    pred=pred[idx_exp],
                    target=target,
                    node_weights=node_class_weights_train,
                    node_names=exp_node_names,
                    min_range=min_range,
                    max_range=max_range,
                )

                train_loss.backward()
                # Apply gradient adjustments based on edge signs
                for i, (sign) in enumerate(edge_signs):
                    if sign < 0:  # For inhibitory edges
                        if model.edge_weight.grad is not None:
                            # Invert the gradient direction for inhibitory edges
                            model.edge_weight.grad[i] = -model.edge_weight.grad[i]
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip_max_norm
                    )
                opt.step()

                if not allow_sign_flip:
                    with torch.no_grad():
                        for param in model.parameters():
                            param.data.clamp_(min=0)

                train_loss_val = train_loss.detach().cpu().numpy()  # Ensure it's on CPU
                train_epoch_loss.append(train_loss_val)
                total_train_loss.append(train_loss_val)

                if model.edge_weight.grad is None:
                    raise ValueError("model.edge_weight.grad is None")
                sum_grad.append(
                    model.edge_weight.grad.sum().detach().cpu().numpy()
                )  # Ensure it's on CPU
                # batch_pbar.update(1) # Update batch pbar if using nested tqdm

            # Calculate epoch average loss
            avg_train_epoch_loss = np.mean(train_epoch_loss)
            train_epoch_losses.append(avg_train_epoch_loss)

            # Step the ReduceLROnPlateau scheduler with train loss
            scheduler.step(avg_train_epoch_loss)

            # --- Test Loop or Monitoring Logic ---
            avg_test_epoch_loss = np.nan  # Initialize as NaN

            if (
                not no_test_set
                and test_pert_items is not None
                and test_pert_dic_small is not None
                and test_mask_dic is not None
                and test_data is not None
                and node_class_weights_test is not None
            ):
                model.eval()
                test_epoch_loss = []
                with torch.no_grad():
                    for j, k in test_pert_items:
                        idx_exp = [
                            node_dic[ele] for ele in test_pert_dic_small[k]["exp"]
                        ]
                        edge_idx_testing = edge_idx_original.detach().clone()

                        if test_data.x is None:
                            raise ValueError(
                                "test_data.x is None"
                            )  # Check test_data specifically
                        input_features = test_data.x[:, j].reshape([-1, 1])

                        if not isinstance(test_data.y, torch.Tensor):
                            raise ValueError(
                                "test_data.y is not a tensor"
                            )  # Check test_data specifically
                        target = test_data.y[idx_exp, j].reshape([-1, 1])

                        # Get node names for the specific test experiment
                        exp_node_names_test = [  # Use a different name to avoid confusion
                            node for node, idx in node_dic.items() if idx in idx_exp
                        ]

                        pred = model(
                            x=input_features,
                            edge_index=edge_idx_testing,
                            edge_mask=test_mask_dic[k],
                            edge_scale=edge_scale,
                            pert_mask=pert_mask,
                        )

                        # Calculate test loss for this experiment
                        test_loss = weighted_node_earth_mover_loss(
                            pred=pred[idx_exp],
                            target=target,
                            node_weights=node_class_weights_test,  # Use test weights
                            node_names=exp_node_names_test,  # Use test node names
                            min_range=min_range,
                            max_range=max_range,
                        )

                        test_loss_val = (
                            test_loss.detach().cpu().numpy()
                        )  # Ensure it's on CPU
                        # Optional: print individual test experiment loss
                        # print(f"  Epoch {epoch + 1}, Test Exp {k}, Loss: {test_loss_val:.4f}")
                        test_epoch_loss.append(test_loss_val)

                avg_test_epoch_loss = (
                    np.mean(test_epoch_loss) if test_epoch_loss else np.nan
                )
                test_epoch_losses.append(avg_test_epoch_loss)
                # Set model back to train mode after evaluation
                model.train()
            elif no_test_set and node_class_weights_train is None:
                raise ValueError("node_class_weights_train is None")

            # --- Callbacks and Progress Bar Update ---
            # Determine the metric to use for callbacks and reporting
            current_metric_val = (
                avg_train_epoch_loss if no_test_set else avg_test_epoch_loss
            )

            # Checkpoint model (only if current_metric_val is not NaN)
            if epoch > warmup_steps and not np.isnan(current_metric_val):
                metrics_dict = {monitor_metric: float(current_metric_val)}
                model_checkpoint(metrics=metrics_dict, epoch=epoch)

            # Early stopping (only if current_metric_val is not NaN)
            if not np.isnan(current_metric_val):
                early_stopping(float(current_metric_val))
                if early_stopping.early_stop:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

            # Update progress bar for the epoch
            postfix_dict = {
                "avg train loss": f"{avg_train_epoch_loss:.6f}",
                "early stop": f"{early_stopping.counter + 1}/{early_stopping.patience}",
            }
            if not no_test_set:
                postfix_dict["avg test loss"] = f"{avg_test_epoch_loss:.6f}"

            pbar.set_postfix(**postfix_dict)  # type: ignore
            pbar.update(1)

    if verbose:
        print("\nTraining complete.")
    # Load the best model found during training (based on monitored metric)

    model_checkpoint.load_best_model()

    # Return losses. test_epoch_losses will be empty if no_test_set is True
    return total_train_loss, sum_grad, train_epoch_losses, test_epoch_losses
