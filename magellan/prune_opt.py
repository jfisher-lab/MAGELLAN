import os
from typing import Any

import pandas as pd
import torch
from tqdm.autonotebook import tqdm


# Add early stopping class before the training section
class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0,
        verbose: bool = True,
        enabled: bool = True,
        pbar: Any | None = None,
    ):
        """
        Initialize the EarlyStopping class.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement is seen.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, update progress bar with early stopping info.
            enabled (bool): If True, early stopping is enabled.
            pbar: Optional progress bar object that has a set_postfix method.
        """
        self.enabled = enabled
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.pbar = pbar
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_val_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if not self.enabled:
            return False

        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class ModelCheckpoint:
    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str | os.PathLike,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: bool = True,
        pbar: Any | None = None,
    ):
        """
        Save model checkpoints during training.

        Args:
            model: Model to save
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor ('val_loss', 'val_accuracy', etc.)
            mode: 'min' or 'max' (whether lower or higher values are better)
            save_best_only: Only save when monitored metric improves
            verbose: Whether to print checkpoint messages
            pbar: Optional progress bar to update with status
        """
        self.model = model
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.pbar = pbar

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Initialize best value based on mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1  # Use -1 to indicate no checkpoint saved yet
        self.best_state_dict = None  # Store the best state dict in memory

    def __call__(self, metrics: dict[str, float], epoch: int) -> None:
        """Check if model should be saved based on metrics."""
        if self.monitor not in metrics:
            raise ValueError(
                f"Metric '{self.monitor}' not found in metrics: {metrics.keys()}"
            )

        current_value = metrics[self.monitor]
        improved = False

        if self.mode == "min":
            improved = current_value < self.best_value
        elif self.mode == "max":
            improved = current_value > self.best_value
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {self.mode}")

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.save_checkpoint(epoch)
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

            if self.pbar:
                self.pbar.set_postfix(
                    **{
                        "best_value": f"{self.best_value:.6f}",
                        "best_epoch": self.best_epoch,
                    }
                )

    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint to file."""
        filepath = os.path.join(self.save_dir, f"model_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_value": self.best_value,
            },
            filepath,
        )

        # Also save as best model
        best_filepath = os.path.join(self.save_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_value": self.best_value,
            },
            best_filepath,
        )

        if self.verbose:
            print(
                f"Saved checkpoint at epoch {epoch} with {self.monitor}: {self.best_value:.6f}"
            )

    def load_best_model(self) -> None:
        """Load the best model back into the model object."""
        best_filepath = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(best_filepath):
            checkpoint = torch.load(best_filepath)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.verbose:
                print(
                    f"Loaded best model from epoch {checkpoint['epoch']} with {self.monitor}: {checkpoint['best_value']:.6f}"
                )
        else:
            print("No best model found. Using current model.")


def generate_curriculum_stages(
    category_curriculum: bool,
    phenotype_nodes_first: bool,
    total_epochs: int = 500,
    critical_weight: float = 5.0,
    base_weight: float = 1.0,
    include_extreme_values_stage: bool = False,
    critical_nodes: list[str] | None = None,
    extreme_values_weight: float | None = None,
) -> list[dict]:
    """
    Generate curriculum learning stages based on configuration parameters.

    Args:
        category_curriculum: Whether to use curriculum learning for categories
        phenotype_nodes_first: Whether to prioritize phenotype nodes in curriculum
        include_extreme_values_stage: Whether to include a stage focusing on extreme values
        critical_nodes: List of critical node names, defaults to ["Proliferation", "Apoptosis"]
        total_epochs: Total number of epochs to divide between stages
        critical_weight: Weight for critical nodes stage
        base_weight: Weight for base all-nodes stage
        extreme_values_weight: Weight for extreme values stage

    Returns:
        list of dictionaries containing curriculum stage configurations
    """
    if not category_curriculum and phenotype_nodes_first:
        raise ValueError(
            "If category curriculum is False, phenotype nodes first has no effect"
        )

    if critical_nodes is None:
        critical_nodes = ["Proliferation", "Apoptosis"]

    # Calculate epochs per stage based on the number of stages
    num_stages = 1
    if category_curriculum:
        num_stages += 1
    if include_extreme_values_stage:
        num_stages += 1

    epochs_per_stage = total_epochs // num_stages

    # Base stage for all nodes
    all_nodes_stage = {
        "name": "all_nodes",
        "epochs": epochs_per_stage,
        "nodes": None,  # None means all nodes
        "weight": base_weight,
        "focus_extreme_values": False,
    }

    # Stage focusing on extreme values (0 and 2)
    extreme_values_stage = {
        "name": "extreme_values",
        "epochs": epochs_per_stage,
        "nodes": None,  # Still all nodes, but with focus on extreme values
        "weight": base_weight,
        "focus_extreme_values": True,  # New flag to focus on extreme values
        "extreme_values_weight": extreme_values_weight,
    }

    # Stage focusing on critical nodes
    critical_nodes_stage = {
        "name": "critical_nodes",
        "epochs": epochs_per_stage,
        "nodes": critical_nodes,
        "weight": critical_weight,
        "focus_extreme_values": False,
    }

    if not category_curriculum and not include_extreme_values_stage:
        return [all_nodes_stage]

    if not category_curriculum and include_extreme_values_stage:
        return [all_nodes_stage, extreme_values_stage]

    # Build complete curriculum with all stages
    if include_extreme_values_stage:
        if extreme_values_weight is None:
            raise ValueError(
                "extreme_values_weight must be provided if include_extreme_values_stage is True"
            )
        if phenotype_nodes_first:
            return [critical_nodes_stage, extreme_values_stage, all_nodes_stage]
        else:
            return [all_nodes_stage, extreme_values_stage]
    else:
        if phenotype_nodes_first:
            return [critical_nodes_stage, all_nodes_stage]
        else:
            return [all_nodes_stage, critical_nodes_stage]



def get_curric_node_weights(
    y: torch.Tensor | int | float | None, stage: dict, node_dic: dict
) -> torch.Tensor:
    """Create weight tensor for loss calculation based on curriculum stage"""
    if y is None or isinstance(y, (int, float)):
        raise ValueError("Expected y to be a tensor, but got None or int/float")

    weights = torch.ones(y.shape[0])
    if stage["nodes"] is not None:
        # Set higher weights for critical nodes
        for node in stage["nodes"]:
            if node in node_dic:
                weights[node_dic[node]] = stage["weight"]
    return weights




def calculate_node_class_weights(
    y_data: pd.DataFrame,
    min_range: int,
    max_range: int,
    method: str = "inverse_freq",
    extreme_boost: float = 1.0,
) -> dict[str, list[float]]:
    """Calculate class weights for each node with value-aware weighting.

    Available methods:
    - inverse_freq: Original method that weights all possible values. This decreases the weight of nodes with sparse values in the spec (e.g. only ever 0) which may lead to overfitting to diverse valued nodes.
    - inverse_freq_sparsity_stable: Only weights observed values, preserving node importance, but this strongly penalises unobserved values.
    - soft_inverse_freq: Baseline weight for unobserved values, but increases weight for observed values.
    - balanced: weights as though there were no class imbalance.
    - no_weighting: Unit weights for all values.

    Args:
        y_data: DataFrame containing target values
        max_range: Maximum value in the range (0 to max_range, inclusive)
        method: Method to calculate weights
        extreme_boost: Factor to boost weights of extreme values (0 and max_range)

    Returns:
        Dictionary mapping node names to lists of weights for each possible value
    """
    if max_range < 0:
        raise ValueError("max_range must be non-negative")
    if min_range < 0:
        raise ValueError("min_range must be non-negative")
    if min_range > max_range:
        raise ValueError("min_range must be less than max_range")
    if not isinstance(y_data, pd.DataFrame):
        raise TypeError("y_data must be a pandas DataFrame")
    if y_data.max().max() > max_range or y_data.min().min() < min_range:
        raise ValueError(
            f"Data contains values outside valid range [{min_range}, {max_range}]"
        )

    node_weights = {}

    for node in tqdm(y_data.index, desc="Calculating node class weights", leave=False):
        node_values = y_data.loc[node]

        # Skip nodes with no data
        if node_values.isna().all():
            continue

        # Count occurrences of each value (0 to max_range)
        value_counts = {}
        observed_values = set()
        for i in range(min_range, max_range + 1):
            count = (node_values == i).sum()
            value_counts[i] = count
            if count > 0:
                observed_values.add(i)

        total = sum(value_counts.values())
        if total == 0:
            continue

        if method == "inverse_freq":
            # Original method - weights all possible values
            weights = []
            for i in range(min_range, max_range + 1):
                count = value_counts.get(i, 0)
                w = 1 / (count / total) if count > 0 else max_range + 1.0
                weights.append(w)

            # Extra normalization to emphasize extreme values
            middle_values = value_counts.get(max_range // 2, 0)
            extreme_values = value_counts.get(min_range, 0) + value_counts.get(
                max_range, 0
            )

            if extreme_values > 0 and middle_values > extreme_values:
                weights[min_range] *= extreme_boost
                weights[max_range - min_range] *= extreme_boost

            # Normalize weights
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

            node_weights[node] = weights

        elif method == "inverse_freq_sparsity_stable":
            # Initialize all weights to 0
            weights = [0.0] * (max_range - min_range + 1)

            # Calculate total observations of actual values
            total_observed = sum(
                count
                for value, count in value_counts.items()
                if value in observed_values
            )

            # Calculate inverse frequency weights only for observed values
            for value in observed_values:
                count = value_counts[value]
                weights[value] = 1 / (count / total_observed)

            # Apply extreme value boosting if both 0 and max_range are observed
            if 0 in observed_values and max_range in observed_values:
                middle_values = value_counts.get(max_range // 2, 0)
                extreme_values = value_counts.get(min_range, 0) + value_counts.get(
                    max_range, 0
                )

                if extreme_values > 0 and middle_values > extreme_values:
                    weights[min_range] *= extreme_boost
                    weights[max_range - min_range] *= extreme_boost

            # Normalize weights of observed values only
            weight_sum = sum(w for i, w in enumerate(weights) if i in observed_values)
            weights = [
                w / weight_sum if i in observed_values else 0.0
                for i, w in enumerate(weights)
            ]

            node_weights[node] = weights

        elif method == "soft_inverse_freq":
            # Small baseline weight for unobserved values
            baseline_weight = 0.1
            weights = [baseline_weight] * (max_range - min_range + 1)

            # Calculate weights for observed values
            total_observed = sum(value_counts.values())
            for value, count in value_counts.items():
                if count > 0:
                    weights[value] = 1 / (count / total_observed)

            # Apply extreme boosting if needed
            if 0 in observed_values and max_range in observed_values:
                middle_values = value_counts.get(max_range // 2, 0)
                extreme_values = value_counts.get(min_range, 0) + value_counts.get(
                    max_range, 0
                )
                if extreme_values > 0 and middle_values > extreme_values:
                    weights[0] *= extreme_boost
                    weights[max_range - min_range] *= extreme_boost

            # Normalize
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

            node_weights[node] = weights

        elif method == "balanced":
            # Equal weighting for all observed values
            weights = []
            for i in range(min_range, max_range + 1):
                weights.append(1.0 if value_counts.get(i, 0) > 0 else 0.0)

            # Normalize
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
            node_weights[node] = weights

        elif method == "no_weighting":
            node_weights[node] = [1.0] * (max_range - min_range + 1)
        else:
            raise ValueError(
                f"Invalid weight method: {method}. Must be one of: "
                "'inverse_freq', 'inverse_freq_sparsity_stable', 'balanced', 'no_weighting'"
            )

    return node_weights


def get_node_class_weight_tensor(
    target: torch.Tensor,
    pred: torch.Tensor,
    node_weights: dict[str, list[float]],
    node_names: list[str],
    min_range: int,
    max_range: int,
) -> torch.Tensor:
    """Create weight tensor based on value-specific node weights.

    Args:
        target: Target values tensor
        pred: Prediction tensor (for shape reference)
        node_weights: Dict mapping nodes to lists of weights
        node_names: List of node names in order matching tensor indices
        min_range: Minimum value in the range (0 to max_range, inclusive)
        max_range: Maximum value in the range (0 to max_range, inclusive)

    Returns:
        Tensor of weights for each target value
    """
    weights = torch.ones_like(pred, dtype=torch.float32)

    # add check that tensor is 0 or 1d
    if target.ndim > 1:
        raise ValueError("target tensor must be 0 or 1 dimensional")

    # Handle 0-dimensional tensor case (single value)
    if target.ndim == 0:
        if len(node_names) > 0:
            node = node_names[0]
            if node in node_weights:
                value = min(max(int(target.item()), min_range), max_range)
                weights = torch.tensor(node_weights[node][value])
        return weights

    # Handle multi-dimensional case
    for i in range(len(target)):
        if i < len(node_names):
            node = node_names[i]
            if node in node_weights:
                value = min(max(int(target[i].item()), min_range), max_range)
                weights[i] = node_weights[node][value]

    return weights


def weighted_node_earth_mover_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_weights: dict[str, list[float]],
    node_names: list[str],
    min_range: int,
    max_range: int,
    smooth_factor: float = 0.1,
) -> torch.Tensor:
    """Node-wise weighted Earth Mover's Distance (EMD) loss for ordinal data."""

    if max_range < 1:
        raise ValueError("max_range must be at least 1")
    if min_range < 0:
        raise ValueError("min_range must be non-negative")
    if min_range > max_range:
        raise ValueError("min_range must be less than max_range")
    if not 0 <= smooth_factor <= 1:
        raise ValueError("smooth_factor must be between 0 and 1")

    pred = pred.squeeze()
    target = target.squeeze()

    # Handle scalar/0-dim tensor case
    if pred.ndim == 0:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Get node-specific weights
    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Clamp predictions to valid range
    pred_clamped = torch.clamp(pred, min_range, max_range)

    # Convert predictions to proper probability distributions
    pred_probs = torch.zeros(
        pred.shape[0], max_range - min_range + 1, device=pred.device
    )
    for i in range(max_range - min_range + 1):
        # Calculate probability mass for each discrete value
        if i == 0:
            pred_probs[:, i] = torch.clamp(1.0 - pred_clamped, 0, 1)
        elif i == max_range - min_range:
            pred_probs[:, i] = torch.clamp(pred_clamped - (max_range - 1), 0, 1)
        else:
            pred_probs[:, i] = torch.clamp(1.0 - torch.abs(pred_clamped - i), 0, 1)

    # Normalize prediction probabilities
    pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)

    # More efficient probability distribution calculation
    # dists = torch.abs(
    #     pred_clamped.unsqueeze(-1) - torch.arange(max_range + 1, device=pred.device)
    # )
    # pred_probs = torch.softmax(-dists, dim=-1)

    # Convert targets to one-hot with smoothing
    target_dist = torch.zeros(
        target.shape[0], max_range - min_range + 1, device=pred.device
    )
    for i in range(max_range - min_range + 1):
        target_dist[:, i] = (target == i).float()

    if smooth_factor > 0:
        target_dist = target_dist * (1 - smooth_factor) + smooth_factor / (
            max_range - min_range + 1
        )

    # Calculate CDFs
    pred_cdf = torch.cumsum(pred_probs, dim=1)
    target_cdf = torch.cumsum(target_dist, dim=1)

    # Calculate EMD as weighted L1 distance between CDFs
    emd = torch.abs(pred_cdf - target_cdf).mean(dim=1)
    weighted_emd = weights * emd

    return weighted_emd.mean()


# Add after the scheduler definition and before the curriculum stages
class WarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        initial_lr: float,
        target_lr: float,
    ):
        """
        Args:
            optimizer: The optimizer to adjust learning rate for
            warmup_steps: Number of steps for warmup
            initial_lr: Starting learning rate
            target_lr: Target learning rate after warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_step = 0

    def step(self) -> None:
        """Perform a scheduler step"""
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (
                self.current_step / self.warmup_steps
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        self.current_step += 1


def node_weight_dict_to_df(node_class_weights: dict) -> pd.DataFrame:
    """Convert node class weights dictionary to a DataFrame with node names.

    Args:
        node_class_weights: Dictionary containing node weights in format
            {'node_name': [weight_0, weight_1, weight_2], ...}

    Returns:
        DataFrame with node names and their corresponding weights
    """
    # Create DataFrame directly from the dictionary
    df = pd.DataFrame.from_dict(
        node_class_weights,
        orient="index",
        columns=[
            f"weight_{i}" for i in range(len(next(iter(node_class_weights.values()))))
        ],
    )

    # Reset index and rename it to 'node'
    df = df.reset_index().rename(columns={"index": "node"})

    return df
