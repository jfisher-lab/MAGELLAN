import pytest
import torch

from magellan.prune_opt import (
    weighted_node_earth_mover_loss,
)


@pytest.fixture
def sample_data():
    pred = torch.tensor([1.2, 0.8, 1.5], dtype=torch.float32)
    target = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)
    node_names = ["A", "B", "C"]
    node_weights = {"A": [0.2, 0.3, 0.5], "B": [0.3, 0.4, 0.3], "C": [0.1, 0.4, 0.5]}
    min_range = 0
    max_range = 2
    return pred, target, node_names, node_weights, min_range, max_range


def test_basic_functionality(sample_data):
    pred, target, node_names, node_weights, min_range, max_range = sample_data
    loss = weighted_node_earth_mover_loss(
        pred=pred,
        target=target,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0.0


def test_scalar_input():
    pred = torch.tensor(1.2)
    target = torch.tensor(1.0)
    node_names = ["A"]
    node_weights = {"A": [0.3, 0.4, 0.3]}
    min_range = 0
    max_range = 2

    loss = weighted_node_earth_mover_loss(
        pred=pred,
        target=target,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0.0


def test_perfect_prediction(sample_data):
    _, target, node_names, node_weights, min_range, max_range = sample_data
    loss = weighted_node_earth_mover_loss(
        pred=target,
        target=target,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )
    assert loss < 0.1  # Small due to label smoothing


def test_smoothing_effect(sample_data):
    pred, target, node_names, node_weights, min_range, max_range = sample_data

    # Compare losses with different smoothing factors
    loss_no_smooth = weighted_node_earth_mover_loss(
        pred=pred,
        target=target,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
        smooth_factor=0.0,
    )

    loss_with_smooth = weighted_node_earth_mover_loss(
        pred=pred,
        target=target,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
        smooth_factor=0.1,
    )

    assert loss_with_smooth < loss_no_smooth


def test_invalid_inputs(sample_data):
    pred, target, node_names, node_weights, min_range, max_range = sample_data

    # Test invalid max_range
    with pytest.raises(ValueError, match="max_range must be at least 1"):
        weighted_node_earth_mover_loss(
            pred=pred,
            target=target,
            node_weights=node_weights,
            node_names=node_names,
            min_range=min_range,
            max_range=0,
        )

    # Test invalid smooth_factor
    with pytest.raises(ValueError, match="smooth_factor must be between 0 and 1"):
        weighted_node_earth_mover_loss(
            pred=pred,
            target=target,
            node_weights=node_weights,
            node_names=node_names,
            min_range=min_range,
            max_range=max_range,
            smooth_factor=1.5,
        )


def test_extremal_predictions(sample_data):
    _, _, node_names, node_weights, min_range, max_range = sample_data

    # Test all zeros
    zeros = torch.zeros(3)
    loss_zeros = weighted_node_earth_mover_loss(
        pred=zeros,
        target=zeros,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )
    assert loss_zeros < 0.1  # Small due to label smoothing

    # Test all max values
    maxes = torch.ones(3) * max_range
    loss_maxes = weighted_node_earth_mover_loss(
        pred=maxes,
        target=maxes,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )
    assert loss_maxes < 0.1  # Small due to label smoothing


def test_out_of_range_predictions(sample_data):
    _, target, node_names, node_weights, min_range, max_range = sample_data

    # Test predictions below 0
    pred_below = torch.tensor([-1.0, -0.5, 0.0])
    loss_below = weighted_node_earth_mover_loss(
        pred=pred_below,
        target=target,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )
    assert not torch.isnan(loss_below)

    # Test predictions above max_range
    pred_above = torch.tensor([max_range + 1.0, max_range + 0.5, max_range])
    loss_above = weighted_node_earth_mover_loss(
        pred=pred_above,
        target=target,
        node_weights=node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )
    assert not torch.isnan(loss_above)
