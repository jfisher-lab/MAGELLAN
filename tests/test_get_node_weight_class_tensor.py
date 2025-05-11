import pytest
import torch

from magellan.prune_opt import get_node_class_weight_tensor


@pytest.fixture
def sample_node_weights():
    """Create sample node weights for testing."""
    return {
        "node1": [0.1, 0.2, 0.7],  # Weights for values 0, 1, 2
        "node2": [0.4, 0.3, 0.3],
        "node3": [0.6, 0.3, 0.1],
    }


@pytest.fixture
def sample_node_names():
    """Create sample node names for testing."""
    return ["node1", "node2", "node3"]


def test_basic_functionality(sample_node_weights, sample_node_names):
    """Test basic functionality with a simple tensor."""
    target = torch.tensor([0, 1, 2], dtype=torch.float32)
    pred = torch.ones_like(target)
    min_range = 0
    max_range = 2

    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=sample_node_weights,
        node_names=sample_node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Expected weights based on the values in target and node_weights
    expected = torch.tensor([0.1, 0.3, 0.1], dtype=torch.float32)

    assert weights.shape == pred.shape
    assert torch.allclose(weights, expected)


def test_out_of_range_values(sample_node_weights, sample_node_names):
    """Test handling of values outside the valid range."""
    target = torch.tensor([-1, 3, 2], dtype=torch.float32)
    pred = torch.ones_like(target)
    min_range = 0
    max_range = 2

    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=sample_node_weights,
        node_names=sample_node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Expected weights: -1 should be clamped to 0, 3 should be clamped to 2
    expected = torch.tensor([0.1, 0.3, 0.1], dtype=torch.float32)

    assert weights.shape == pred.shape
    assert torch.allclose(weights, expected)


def test_missing_node_weights(sample_node_weights):
    """Test handling of nodes not in the node_weights dictionary."""
    target = torch.tensor([0, 1], dtype=torch.float32)
    pred = torch.ones_like(target)
    min_range = 0
    max_range = 2
    node_names = ["node1", "unknown_node"]

    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=sample_node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Expected weights: node1 gets 0.1, unknown_node gets default weight 1.0
    expected = torch.tensor([0.1, 1.0], dtype=torch.float32)

    assert weights.shape == pred.shape
    assert torch.allclose(weights, expected)


def test_empty_node_names(sample_node_weights):
    """Test behavior with empty node names list."""
    target = torch.tensor([0, 1], dtype=torch.float32)
    pred = torch.ones_like(target)
    node_names = []
    min_range = 0
    max_range = 2

    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=sample_node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Expected weights: all nodes get default weight 1.0
    expected = torch.ones_like(pred)

    assert weights.shape == pred.shape
    assert torch.allclose(weights, expected)


def test_more_node_names_than_targets(sample_node_weights):
    """Test behavior when there are more node names than target elements."""
    target = torch.tensor([0], dtype=torch.float32)
    pred = torch.ones_like(target)
    node_names = ["node1", "node2", "node3"]
    min_range = 0
    max_range = 2

    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=sample_node_weights,
        node_names=node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Expected weights: only the first node weight is used
    expected = torch.tensor([0.1], dtype=torch.float32)

    assert weights.shape == pred.shape
    assert torch.allclose(weights, expected)


def test_scalar_tensor(sample_node_weights, sample_node_names):
    """Test handling of scalar tensors."""
    target = torch.tensor(1)  # scalar tensor
    pred = torch.tensor(1.0)  # scalar tensor
    min_range = 0
    max_range = 2

    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=sample_node_weights,
        node_names=sample_node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Expected: scalar tensor with the weight for node1, value 1
    expected = torch.tensor(0.2)

    assert weights.ndim == 0  # scalar tensor
    assert weights.item() == expected.item()


def test_different_shapes(sample_node_weights, sample_node_names):
    """Test behavior when pred and target have different shapes."""
    target = torch.tensor([0, 1, 2], dtype=torch.float32)
    pred = torch.ones((3, 1), dtype=torch.float32)  # Different shape than target
    min_range = 0
    max_range = 2

    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=sample_node_weights,
        node_names=sample_node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Expected weights: should match pred's shape
    assert weights.shape == pred.shape

    # Check first dimension values
    expected_values = torch.tensor([0.1, 0.3, 0.1], dtype=torch.float32).reshape(-1, 1)
    assert torch.allclose(weights, expected_values)


def test_empty_targets_and_preds(sample_node_weights, sample_node_names):
    """Test behavior with empty tensors."""
    target = torch.tensor([], dtype=torch.float32)
    pred = torch.tensor([], dtype=torch.float32)
    min_range = 0
    max_range = 2

    weights = get_node_class_weight_tensor(
        target=target,
        pred=pred,
        node_weights=sample_node_weights,
        node_names=sample_node_names,
        min_range=min_range,
        max_range=max_range,
    )

    # Expected: empty tensor with same shape as pred
    assert weights.shape == pred.shape
    assert len(weights) == 0


# def test_2d_tensors(sample_node_weights, sample_node_names):
#     """Test behavior with 2D tensors."""
#     target = torch.tensor([[0, 1], [2, 0]], dtype=torch.float32)
#     pred = torch.ones_like(target)
#     max_range = 2

#     # With 2D tensors, the function should treat the first dimension as the batch dimension
#     # and apply weights accordingly
#     weights = get_node_class_weight_tensor(
#         target=target,
#         pred=pred,
#         node_weights=sample_node_weights,
#         node_names=sample_node_names,
#         max_range=max_range,
#     )

#     # Expecting the original behavior - applying weights based on first dimension only
#     # First dimension (0) gets weights for node1, node2, with values 0, 2
#     expected = torch.ones_like(pred)
#     expected[0, 0] = 0.1  # node1, value 0
#     expected[1, 0] = 0.1  # node3, value 2 (if index in range of node_names)

#     assert weights.shape == pred.shape
#     assert torch.allclose(weights[:2, 0], torch.tensor([0.1, 0.1]))
