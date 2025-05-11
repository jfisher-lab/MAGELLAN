import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from magellan.gnn_model import Net
from magellan.prune import predict_nn


@pytest.fixture
def mock_data():
    # Create sample input data
    y = pd.DataFrame(
        {"exp1": [0, 1, 2], "exp2": [2, 1, 0]}, index=["node1", "node2", "node3"]
    )

    # Create x with shape [num_nodes, num_experiments]
    x = torch.tensor([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    # Add y to the Data object
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    data = Data(x=x, y=y_tensor)

    pert_dic_small = {
        "exp1": {"pert": {"node1": 1}, "exp": {"node3": 2}},
        "exp2": {"pert": {"node2": 1}, "exp": {"node1": 2}},
    }

    A_mult = pd.DataFrame(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        index=["node1", "node2", "node3"],
        columns=["node1", "node2", "node3"],
    )

    pert_idx = [0, 1]

    mask_dic = {"exp1": torch.ones(5), "exp2": torch.ones(5)}

    edge_scale = torch.ones(5)
    pert_mask = torch.zeros(5)

    return y, data, pert_dic_small, A_mult, pert_idx, mask_dic, edge_scale, pert_mask


def test_predict_nn_basic(mock_data):
    """Test basic functionality of predict_nn"""
    y, data, pert_dic_small, A_mult, pert_idx, mask_dic, edge_scale, pert_mask = (
        mock_data
    )

    class MockNet(Net):
        def forward(self, x, edge_index, edge_mask, edge_scale, pert_mask):
            # Return fixed values matching input dimensions
            return torch.ones(3, 1) * x[0]

    model = MockNet(edge_dim=5)
    pred = predict_nn(
        model,
        y,
        data,
        pert_dic_small,
        A_mult,
        pert_idx,
        mask_dic,
        edge_scale,
        pert_mask,
    )

    assert pred.shape == y.shape
    assert list(pred.index) == list(y.index)
    assert list(pred.columns) == list(y.columns)


def test_predict_nn_none_x():
    """Test that predict_nn raises error when data.x is None"""
    data = Data(x=None)
    with pytest.raises(ValueError, match="data.x is None"):
        predict_nn(None, None, data, None, None, None, None, None, None)  # type: ignore


def test_predict_nn_values(mock_data):
    """Test that predict_nn produces expected values"""
    y, data, pert_dic_small, A_mult, pert_idx, mask_dic, edge_scale, pert_mask = (
        mock_data
    )

    class MockNet(Net):
        def forward(self, x, edge_index, edge_mask, edge_scale, pert_mask):
            # Return deterministic values based on input x
            return torch.ones(3, 1) * x[0]

    model = MockNet(edge_dim=5)
    pred = predict_nn(
        model,
        y,
        data,
        pert_dic_small,
        A_mult,
        pert_idx,
        mask_dic,
        edge_scale,
        pert_mask,
    )

    # For exp1, x[0] = 1, so all values should be 1
    assert np.allclose(pred["exp1"].values, np.ones(3))
    # For exp2, x[0] = 2, so all values should be 2
    assert np.allclose(pred["exp2"].values, np.ones(3) * 2)
