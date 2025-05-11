
import networkx as nx
import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from magellan.gnn_model import Net, train_model
from magellan.prune_opt import WarmupScheduler


@pytest.fixture
def mock_data():
    # Create simple mock data with 3 nodes and 2 experiments
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index)


@pytest.fixture
def mock_save_dir(tmp_path):
    return tmp_path / "mock_save_dir"


@pytest.fixture
def mock_pert_dic():
    return {
        "exp1": {"pert": {"node1": 1}, "exp": {"node2": 0}},
        "exp2": {"pert": {"node2": 1}, "exp": {"node1": 0}},
    }


@pytest.fixture
def mock_train_pert_dic():
    return {
        "exp1": {"pert": {"node1": 1}, "exp": {"node2": 0}},
    }


@pytest.fixture
def mock_test_pert_dic():
    return {
        "exp2": {"pert": {"node2": 1}, "exp": {"node1": 0}},
    }


@pytest.fixture
def mock_node_dic():
    return {"node1": 0, "node2": 1, "node3": 2}


@pytest.fixture
def mock_edge_idx():
    return torch.tensor([[0, 1], [1, 2]], dtype=torch.long)


@pytest.fixture
def mock_G():
    G = nx.DiGraph()
    G.add_edge("node1", "node2", sign="Activator")
    G.add_edge("node2", "node3", sign="Inhibitor")
    return G


@pytest.fixture
def mock_mask_dic():
    mask1 = torch.ones(2)  # Number of edges
    mask2 = torch.ones(2)
    return {"exp1": mask1, "exp2": mask2}


@pytest.fixture
def mock_edge_scale():
    return torch.ones(2)  # Number of edges


@pytest.fixture
def mock_pert_mask():
    return torch.zeros(2)  # Number of edges


@pytest.fixture
def mock_curriculum_stages():
    # This fixture is no longer used by the updated train_model
    # return [{"name": "all_nodes", "epochs": 2, "nodes": None, "weight": 1.0}]
    return None


@pytest.fixture
def mock_edge_signs():
    return torch.tensor([1.0, -1.0])  # Activator, Inhibitor


@pytest.fixture
def mock_node_class_weights():
    return {
        "node1": [0.5, 0.25, 0.25],  # weights for values 0,1,2
        "node2": [0.3, 0.4, 0.3],
        "node3": [0.4, 0.3, 0.3],
    }


def test_train_model_basic(
    mock_data,
    mock_pert_dic,
    mock_train_pert_dic,
    mock_test_pert_dic,
    mock_node_dic,
    mock_edge_idx,
    mock_mask_dic,
    mock_edge_scale,
    mock_pert_mask,
    mock_G,
    mock_save_dir,
    mock_edge_signs,
    mock_node_class_weights,
):
    # Initialize model and optimizer
    model = Net(edge_weight=torch.ones(2), min_val=0, max_val=2, n_iter=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Run training
    total_train_loss, sum_grad, train_epoch_losses, test_epoch_losses = train_model(
        model=model,
        train_data=mock_data,
        test_data=mock_data,
        train_pert_dic=mock_train_pert_dic,
        test_pert_dic=mock_test_pert_dic,
        node_dic=mock_node_dic,
        edge_idx_original=mock_edge_idx,
        edge_scale=mock_edge_scale,
        pert_mask=mock_pert_mask,
        train_mask_dic=mock_mask_dic,
        test_mask_dic=mock_mask_dic,
        node_class_weights_train=mock_node_class_weights,
        node_class_weights_test=mock_node_class_weights,
        save_dir=mock_save_dir,
        opt=optimizer,
        scheduler=scheduler,
        edge_signs=mock_edge_signs,
        max_range=2,
        min_range=0,
        epochs=2,
        warmup_scheduler=None,
        early_stopping_enabled=True,
        early_stopping_patience=5,
        verbose=False,
    )

    # Basic checks
    assert len(total_train_loss) > 0, "Training should produce losses"
    assert len(sum_grad) > 0, "Training should produce gradients"
    assert len(train_epoch_losses) > 0, "Training should produce epoch losses"
    assert all(not np.isnan(loss) for loss in total_train_loss), (
        "No losses should be NaN"
    )
    assert all(not np.isinf(loss) for loss in total_train_loss), (
        "No losses should be infinite"
    )


def test_train_model_with_warmup(
    mock_data,
    mock_train_pert_dic,
    mock_test_pert_dic,
    mock_node_dic,
    mock_edge_idx,
    mock_mask_dic,
    mock_edge_scale,
    mock_pert_mask,
    mock_edge_signs,
    mock_node_class_weights,
    mock_save_dir,
):
    model = Net(edge_weight=torch.ones(2), min_val=0, max_val=2, n_iter=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    warmup_scheduler = WarmupScheduler(
        optimizer=optimizer, warmup_steps=10, initial_lr=0.001, target_lr=0.01
    )

    total_train_loss, sum_grad, train_epoch_losses, test_epoch_losses = train_model(
        model=model,
        train_data=mock_data,
        test_data=mock_data,
        train_pert_dic=mock_train_pert_dic,
        test_pert_dic=mock_test_pert_dic,
        node_dic=mock_node_dic,
        edge_idx_original=mock_edge_idx,
        train_mask_dic=mock_mask_dic,
        test_mask_dic=mock_mask_dic,
        edge_scale=mock_edge_scale,
        pert_mask=mock_pert_mask,
        opt=optimizer,
        scheduler=scheduler,
        edge_signs=mock_edge_signs,
        max_range=2,
        min_range=0,
        node_class_weights_train=mock_node_class_weights,
        node_class_weights_test=mock_node_class_weights,
        save_dir=mock_save_dir,
        epochs=2,
        warmup_scheduler=warmup_scheduler,
        early_stopping_enabled=True,
        early_stopping_patience=5,
        verbose=False,
    )

    assert len(total_train_loss) > 0, "Training should produce losses"
    assert len(sum_grad) > 0, "Training should produce gradients"


def test_train_model_gradient_clipping(
    mock_data,
    mock_train_pert_dic,
    mock_test_pert_dic,
    mock_node_dic,
    mock_edge_idx,
    mock_mask_dic,
    mock_edge_scale,
    mock_pert_mask,
    mock_edge_signs,
    mock_node_class_weights,
    mock_save_dir,
):
    model = Net(edge_weight=torch.ones(2), min_val=0, max_val=2, n_iter=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    total_train_loss, sum_grad, train_epoch_losses, test_epoch_losses = train_model(
        model=model,
        train_data=mock_data,
        test_data=mock_data,
        train_pert_dic=mock_train_pert_dic,
        test_pert_dic=mock_test_pert_dic,
        node_dic=mock_node_dic,
        edge_idx_original=mock_edge_idx,
        train_mask_dic=mock_mask_dic,
        test_mask_dic=mock_mask_dic,
        edge_scale=mock_edge_scale,
        pert_mask=mock_pert_mask,
        opt=optimizer,
        scheduler=scheduler,
        edge_signs=mock_edge_signs,
        max_range=2,
        min_range=0,
        node_class_weights_train=mock_node_class_weights,
        node_class_weights_test=mock_node_class_weights,
        save_dir=mock_save_dir,
        epochs=2,
        grad_clip_max_norm=1.0,
        early_stopping_enabled=True,
        early_stopping_patience=5,
        verbose=False,
    )

    assert len(total_train_loss) > 0, "Training should produce losses"
    assert len(sum_grad) > 0, "Training should produce gradients"
    assert all(abs(g) <= 1.0 for g in sum_grad), "Gradients should be clipped"


def test_train_model_sign_flip_prevention(
    mock_data,
    mock_train_pert_dic,
    mock_test_pert_dic,
    mock_node_dic,
    mock_edge_idx,
    mock_mask_dic,
    mock_edge_scale,
    mock_pert_mask,
    mock_edge_signs,
    mock_node_class_weights,
    mock_save_dir,
):
    model = Net(edge_weight=torch.ones(2), min_val=0, max_val=2, n_iter=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    total_train_loss, sum_grad, train_epoch_losses, test_epoch_losses = train_model(
        model=model,
        train_data=mock_data,
        test_data=mock_data,
        train_pert_dic=mock_train_pert_dic,
        test_pert_dic=mock_test_pert_dic,
        node_dic=mock_node_dic,
        edge_idx_original=mock_edge_idx,
        train_mask_dic=mock_mask_dic,
        test_mask_dic=mock_mask_dic,
        edge_scale=mock_edge_scale,
        pert_mask=mock_pert_mask,
        opt=optimizer,
        scheduler=scheduler,
        edge_signs=mock_edge_signs,
        max_range=2,
        min_range=0,
        node_class_weights_train=mock_node_class_weights,
        node_class_weights_test=mock_node_class_weights,
        save_dir=mock_save_dir,
        epochs=2,
        allow_sign_flip=False,
        early_stopping_enabled=True,
        early_stopping_patience=5,
        verbose=False,
    )

    # Check that edge weights remain non-negative
    assert all(w >= 0 for w in model.edge_weight.data), (
        "Edge weights should be non-negative"
    )
    assert len(total_train_loss) > 0  # Added basic check


def test_train_model_input_validation(
    mock_data,
    mock_train_pert_dic,
    mock_node_dic,
    mock_edge_idx,
    mock_mask_dic,
    mock_edge_scale,
    mock_pert_mask,
    mock_edge_signs,
    mock_node_class_weights,
    mock_save_dir,
):
    model = Net(edge_weight=torch.ones(2), min_val=0, max_val=2, n_iter=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Test with invalid data.x
    invalid_data_x = Data(x=None, y=mock_data.y, edge_index=mock_data.edge_index)
    with pytest.raises(ValueError, match="data.x is None"):
        train_model(
            model=model,
            train_data=invalid_data_x,
            test_data=None,
            train_pert_dic=mock_train_pert_dic,
            test_pert_dic=None,
            node_dic=mock_node_dic,
            edge_idx_original=mock_edge_idx,
            train_mask_dic=mock_mask_dic,
            test_mask_dic=None,
            edge_scale=mock_edge_scale,
            pert_mask=mock_pert_mask,
            opt=optimizer,
            scheduler=scheduler,
            edge_signs=mock_edge_signs,
            max_range=2,
            min_range=0,
            node_class_weights_train=mock_node_class_weights,
            node_class_weights_test=None,
            save_dir=mock_save_dir,
            epochs=1,
            verbose=False,
        )

    # Test with invalid data.y (non-tensor)
    # Note: The error for y=[1,2,3] is a TypeError in Data constructor, not directly in train_model logic for y.
    # To test train_model's handling, we'd need data.y to be non-Tensor *after* Data creation.
    # For now, this tests Data constructor. If train_model has its own check, that's separate.
    # The current train_model checks `isinstance(train_data.y, torch.Tensor)`.
    invalid_data_y = Data(
        x=mock_data.x,
        y=torch.tensor([1, 2, 3], dtype=torch.float32),
        edge_index=mock_data.edge_index,
    )  # Valid y for Data
    invalid_data_y.y = [1, 2, 3]  # type: ignore # Make y invalid after Data object creation

    with pytest.raises(
        ValueError, match="data.y is not a tensor"
    ):  # This should match train_model's internal check
        train_model(
            model=model,
            train_data=invalid_data_y,
            test_data=None,
            train_pert_dic=mock_train_pert_dic,
            test_pert_dic=None,
            node_dic=mock_node_dic,
            edge_idx_original=mock_edge_idx,
            train_mask_dic=mock_mask_dic,
            test_mask_dic=None,
            edge_scale=mock_edge_scale,
            pert_mask=mock_pert_mask,
            opt=optimizer,
            scheduler=scheduler,
            edge_signs=mock_edge_signs,
            max_range=2,
            min_range=0,
            node_class_weights_train=mock_node_class_weights,
            node_class_weights_test=None,
            save_dir=mock_save_dir,
            epochs=1,
            verbose=False,
        )
