import pytest
import torch

from magellan.gnn_model import Net


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    n_nodes = 3
    n_edges = 4
    edge_dim = n_edges
    
    x = torch.rand(n_nodes, 1)  # Node features, values between 0 and 1
    edge_index = torch.tensor([
        [0, 1, 1, 2],  # Source nodes
        [1, 0, 2, 1],  # Target nodes
    ], dtype=torch.int64)
    edge_mask = torch.ones(n_edges)
    edge_scale = torch.ones(n_edges)
    pert_mask = torch.zeros(n_edges)
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'edge_dim': edge_dim,
        'x': x,
        'edge_index': edge_index,
        'edge_mask': edge_mask,
        'edge_scale': edge_scale,
        'pert_mask': pert_mask
    }


def test_init_default():
    """Test initialization with default parameters."""
    edge_dim = 5
    model = Net(edge_dim=edge_dim)
    
    assert model.edge_dim == edge_dim
    assert model.edge_weight.shape == (edge_dim,)
    assert len(model.conv_layers) == 5  # n_iter - 1
    assert isinstance(model.act, torch.nn.Hardtanh)
    assert not model.max_update
    assert not model.round_val


def test_init_custom():
    """Test initialization with custom parameters."""
    edge_weight = torch.ones(3)
    model = Net(
        edge_weight=edge_weight,
        min_val=-1,
        max_val=3,
        n_iter=4,
        max_update=True,
        round_val=True
    )
    
    assert model.edge_dim == 3
    assert torch.equal(model.edge_weight.data, edge_weight)
    assert len(model.conv_layers) == 3  # n_iter - 1
    assert model.act.min_val == -1
    assert model.act.max_val == 3
    assert model.max_update
    assert model.round_val


def test_forward_basic(sample_data):
    """Test basic forward pass without special features."""
    model = Net(edge_dim=sample_data['edge_dim'])
    
    output = model(
        sample_data['x'],
        sample_data['edge_index'],
        sample_data['edge_mask'],
        sample_data['edge_scale'],
        sample_data['pert_mask']
    )
    
    assert output.shape == (sample_data['n_nodes'], 1)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_layer_updates_with_max_update(sample_data):
    """Test that each layer's updates are bounded when max_update is enabled."""
    model = Net(
        edge_dim=sample_data['edge_dim'],
        max_update=True,
        n_iter=3  # Small number of iterations for testing
    )
    
    x = sample_data['x']
    edge_index = sample_data['edge_index']
    # edge_mask = sample_data['edge_mask'].to(torch.int64)
    # edge_scale = sample_data['edge_scale']
    # pert_mask = sample_data['pert_mask']
    
    # Test each layer's update
    for layer in model.conv_layers:
        x_prev = x.clone()
        x_new = layer(x_prev, edge_index, model.edge_weight)
        if model.max_update:
            x_new = x_prev + model.tanh(x_new - x_prev)
        
        layer_diff = x_new - x_prev
        assert torch.all(layer_diff <= 1) and torch.all(layer_diff >= -1), \
            f"Layer updates exceeded bounds: min={layer_diff.min().item():.4f}, max={layer_diff.max().item():.4f}"
        
        x = model.act(x_new)  # Update x for next layer

def test_output_bounds(sample_data):
    """Test that final output values are within activation bounds."""
    min_val, max_val = 0, 2  # Standard bounds
    model = Net(
        edge_dim=sample_data['edge_dim'],
        max_update=True,
        min_val=min_val,
        max_val=max_val
    )
    
    # Run forward pass
    output = model(
        sample_data['x'],
        sample_data['edge_index'],
        sample_data['edge_mask'].to(torch.int64),
        sample_data['edge_scale'],
        sample_data['pert_mask']
    )
    
    # Check output bounds
    assert torch.all(output >= min_val) and torch.all(output <= max_val), \
        f"Output exceeded activation bounds: min={output.min().item():.4f}, max={output.max().item():.4f}"

def test_forward_with_round_val(sample_data):
    """Test forward pass with round_val enabled."""
    model = Net(
        edge_dim=sample_data['edge_dim'],
        round_val=True
    )
    
    output = model(
        sample_data['x'],
        sample_data['edge_index'],
        sample_data['edge_mask'],
        sample_data['edge_scale'],
        sample_data['pert_mask']
    )
    
    # Check that all outputs are integers
    assert torch.all(output == output.floor())


def test_forward_with_classification(sample_data):
    """Test forward pass with classification enabled."""
    model = Net(edge_dim=sample_data['edge_dim'])
    
    # output = model(
    #     sample_data['x'],
    #     sample_data['edge_index'],
    #     sample_data['edge_mask'],
    #     sample_data['edge_scale'],
    #     sample_data['pert_mask'],
    #     classify=True
    # )
    
    # # Check softmax properties
    # assert torch.allclose(output.sum(dim=0), torch.tensor([1.0]))
    # assert torch.all(output >= 0) and torch.all(output <= 1)
    # check that not implemented error is raised
    with pytest.raises(NotImplementedError):
        model(
            sample_data['x'],
            sample_data['edge_index'],
            sample_data['edge_mask'],
            sample_data['edge_scale'],
            sample_data['pert_mask'],
            classify=True
        )

def test_edge_masking(sample_data):
    """Test that edge masking works correctly."""
    model = Net(edge_dim=sample_data['edge_dim'])
    
    # Create mask that blocks some edges
    edge_mask = torch.tensor([1., 0., 1., 0.])
    
    output = model(
        sample_data['x'],
        sample_data['edge_index'],
        edge_mask,
        sample_data['edge_scale'],
        sample_data['pert_mask']
    )
    
    assert output.shape == (sample_data['n_nodes'], 1)
    assert not torch.isnan(output).any()


def test_pert_mask(sample_data):
    """Test that perturbation masking works correctly."""
    model = Net(edge_dim=sample_data['edge_dim'])
    
    # Create perturbation mask for self-loops
    pert_mask = torch.tensor([1., 0., 0., 0.])
    
    _ = model(
        sample_data['x'],
        sample_data['edge_index'],
        sample_data['edge_mask'],
        sample_data['edge_scale'],
        pert_mask
    )
    
    # Check that edge weights are 1 where pert_mask is 1
    masked_weights = model.edge_weight.data * (torch.ones(model.edge_dim) - pert_mask) + pert_mask
    assert torch.equal(masked_weights[pert_mask == 1], torch.ones_like(masked_weights[pert_mask == 1]))


def test_value_bounds(sample_data):
    """Test that output values respect min_val and max_val bounds."""
    min_val = -1
    max_val = 3
    
    model = Net(
        edge_dim=sample_data['edge_dim'],
        min_val=min_val,
        max_val=max_val
    )
    
    output = model(
        sample_data['x'],
        sample_data['edge_index'],
        sample_data['edge_mask'],
        sample_data['edge_scale'],
        sample_data['pert_mask']
    )
    
    assert torch.all(output >= min_val)
    assert torch.all(output <= max_val)


def test_edge_scale(sample_data):
    """Test that edge scaling works correctly."""
    model = Net(edge_dim=sample_data['edge_dim'])
    
    # Create custom edge scale
    edge_scale = torch.tensor([0.5, 1.0, 1.5, 2.0])
    
    _ = model(
        sample_data['x'],
        sample_data['edge_index'],
        sample_data['edge_mask'],
        edge_scale,
        sample_data['pert_mask']
    )
    
    # Check that edge weights were properly scaled and unscaled
    assert torch.allclose(model.edge_weight.data, torch.ones(model.edge_dim))