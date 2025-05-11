import numpy as np
import pandas as pd
import pytest
import torch

from magellan.prune import create_edge_scale


def test_create_edge_scale_basic():
    # Create simple test inputs
    A_mult = pd.DataFrame([
        [0, 2, 0],
        [1, 0, 3],
        [0, 4, 0]
    ])
    pert_idx = [0, 1]
    
    result = create_edge_scale(A_mult, pert_idx)
    
    # Expected non-zero values from A_mult.T: [1, 2, 3, 4]
    # Plus ones for each pert_idx: [1, 1]
    expected = torch.tensor([1., 2., 4., 3., 1., 1.], dtype=torch.float32)
    
    assert torch.allclose(result, expected)
    assert result.dtype == torch.float32

def test_create_edge_scale_empty_matrix():
    A_mult = pd.DataFrame([[0, 0], [0, 0]])
    pert_idx = [0]
    
    result = create_edge_scale(A_mult, pert_idx)
    
    # Should only contain ones for pert_idx
    expected = torch.tensor([1.], dtype=torch.float32)
    
    assert torch.allclose(result, expected)

def test_create_edge_scale_no_perturbations():
    A_mult = pd.DataFrame([[0, 1], [1, 0]])
    pert_idx = []
    
    result = create_edge_scale(A_mult, pert_idx)
    
    # Should only contain values from A_mult
    expected = torch.tensor([1., 1.], dtype=torch.float32)
    
    assert torch.allclose(result, expected)

def test_create_edge_scale_large_values():
    A_mult = pd.DataFrame([[0, 1e6], [1e6, 0]])
    pert_idx = [0]
    
    result = create_edge_scale(A_mult, pert_idx)
    
    expected = torch.tensor([1e6, 1e6, 1.], dtype=torch.float32)
    
    assert torch.allclose(result, expected)

def test_create_edge_scale_input_types():
    # Test with numpy array converted to DataFrame
    A_mult_np = np.array([[0, 1], [1, 0]])
    A_mult = pd.DataFrame(A_mult_np)
    pert_idx = [0]
    
    result = create_edge_scale(A_mult, pert_idx)
    
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32

@pytest.mark.parametrize("invalid_input", [
    None,
    "invalid",
    [[1, 2], [3, 4]]  # List of lists instead of DataFrame
])
def test_create_edge_scale_invalid_input(invalid_input):
    with pytest.raises((AttributeError, TypeError)):
        create_edge_scale(invalid_input, [0])