import numpy as np
import pandas as pd
import pytest
import torch

from magellan.prune import make_edge_idx


def test_make_edge_idx_basic():
    # Create a simple adjacency matrix
    matrix = pd.DataFrame([
        [0, 1, 0], # 0 --> 2
        [0, 0, 1], # 1 --> 0
        [1, 0, 0], # 2 --> 1
    ])
    pert_idx = [0, 2]  # Nodes 0 and 2 are perturbed so we need 0 --> 0 and 2 --> 2
    
    expected_edge_idx = torch.tensor([
        [0, 1, 2, 0, 2],  # Source nodes
        [2, 0, 1, 0, 2],  # Target nodes
    ], dtype=torch.int64)
    
    result = make_edge_idx(matrix, pert_idx)
    
    assert torch.equal(result, expected_edge_idx)


def test_make_edge_idx_empty_matrix():
    # Test with matrix with no edges
    matrix = pd.DataFrame(np.zeros((3, 3)))
    pert_idx = [0]
    
    expected_edge_idx = torch.tensor([[0], [0]], dtype=torch.int64)
    
    result = make_edge_idx(matrix, pert_idx)
    
    assert torch.equal(result, expected_edge_idx)


def test_make_edge_idx_no_perturbations():
    # Test with no perturbed nodes
    matrix = pd.DataFrame([
        [0, 1],
        [1, 0],
    ])
    pert_idx = []
    
    expected_edge_idx = torch.tensor([
        [0, 1],  # Source nodes
        [1, 0],  # Target nodes
    ], dtype=torch.int64)
    
    result = make_edge_idx(matrix, pert_idx)
    
    assert torch.equal(result, expected_edge_idx)


def test_make_edge_idx_input_validation():
    # Test with invalid input types
    with pytest.raises(Exception):
        make_edge_idx(None, [0]) # type: ignore
    
    with pytest.raises(Exception):
        make_edge_idx(pd.DataFrame([[0]]), None) # type: ignore