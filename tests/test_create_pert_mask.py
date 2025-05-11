import pytest
import torch

from magellan.prune import create_pert_mask


def test_self_loops():
    """Test that self-loops are correctly identified in the perturbation mask"""
    edge_idx = torch.tensor([
        [0, 0, 1, 1, 2],  # source nodes
        [1, 0, 2, 1, 0]   # target nodes
    ])
    node_dic = {"node0": 0, "node1": 1, "node2": 2}
    
    pert_mask = create_pert_mask(edge_idx, node_dic)
    
    # Check that self-loops (0->0 and 1->1) are marked as 1
    expected_mask = torch.tensor([0, 1, 0, 1, 0])
    assert torch.equal(pert_mask, expected_mask)

def test_dummy_nodes():
    """Test that edges from dummy nodes are correctly marked"""
    edge_idx = torch.tensor([
        [0, 1, 2, 3],     # source nodes
        [1, 2, 3, 0]      # target nodes
    ])
    node_dic = {
        "dummy_0": 0,
        "node1": 1,
        "node2": 2,
        "node3": 3
    }
    
    pert_mask = create_pert_mask(edge_idx, node_dic)
    
    # Check that edges from dummy node (0->1) are marked as 1
    expected_mask = torch.tensor([1, 0, 0, 0])
    assert torch.equal(pert_mask, expected_mask)

def test_combined_dummy_and_self_loops():
    """Test both dummy nodes and self-loops together"""
    edge_idx = torch.tensor([
        [0, 0, 1, 1, 2],  # source nodes
        [1, 0, 2, 1, 2]   # target nodes
    ])
    node_dic = {
        "dummy_0": 0,
        "node1": 1,
        "node2": 2
    }
    
    pert_mask = create_pert_mask(edge_idx, node_dic)
    
    # Check that both self-loops and dummy node edges are marked
    expected_mask = torch.tensor([1, 1, 0, 1, 1])
    assert torch.equal(pert_mask, expected_mask)

def test_empty_graph():
    """Test behavior with empty edge index"""
    edge_idx = torch.tensor([[], []], dtype=torch.int64)
    node_dic = {"node0": 0}
    
    pert_mask = create_pert_mask(edge_idx, node_dic)
    
    assert pert_mask.shape[0] == 0
    assert pert_mask.dtype == torch.int64

def test_no_dummy_nodes():
    """Test behavior when there are no dummy nodes"""
    edge_idx = torch.tensor([
        [0, 1, 2],  # source nodes
        [1, 2, 0]   # target nodes
    ])
    node_dic = {"node0": 0, "node1": 1, "node2": 2}
    
    pert_mask = create_pert_mask(edge_idx, node_dic)
    
    # Check that no edges are marked (no self-loops or dummy nodes)
    expected_mask = torch.tensor([0, 0, 0])
    assert torch.equal(pert_mask, expected_mask)



def test_invalid_node_indices():
    """Test that function handles invalid node indices gracefully"""
    edge_idx = torch.tensor([
        [0, 1, 2],  # source nodes
        [1, 2, 3]   # target nodes
    ])
    node_dic = {"dummy_0": 0, "node1": 1}  # Missing indices 2 and 3
    
    with pytest.raises(Exception):
        _ = create_pert_mask(edge_idx, node_dic)