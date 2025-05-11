import pytest
import torch

from magellan.prune import construct_mask_dic


@pytest.fixture
def sample_data():
    # Create sample edge index representing a simple graph
    # Edge index format: [[source nodes], [target nodes]]
    edge_idx = torch.tensor([
        [0, 1, 1, 2, 0, 1, 2],  # source nodes
        [1, 2, 0, 0, 0, 1, 2],  # target nodes
    ])
    
    # Node dictionary mapping names to indices
    node_dic = {
        'A': 0,
        'B': 1,
        'C': 2
    }
    
    # Perturbation dictionary with two experiments
    pert_dic = {
        'exp1': {'pert': ['A'], 'exp': ['B', 'C']},
        'exp2': {'pert': ['B'], 'exp': ['A', 'C']}
    }
    
    return edge_idx, node_dic, pert_dic


def test_construct_mask_dic_basic(sample_data):
    edge_idx, node_dic, pert_dic = sample_data
    mask_dic = construct_mask_dic(pert_dic, node_dic, edge_idx)
    
    # Check if mask_dic has correct keys
    assert set(mask_dic.keys()) == {'exp1', 'exp2'}
    
    # Check if masks have correct length (equal to number of edges)
    assert len(mask_dic['exp1']) == edge_idx.shape[1]
    assert len(mask_dic['exp2']) == edge_idx.shape[1]


def test_construct_mask_dic_values(sample_data):
    edge_idx, node_dic, pert_dic = sample_data
    mask_dic = construct_mask_dic(pert_dic, node_dic, edge_idx)
    
    # For exp1 (A is perturbed):
    # - Self-loop on A should be 1
    # - Parents of A should be 0
    # - Self-loops on non-perturbed nodes (B, C) should be 0
    exp1_mask = mask_dic['exp1']
    assert exp1_mask[4] == 1  # Self-loop on A (0->0)
    assert exp1_mask[5] == 0  # Self-loop on B (1->1)
    assert exp1_mask[6] == 0  # Self-loop on C (2->2)


def test_construct_mask_dic_debug_output(sample_data, capsys):
    edge_idx, node_dic, pert_dic = sample_data
    _ = construct_mask_dic(pert_dic, node_dic, edge_idx, mask_debug=True)
    
    captured = capsys.readouterr()
    assert "Processing perturbation: exp1" in captured.out
    assert "Processing perturbation: exp2" in captured.out


def test_construct_mask_dic_empty_pert():
    edge_idx = torch.tensor([[0, 1], [1, 0]])
    node_dic = {'A': 0, 'B': 1}
    pert_dic = {'exp1': {'pert': [], 'exp': ['B']}}
    
    mask_dic = construct_mask_dic(pert_dic, node_dic, edge_idx)
    assert len(mask_dic['exp1']) == edge_idx.shape[1]


def test_construct_mask_dic_invalid_node():
    edge_idx = torch.tensor([[0, 1], [1, 0]])
    node_dic = {'A': 0, 'B': 1}
    pert_dic = {'exp1': {'pert': ['C'], 'exp': ['B']}}
    
    with pytest.raises(KeyError):
        construct_mask_dic(pert_dic, node_dic, edge_idx)