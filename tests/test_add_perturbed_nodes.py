import pandas as pd
import pytest

from magellan.sci_opt import add_perturbed_nodes


@pytest.fixture
def sample_data():
    # Create sample prediction DataFrame
    pred_bound_zero_perturbed = pd.DataFrame({
        'exp1': [0.0, 0.0, 0.5],
        'exp2': [0.0, 0.0, 0.8]
    }, index=['node1', 'node2', 'node3'])
    
    # Create sample perturbation dictionary
    pert_dic_small = {
        'exp1': {
            'pert': {'node1': 1.0, 'node2': 2.0},
            'exp': {'node3': 0.5}
        },
        'exp2': {
            'pert': {'node1': 0.0, 'node2': 1.0},
            'exp': {'node3': 0.8}
        }
    }
    
    # Create sample y DataFrame matching the structure
    y = pd.DataFrame({
        'exp1': [-1, -1, 0.5],
        'exp2': [-1, -1, 0.8]
    }, index=['node1', 'node2', 'node3'])
    
    return pred_bound_zero_perturbed, pert_dic_small, y

def test_basic_perturbation_correction(sample_data):
    """Test that perturbed nodes are correctly updated with their perturbation values."""
    pred_bound_zero_perturbed, pert_dic_small, y = sample_data
    
    result = add_perturbed_nodes(pred_bound_zero_perturbed, pert_dic_small, y)
    
    # Check exp1 perturbations
    assert result.loc['node1', 'exp1'] == 1.0
    assert result.loc['node2', 'exp1'] == 2.0
    
    # Check exp2 perturbations
    assert result.loc['node1', 'exp2'] == 0.0
    assert result.loc['node2', 'exp2'] == 1.0
    
    # Check non-perturbed values remain unchanged
    assert result.loc['node3', 'exp1'] == 0.5
    assert result.loc['node3', 'exp2'] == 0.8

def test_input_not_modified(sample_data):
    """Test that the input DataFrame is not modified."""
    pred_bound_zero_perturbed, pert_dic_small, y = sample_data
    original = pred_bound_zero_perturbed.copy()
    
    _ = add_perturbed_nodes(pred_bound_zero_perturbed, pert_dic_small, y)
    
    pd.testing.assert_frame_equal(pred_bound_zero_perturbed, original)

def test_output_shape(sample_data):
    """Test that output DataFrame maintains the same shape and indices as input."""
    pred_bound_zero_perturbed, pert_dic_small, y = sample_data
    
    result = add_perturbed_nodes(pred_bound_zero_perturbed, pert_dic_small, y)
    
    assert result.shape == pred_bound_zero_perturbed.shape
    assert all(result.index == pred_bound_zero_perturbed.index)
    assert all(result.columns == pred_bound_zero_perturbed.columns)

def test_empty_perturbation_dict():
    """Test behavior with empty perturbation dictionary."""
    pred_bound_zero_perturbed = pd.DataFrame({
        'exp1': [0.0, 0.5],
        'exp2': [0.0, 0.8]
    }, index=['node1', 'node2'])
    
    pert_dic_small = {}
    
    y = pd.DataFrame({
        'exp1': [-1, 0.5],
        'exp2': [-1, 0.8]
    }, index=['node1', 'node2'])
    
    result = add_perturbed_nodes(pred_bound_zero_perturbed, pert_dic_small, y)
    
    pd.testing.assert_frame_equal(result, pred_bound_zero_perturbed)

def test_float_values(sample_data):
    """Test handling of float values in perturbations."""
    pred_bound_zero_perturbed, pert_dic_small, y = sample_data
    
    # Modify perturbation values to floats
    pert_dic_small['exp1']['pert']['node1'] = 1.5
    pert_dic_small['exp2']['pert']['node2'] = 0.7
    
    result = add_perturbed_nodes(pred_bound_zero_perturbed, pert_dic_small, y)
    
    assert result.loc['node1', 'exp1'] == 1.5
    assert result.loc['node2', 'exp2'] == 0.7

def test_missing_indices():
    """Test behavior when perturbation dictionary contains nodes not in DataFrame."""
    pred_bound_zero_perturbed = pd.DataFrame({
        'exp1': [0.0, 0.5],
        'exp2': [0.0, 0.8]
    }, index=['node1', 'node2'])
    
    pert_dic_small = {
        'exp1': {
            'pert': {'node3': 1.0},  # Node3 doesn't exist in DataFrame
            'exp': {}
        }
    }
    
    y = pd.DataFrame({
        'exp1': [-1, 0.5],
        'exp2': [-1, 0.8]
    }, index=['node1', 'node2'])
    
    with pytest.raises(KeyError):
        _ = add_perturbed_nodes(pred_bound_zero_perturbed, pert_dic_small, y)

def test_missing_experiments():
    """Test behavior when perturbation dictionary contains experiments not in DataFrame."""
    pred_bound_zero_perturbed = pd.DataFrame({
        'exp1': [0.0, 0.5]
    }, index=['node1', 'node2'])
    
    pert_dic_small = {
        'exp1': {
            'pert': {'node1': 1.0},
            'exp': {}
        },
        'exp2': {  # exp2 doesn't exist in DataFrame
            'pert': {'node1': 0.0},
            'exp': {}
        }
    }
    
    y = pd.DataFrame({
        'exp1': [-1, 0.5]
    }, index=['node1', 'node2'])
    
    with pytest.raises(KeyError):
        _ = add_perturbed_nodes(pred_bound_zero_perturbed, pert_dic_small, y)