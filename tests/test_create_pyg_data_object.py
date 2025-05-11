import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from magellan.prune import create_pyg_data_object


@pytest.fixture
def sample_data():
    # Create sample input data
    X = pd.DataFrame({
        'feat1': [1.0, 2.0, 3.0],
        'feat2': [0.1, 0.2, 0.3]
    })
    y = pd.DataFrame({'target': [0, 1, 0]})
    edge_idx = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    return X, y, edge_idx

def test_output_type(sample_data):
    """Test if the function returns a PyG Data object"""
    X, y, edge_idx = sample_data
    data = create_pyg_data_object(X, y, edge_idx)
    assert isinstance(data, Data)

def test_tensor_dtypes(sample_data):
    """Test if output tensors have correct dtypes"""
    X, y, edge_idx = sample_data
    data = create_pyg_data_object(X, y, edge_idx)
    
    assert data.x.dtype == torch.float32 # type: ignore
    assert data.y.dtype == torch.float32 # type: ignore
    assert data.edge_index.dtype == torch.long # type: ignore

def test_tensor_shapes(sample_data):
    """Test if output tensors have correct shapes"""
    X, y, edge_idx = sample_data
    data = create_pyg_data_object(X, y, edge_idx)
    
    assert data.x.shape == (3, 2)  # 3 nodes, 2 features # type: ignore
    assert data.y.shape == (3, 1)  # 3 nodes, 1 target # type: ignore
    assert data.edge_index.shape == (2, 4)  # 2 rows (source/target), 4 edges # type: ignore

def test_tensor_values(sample_data):
    """Test if tensor values match input data"""
    X, y, edge_idx = sample_data
    data = create_pyg_data_object(X, y, edge_idx)
    
    np.testing.assert_array_almost_equal(
        data.x.numpy(),  # type: ignore
        X.values 
    )
    np.testing.assert_array_almost_equal(
        data.y.numpy(),  # type: ignore
        y.values
    )
    np.testing.assert_array_equal(
        data.edge_index.numpy(),  # type: ignore
        edge_idx.numpy()
    )

def test_empty_input():
    """Test handling of empty input data"""
    X = pd.DataFrame({'feat1': [], 'feat2': []})
    y = pd.DataFrame({'target': []})
    edge_idx = torch.tensor([[],[]], dtype=torch.long)
    
    data = create_pyg_data_object(X, y, edge_idx)
    assert data.x.shape[0] == 0 # type: ignore
    assert data.y.shape[0] == 0 # type: ignore
    assert data.edge_index.shape[1] == 0 # type: ignore

def test_single_feature():
    """Test with single feature input"""
    X = pd.DataFrame({'feat1': [1.0, 2.0]})
    y = pd.DataFrame({'target': [0, 1]})
    edge_idx = torch.tensor([[0], [1]], dtype=torch.long)
    
    data = create_pyg_data_object(X, y, edge_idx)
    assert data.x.shape == (2, 1) # type: ignore
    assert data.y.shape == (2, 1) # type: ignore
    assert data.edge_index.shape == (2, 1) # type: ignore