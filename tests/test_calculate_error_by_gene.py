import numpy as np
import pandas as pd
import pytest

from magellan.prune import calculate_error_by_gene


@pytest.fixture
def sample_data():
    # Create sample DataFrames
    y = pd.DataFrame({
        'exp1': [1.0, 2.0, 3.0],
        'exp2': [2.0, 3.0, 4.0],
        'exp3': [3.0, 4.0, 5.0]
    }, index=['gene1', 'gene2', 'gene3'])
    
    pred = pd.DataFrame({
        'exp1': [1.1, 2.2, 2.8],
        'exp2': [1.9, 3.1, 4.2],
        'exp3': [3.2, 3.8, 4.9]
    }, index=['gene1', 'gene2', 'gene3'])
    
    idx_real = ['gene1', 'gene2', 'gene3']
    
    return y, pred, idx_real

def test_basic_error_calculation(sample_data):
    """Test basic error calculation functionality"""
    y, pred, idx_real = sample_data
    rmse, mae = calculate_error_by_gene(y, pred, idx_real)
    
    # Check output types
    assert isinstance(rmse, pd.Series)
    assert isinstance(mae, pd.Series)
    
    # Check indices
    assert list(rmse.index) == idx_real
    assert list(mae.index) == idx_real
    
    # Manually calculate expected values
    expected_rmse = np.array([
        np.sqrt(((1.1-1.0)**2 + (1.9-2.0)**2 + (3.2-3.0)**2)/3),  # gene1
        np.sqrt(((2.2-2.0)**2 + (3.1-3.0)**2 + (3.8-4.0)**2)/3),  # gene2
        np.sqrt(((2.8-3.0)**2 + (4.2-4.0)**2 + (4.9-5.0)**2)/3)   # gene3
    ])
    
    expected_mae = np.array([
        (abs(1.1-1.0) + abs(1.9-2.0) + abs(3.2-3.0))/3,  # gene1
        (abs(2.2-2.0) + abs(3.1-3.0) + abs(3.8-4.0))/3,  # gene2
        (abs(2.8-3.0) + abs(4.2-4.0) + abs(4.9-5.0))/3   # gene3
    ])
    
    # Check values
    np.testing.assert_array_almost_equal(rmse.values, expected_rmse) # type: ignore
    np.testing.assert_array_almost_equal(mae.values, expected_mae) # type: ignore

def test_subset_indices():
    """Test calculation with subset of indices"""
    y = pd.DataFrame({
        'exp1': [1.0, 2.0, 3.0],
        'exp2': [2.0, 3.0, 4.0]
    }, index=['gene1', 'gene2', 'gene3'])
    
    pred = pd.DataFrame({
        'exp1': [1.0, 2.0, 3.0],
        'exp2': [2.0, 3.0, 4.0]
    }, index=['gene1', 'gene2', 'gene3'])
    
    idx_real = ['gene1', 'gene3']  # Only using subset
    
    rmse, mae = calculate_error_by_gene(y, pred, idx_real)
    
    # Check only requested indices are present
    assert list(rmse.index) == idx_real
    assert list(mae.index) == idx_real
    
    # For identical predictions, errors should be 0
    np.testing.assert_array_equal(rmse.values, [0.0, 0.0])
    np.testing.assert_array_equal(mae.values, [0.0, 0.0])

def test_perfect_prediction():
    """Test when predictions exactly match true values"""
    y = pd.DataFrame({
        'exp1': [1.0, 2.0],
        'exp2': [3.0, 4.0]
    }, index=['gene1', 'gene2'])
    
    pred = y.copy()  # Perfect predictions
    idx_real = ['gene1', 'gene2']
    
    rmse, mae = calculate_error_by_gene(y, pred, idx_real)
    
    # All errors should be 0
    np.testing.assert_array_equal(rmse.values, [0.0, 0.0])
    np.testing.assert_array_equal(mae.values, [0.0, 0.0])

def test_empty_data():
    """Test behavior with empty DataFrames"""
    y = pd.DataFrame(columns=['exp1'], index=['gene1'])
    pred = pd.DataFrame(columns=['exp1'], index=['gene1'])
    idx_real = ['gene1']
    
    with pytest.raises(ValueError):
        calculate_error_by_gene(y, pred, idx_real)

def test_single_experiment():
    """Test with single experiment"""
    y = pd.DataFrame({'exp1': [1.0, 2.0]}, index=['gene1', 'gene2'])
    pred = pd.DataFrame({'exp1': [1.1, 1.9]}, index=['gene1', 'gene2'])
    idx_real = ['gene1', 'gene2']
    
    rmse, mae = calculate_error_by_gene(y, pred, idx_real)
    
    # Check values
    expected_rmse = np.array([0.1, 0.1])
    expected_mae = np.array([0.1, 0.1])
    
    np.testing.assert_array_almost_equal(rmse.values, expected_rmse) # type: ignore
    np.testing.assert_array_almost_equal(mae.values, expected_mae) # type: ignore

def test_invalid_indices():
    """Test error handling for invalid indices"""
    y = pd.DataFrame({'exp1': [1.0]}, index=['gene1'])
    pred = pd.DataFrame({'exp1': [1.0]}, index=['gene1'])
    idx_real = ['invalid_gene']  # Index that doesn't exist
    
    with pytest.raises(KeyError):
        calculate_error_by_gene(y, pred, idx_real)

def test_mismatched_dataframes():
    """Test error handling for mismatched DataFrames"""
    y = pd.DataFrame({'exp1': [1.0]}, index=['gene1'])
    pred = pd.DataFrame({'exp2': [1.0]}, index=['gene1'])  # Different column name
    idx_real = ['gene1']
    
    with pytest.raises(ValueError):
        calculate_error_by_gene(y, pred, idx_real)