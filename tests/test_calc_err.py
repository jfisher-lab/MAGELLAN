import numpy as np
import pandas as pd
import pytest

from magellan.prune import calc_err


def test_calc_err_basic_arrays():
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 3.0, 2.0])
    
    sum_err, mean_err = calc_err(y_pred, y_true)
    
    assert np.isclose(sum_err, 2.0)
    assert np.isclose(mean_err, 2/3)

def test_calc_err_dataframes():
    y_pred = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
    y_true = pd.DataFrame({'val': [1.0, 3.0, 2.0]})
    
    sum_err, mean_err = calc_err(y_pred, y_true)
    
    assert np.isclose(sum_err, 2.0)
    assert np.isclose(mean_err, 2/3)

def test_calc_err_with_mask():
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 3.0, 2.0])
    mask = np.array(['keep', 'remove', 'keep'])
    
    sum_err, mean_err = calc_err(y_pred, y_true, mask, mask_val='keep')
    
    assert np.isclose(sum_err, 1.0)  # Only counting errors where mask is 'keep'
    assert np.isclose(mean_err, 0.5)  # Mean of remaining errors

def test_calc_err_all_masked():
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 3.0, 2.0])
    mask = np.array(['remove', 'remove', 'remove'])
    
    # Test default behavior
    sum_err, mean_err = calc_err(y_pred, y_true, mask, mask_val='keep')
    assert sum_err == 0.0
    assert np.isnan(mean_err)
    
    # Test with return_nan_if_all_masked=True
    sum_err, mean_err = calc_err(y_pred, y_true, mask, mask_val='keep', return_nan_if_all_masked=True)
    assert np.isnan(sum_err)
    assert np.isnan(mean_err)

def test_calc_err_mixed_types():
    y_pred = np.array([1, 2, 3], dtype=float)
    y_true = pd.Series([1, 3, 2], dtype=float)
    
    sum_err, mean_err = calc_err(y_pred, y_true)
    
    assert np.isclose(sum_err, 2.0)
    assert np.isclose(mean_err, 2/3)

def test_calc_err_empty():
    y_pred = np.array([], dtype=float)
    y_true = np.array([], dtype=float)
    
    with pytest.raises(ValueError):
        calc_err(y_pred, y_true)


def test_calc_err_with_nans():
    y_pred = np.array([1.0, np.nan, 3.0])
    y_true = np.array([1.0, 3.0, 2.0])
    
    sum_err, mean_err = calc_err(y_pred, y_true)
    
    assert np.isclose(sum_err, 1.0)  # Only counting non-nan differences
    assert np.isclose(mean_err, 0.5)

@pytest.mark.parametrize("shape", [(3,), (3,1), (1,3)])
def test_calc_err_different_shapes(shape):
    y_pred = np.ones(shape)
    y_true = np.zeros(shape)
    
    sum_err, mean_err = calc_err(y_pred, y_true)
    
    assert np.isclose(sum_err, 3.0)  # All differences are 1
    assert np.isclose(mean_err, 1.0)

def test_calc_err_mismatched_shapes():
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError):
        calc_err(y_pred, y_true)