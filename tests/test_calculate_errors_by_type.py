import pandas as pd
import pytest

from magellan.prune import calculate_errors_by_type


@pytest.fixture
def sample_data():
    # Create test DataFrames
    y = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=['x', 'y', 'z'])
    
    pred1 = pd.DataFrame({
        'A': [1.1, 2.1, 3.1], 
        'B': [4.1, 5.1, 6.1]
    }, index=['x', 'y', 'z'])
    
    pred2 = pd.DataFrame({
        'A': [1.2, 2.2, 3.2],
        'B': [4.2, 5.2, 6.2] 
    }, index=['x', 'y', 'z'])
    
    annot = pd.DataFrame({
        'A': ['valid', 'invalid', 'valid'],
        'B': ['valid', 'valid', 'invalid']
    }, index=['x', 'y', 'z'])
    
    predictions = [
        (pred1, 'pred1'),
        (pred2, 'pred2')
    ]
    
    return y, predictions, annot

def test_basic_functionality(sample_data):
    y, predictions, annot = sample_data
    idx_real = ['x', 'y', 'z']
    
    errors = calculate_errors_by_type(
        predictions=predictions,
        y=y, 
        annot=annot,
        mask_val='valid',
        idx_real=idx_real
    )
    
    # Should return dict with errors for each prediction
    assert isinstance(errors, dict)
    assert 'pred1' in errors
    assert 'pred2' in errors
    
    # Each value should be a tuple of (sum_error, mean_error)
    for error_tuple in errors.values():
        assert isinstance(error_tuple, tuple)
        assert len(error_tuple) == 2
        assert all(isinstance(x, float) for x in error_tuple)

def test_empty_predictions():
    y = pd.DataFrame({'A': [1,2]})
    annot = pd.DataFrame({'A': ['valid', 'valid']})
    
    with pytest.raises(ValueError):
        calculate_errors_by_type(
            predictions=[],
            y=y,
            annot=annot, 
            mask_val='valid',
            idx_real=['x','y']
        )

def test_mismatched_indices():
    y = pd.DataFrame({'A': [1,2]}, index=['x','y'])
    pred = pd.DataFrame({'A': [1,2]}, index=['a','b'])
    annot = pd.DataFrame({'A': ['valid','valid']}, index=['x','y'])
    
    with pytest.raises(KeyError):
        calculate_errors_by_type(
            predictions=[(pred, 'pred')],
            y=y,
            annot=annot,
            mask_val='valid', 
            idx_real=['x','y']
        )

def test_error_calculation():
    # Create test data where we can calculate exact expected errors
    y = pd.DataFrame({'A': [1.0, 2.0]}, index=['x','y'])
    pred = pd.DataFrame({'A': [2.0, 3.0]}, index=['x','y'])  # Each value off by 1
    annot = pd.DataFrame({'A': ['valid','valid']}, index=['x','y'])
    
    errors = calculate_errors_by_type(
        predictions=[(pred, 'pred')],
        y=y,
        annot=annot,
        mask_val='valid',
        idx_real=['x','y']
    )
    
    # Sum error should be 2.0 (1.0 diff * 2 values)
    # Mean error should be 1.0 (2.0 sum / 2 values)
    assert errors['pred'][0] == pytest.approx(2.0)
    assert errors['pred'][1] == pytest.approx(1.0)

def test_masked_values():
    y = pd.DataFrame({'A': [1.0, 2.0, 3.0]}, index=['x','y','z'])
    pred = pd.DataFrame({'A': [2.0, 3.0, 4.0]}, index=['x','y','z'])
    annot = pd.DataFrame({'A': ['valid','invalid','valid']}, index=['x','y','z'])
    
    errors = calculate_errors_by_type(
        predictions=[(pred, 'pred')],
        y=y,
        annot=annot,
        mask_val='valid',
        idx_real=['x','y','z']
    )
    
    # Should only include errors for 'valid' values (x and z)
    assert errors['pred'][0] == pytest.approx(2.0)  # Sum of 1.0 diff * 2 valid values
    assert errors['pred'][1] == pytest.approx(1.0)  # Mean of 2.0 sum / 2 valid values

def test_all_masked():
    y = pd.DataFrame({'A': [1.0, 2.0]}, index=['x','y'])
    pred = pd.DataFrame({'A': [2.0, 3.0]}, index=['x','y'])
    annot = pd.DataFrame({'A': ['invalid','invalid']}, index=['x','y'])
    
    errors = calculate_errors_by_type(
        predictions=[(pred, 'pred')],
        y=y,
        annot=annot,
        mask_val='valid',
        idx_real=['x','y']
    )
    
    # When all values are masked, should return (0.0, nan)
    assert errors['pred'][0] == 0.0
    assert pd.isna(errors['pred'][1])