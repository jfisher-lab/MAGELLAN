import numpy as np
import pandas as pd
import pytest

from magellan.plot import create_annotation_matrix


@pytest.fixture
def base_dataframe():
    return pd.DataFrame(
        np.zeros((3, 2)),
        index=['gene1', 'gene2', 'gene3'],
        columns=['condition1', 'condition2']
    )

@pytest.fixture
def simple_perturbation_dict():
    return {
        'condition1': {
            'pert': {'gene1': 1},
            'exp': {'gene2': 0}
        },
        'condition2': {
            'pert': {'gene3': 1},
            'exp': {'gene1': 0}
        }
    }

def test_basic_annotation():
    """Test basic annotation matrix creation with default symbols"""
    df = pd.DataFrame(np.zeros((2,2)), index=['A', 'B'], columns=['C1', 'C2'])
    pert_dict = {
        'C1': {
            'pert': {'A': 1},
            'exp': {'B': 0}
        }
    }
    
    result = create_annotation_matrix(df, pert_dict)
    
    assert result.loc['A', 'C1'] == '•'
    assert result.loc['B', 'C1'] == '-'
    assert result.loc['A', 'C2'] == ''
    assert result.loc['B', 'C2'] == ''

def test_custom_symbols():
    """Test annotation matrix creation with custom symbols"""
    df = pd.DataFrame(np.zeros((2,1)), index=['A', 'B'], columns=['C1'])
    pert_dict = {
        'C1': {
            'pert': {'A': 1},
            'exp': {'B': 0}
        }
    }
    custom_symbols = {'pert': '*', 'exp': '>', 'tst': '.'}
    
    result = create_annotation_matrix(df, pert_dict, custom_symbols)
    
    assert result.loc['A', 'C1'] == '*'
    assert result.loc['B', 'C1'] == '>'

def test_empty_perturbation_dict():
    """Test with empty perturbation dictionary"""
    df = pd.DataFrame(np.zeros((2,2)), index=['A', 'B'], columns=['C1', 'C2'])
    empty_dict = {}
    
    result = create_annotation_matrix(df, empty_dict)
    
    assert (result == '').all().all()

def test_missing_perturbation_types(base_dataframe):
    """Test with missing perturbation or expectation entries"""
    pert_dict = {
        'condition1': {
            'pert': {'gene1': 1}
            # No 'exp' entry
        },
        'condition2': {
            'exp': {'gene2': 0}
            # No 'pert' entry
        }
    }
    
    result = create_annotation_matrix(base_dataframe, pert_dict)
    
    assert result.loc['gene1', 'condition1'] == '•'
    assert result.loc['gene2', 'condition2'] == '-'

def test_nonexistent_genes():
    """Test behavior with genes not present in base DataFrame"""
    df = pd.DataFrame(np.zeros((2,1)), index=['A', 'B'], columns=['C1'])
    pert_dict = {
        'C1': {
            'pert': {'nonexistent': 1},
            'exp': {'B': 0}
        }
    }
    
    with pytest.raises(KeyError):
        create_annotation_matrix(df, pert_dict)

def test_nonexistent_conditions():
    """Test behavior with conditions not present in base DataFrame"""
    df = pd.DataFrame(np.zeros((2,1)), index=['A', 'B'], columns=['C1'])
    print(df)
    pert_dict = {
        'nonexistent': {
            'pert': {'A': 1},
            'exp': {'B': 0}
        }
    }
    
    with pytest.raises(KeyError):
        create_annotation_matrix(df, pert_dict)

def test_none_annotation_symbols():
    """Test that default symbols are used when annotation_symbols is None"""
    df = pd.DataFrame(np.zeros((2,1)), index=['A', 'B'], columns=['C1'])
    pert_dict = {
        'C1': {
            'pert': {'A': 1},
            'exp': {'B': 0}
        }
    }
    
    result = create_annotation_matrix(df, pert_dict, None)
    
    assert result.loc['A', 'C1'] == '•'
    assert result.loc['B', 'C1'] == '-'

def test_overlapping_pert_exp():
    """Test behavior when a gene is both perturbed and expected"""
    df = pd.DataFrame(np.zeros((2,1)), index=['A', 'B'], columns=['C1'])
    pert_dict = {
        'C1': {
            'pert': {'A': 1},
            'exp': {'A': 0}
        }
    }
    
    result = create_annotation_matrix(df, pert_dict)
    
    # Last assignment should take precedence
    assert result.loc['A', 'C1'] == '-'
    
def test_missing_indices():
    """Test behavior with indices in perturbation_dict that are not present in base DataFrame"""
    df = pd.DataFrame(np.zeros((2, 1)), index=['A', 'B'], columns=['C1'])
    pert_dict = {
        'C1': {
            'pert': {'C': 1},  # 'C' is not in the index of df
            'exp': {'A': 0}
        }
    }
    
    with pytest.raises(KeyError) as excinfo:
        create_annotation_matrix(df, pert_dict)
    
    assert "Missing indices in perturbation_dict for pert" in str(excinfo.value)