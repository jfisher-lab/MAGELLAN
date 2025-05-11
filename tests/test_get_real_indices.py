import pandas as pd
import pytest

from magellan.prune import get_real_indices


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        index=['node01', 'dummy_node', 'node00', 'node02']
    )

def test_get_real_indices(sample_df):
    result = get_real_indices(sample_df)
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)
    assert result == ['node01', 'node02']
    
def test_get_real_indices_empty():
    empty_df = pd.DataFrame(index=['dummy_node', 'node00'])
    assert get_real_indices(empty_df) == []

def test_get_real_indices_all_real():
    real_df = pd.DataFrame(index=['node01', 'node02', 'node03'])
    result = get_real_indices(real_df)
    assert len(result) == 3
    assert all('dummy' not in x and 'node00' not in x for x in result)