import pytest

from magellan.prune import check_pert_dic_values


def test_check_pert_dic_values():
    # Valid case - all values within bounds
    valid_dic = {
        "node1": {
            "pert": {
                "param1": 0.5,
                "param2": 0.3
            }
        },
        "node2": {
            "pert": {
                "param1": 0.7,
                "param2": 0.2
            }
        }
    }
    
    # Should not raise error
    check_pert_dic_values(valid_dic, 0.0, 1.0)
    
    # Invalid case - value below min_val
    invalid_low = {
        "node1": {
            "pert": {
                "param1": -0.1,
                "param2": 0.3
            }
        }
    }
    
    with pytest.raises(ValueError) as exc_info:
        check_pert_dic_values(invalid_low, 0.0, 1.0)
    assert "Value -0.1 for param1 in node1" in str(exc_info.value)
    
    # Invalid case - value above max_val
    invalid_high = {
        "node1": {
            "pert": {
                "param1": 1.2,
                "param2": 0.3
            }
        }
    }
    
    with pytest.raises(ValueError) as exc_info:
        check_pert_dic_values(invalid_high, 0.0, 1.0)
    assert "Value 1.2 for param1 in node1" in str(exc_info.value)
    
    # Test with different min_val and max_val
    custom_bounds = {
        "node1": {
            "pert": {
                "param1": 2.5,
                "param2": 3.0
            }
        }
    }
    
    # Should not raise error
    check_pert_dic_values(custom_bounds, 2.0, 4.0)
    
    # Test empty dictionary
    empty_dic = {}
    check_pert_dic_values(empty_dic, 0.0, 1.0)  # Should not raise error