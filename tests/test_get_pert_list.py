from magellan.prune import get_pert_list


def test_get_pert_list_basic():
    # Basic test case with simple perturbation dictionary
    pert_dic = {
        'exp1': {'pert': {'node1': 1, 'node2': 0}},
        'exp2': {'pert': {'node2': 1, 'node3': 1}}
    }
    inh = []
    
    result = get_pert_list(pert_dic, inh)
    assert result == ['node1', 'node2', 'node3']

def test_get_pert_list_with_inhibitors():
    # Test with inhibitors that should be excluded
    pert_dic = {
        'exp1': {'pert': {'node1': 1, 'inh1': 0}},
        'exp2': {'pert': {'node2': 1, 'inh1': 1, 'inh2': 0}}
    }
    inh = ['inh1', 'inh2']
    
    result = get_pert_list(pert_dic, inh)
    assert result == ['node1', 'node2']

def test_get_pert_list_empty():
    # Test with empty perturbation dictionary
    pert_dic = {}
    inh = []
    
    result = get_pert_list(pert_dic, inh)
    assert result == []

def test_get_pert_list_only_inhibitors():
    # Test when all perturbations are inhibitors
    pert_dic = {
        'exp1': {'pert': {'inh1': 1, 'inh2': 0}},
        'exp2': {'pert': {'inh1': 0, 'inh2': 1}}
    }
    inh = ['inh1', 'inh2']
    
    result = get_pert_list(pert_dic, inh)
    assert result == []