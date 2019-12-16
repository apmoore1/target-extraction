import pytest

from target_extraction.analysis import statistical_analysis

@pytest.mark.parametrize('method', ('B', ))
def test_find_k_estimator(method: str):
    # Case where none of the p-values are significant as they are all above 
    # alpha
    p_values = [0.05, 0.06, 0.07]
    print(p_values)
    assert 0 == statistical_analysis.find_k_estimator(p_values, alpha=0.05, method=method)
    # Case where we have zero p-values
    p_values = []
    assert 0 == statistical_analysis.find_k_estimator(p_values, alpha=0.05, method=method)
    # Case that 1 should be returned as the p-value is less than alpha and 
    # there is only one p-value
    p_values = [0.04]
    assert 1 == statistical_analysis.find_k_estimator(p_values, alpha=0.05, method=method)
    # Ensure that changining alpha does not make a difference when the p_values 
    # also change
    p_values = [0.009]
    assert 1 == statistical_analysis.find_k_estimator(p_values, alpha=0.01, method=method)
    
    p_values = [0.0003, 0.002, 0.007]
    if method == 'B':
        assert 3 == statistical_analysis.find_k_estimator(p_values, alpha=0.01, method=method)
    else:
        assert 2 == statistical_analysis.find_k_estimator(p_values, alpha=0.01, method=method)
    # Ensure that it can return a number of p_values > 0 and less than the total 
    # number of p_values given
    p_values = [0.0003, 0.002, 0.002, 0.0055,0.0095]
    if method == 'B':
        assert 3 == statistical_analysis.find_k_estimator(p_values, alpha=0.01, method=method)
    else:
        assert 2 == statistical_analysis.find_k_estimator(p_values, alpha=0.01, method=method)

def test_calc_partial_cunjunction():
    # Test that the a ValueError is raisesd when the method argument is not 
    # `B` or `F`
    p_values = [0.001, 0.002]
    u = 1
    with pytest.raises(ValueError):
        statistical_analysis._calc_partial_cunjunction(p_values, u, 'A')
    with pytest.raises(NotImplementedError):
        statistical_analysis._calc_partial_cunjunction(p_values, u, 'F')