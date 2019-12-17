from typing import List

import numpy as np
from scipy import stats

def find_k_estimator(p_values: List[float], alpha: float,
                     method: str = 'B') -> int:
    '''
    Given a list of p-values returns the number of those p-values that are
    significant at the level of alpha according to either the Bonferroni or
    Fisher correction method.
    This code has come from `Dror et al. 2017 paper 
    <https://aclanthology.coli.uni-saarland.de/papers/Q17-1033/q17-1033>`_.
    Code base for the paper `here 
    <https://github.com/rtmdrr/replicability-analysis-NLP/blob/master/Replicability_Analysis.py>`_
    Fisher is used if the p-values have come from an independent set i.e. method
    p-values results from independent datasets. Bonferroni used if this
    independent assumption is not True.

    **Fisher** is currently not implemented.
    
    :param p_values: list of p-values.
    :param alpha: significance level.
    :param method: 'B' for Bonferroni
    :return: Number of datasets that are significant at the level of alpha for
             the p_values given.
    :raises NotImplementedError: If `F` is given for the `method` argument.
    '''
    n = len(p_values)
    pc_vec = [1] * n
    k_hat = 0
    p_values = sorted(p_values, reverse=True)
    for u in range(0, n):
        if (u == 0):
            pc_vec[u] = _calc_partial_cunjunction(p_values, u + 1, method)
        else:
            pc_vec[u] = max(_calc_partial_cunjunction(p_values, u + 1, method),
                            pc_vec[u - 1])
    k_hat = len([i for i in pc_vec if i <= alpha])
    return k_hat

def _calc_partial_cunjunction(p_values: List[float], u: int,
                              method: str = 'B') -> float:
    '''
    This function calculates the partial conjunction p-value of u out of n.
    This code has come from `Dror et al. 2017 paper 
    <https://aclanthology.coli.uni-saarland.de/papers/Q17-1033/q17-1033>`_.
    Code base for the paper `here 
    <https://github.com/rtmdrr/replicability-analysis-NLP/blob/master/Replicability_Analysis.py>`_
    
    :param p_values: list of p-values.
    :param u: number of hypothesized true null hypotheses.
    :param method: 'B' for Bonferroni or 'F' for Fisher default Bonferroni.
    :returns: Number of datasets that are significant at the level of alpha for
              the p_values given.
    :raises NotImplementedError: If `F` is given for the `method` argument.
    :raises ValueError: If the `method` argument is not `B` or `F`.
    '''
    n = len(p_values)
    sorted_pvlas = p_values[0:(n - u + 1)]
    if method == 'B':
        p_u_n = (n - u + 1) * min(sorted_pvlas)
    elif method == 'F':
        raise NotImplementedError('Currently the Fisher test is under '
                                  're-factoring and is not working')
        #sum_chi_stat = 0
        #for p in sorted_pvlas:
        #    sum_chi_stat = sum_chi_stat - 2 * np.log(p)
        #p_u_n = 1 - stats.chi2.cdf(sum_chi_stat, 2 * (n - u + 1))
    else:
        method_err = ('The method argument has to be either `B` or `F` and '
                      f'not {method}')
        raise ValueError(method_err)

    return p_u_n

def one_tailed_p_value(scores_1: List[float], scores_2: List[float], 
                       assume_normal: bool) -> float:
    '''
    :param scores_1: The scores e.g. list of accuracy values that reprsent one 
                     model/methods results (multiple scores can come from running 
                     the same model/method over different random seeds and/or 
                     dataset splits).
    :param scores_2: Same as `scores_1` but coming from a different method/model
    :param assume_normal: If the the scores are assumed to come from a normal 
                          distribution. See the following guide by 
                          `Dror and Reichart 2018 <https://arxiv.org/pdf/1809.01448.pdf>`_
                          to know if your metric/scores can be assumed to be normal or 
                          not. The test used when the scores are normal is the 
                          Welch’s t-test. When not normal it is the 
                          Wilcoxon signed-rank test.
    :returns: The p-value of a one-tailed test to determine if `scores_1` is 
              better than `scores_2`.
    '''
    p_value = -1.0
    if assume_normal:
        stat_result = stats.ttest_ind(scores_1, scores_2, equal_var=False, 
                                      nan_policy='raise')
        # one tailed hence why divided by 2
        p_value = stat_result.pvalue / 2
        # This is to take into account if the other is better
        if stat_result.statistic < 0:
            p_value = 1 - p_value
    else:
        stat_result = stats.wilcoxon(scores_1, scores_2, correction=False, 
                                     alternative='greater', zero_method='wilcox')
        p_value = stat_result.pvalue
    if p_value == -1.0:
        raise ValueError('P-Value was not calculated for these scores: '
                         f'score_1 {score_1} and scores_2 {scores_2} where '
                         f'the assumption of normality is {assume_normal}')
    return p_value