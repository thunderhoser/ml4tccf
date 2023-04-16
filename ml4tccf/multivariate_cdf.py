"""Helper methods for computing empirical multivariate CDF.

CDF = cumulative density function

Downloaded from: https://github.com/statsmodels/statsmodels/blob/main/
                 statsmodels/distributions/tools.py
"""

import numpy as np


def _rankdata_no_ties(x):
    """rankdata without ties for 2-d array
    This is a simplified version for ranking data if there are no ties.
    Works vectorized across columns.
    See Also
    --------
    scipy.stats.rankdata
    """
    nobs, k_vars = x.shape
    ranks = np.ones((nobs, k_vars))
    sidx = np.argsort(x, axis=0)
    ranks[sidx, np.arange(k_vars)] = np.arange(1, nobs + 1)[:, None]
    return ranks


def _ecdf_mv(data, method="seq", use_ranks=True):
    """
    Multivariate empiricial distribution function, empirical copula
    Notes
    -----
    Method "seq" is faster than method "brute", but supports mainly bivariate
    case. Speed advantage of "seq" is increasing in number of observations
    and decreasing in number of variables.
    (see Segers ...)
    Warning: This does not handle ties. The ecdf is based on univariate ranks
    without ties. The assignment of ranks to ties depends on the sorting
    algorithm and the initial ordering of the data.
    When the original data is used instead of ranks, then method "brute"
    computes the correct ecdf counts even in the case of ties.
    """
    x = np.asarray(data)
    n = x.shape[0]
    if use_ranks:
        x = _rankdata_no_ties(x) / n
    if method == "brute":
        count = [((x <= x[i]).all(1)).sum() for i in range(n)]
        count = np.asarray(count)
    elif method.startswith("seq"):
        sort_idx0 = np.argsort(x[:, 0])
        x_s0 = x[sort_idx0]
        x1 = x_s0[:, 1:]
        count_smaller = [(x1[:i] <= x1[i]).all(1).sum() + 1 for i in range(n)]
        count = np.empty(x.shape[0])
        count[sort_idx0] = count_smaller
    else:
        raise ValueError("method not available")

    return count, x
