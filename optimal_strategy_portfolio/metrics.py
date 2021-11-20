from timeit import timeit

import numpy as np
from numba import njit
from pypfopt import expected_returns, risk_models

from optimal_strategy_portfolio.indicators import ema


@njit
def mean(x):
    return np.mean(x)


@njit
def std(x):
    return np.std(x)


@njit
def ew_mean(x, n=500):
    length = len(x) - n
    result = np.sum(x[0:n]) / n
    for i in range(length):
        result = result + (x[i + n] - result) * (2 / (n + 1))

    return result


@njit
def sharpe(r, mean_func=mean, risk_free_rate=0.02/252):
    if not np.any(r):
        return -np.inf  # return -infinity if there are no returns
    return (mean_func(r) - risk_free_rate) / std(r)


@njit
def sortino(r, mean_func=mean, risk_free_rate=0.02/252):
    if not np.any(r):
        return -np.inf  # return -infinity if there are no returns
    neg_r = r[r < risk_free_rate]
    if not neg_r.shape[0]:
        return np.inf
    sigma = std(neg_r)
    if not sigma:
        return np.inf
    return (mean_func(r) - risk_free_rate) / std(neg_r)


@njit
def trend(y):
    length = len(y)
    x = np.arange(length)
    ones = np.ones(length)
    a = np.vstack((x, ones)).T
    return np.linalg.lstsq(a, y)[0][0]


@njit
def annualized_sharpe(r, mean_func=mean, freq=252, risk_free_rate=0.02):
    if not np.any(r):
        return -np.inf  # return -infinity if there are no returns
    return (mean_func(r) * freq - risk_free_rate) / (std(r) * freq ** 0.5)


if __name__ == '__main__':
    returns = np.random.random(10_000)

    print(trend(returns))

