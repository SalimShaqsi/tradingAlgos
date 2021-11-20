from timeit import timeit

import numpy as np
from numba import jit
import pandas as pd
from pandas_datareader import data
from scipy.signal import lfilter
from scipy.ndimage.filters import uniform_filter1d

from os.path import join, dirname
from datetime import datetime, timedelta


yesterday = datetime.now() - timedelta(days=1)
yesterday_str = yesterday.strftime('%Y-%m-%d')
dir = dirname(__file__)
universe = dict()


def set_universe(symbols, start_date='2014-01-01', end_date=yesterday_str, test_start_date=None):
    universe.clear()
    data = get_securities_data(symbols, start_date=start_date, end_date=end_date)
    universe.update({'HAS_TEST_DATA': test_start_date is not None})
    if test_start_date:
        train_data = {f'{symb}': d[:test_start_date] for symb, d in data.items()}
        test_data = {f'{symb}_TEST': d for symb, d in data.items()}
        test_data_for_returns = {f'{symb}_TEST_R': d[test_start_date:] for symb, d in data.items()}
        universe.update(train_data)
        universe.update(test_data)
        universe.update(test_data_for_returns)
    else:
        universe.update(data)


def get_data(symbols, start_date='2014-01-01', end_date='2020-01-01'):
    symbols_expr = str(symbols).replace('[', '').replace(']', '').replace(' ', '').replace(',', '_').replace("'", "")
    SRC_DATA_FILENAME = f'old__{symbols_expr}_{start_date}_{end_date}_data.pkl'
    SRC_DATA_FILENAME = join(dir, 'cache', SRC_DATA_FILENAME)
    try:
        d = pd.read_pickle(SRC_DATA_FILENAME)
    except FileNotFoundError:
        d = data.DataReader(symbols, 'yahoo', start_date, end_date)
        d.to_pickle(SRC_DATA_FILENAME)

    return d['Adj Close']


def get_securities_data(symbols, start_date='2014-01-01', end_date='2020-01-01'):
    symbols_expr = str(symbols).replace('[', '').replace(']', '').replace(' ', '').replace(',', '_').replace("'", "")
    SRC_DATA_FILENAME = f'{symbols_expr}_{start_date}_{end_date}_data.pkl'
    SRC_DATA_FILENAME = join(dir, 'cache', SRC_DATA_FILENAME)
    try:
        d = pd.read_pickle(SRC_DATA_FILENAME)
    except FileNotFoundError:
        d = data.DataReader(symbols, 'yahoo', start_date, end_date)
        d.to_pickle(SRC_DATA_FILENAME)

    n_s = len(symbols)
    r_data = dict()
    for i, symb in enumerate(symbols):
        r_data[symb] = d.iloc[:, range(i, n_s*6, n_s)]
        r_data[symb].columns = [c[0] for c in r_data[symb].columns]
    return r_data


def exponential_ma(array, alpha):
    alpha = 1 - alpha
    zi = [array[0]]
    y, zi = lfilter([1. - alpha], [1., -alpha], array, zi=zi)
    y[0] = array[0]
    return y


def random_walk(size):
    walk = np.random.choice([-1, 1], size).cumsum()
    return walk - min(walk) + 1


def moving_average(x, w):
    len_x = len(x)
    assert w > 0, "Moving average window must be greater than 0"
    #assert x.dim == 1, "Data must be one dimensional"
    assert len_x > 2, "Data must have at least 2 points"
    return np.convolve(x, np.ones(w), 'valid') / w


def moving_average2(x, n):
    return uniform_filter1d(x, n, mode='constant', origin=-(n//2))[:-(n-1)]


def moving_average3(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / n


@jit(nopython=True)
def mult(x, y):
    return x * y


@jit(nopython=True)
def moving_average4(x, n):
    cumsum = np.cumsum(np.append(0, x))
    return (cumsum[n:] - cumsum[:-n]) / n


@jit(nopython=True)
def moving_averages(data, bounds):
    zeros = np.zeros((data.shape[0], 1))
    data = np.hstack((zeros, data))
    x_shape = data.shape
    n_s = x_shape[0]
    n_t = x_shape[1]
    min_window = bounds[0]
    max_window = bounds[1]
    arr = np.zeros((n_s, max_window - min_window + 1, n_t - max_window))
    cumsums = [np.cumsum(d) for d in data]

    for i, cumsum in enumerate(cumsums):
        for j in range(1 + max_window - min_window):
            n = j + min_window
            arr[i, j] = (cumsum[n:] - cumsum[:-n])[-n_t + max_window:] / n

    return arr


@jit(nopython=True)
def returns_from_prices(prices):
    return np.diff(prices) / prices[:-1]


if __name__ == '__main__':
    symbs = ['GOOG', 'FB', 'AMZN', "MSFT", 'TSLA']
    set_universe(symbs)

    print(universe['GOOG'])



