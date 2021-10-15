from operator import __add__, __gt__, __lt__, __ge__, __le__, __eq__

import numpy as np
import pandas as pd
from numba import njit

from shared_utils.data import returns_from_prices, moving_averages


class Strategy(object):
    n_args = 0
    name = 'Buy & Hold'
    is_composite = False

    def __init__(self, prices: pd.Series, returns=None, meta_data=None, trimmed_length=None,
                 test_prices=None, test_returns=None, test_meta_data=None, test_trimmed_length=None):
        self.symbol = prices.name
        self.n_t = len(prices)
        self.positions = np.ones(self.n_t - 1)
        self.prices = prices
        self.prices_np = prices.to_numpy()

        self.base_returns = returns or self._returns_from_prices()
        self.meta_data = meta_data
        self.trimmed_length = trimmed_length or self.n_t - 1
        self.returns = np.zeros(self.trimmed_length)
        self.args = ''
        self.test_instance = self.__class__(test_prices, returns=test_returns, meta_data=test_meta_data,
                                            trimmed_length=test_trimmed_length) if test_prices is not None else None

    def __str__(self):
        return f'{self.symbol} - {self.name} {self.args}'.rstrip()

    def _returns_from_prices(self):
        return returns_from_prices(self.prices_np)

    def get_positions(self, *args):
        return self.positions

    def execute(self, *args):
        self.args = str(args)
        self.returns = self.get_positions(*args)[-self.trimmed_length:] * \
            self.base_returns[-self.trimmed_length:]
        return self.returns

    def test(self, *args):
        assert self.test_instance, "Test data was not provided"
        return self.test_instance.execute(*args)


class StrategyFromSignal(Strategy):
    def __init__(self, signal, transform_function):
        super().__init__(signal.prices)
        self.n_args = signal.n_args
        self.transform_function = transform_function
        self.signal = signal
        self.trimmed_length = signal.n_t

    def get_positions(self, *args):
        return self.transform_function(self.signal.__call__(*args))


class PriceMovingAverageCrossover(Strategy):
    n_args = 2
    name = 'Price MA Crossover'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_data = self.meta_data or self.get_meta_data()
        self.trimmed_length = kwargs.get('trimmed_length', self.n_t - self.meta_data['ma_bounds'][1])
        self.returns = np.zeros(self.trimmed_length)

    def get_meta_data(self):
        meta_data = dict()
        meta_data['ma_bounds'] = bounds = np.array([1, 300])

        meta_data['mas'] = moving_averages(np.array([self.prices_np]), bounds)

        return meta_data

    def _ma(self, window_size):
        window_size = int(window_size)
        return self.meta_data['mas'][0, window_size - self.meta_data['ma_bounds'][0]][-self.trimmed_length-1:-1]

    def get_positions(self, fast_window, slow_window):
        return self._ma(fast_window) - self._ma(slow_window) > 0

