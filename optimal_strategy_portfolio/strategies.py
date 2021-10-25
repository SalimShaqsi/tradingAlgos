from operator import __add__, __gt__, __lt__, __ge__, __le__, __eq__

import numpy as np
import pandas as pd
from numba import njit


from shared_utils.data import returns_from_prices, moving_averages


class Strategy(object):
    n_args = 0
    name = 'Buy & Hold'
    is_composite = False

    def __init__(self, price_data: pd.DataFrame, returns=None, meta_data=None, trimmed_length=None):
        assert type(price_data) == pd.DataFrame, "Price data should be provided as a Pandas DataFrame"
        self.price_data = price_data
        self.prices = price_data['Adj Close'].to_numpy()
        self.highs = price_data['High'].to_numpy()
        self.lows = price_data['Low'].to_numpy()
        self.opens = price_data['Open'].to_numpy()
        self.volume = price_data['Volume'].to_numpy()
        self.close = price_data['Close'].to_numpy()
        self.index = price_data.index
        self.n_t = len(self.prices)
        self.positions = np.ones(self.n_t - 1)

        self.base_returns = returns or self._returns_from_prices()
        self.meta_data = meta_data
        self.trimmed_length = trimmed_length or self.n_t - 1
        self.returns = np.zeros(self.trimmed_length)
        self.args = ''

    def __str__(self):
        return f'{self.name} {self.args}'.rstrip()

    def _returns_from_prices(self):
        return returns_from_prices(self.prices)

    def get_positions(self, *args):
        return self.positions

    def execute(self, *args):
        self.args = str(args)
        self.returns = self.get_positions(*args)[-self.trimmed_length:] * \
            self.base_returns[-self.trimmed_length:]
        return self.returns

    def __matmul__(self, other):
        from optimal_strategy_portfolio.strategy_portforlios import StrategyPortfolio
        if isinstance(other, Strategy):
            if self.n_args != other.n_args:
                raise ValueError("Strategies with different numbers of arguments cannot be linked")
            return StrategyPortfolio([[self, other]])
        elif isinstance(other, StrategyPortfolio):
            return StrategyPortfolio([[self]]) @ other
        else:
            raise TypeError(f"@ operator with type {type(other)} is not supported")

    def __add__(self, other):
        from optimal_strategy_portfolio.strategy_portforlios import StrategyPortfolio
        if isinstance(other, Strategy):
            return StrategyPortfolio([[self], [other]])
        elif isinstance(other, StrategyPortfolio):
            return StrategyPortfolio([[self]]) + other
        else:
            raise TypeError(f"+ operator with type {type(other)} is not supported")


class StrategyFromSignal(Strategy):
    def __init__(self, signal, transform_function, price_data=None, **transform_function_kwargs):
        assert price_data or 'price_data' in signal.__dict__, \
            "Either a signal_arr with price data or separate price data must be provided"
        price_data = price_data or signal.price_data
        super().__init__(price_data)
        self.n_args = signal.n_args
        self.transform_function = transform_function
        self.signal = signal
        self.trimmed_length = signal.n_t
        self.transform_function_kwargs = transform_function_kwargs

    def get_positions(self, *args):
        return self.transform_function(self.signal(*args), **self.transform_function_kwargs)


class PriceMovingAverageCrossover(Strategy):
    n_args = 2
    name = 'Price MA Crossover'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_data = self.meta_data or self.get_meta_data()
        self.trimmed_length = kwargs.get('trimmed_length', self.n_t - self.meta_data['ma_bounds'][1] - 1)
        self.returns = np.zeros(self.trimmed_length)

    def get_meta_data(self):
        meta_data = dict()
        meta_data['ma_bounds'] = bounds = np.array([1, 300])

        meta_data['mas'] = moving_averages(np.array([self.prices]), bounds)

        return meta_data

    def _ma(self, window_size):
        window_size = int(window_size)
        return self.meta_data['mas'][0, window_size - self.meta_data['ma_bounds'][0]]

    def get_positions(self, fast_window, slow_window):
        return (self._ma(fast_window) - self._ma(slow_window) > 0)[:-2]

