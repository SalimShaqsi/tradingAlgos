from functools import cache
from timeit import timeit
from typing import Union

import numpy as np
import pandas as pd

from optimal_strategy_portfolio.signals import Signal
from shared_utils.data import moving_average4, get_data, get_securities_data, returns_from_prices, universe, \
    set_universe
from optimal_strategy_portfolio.indicators import moving_average, moving_averages, ema, rsi


class PriceSignal(Signal):
    is_array = True
    lag = 2  # this is set to 2 because we can only react to close data the following day

    def __init__(self, price_data: Union[pd.DataFrame, str], name='price', bounds=(), arg_types=()):
        super(PriceSignal, self).__init__(name=name, bounds=bounds, arg_types=arg_types)
        assert type(price_data) in [pd.DataFrame, str], \
            "Price data should be provided as a Pandas DataFrame or a string referencing a security in the " \
            "securities universe"
        if type(price_data) == str:
            symb = price_data
            self.name = f'{symb}_{self.name}'
            price_data = universe[symb]
            if universe['HAS_TEST_DATA'] and '_TEST' not in symb:
                self.test_instance = self.__class__(f'{symb}_TEST', name=name)
        self.price_data = price_data
        self.prices = price_data['Adj Close'].to_numpy()
        self.highs = price_data['High'].to_numpy()
        self.lows = price_data['Low'].to_numpy()
        self.opens = price_data['Open'].to_numpy()
        self.volume = price_data['Volume'].to_numpy()
        self.close = price_data['Close'].to_numpy()
        self.index = price_data.index
        self.n_t = len(self.prices)

    def transform(self, **kwargs):
        return self.prices


class PriceMASignal(PriceSignal):
    n_args = 1
    is_array = True

    def __init__(self, price_data, bounds=((1, 300),), name='price_sma'):
        super().__init__(price_data, name=name, bounds=bounds, arg_types=(int,))
        self.n_t = self.n_t - bounds[0][1]

    @cache
    def transform(self, window_size, **kwargs):
        return moving_average4(self.prices, window_size)


class PriceEMASignal(PriceSignal):
    n_args = 1
    is_array = True

    def __init__(self, price_data, bounds=((1, 300),), name='price_ema'):
        super().__init__(price_data, name=name, bounds=bounds, arg_types=(int,))
        self.n_t = self.n_t - bounds[0][1]

    @cache
    def transform(self, window_size, **kwargs):
        return ema(self.prices, window_size)


class RSISignal(PriceSignal):
    n_args = 1
    is_array = True

    def __init__(self, price_data, bounds=((2, 300),), name='rsi'):
        super().__init__(price_data, name=name, bounds=bounds, arg_types=(int,))
        self.n_t = self.n_t - bounds[0][1]

    @cache
    def transform(self, window_size, **kwargs):
        return rsi(self.prices, window_size)


class ReturnsSignal(PriceSignal):
    def __init__(self, price_data, name='returns'):
        super().__init__(price_data, name=name)
        self.returns = returns_from_prices(self.prices)
        self.n_t = len(self.returns)

    def transform(self, *args, **kwargs):
        return self.returns


class VolumeSignal(PriceSignal):
    def __init__(self, price_data, name='volume'):
        super().__init__(price_data, name=name)
        self.n_t = len(self.volume)

    def transform(self, *args, **kwargs):
        return self.volume


Price = P = PriceSignal
PriceSMA = PriceMA = PMA = PSMA = PriceMASignal
PriceEMA = PEMA = PriceEMASignal
Returns = R = ReturnsSignal
Volume = V = VolumeSignal
RSI = RSISignal


if __name__ == '__main__':
    symbols = ['GOOG', 'FB', "TSLA", "AMZN"]
    set_universe(symbols)
    fast_pmas = [PMA(symb) for symb in symbols]
    slow_pmas = [PMA(symb) for symb in symbols]
    signals = [(fast > slow) - (slow > fast) for fast, slow in zip(fast_pmas, slow_pmas)]
    strats = [s.to_strategy() for s in signals]




