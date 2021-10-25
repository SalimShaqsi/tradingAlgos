from timeit import timeit

import numpy as np
import pandas as pd

from optimal_strategy_portfolio.signals import Signal
from shared_utils.data import moving_average4, get_data, get_securities_data
from optimal_strategy_portfolio.indicators import moving_average, moving_averages, ema, rsi


class PriceSignal(Signal):
    is_array = True
    lag = 2

    def __init__(self, price_data: pd.DataFrame):
        assert type(price_data) == pd.DataFrame, "Price data should be provided as a Pandas DataFrame"
        self.price_data = price_data
        self.prices = price_data['Adj Close'].to_numpy()
        self.highs = price_data['High'].to_numpy()
        self.lows = price_data['Low'].to_numpy()
        self.opens = price_data['Open'].to_numpy()
        self.volume = price_data['Volume'].to_numpy()
        self.close = price_data['Close'].to_numpy()
        self.index = price_data.index
        self.n_t = len(self.prices)  # last data point is dropped

    def transform(self, *args):
        return self.prices


class PriceMASignal(PriceSignal):
    n_args = 1
    is_array = True

    def __init__(self, price_data, lower_ma_bound=1, upper_ma_bound=300):
        super().__init__(price_data)
        self.n_t = self.n_t - upper_ma_bound

    def transform(self, window_size):
        return moving_average4(self.prices, window_size)


class PriceEMASignal(PriceSignal):
    n_args = 1
    is_array = True

    def __init__(self, price_data, lower_ma_bound=1, upper_ma_bound=300):
        super().__init__(price_data)
        self.n_t = self.n_t - upper_ma_bound

    def transform(self, window_size):
        return ema(self.prices, window_size)


class RSISignal(PriceSignal):
    n_args = 1
    is_array = True

    def __init__(self, price_data, lower_ma_bound=1, upper_ma_bound=300):
        super().__init__(price_data)
        self.n_t = self.n_t - upper_ma_bound

    def transform(self, window_size):
        return rsi(self.prices, window_size)


if __name__ == '__main__':
    prices = get_securities_data(['AMZN'])['AMZN']

    fast_ma, slow_ma = PriceMASignal(prices), PriceMASignal(prices)

    signal = (fast_ma > slow_ma) - (slow_ma > fast_ma)

    t1 = timeit('signal(20,200)', number=100_000, globals=globals())
    s = """
slow = fast_ma(200)
l = len(slow)
fast = fast_ma(20)[-l:]
(fast > slow) * 1 + (slow > fast) * -1
    """
    t2 = timeit(s, number=100_000, globals=globals())

    print(t1 / t2)

    cols = {'High', "Low", "Open", "Close", "Volume", "Adj Close"}
    data = pd.DataFrame({col: np.random.random(1_000_000) for col in cols})

    fast_ma, slow_ma = PriceMASignal(data), PriceMASignal(data)

    signal2 = (fast_ma > slow_ma) - (slow_ma > fast_ma)

    strat = signal2.to_strategy()

    t3 = timeit('strat.execute(20,200)', number=1, globals=globals())



