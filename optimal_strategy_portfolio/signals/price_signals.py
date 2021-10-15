from optimal_strategy_portfolio.signals import Signal
from shared_utils.data import moving_average4


class PriceSignal(Signal):
    is_array = True

    def __init__(self, prices):
        self.prices = prices
        self.index = prices.index
        self.n_t = len(prices) - 1
        self.prices_np = prices.to_numpy()

    def transform(self, *args):
        return self.prices_np[:-1]


class PriceMASignal(PriceSignal):
    n_args = 1
    is_array = True

    def __init__(self, prices, lower_ma_bound=1, upper_ma_bound=300):
        super().__init__(prices)
        self.n_t = self.n_t - upper_ma_bound

    def transform(self, window_size):
        return moving_average4(self.prices_np[:-1], window_size)[-self.n_t:]
