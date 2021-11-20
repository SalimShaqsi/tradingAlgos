from operator import __add__, __gt__, __lt__, __ge__, __le__, __eq__

import numpy as np
import pandas as pd
from numba import njit

from optimal_strategy_portfolio.commision_models.ibkr_commission_models import ibkr_commission
from optimal_strategy_portfolio.single_strategy_optimizers import SingleStrategyOptimizer
from shared_utils.data import returns_from_prices, moving_averages
import quantstats as qs


class Strategy(object):
    n_args = 0
    is_composite = False
    bounds = ()

    def __init__(self, price_data: pd.DataFrame, bounds=(), arg_types=(), returns=None, meta_data=None,
                 trimmed_length=None, avg_cash=10_000, commission_func=ibkr_commission, slippage_factor=0.05,
                 name='buy_and_hold'):
        assert type(price_data) == pd.DataFrame, "Price data should be provided as a Pandas DataFrame"
        self.name = name
        self.price_data = price_data
        self.prices = price_data['Adj Close'].to_numpy()
        self.highs = price_data['High'].to_numpy()
        self.lows = price_data['Low'].to_numpy()
        self.opens = price_data['Open'].to_numpy()
        self.volume = price_data['Volume'].to_numpy()
        self.close = price_data['Close'].to_numpy()
        self.price_index = price_data.index
        self.n_t = len(self.prices)

        self.positions = np.ones(self.n_t - 1)

        self.base_returns = returns if returns is not None else returns_from_prices(self.prices)

        # returns when accounting for buy slippage and commission on order execution
        buy_prices = (self.highs - self.prices) * slippage_factor + self.prices
        buy_trade_values = buy_prices * avg_cash
        buy_share_amounts = avg_cash / buy_prices
        buy_commissions = commission_func(buy_trade_values, buy_share_amounts) if commission_func \
            else np.zeros(self.n_t)
        self.buy_returns = (self.prices[1:] - buy_prices[:-1] - buy_commissions[:-1]) / buy_prices[:-1]

        # returns when accounting for sell slippage and commission on order execution
        sell_prices = self.prices - (self.prices - self.lows) * slippage_factor
        sell_trade_values = sell_prices * avg_cash
        sell_share_amounts = avg_cash / sell_prices
        sell_commissions = commission_func(sell_trade_values, sell_share_amounts) if commission_func \
            else np.zeros(self.n_t)
        self.sell_returns = (sell_prices[1:] - self.prices[:-1] - sell_commissions[:-1]) / self.prices[:-1]

        self.meta_data = meta_data
        self.trimmed_length = trimmed_length or self.n_t - 1
        self.returns = np.zeros(self.trimmed_length)
        self.optimization_results = None

        self.bounds = bounds
        self.arg_types = arg_types

    def __str__(self):
        return f'{self.name}'.rstrip()

    def get_positions(self, *args):
        return self.positions

    # TODO: 'Numbaize' this and create matrix version for StrategyPortfolio
    def execute(self, *args):
        positions = self.get_positions(*args)[-self.trimmed_length:]

        diff = np.append(positions[0], np.diff(positions))

        longs = (diff > 0) * 1   # long trades
        shorts = (diff < 0) * 1  # short trades
        holds = (diff == 0) * 1  # holds

        self.returns = positions * \
            (holds * self.base_returns[-self.trimmed_length:] +
             longs * self.buy_returns[-self.trimmed_length:] +
             shorts * self.sell_returns[-self.trimmed_length:])
        return self.returns

    def test(self, *args):
        strat = self.test_instance if hasattr(self, 'test_instance') else self
        if not args and self.optimization_results:
            self.test_results = strat.execute(*self.optimization_results[0])
            return self
        else:
            return strat.execute(*args)

    def execute_to_pandas(self, *args):
        return pd.Series(self.execute(*args), index=self.price_index[-self.trimmed_length:])

    def optimize(self, optimizer_class=SingleStrategyOptimizer,
                 **optimizer_kwargs):
        optimizer = optimizer_class(self, **optimizer_kwargs)
        self.optimization_results = optimizer.run()

        return self

    def optimization_report(self, benchmark="SPY"):
        x = self.optimization_results[0]
        results = self.execute_to_pandas(*x)
        qs.reports.full(results, benchmark)

    def test_report(self, benchmark="SPY"):
        strat = self.test_instance if hasattr(self, 'test_instance') else self
        test_results_pd = pd.Series(self.test_results,
                                    index=strat.price_index[-strat.trimmed_length:])
        qs.reports.full(test_results_pd, benchmark)

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

    def __radd__(self, other):
        from optimal_strategy_portfolio.strategy_portforlios import StrategyPortfolio
        if isinstance(other, Strategy):
            return StrategyPortfolio([[other], [self]])
        elif isinstance(other, StrategyPortfolio):
            return other + StrategyPortfolio([[self]])
        else:
            raise TypeError(f"+ operator with type {type(other)} is not supported")


class StrategyFromSignal(Strategy):
    def __init__(self, signal, transform_function, trimmed_length=None, price_data=None, test_price_data=None,
                 **transform_function_kwargs):
        assert price_data or 'price_data' in signal.__dict__, \
            "Either a signal with price data or separate price data must be provided"
        price_data = price_data or signal.price_data
        super().__init__(price_data, trimmed_length=trimmed_length, bounds=signal.bounds, arg_types=signal.arg_types)
        self.n_args = signal.n_args
        self.transform_function = transform_function
        self.signal = signal
        self.trimmed_length = trimmed_length or signal.n_t - signal.lag
        self.transform_function_kwargs = transform_function_kwargs

        if hasattr(signal, 'test_instance'):
            test_price_data = test_price_data or signal.test_instance.price_data
            trimmed_length = len(test_price_data) - len(price_data) - 1
            self.test_instance = self.__class__(signal.test_instance, transform_function,
                                                trimmed_length=trimmed_length)

    @property
    def arg_names(self):
        return self.signal.arg_names

    def get_positions(self, *args):
        return self.transform_function(self.signal(*args), **self.transform_function_kwargs)


class PriceMovingAverageCrossover(Strategy):
    n_args = 2

    def __init__(self, *args, name='price_ma_crossover', **kwargs):
        super().__init__(*args, name=name, arg_types=(int, int),
                         bounds=((1, 300), (1, 300)), **kwargs)
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

