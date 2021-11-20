import unittest

import numpy as np

from optimal_strategy_portfolio.metrics import sharpe, annualized_sharpe
from optimal_strategy_portfolio.signals.aggregators import HMean, HStd
from optimal_strategy_portfolio.signals.price_signals import PriceMASignal, R, RSI
from optimal_strategy_portfolio.signals.transforms import SMA
from optimal_strategy_portfolio.signals.variables import Var, BoolVar
from optimal_strategy_portfolio.single_strategy_optimizers import SingleStrategyOptimizer
from optimal_strategy_portfolio.solver_wrappers import pygad_wrapper
from optimal_strategy_portfolio.strategies import PriceMovingAverageCrossover
from shared_utils.data import get_data, get_securities_data, set_universe

from matplotlib import pyplot as plt


class TestSingleStrategyOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.symbs = ['GOOG', 'AMZN', 'FB', 'TSLA']
        self.data = get_securities_data(self.symbs)

        self.n_t = len(self.data['GOOG'])

    def test_price_moving_average_crossover_optimizer(self):
        prices = self.data['GOOG']
        strat = PriceMovingAverageCrossover(prices)
        bounds = np.array([[1, 300], [1, 300]])
        optimizer = SingleStrategyOptimizer(strat, bounds)
        x, f = optimizer.run()
        sharpe1 = -f

        optimizer2 = SingleStrategyOptimizer(strat, bounds, solver=pygad_wrapper)
        x, f = optimizer2.run()

        sharpe2 = f

        self.assertAlmostEqual(sharpe1 / 5, sharpe2 / 5, 1)

        fast_ma = PriceMASignal(prices)
        slow_ma = PriceMASignal(prices)

        crossover = (fast_ma > slow_ma) - (slow_ma > fast_ma)
        strat2 = crossover.to_strategy()

        optimizer3 = SingleStrategyOptimizer(strat2, bounds)

        x, f = optimizer3.run()

        sharpe3 = -f

        self.assertAlmostEqual(sharpe2 / 5, sharpe3 / 5, 1)

    def test_complex_strategy_optimization(self):
        fast_price_mas = \
            [PriceMASignal(self.data[symb], name=f'{symb}_fast_ma') for symb in self.symbs]
        slow_price_mas = \
            [PriceMASignal(self.data[symb], name=f'{symb}_slow_ma') for symb in self.symbs]
        return_signals = [R(self.data[symb], name=f'{symb}_returns') for symb in self.symbs]
        slow_sma_returns = [SMA(r) for r in return_signals]
        fast_sma_returns = [SMA(r) for r in return_signals]
        z = HMean(slow_sma_returns) + Var(bounds=((0, 5),)) * HStd(slow_sma_returns)

        price_crossover_buys = [fast > slow for fast, slow in zip(fast_price_mas, slow_price_mas)]
        price_crossover_sells = [slow > fast for fast, slow in zip(fast_price_mas, slow_price_mas)]

        relative_crossover_buys = [(fast > z) for fast, slow in zip(fast_sma_returns, slow_sma_returns)]
        relative_crossover_sells = [(fast < z) for fast, slow in zip(fast_sma_returns, slow_sma_returns)]

        abs_crossover_buys = [(fast > slow) for fast, slow in zip(fast_sma_returns, slow_sma_returns)]
        abs_crossover_sells = [(fast < slow) for fast, slow in zip(fast_sma_returns, slow_sma_returns)]

        buys = [(BoolVar() | b1) * (BoolVar() | b2) * (BoolVar() | b3)
                for b1, b2, b3 in zip(relative_crossover_buys, abs_crossover_buys, price_crossover_buys)]
        sells = [(BoolVar() | b1) * (BoolVar() | b2) * (BoolVar() | b3)
                 for b1, b2, b3 in zip(relative_crossover_sells, abs_crossover_sells, price_crossover_sells)]

        signal = buys[1] - sells[1]

        strat = signal.to_strategy()

        r = strat.optimize(solver=pygad_wrapper)

        optimal_x = r[0]
        optimal_returns = strat.execute(*optimal_x)
        optimal_sharpe = annualized_sharpe(optimal_returns)

        suboptimal_x = [(a + b) / 2 for a, b in strat.bounds]
        suboptimal_returns = strat.execute(*suboptimal_x)
        suboptimal_sharpe = annualized_sharpe(suboptimal_returns)

        self.assertGreaterEqual(optimal_sharpe, suboptimal_sharpe)

    def test_s(self):
        set_universe(['GOOG'])

        rsi = RSI('GOOG')
        s = rsi < 0.3 - rsi > 0.7

        s.to_strategy().optimize()













