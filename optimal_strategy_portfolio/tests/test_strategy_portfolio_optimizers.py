import unittest

import numpy as np

from optimal_strategy_portfolio.metrics import sharpe, mean, ew_mean, std
from optimal_strategy_portfolio.signals.price_signals import PMA
from optimal_strategy_portfolio.single_strategy_optimizers import SingleStrategyOptimizer
from optimal_strategy_portfolio.solver_wrappers import pygad_wrapper
from optimal_strategy_portfolio.strategies import PriceMovingAverageCrossover, Strategy
from optimal_strategy_portfolio.strategy_portfolio_optimizers import BilevelStrategyPortfolioOptimizer
from optimal_strategy_portfolio.strategy_portforlios import StrategyPortfolio
from shared_utils.data import get_data, get_securities_data, set_universe


class TestSingleStrategyOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.symbols = ['GOOG', 'TSLA', 'AMZN', 'FB', 'SPY']
        self.price_data = get_securities_data(self.symbols)

    def test_strategy_portfolio_optimizer(self):
        set_universe(self.symbols)

        fast_mas = [PMA(symb) for symb in self.symbols]
        slow_mas = [PMA(symb) for symb in self.symbols]

        signals = [(fast > slow) - (slow > fast) for fast, slow in
                   zip(fast_mas, slow_mas)]

        port = StrategyPortfolio([signal.to_strategy() for signal in signals])




    def test_bilevel_strategy_portfolio_optimizer(self):
        strats = [PriceMovingAverageCrossover(self.price_data[symb], name=f'pma_{symb}1') for symb in self.symbols]
        strats += [PriceMovingAverageCrossover(self.price_data[symb], name=f'pma_{symb}2') for symb in self.symbols]
        strats += [Strategy(self.price_data[symb], name=f'bnh_{symb}') for symb in self.symbols]

        portfolio = StrategyPortfolio(strats)

        bounds = np.array([[1, 300]] * len(self.symbols)*4)
        strategy_portfolio_optimizer = BilevelStrategyPortfolioOptimizer(portfolio, bounds)

        self.assertEqual(strategy_portfolio_optimizer.strategy_portfolio, portfolio)
        self.assertTrue(np.array_equal(strategy_portfolio_optimizer.bounds, bounds))

        r = strategy_portfolio_optimizer.run()
        print(r)
        args = r[0]

        w = strategy_portfolio_optimizer.get_optimal_weights(r[0])
        weights = np.array(list(w.values()))
        returns = portfolio.execute_with_weights(args, weights)
