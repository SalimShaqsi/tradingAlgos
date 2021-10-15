import unittest

import numpy as np

from optimal_strategy_portfolio.single_strategy_optimizer import SingleStrategyOptimizer
from optimal_strategy_portfolio.solver_wrappers import pygad_wrapper
from optimal_strategy_portfolio.strategies import PriceMovingAverageCrossover, Strategy
from optimal_strategy_portfolio.strategy_portfolio_optimizers import BilevelStrategyPortfolioOptimizer
from optimal_strategy_portfolio.strategy_portforlios import StrategyPortfolio
from shared_utils.data import get_data


class TestSingleStrategyOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.symbols = ['GOOG', 'TSLA', 'AMZN', 'FB', 'SPY']
        self.price_data = get_data(self.symbols)

    def test_strategy_portfolio_optimizer(self):
        strats = [PriceMovingAverageCrossover(symb, self.price_data[symb]) for symb in self.symbols]
        strats += [Strategy(symb, self.price_data[symb]) for symb in self.symbols]

        portfolio = StrategyPortfolio(strats)

        bounds = np.array([[1, 300]]*len(self.symbols)*2)
        strategy_portfolio_optimizer = BilevelStrategyPortfolioOptimizer(portfolio, bounds)

        self.assertEqual(strategy_portfolio_optimizer.strategy_portfolio, portfolio)
        self.assertTrue(np.array_equal(strategy_portfolio_optimizer.bounds, bounds))

        strategy_portfolio_optimizer.run()
