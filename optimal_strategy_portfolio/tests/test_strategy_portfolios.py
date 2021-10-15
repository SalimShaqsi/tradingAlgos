import unittest

import numpy as np

from optimal_strategy_portfolio.single_strategy_optimizer import SingleStrategyOptimizer
from optimal_strategy_portfolio.solver_wrappers import pygad_wrapper
from optimal_strategy_portfolio.strategies import PriceMovingAverageCrossover, Strategy
from optimal_strategy_portfolio.strategy_portforlios import StrategyPortfolio
from shared_utils.data import get_data


class TestStrategyPortfolios(unittest.TestCase):
    def setUp(self) -> None:
        self.symbols = ['GOOG', 'TSLA', 'AMZN', 'FB', 'SPY']
        self.price_data = get_data(self.symbols)

    def test_strategy_portfolio(self):
        strats = [PriceMovingAverageCrossover(symb, self.price_data[symb]) for symb in self.symbols]
        strats += [Strategy(symb, self.price_data[symb]) for symb in self.symbols]

        portfolio = StrategyPortfolio(strats)

        self.assertEqual(portfolio.strats, strats)
        self.assertEqual(portfolio.n_args, sum([strat.n_args for strat in strats]))

        returns_matrix = portfolio.execute(np.array([50, 200]*len(self.symbols)))

        self.assertEqual(returns_matrix.shape, (len(strats), strats[0].trimmed_length))
