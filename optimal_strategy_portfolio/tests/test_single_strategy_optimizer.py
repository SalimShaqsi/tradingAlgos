import unittest

import numpy as np

from optimal_strategy_portfolio.single_strategy_optimizer import SingleStrategyOptimizer
from optimal_strategy_portfolio.solver_wrappers import pygad_wrapper
from optimal_strategy_portfolio.strategies import PriceMovingAverageCrossover
from shared_utils.data import get_data


class TestSingleStrategyOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.symb = 'GOOG'
        self.price_data = get_data(self.symb)
        self.n_t = len(self.price_data)

    def test_price_moving_average_crossover_optimizer(self):
        strat = PriceMovingAverageCrossover(self.symb, self.price_data)
        bounds = np.array([[1, 300], [1, 300]])
        optimizer = SingleStrategyOptimizer(strat, bounds)
        x, f = optimizer.run()
        sharpe1 = -f

        optimizer2 = SingleStrategyOptimizer(strat, bounds, solver=pygad_wrapper)
        x, f = optimizer2.run()

        sharpe2 = f

        self.assertAlmostEqual(sharpe1, sharpe2, 1)



