import unittest

import numpy as np

from optimal_strategy_portfolio.signals.price_signals import RSISignal
from optimal_strategy_portfolio.single_strategy_optimizer import SingleStrategyOptimizer
from optimal_strategy_portfolio.solver_wrappers import pygad_wrapper
from optimal_strategy_portfolio.strategies import PriceMovingAverageCrossover, Strategy
from optimal_strategy_portfolio.strategy_portforlios import StrategyPortfolio
from shared_utils.data import get_data, get_securities_data


class TestStrategyPortfolios(unittest.TestCase):
    def setUp(self) -> None:
        self.symbols = ['GOOG', 'TSLA', 'AMZN', 'FB', 'SPY']
        self.price_data = get_securities_data(self.symbols)

    def test_strategy_portfolio(self):
        strats = [PriceMovingAverageCrossover(self.price_data[symb]) for symb in self.symbols]
        strats += [Strategy(self.price_data[symb]) for symb in self.symbols]

        portfolio = StrategyPortfolio(strats)

        self.assertEqual(portfolio.n_args, sum([strat.n_args for strat in strats]))
        self.assertEqual(portfolio.strat_structure, [[strat] for strat in strats])

        self.assertEqual(portfolio.strats, strats)
        self.assertEqual(portfolio.n_args, sum([strat.n_args for strat in strats]))
        self.assertEqual(portfolio.trimmed_length, min(strat.trimmed_length for strat in strats))

        returns_matrix = portfolio.execute(np.array([50, 200]*len(self.symbols)))
        returns_matrix_test = np.array([strat.execute(50, 200)[-portfolio.trimmed_length:] for strat in strats])

        self.assertEqual(returns_matrix.shape, (len(strats), strats[0].trimmed_length))

        self.assertTrue(np.array_equal(
            returns_matrix, returns_matrix_test
        ))

    def test_linked_strategies(self):
        strats_1: [Strategy] = [PriceMovingAverageCrossover(self.price_data[symb]) for symb in self.symbols]
        strats_2: [Strategy] = [Strategy(self.price_data[symb]) for symb in self.symbols]
        strats = strats_1 + strats_2

        struct = [strats_1, strats_2]

        portfolio = StrategyPortfolio(struct)

        self.assertEqual(portfolio.n_args, PriceMovingAverageCrossover.n_args)
        self.assertEqual(portfolio.strat_structure, struct)
        self.assertEqual(portfolio.strats, strats)

        returns_matrix = portfolio.execute(np.array([50, 200]))
        returns_matrix_test = np.array([strat.execute(50, 200)[-portfolio.trimmed_length:] for strat in strats])

        self.assertEqual(returns_matrix.shape, (len(strats), strats[0].trimmed_length))

        self.assertTrue(np.array_equal(
            returns_matrix, returns_matrix_test
        ))

    def test_portfolio_add_operator(self):
        prices = [self.price_data[symb] for symb in self.symbols]
        strat1 = PriceMovingAverageCrossover(prices[0])
        strat2 = PriceMovingAverageCrossover(prices[1])
        rsi = RSISignal(prices[2])

        strat3 = (rsi < 0.3 - rsi > 0.7).to_strategy()

        port1 = strat1 + strat2

        self.assertIsInstance(port1, StrategyPortfolio)
        self.assertEqual(port1.n_args, strat1.n_args + strat2.n_args)
        self.assertEqual(port1.strat_structure, [[strat1], [strat2]])
        self.assertEqual(port1.strats, [strat1, strat2])
        self.assertEqual(port1.trimmed_length, min(strat1.trimmed_length, strat2.trimmed_length))
        self.assertEqual(port1.reshaped_args, [[0] * strat1.n_args, [0]*strat2.n_args])
        self.assertEqual(port1._reshape_args([20, 200, 20, 200]), [[20, 200], [20, 200]])

        self.assertTrue(np.array_equal(
            port1.execute([20, 200, 20, 200]),
            np.array([strat1.execute(20, 200), strat2.execute(20, 200)])
        ))

        port1 += strat3

        self.assertIsInstance(port1, StrategyPortfolio)
        self.assertEqual(port1.n_args, strat1.n_args + strat2.n_args + strat3.n_args)
        self.assertEqual(port1.strat_structure, [[strat1], [strat2], [strat3]])
        self.assertEqual(port1.strats, [strat1, strat2, strat3])
        self.assertEqual(port1.trimmed_length, min(strat1.trimmed_length, strat2.trimmed_length, strat3.trimmed_length))
        self.assertEqual(port1.reshaped_args, [[0] * strat1.n_args, [0] * strat2.n_args, [0] * strat3.n_args])
        self.assertEqual(port1._reshape_args([20, 200, 30, 300, 50]), [[20, 200], [30, 300], [50]])

        self.assertTrue(np.array_equal(
            port1.execute([20, 200, 20, 300, 50]),
            np.array([strat1.execute(20, 200)[-port1.trimmed_length:],
                      strat2.execute(30, 300)[-port1.trimmed_length:],
                      strat3.execute(50)][-port1.trimmed_length:])
        ))

    def test_portfolio_matmul_operator(self):
        prices = [self.price_data[symb] for symb in self.symbols]
        strat1 = PriceMovingAverageCrossover(prices[0])
        strat2 = PriceMovingAverageCrossover(prices[1])

        port1 = strat1@strat2  # linked inputs

        self.assertIsInstance(port1, StrategyPortfolio)
        self.assertEqual(port1.n_args, strat1.n_args)
        self.assertEqual(port1.strat_structure, [[strat1, strat2]])
        self.assertEqual(port1.strats, [strat1, strat2])
        self.assertEqual(port1.trimmed_length, min(strat1.trimmed_length, strat2.trimmed_length))
        self.assertEqual(port1.reshaped_args, [[0] * strat1.n_args, [0] * strat2.n_args])
        self.assertEqual(port1._reshape_args([20, 200]), [[20, 200], [20, 200]])

        self.assertTrue(np.array_equal(
            port1.execute([20, 200]),
            np.array([strat1.execute(20, 200), strat2.execute(20, 200)])
        ))

        strat3 = PriceMovingAverageCrossover(prices[3])

        port2 = strat1@strat2@strat3

        self.assertIsInstance(port2, StrategyPortfolio)
        self.assertEqual(port2.n_args, strat1.n_args)
        self.assertEqual(port2.strat_structure, [[strat1, strat2, strat3]])

        self.assertTrue(np.array_equal(
            port2.execute([20, 200]),
            np.array([strat1.execute(20, 200), strat2.execute(20, 200), strat3.execute(20, 200)])
        ))

        # + and @ together

        strat1_1 = PriceMovingAverageCrossover(prices[0])
        rsi1 = RSISignal(prices[0])
        strat2 = (rsi1 < 0.3 - rsi1 > 0.7).to_strategy()

        strat3 = PriceMovingAverageCrossover(prices[1])
        rsi2 = RSISignal(prices[1])
        strat4 = (rsi1 < 0.3 - rsi1 > 0.7).to_strategy()

        port = (strat1 + strat2) @ (strat3 + strat4)

        self.assertIsInstance(port, StrategyPortfolio)
        self.assertEqual(port.n_args, strat1.n_args + strat2.n_args)
        self.assertEqual(port.strat_structure, [[strat1, strat3], [strat2, strat4]])
        self.assertEqual(port.strats, [strat1, strat3, strat2, strat4])
        self.assertEqual(port.trimmed_length, min(strat1.trimmed_length, strat2.trimmed_length))
        self.assertEqual(port.reshaped_args,
                         [[0] * strat1.n_args, [0] * strat3.n_args, [0] * strat2.n_args, [0] * strat4.n_args])
        self.assertEqual(port._reshape_args([20, 200, 50]), [[20, 200], [20, 200], [50], [50]])
