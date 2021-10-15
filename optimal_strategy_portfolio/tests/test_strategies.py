import unittest

import numpy as np

from optimal_strategy_portfolio.strategies import Strategy, PriceMovingAverageCrossover
from shared_utils.data import get_data, returns_from_prices, moving_averages


class TestStrategies(unittest.TestCase):
    def setUp(self) -> None:
        self.symb = 'GOOG'
        self.price_data = get_data(self.symb)
        self.price_data.name = 'GOOG'
        self.n_t = len(self.price_data)

    def test_buy_and_hold(self):
        strat = Strategy(self.price_data)
        self.assertEqual(str(strat), f'{self.symb} - Buy & Hold')
        self.assertTrue(np.array_equal(strat.get_positions(), np.ones(self.n_t - 1)))
        prices = strat.prices_np
        returns = np.diff(prices) / prices[:-1]
        self.assertTrue(np.array_equal(returns, strat.execute()))

    def test_price_moving_average_crossover(self):
        strat = PriceMovingAverageCrossover(self.price_data)
        self.assertIsNotNone(strat.meta_data)
        returns = strat.execute(50, 200)

        base_returns = returns_from_prices(self.price_data.to_numpy())[-self.n_t + 300:]
        mas = moving_averages(np.array([self.price_data.to_numpy()]), np.array([1, 300]))
        test_returns = (mas[0, 50-1] - mas[0, 200-1] > 0)[:-1] * base_returns

        self.assertTrue(np.array_equal(returns, test_returns))

    def test_test_strategy(self):
        train_data = self.price_data[:int(self.n_t*0.8)]
        test_data = self.price_data
        test_length = self.n_t - int(self.n_t*0.8)
        strat = Strategy(train_data, test_prices=test_data, test_trimmed_length=test_length)

        self.assertTrue(np.array_equal(strat.test_instance.prices_np, test_data.to_numpy()))
        prices = strat.test_instance.prices_np
        returns = (np.diff(prices) / prices[:-1])[-test_length:]
        self.assertTrue(np.array_equal(returns, strat.test()))


