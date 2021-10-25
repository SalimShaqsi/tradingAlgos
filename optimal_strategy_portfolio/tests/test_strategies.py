import unittest

import numpy as np

from optimal_strategy_portfolio.signals import Variable
from optimal_strategy_portfolio.signals.price_signals import RSISignal
from optimal_strategy_portfolio.strategies import Strategy, PriceMovingAverageCrossover
from optimal_strategy_portfolio.transforms import signal_to_positions
from shared_utils.data import get_data, returns_from_prices, moving_averages, get_securities_data


class TestStrategies(unittest.TestCase):
    def setUp(self) -> None:
        self.symb = 'AMZN'
        self.price_data = get_data(self.symb)
        self.price_data.name = 'AMZN'
        self.n_t = len(self.price_data)

        self.price_data_full = get_securities_data(['AMZN'])['AMZN']

    def test_buy_and_hold(self):
        strat = Strategy(self.price_data_full)
        self.assertEqual(str(strat), f'Buy & Hold')
        self.assertTrue(np.array_equal(strat.get_positions(), np.ones(self.n_t - 1)))
        prices = strat.prices
        returns = np.diff(prices) / prices[:-1]
        self.assertTrue(np.array_equal(returns, strat.execute()))

    def test_price_moving_average_crossover(self):
        strat = PriceMovingAverageCrossover(self.price_data_full)
        self.assertIsNotNone(strat.meta_data)
        returns = strat.execute(50, 200)

        base_returns = returns_from_prices(self.price_data.to_numpy())[-self.n_t + 300 + 1:]
        mas = moving_averages(np.array([self.price_data.to_numpy()]), np.array([1, 300]))
        test_returns = (mas[0, 50-1] - mas[0, 200-1] > 0)[:-2] * base_returns

        self.assertTrue(np.array_equal(returns, test_returns))

    def test_strategy_from_signal(self):
        rsi = RSISignal(self.price_data_full)
        signal = (rsi > 0.7) * -1 or (rsi < 0.3)

        strat = signal.to_strategy(lower_bound=-1, upper_bound=1)

        self.assertTrue(np.array_equal(
            strat.price_data.to_numpy(),
            rsi.price_data.to_numpy()
        ))

        self.assertEqual(
            strat.n_args,
            signal.n_args
        )

        sig = signal(50)

        positions = signal_to_positions(sig, lower_bound=-1, upper_bound=1)

        self.assertTrue(np.array_equal(
            strat.get_positions(50),
            positions
        ))

        ceil = Variable()
        floor = Variable()
        signal2 = (rsi > ceil) * -1 or (rsi < floor)

        print([id(rsi), id(ceil), id(floor)], [id(s) for s in signal2.leaf_signals])





