import unittest

import numpy as np

from optimal_strategy_portfolio.signals import *
from optimal_strategy_portfolio.signals.price_signals import PriceMASignal, PriceSignal
from shared_utils.data import get_data, returns_from_prices, moving_averages, get_securities_data, moving_average4


class TestSignals(unittest.TestCase):
    def setUp(self) -> None:
        self.symb = ['AMZN', "GOOG"]
        self.data = get_securities_data(self.symb)
        self.price_data = self.data['AMZN']
        self.price_data2 = self.data['GOOG']
        self.n_t = len(self.price_data)
        self.prices = self.price_data['Adj Close']
        self.prices2 = self.price_data2['Adj Close']

    def test_price_signal(self):
        signal = PriceSignal(self.price_data)
        self.assertEqual(signal.lag, 2)
        self.assertTrue(np.array_equal(self.prices.to_numpy()[:-2], signal.__call__()))

    def test_signal_operators(self):
        signal1 = PriceSignal(self.price_data)

        price_data2 = self.price_data2
        signal2 = PriceSignal(price_data2)

        signal = signal1 + signal2

        self.assertIsInstance(signal, Signal)
        self.assertIsInstance(signal, CombinedSignal)

        self.assertEqual(signal.signal, signal1)
        self.assertEqual(signal.other, signal2)

        self.assertTrue(np.array_equal(
            self.prices.to_numpy()[:-2] + self.prices2.to_numpy()[:-2],
            signal.__call__())
        )

        signala = signal1 + 1
        signalb = 1 + signal1

        self.assertTrue(np.array_equal(
            signala.__call__(), signalb.__call__()
        ))

        signal = signal1 > signal2

        self.assertTrue(np.array_equal(
            self.prices.to_numpy()[:-2] > self.prices2.to_numpy()[:-2],
            signal.__call__())
        )

        signal = signal1 - signal2

        self.assertTrue(np.array_equal(
            self.prices.to_numpy()[:-2] - self.prices2.to_numpy()[:-2],
            signal.__call__())
        )

    def test_price_ma_signal(self):
        ma_signal = PriceMASignal(self.price_data)

        self.assertEqual(ma_signal.lag, 2)

        self.assertTrue(np.array_equal(
            ma_signal(300),
            moving_average4(self.prices.to_numpy()[:-1], 300)[-len(self.price_data) + 1 + 300:-1]
        ))

        self.assertTrue(np.array_equal(
            ma_signal(20),
            moving_average4(self.prices.to_numpy()[:-1], 20)[-len(self.price_data) + 1 + 300:-1]
        ))

        self.assertTrue(np.array_equal(
            ma_signal(1),
            moving_average4(self.prices.to_numpy()[:-1], 1)[-len(self.price_data) + 1 + 300:-1]
        ))

        slow_ma = PriceMASignal(self.price_data)
        fast_ma = PriceMASignal(self.price_data)

        signal = (fast_ma > slow_ma) - (slow_ma > fast_ma)  # ma crossover

        self.assertEqual(
            [fast_ma, slow_ma],
            [s for s in signal.leaf_signals]
        )

        # the ordering of variables matched the order in which they appear the the signal_arr expression above

        slow = fast_ma(200)
        l = len(slow)
        fast = fast_ma(20)[-l:]

        self.assertTrue(np.array_equal(
            (fast > slow) * 1.0 + (slow > fast) * -1.0,
            signal(20, 200)
        ))

        self.assertTrue(np.array_equal(
            (fast > slow) * 1 + (slow > fast) * -1,
            signal(20, 200)
        ))




