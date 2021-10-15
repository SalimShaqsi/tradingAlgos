import unittest

import numpy as np

from optimal_strategy_portfolio.signals import *
from optimal_strategy_portfolio.signals.price_signals import PriceMASignal, PriceSignal
from shared_utils.data import get_data, returns_from_prices, moving_averages


class TestSignals(unittest.TestCase):
    def setUp(self) -> None:
        self.symb = 'GOOG'
        self.price_data = get_data(self.symb)
        self.n_t = len(self.price_data)

    def test_signal(self):
        signal = PriceSignal(self.price_data)
        self.assertTrue(np.array_equal(self.price_data.to_numpy()[:-1], signal.__call__()))

    def test_signal_operators(self):
        signal1 = PriceSignal(self.price_data)

        symb2 = 'AMZN'
        price_data2 = get_data(symb2)
        signal2 = PriceSignal(price_data2)

        signal = signal1 + signal2

        self.assertIsInstance(signal, Signal)
        self.assertIsInstance(signal, CombinedSignal)

        self.assertEqual(signal.signal, signal1)
        self.assertEqual(signal.other, signal2)

        self.assertTrue(np.array_equal(
            self.price_data.to_numpy()[:-1] + price_data2.to_numpy()[:-1],
            signal.__call__())
        )

        signala = signal1 + 1
        signalb = 1 + signal1

        self.assertTrue(np.array_equal(
            signala.__call__(), signalb.__call__()
        ))

        signal = signal1 > signal2

        self.assertTrue(np.array_equal(
            self.price_data.to_numpy()[:-1] > price_data2.to_numpy()[:-1],
            signal.__call__())
        )

        signal = signal1 - signal2

        self.assertTrue(np.array_equal(
            self.price_data.to_numpy()[:-1] - price_data2.to_numpy()[:-1],
            signal.__call__())
        )

    def test_price_ma_signal(self):
        ma_signal = PriceMASignal(self.price_data)

        self.assertTrue(np.array_equal(
            ma_signal(300),
            moving_average4(self.price_data.to_numpy()[:-1], 300)[-len(self.price_data) + 1 + 300:]
        ))

        self.assertTrue(np.array_equal(
            ma_signal(20),
            moving_average4(self.price_data.to_numpy()[:-1], 20)[-len(self.price_data) + 1 + 300:]
        ))

        self.assertTrue(np.array_equal(
            ma_signal(1),
            moving_average4(self.price_data.to_numpy()[:-1], 1)[-len(self.price_data) + 1 + 300:]
        ))

        slow_ma = PriceMASignal(self.price_data)
        fast_ma = PriceMASignal(self.price_data)

        signal = (fast_ma > slow_ma) * 1 + (slow_ma > fast_ma) * -1  # ma crossover

        self.assertEqual(
            [fast_ma, slow_ma],
            [s for s in signal.leaf_signals]
        )

        print(id(fast_ma) < id(slow_ma))
        # the ordering of variables matched the order in which they appear the the signal expression above

        slow = fast_ma(200)
        l = len(slow)
        fast = fast_ma(20)[-l:]

        self.assertTrue(np.array_equal(
            (fast > slow) * 1 + (slow > fast) * -1,
            signal(20, 200)
        ))




