import unittest
from timeit import timeit
from random import randint, random

import numpy as np

from optimal_strategy_portfolio.signals import *
from optimal_strategy_portfolio.signals.transforms import TransformedSignal, Log, SMA, ExpandingTransformedSignal, \
    RollingTransformedSignal, MA
from optimal_strategy_portfolio.signals.variables import Variable, Var, IntVar, BoolVar
from optimal_strategy_portfolio.signals.aggregators import SignalAggregator, HMax
from optimal_strategy_portfolio.signals.price_signals import PriceMASignal, PriceSignal, RSISignal, PriceEMASignal, \
    ReturnsSignal, R, PMA, P, RSI
from shared_utils.data import get_data, returns_from_prices, moving_averages, get_securities_data, moving_average4, \
    set_universe, universe


class TestSignals(unittest.TestCase):
    def setUp(self) -> None:
        self.symbs = ['AMZN', "GOOG", "MSFT", "FB", "AAPL", "TSLA"]
        self.data = get_securities_data(self.symbs)
        self.price_data = self.data['AMZN']
        self.price_data2 = self.data['GOOG']
        self.n_t = len(self.price_data)
        self.prices = self.price_data['Adj Close']
        self.prices2 = self.price_data2['Adj Close']

    def test_signal(self):
        name = 'test_name'
        s = Signal(name=name)

        self.assertEqual(s.name, name)
        self.assertEqual(str(s), name)
        self.assertEqual(s(), 1)
        self.assertEqual(s.bounds, ())

        set_universe(self.symbs)

        p = P("GOOG")

        self.assertEqual(p.name, 'GOOG_price')
        self.assertEqual(str(p), 'GOOG_price')
        self.assertEqual(repr(p), 'GOOG_price')

        name = 'fast_ma'
        p2 = PMA('GOOG', name=name)

        self.assertEqual(p2.name, f'GOOG_{name}')
        self.assertEqual(str(p2), f'GOOG_{name}')

    def test_signal_arg_names(self):
        set_universe(self.symbs)
        name = 'test_name'
        symb = 'GOOG'
        s = Signal(name=name)

        self.assertEqual(s.arg_names, [])

        s2 = PMA(symb, name=name)

        self.assertEqual(s2.arg_names, ['window_size'])
        self.assertEqual(repr(s2), f'{symb}_{name}(window_size)')

        cross = PMA(symb, name='fast_ma') > PMA(symb, name='slow_ma')

        self.assertEqual(repr(cross), f'({symb}_fast_ma(window_size) > {symb}_slow_ma(window_size))')

        self.assertEqual(cross.arg_names, [f'{symb}_fast_ma__window_size', f'{symb}_slow_ma__window_size'])

        rsi = RSI(symb)
        signal = (rsi < Var(name='lower_bound')) - (rsi > Var(name='upper_bound'))

        self.assertEqual(repr(signal),
                         f'(({symb}_rsi(window_size) < lower_bound(x)) - ({symb}_rsi(window_size) > upper_bound(x)))')
        self.assertEqual(signal.arg_names, [f'{symb}_rsi__window_size', 'lower_bound__x', 'upper_bound__x'])

    def test_signal_arg_types(self):
        fast, slow = PMA(self.price_data, name='fast_ma'), PMA(self.price_data, name='slow_ma')

        self.assertEqual(fast.arg_types, (int,))

        cross = fast > slow

        self.assertEqual(cross.arg_types, (int, int))

        s = cross - ~cross

        self.assertEqual(s.arg_types, (int, int))

        rsi = RSI(self.price_data)

        s = (rsi < Var(name='lower')) - (rsi > Var(name='upper'))

        self.assertEqual(s.arg_types, (int, float, float))

    def test_variables(self):
        x, y, z = Var(), IntVar(), BoolVar()

        self.assertEqual(x.n_args, 1)
        self.assertEqual(y.n_args, 1)
        self.assertEqual(z.n_args, 1)

        self.assertEqual(x.bounds, ((-10_000, 10_000),))
        self.assertEqual(y.bounds, ((-10_000, 10_000),))
        self.assertEqual(z.bounds, ((-1, 1),))

        v = random() - 0.5
        self.assertEqual(x(v), v)
        self.assertEqual(y(v), int(v))
        self.assertEqual(z(v), (v > 0) * 1)

    def test_price_signal(self):
        signal = PriceSignal(self.price_data)
        self.assertEqual(signal.n_args, 0)
        self.assertEqual(signal.bounds, ())
        self.assertEqual(signal.lag, 2)
        self.assertTrue(np.array_equal(self.prices.to_numpy()[:-2], signal()))

    def test_transformed_signal(self):
        signal = PriceSignal(self.price_data)
        t_signal = TransformedSignal(signal, lambda s: s**2)

        self.assertEqual(t_signal.n_args, 0)
        self.assertEqual(t_signal.bounds, ())
        self.assertEqual(t_signal.leaf_signals, {signal: None, t_signal: None})

        self.assertTrue(np.array_equal(
            t_signal(),
            signal() ** 2
        ))

        t_signal = TransformedSignal(signal, lambda s, n: np.emath.logn(n, s), additional_bounds=((2, 10),))

        self.assertEqual(t_signal.n_args, 1)
        self.assertEqual(t_signal.bounds, ((2, 10),))
        self.assertEqual(t_signal.leaf_signals, {signal: None, t_signal: None})

        self.assertTrue(np.array_equal(
            t_signal(2.2),
            np.emath.logn(2.2, signal())
        ))

    def test_step_transformed_signal(self):
        signal = P(self.price_data)

        warmup = 1

        def func(x):
            return np.sum(x) / len(x)

        ma = ExpandingTransformedSignal(signal, func, warmup=warmup)

        self.assertTrue(np.array_equal(
            np.array([func(signal()[:n]) for n in range(warmup, signal.n_t - signal.lag)]),
            ma()
        ))

    def test_rolling_transformed_signal(self):
        signal = P(self.price_data)

        ma = RollingTransformedSignal(signal, np.mean)
        ma2 = MA(signal)

        self.assertTrue(np.array_equal(
            ma(20),
            ma2(20)
        ))

    def test_signal_operators(self):
        signal1 = PriceSignal(self.price_data)

        price_data2 = self.price_data2
        signal2 = PriceSignal(price_data2)

        signal = signal1 + signal2

        self.assertIsInstance(signal, Signal)
        self.assertIsInstance(signal, CombinedSignal)

        self.assertEqual(signal.signal, signal1)
        self.assertEqual(signal.other, signal2)

        self.assertEqual(signal.n_args, 0)
        self.assertEqual(signal.bounds, ())

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

        signal1 = PriceMASignal(self.price_data)
        signal2 = PriceSignal(self.price_data2)

        signal = signal1 + signal2
        self.assertEqual(signal.n_args, signal1.n_args + signal2.n_args)

        signal = signal1 + signal1
        self.assertEqual(signal.n_args, signal1.n_args)

    def test_price_ma_signal(self):
        ma_signal = PriceMASignal(self.price_data)

        self.assertEqual(ma_signal.lag, 2)
        self.assertEqual(ma_signal.n_args, 1)
        self.assertEqual(ma_signal.bounds, ((1, 300),))

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

        self.assertEqual(signal.n_args, 2)
        self.assertEqual(signal.bounds, ((1, 300), (1, 300)))

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

    def test_signal_sum(self):
        ma_signals = [PriceMASignal(prices) for prices in self.data.values()]
        sum_ma = sum(ma_signals)

        self.assertIsInstance(sum_ma, CombinedSignal)
        self.assertEqual(sum_ma.leaf_signals, {s: None for s in ma_signals})
        self.assertEqual(list(sum_ma.leaf_signals.keys()), ma_signals)
        self.assertEqual(sum_ma.n_args, sum([s.n_args for s in ma_signals]))

        self.assertTrue(np.array_equal(
            sum_ma(*[20] * len(ma_signals)),
            sum(s(20) for s in ma_signals)
        ))

    def test_signal_aggregators(self):
        s1 = PriceMASignal(self.price_data)
        s2 = PriceMASignal(self.price_data2)

        s3 = SignalAggregator([s1, s2], np.max, axis=0)

        self.assertEqual(s3.signals, [s1, s2])
        self.assertEqual(s3.leaf_signals.keys(), {s1, s2, s3})
        self.assertEqual(list(s3.leaf_signals.keys()), [s1, s2, s3])
        self.assertEqual(s3.n_args, s1.n_args + s2.n_args)

        self.assertTrue(np.array_equal(
            s3(20, 40),
            np.max(np.array([s1(20), s2(40)]), axis=0)
        ))

        # pre-built signal
        s_max = HMax([s1, s2])

        self.assertTrue(np.array_equal(
            s_max(20, 40),
            np.max(np.array([s1(20), s2(40)]), axis=0)
        ))

        # aggregator with args
        s_agg = SignalAggregator([s1, s2], lambda arr, p: np.max(arr, axis=0) * p, additional_bounds=((0.5, 0.9),))
        self.assertEqual(s_agg.n_args, 3)
        self.assertEqual(s_agg.bounds, s1.bounds + s2.bounds + ((0.5, 0.9),))

        self.assertTrue(np.array_equal(
            s_agg(20, 40, 0.7),
            np.max(np.array([s1(20), s2(40)]), axis=0) * 0.7
        ))

    def test_complex_signals(self):
        ma_signals = [PriceMASignal(prices) for prices in self.data.values()]

        max_ma = HMax(ma_signals)

        self.assertIsInstance(max_ma, SignalAggregator)
        self.assertEqual(max_ma.leaf_signals, {s: None for s in ma_signals + [max_ma]})
        self.assertEqual(list(max_ma.leaf_signals.keys()), ma_signals + [max_ma])
        self.assertEqual(max_ma.n_args, sum([s.n_args for s in ma_signals]))

        self.assertTrue(np.array_equal(
            max_ma(*[20] * len(ma_signals)),
            np.max([s(20) for s in ma_signals], axis=0)
        ))

        relative_crossovers = [(s >= max_ma) - (s < max_ma) for s in ma_signals]

        self.assertIsInstance(relative_crossovers[0], CombinedSignal)
        self.assertEqual(relative_crossovers[0].leaf_signals, {s: None for s in ma_signals + [max_ma]})
        self.assertEqual(list(relative_crossovers[0].leaf_signals.keys()), ma_signals + [max_ma])
        self.assertEqual(relative_crossovers[0].n_args, sum([s.n_args for s in ma_signals]))

        self.assertTrue(np.array_equal(
            relative_crossovers[0](*[20] * len(ma_signals)),
            (ma_signals[0](20) >= np.max([s(20) for s in ma_signals], axis=0)) * 1 -
            (ma_signals[0](20) < np.max([s(20) for s in ma_signals], axis=0)) * 1
        ))

        self.assertTrue(np.array_equal(
            relative_crossovers[0](*[30] * len(ma_signals)),
            (ma_signals[0](30) >= np.max([s(30) for s in ma_signals], axis=0)) * 1 -
            (ma_signals[0](30) < np.max([s(30) for s in ma_signals], axis=0)) * 1
        ))

        rsi_signals = [RSISignal(prices) for prices in self.data.values()]

        h_thresh = Variable()
        l_thresh = Variable()
        rsi_crossover_signals = [(rsi < l_thresh) - (rsi > h_thresh) for rsi in rsi_signals]

        signals = [m * r for m, r in zip(relative_crossovers, rsi_crossover_signals)]

        self.assertEqual(signals[0].n_args, max_ma.n_args + h_thresh.n_args + l_thresh.n_args + rsi_signals[0].n_args)

        slow_mas = [PriceMASignal(prices) for prices in self.data.values()]

        absolute_crossovers = [(fast > slow) - (slow > fast) for fast, slow in zip(ma_signals, slow_mas)]

        self.assertEqual(absolute_crossovers[0].n_args, slow_mas[0].n_args + ma_signals[0].n_args)

    def test_complex_signals_lock_mechanism(self):
        ma_signals = [PriceMASignal(prices) for prices in self.data.values()]
        ema_signals = [PriceEMASignal(prices) for prices in self.data.values()]

        sigs = [ma + ema for ma, ema in zip(ma_signals, ema_signals)]

        self.assertTrue(np.array_equal(
             sigs[0](10, 20),
             ma_signals[0](10) + ema_signals[0](20)
        ))

        self.assertFalse(
            any([s.locked for s in sigs[0].leaf_signals.keys()])
        )

        a_sigs = [abs(ma) + abs(sig) for sig, ma in zip(sigs, ma_signals)]

        a_sigs[0](20, 40)

        self.assertFalse(
            any([s.locked for s in a_sigs[0].leaf_signals.keys()])
        )

        comp_sig = HMax(a_sigs)

        comp_sig(*[20] * 12)

        self.assertFalse(
            any([s.locked for s in comp_sig.leaf_signals.keys()])
        )

        comp_sig = comp_sig + comp_sig

        comp_sig(*[20] * 12)

        self.assertFalse(
            any([s.locked for s in comp_sig.leaf_signals.keys()])
        )

    def test_signal_cache(self):
        returns = R(self.data['GOOG'])
        log_returns = Log(returns)

        returns2 = R(self.data['AMZN'])
        log_returns2 = Log(returns2)

        self.assertFalse(np.array_equal(
            log_returns(),
            log_returns2
        ))

    def test_performance_on_complex_signals(self):
        ma_signals = [PriceMASignal(prices) for prices in self.data.values()]
        max_ma = HMax(ma_signals)
        relative_crossovers = [(s >= max_ma) - (s < max_ma) for s in ma_signals]

        globs = locals()
        globs.update(globals())

        t1 = timeit("""
w = randint(1,100)
max_ma(*[w] * len(ma_signals))
""", number=10_000, globals=globs)

        t2 = timeit("""
w = randint(1,100)
np.max([s(w) for s in ma_signals], axis=0)
""", number=10_000, globals=globs)

        self.assertLess(t1, t2 * 6)

        t1 = timeit("""
w = randint(1,100)
relative_crossovers[0](*[w] * len(ma_signals))
""", number=10_000, globals=globs)
        t2 = timeit("""
w = randint(1,100)
(ma_signals[0](w) >= np.max([s(w) for s in ma_signals], axis=0)) * 1 - \
(ma_signals[0](w) < np.max([s(w) for s in ma_signals], axis=0)) * 1
""", number=10_000, globals=globs)

        self.assertLess(t1, t2 * 6)

    def test_signal_test(self):
        set_universe(self.symbs, test_start_date='2020-01-01')

        ma = PMA('GOOG')

        self.assertIsInstance(ma.test_instance, ma.__class__)
        self.assertGreater(ma.test_instance.n_t, ma.n_t)

        ma2 = PMA('GOOG')

        cross = ma > ma2

        self.assertIsInstance(cross.test_instance, cross.__class__)

        self.assertEqual(repr(cross.test_instance), f'({repr(ma.test_instance)} > {repr(ma2.test_instance)})')

        self.assertTrue(np.array_equal(
            cross.test(20, 30),
            (PMA(universe['GOOG_TEST'])(20) > PMA(universe['GOOG_TEST'])(30))
        ))

