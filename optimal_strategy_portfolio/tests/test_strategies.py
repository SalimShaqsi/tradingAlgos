import unittest

import numpy as np

from optimal_strategy_portfolio.commision_models.ibkr_commission_models import ibkr_commission
from optimal_strategy_portfolio.signals.variables import Variable, Var
from optimal_strategy_portfolio.signals.price_signals import RSISignal
from optimal_strategy_portfolio.strategies import Strategy, PriceMovingAverageCrossover
from optimal_strategy_portfolio.transform_functions import signal_to_positions
from shared_utils.data import get_data, returns_from_prices, moving_averages, get_securities_data, set_universe, \
    universe


class TestStrategies(unittest.TestCase):
    def setUp(self) -> None:
        self.symb = 'AMZN'
        self.price_data = get_data(self.symb)
        self.price_data.name = 'AMZN'
        self.n_t = len(self.price_data)

        self.price_data_full = get_securities_data(['AMZN'])['AMZN']
        self.highs = self.price_data_full['High']
        self.lows = self.price_data_full['Low']

    def test_buy_and_hold(self):
        avg_cash = 10_000
        slippage_factor = 0.03
        strat = Strategy(self.price_data_full, commission_func=ibkr_commission, slippage_factor=slippage_factor)
        self.assertEqual(str(strat), f'buy_and_hold')
        self.assertTrue(np.array_equal(strat.get_positions(), np.ones(self.n_t - 1)))
        prices = strat.prices

        returns = np.diff(prices) / prices[:-1]

        buy_prices = (strat.highs - prices) * slippage_factor + prices
        trade_values = avg_cash * buy_prices
        trade_share_amounts = avg_cash / buy_prices
        buy_commissions = ibkr_commission(trade_values, trade_share_amounts)
        buy_returns = (prices[1:] - buy_prices[:-1] - buy_commissions[:-1]) / buy_prices[:-1]

        sell_prices = prices - (prices - strat.lows) * slippage_factor
        trade_values = avg_cash * sell_prices
        trade_share_amounts = avg_cash / sell_prices
        sell_commissions = ibkr_commission(trade_values, trade_share_amounts)
        sell_returns = (sell_prices[1:] - prices[:-1] - sell_commissions[:-1]) / prices[:-1]

        positions = strat.get_positions()
        diff = np.append(positions[0], np.diff(positions))

        longs = (diff > 0) * 1
        shorts = (diff < 0) * 1
        holds = (diff == 0) * 1

        strat_returns = positions * (longs * buy_returns + shorts * sell_returns + holds * returns)

        self.assertTrue(np.array_equal(strat_returns, strat.execute()))

    def test_price_moving_average_crossover(self):
        strat = PriceMovingAverageCrossover(self.price_data_full, commission_func=None, slippage_factor=0)
        self.assertIsNotNone(strat.meta_data)
        returns = strat.execute(50, 200)

        base_returns = returns_from_prices(self.price_data.to_numpy())[-self.n_t + 300 + 1:]
        mas = moving_averages(np.array([self.price_data.to_numpy()]), np.array([1, 300]))
        test_returns = (mas[0, 50-1] - mas[0, 200-1] > 0)[:-2] * base_returns

        self.assertTrue(np.array_equal(returns, test_returns))

    def test_strategy_from_signal(self):
        rsi = RSISignal(self.price_data_full)
        signal = (rsi > 0.7) * -1 | (rsi < 0.3)

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

        self.assertEqual(strat.arg_names, ['rsi__window_size'])

    def test_strategy_tests(self):
        set_universe([self.symb], test_start_date='2019-01-01')

        rsi = RSISignal(self.symb)
        signal = (rsi < 40) - (rsi > 60)

        strat = signal.to_strategy(lower_bound=-1, upper_bound=1)

        self.assertIsInstance(strat.test_instance, strat.__class__)
        self.assertTrue(
            all(strat.test_instance.price_data == universe[f'{self.symb}_TEST'])
        )
        self.assertEqual(
            strat.test_instance.trimmed_length,
            len(universe[f'{self.symb}_TEST']) - len(universe[self.symb]) - 1
        )

        strat.test(20, report=False)

        strat.optimize().test(report=False)

        strat = (
            (rsi < Var(name='floor')) - (rsi > Var(name='ceiling'))
        ).to_strategy()

        strat.optimize().test(report=False)





