from functools import cache
from inspect import signature
from logging import warn, warning

from optimal_strategy_portfolio.helpers import get_func_arg_names
from optimal_strategy_portfolio.indicators import moving_average, ema
from optimal_strategy_portfolio.metrics import trend
from optimal_strategy_portfolio.signals import Signal, CombinedSignal
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from optimal_strategy_portfolio.signals.price_signals import R
from shared_utils.data import moving_average4, set_universe


class TransformedSignal(Signal):
    lag = 0

    def __init__(self, signal, func, additional_bounds=(), additional_arg_names=(),
                 additional_arg_types=(), name='transform', is_array=True, warmup=0, **func_kwargs):
        super().__init__(name=name)
        self.signal = signal
        self.signal_check()

        self.index = self.signal.index

        if 'price_data' in signal.__dir__():
            self.price_data = signal.price_data

        self.additional_bounds = additional_bounds
        self.additional_arg_types = additional_arg_types
        self.func = func
        self.func_check()
        if type(func) == np.ufunc:
            self.n_additional_args = func.nin - 1
            self.additional_arg_names = list(additional_arg_names)
        else:
            arg_names = get_func_arg_names(func)[1:]
            self.n_additional_args = len(arg_names)
            self.additional_arg_names = list(additional_arg_names) or arg_names

        self.warmup = warmup
        self.n_t = signal.n_t - signal.lag - warmup

        self.func_kwargs = func_kwargs

        self.is_array = is_array
        self.leaf_signals = dict()
        self._set_leaf_signals()

        self._set_bounds()
        self._set_arg_types()

        self.r = np.zeros(self.n_t)

        if self.n_args == 0:
            self.transform()  # cache the result

        if hasattr(signal, 'test_instance'):
            self.test_instance = self.__class__(signal.test_instance, func,
                                                additional_bounds=additional_bounds,
                                                additional_arg_names=additional_arg_names,
                                                additional_arg_types=additional_arg_types,
                                                name=name, is_array=is_array, warmup=warmup, **func_kwargs)

    def __repr__(self):
        return f"{self.name}({repr(self.signal)})"

    def _set_bounds(self):
        bounds = ()
        for s in self.leaf_signals.keys():
            bounds += s.additional_bounds if hasattr(s, 'additional_bounds') else s.bounds

        self.bounds = bounds

    def _set_arg_types(self):
        arg_types = ()
        for s in self.leaf_signals.keys():
            arg_types += s.additional_arg_types if hasattr(s, 'additional_arg_types') else s.arg_types
        self.arg_types = arg_types

    def _set_leaf_signals(self):
        s = self.signal
        if hasattr(s, 'leaf_signals'):
            self.leaf_signals.update(s.leaf_signals)
        else:
            self.leaf_signals[s] = None

        self.leaf_signals[self] = None  # This signal still has to be called even though it contains another signal

        for s in self.leaf_signals.keys():
            if hasattr(s, 'leaf_signals'):
                self.n_args += s.n_additional_args
            else:
                self.n_args += s.n_args

    def signal_check(self):
        if not isinstance(self.signal, Signal):
            raise TypeError(f"Object provided is not a Signal")

        if not self.signal.is_array:
            raise TypeError(f"Signal must be an array")

    def func_check(self):
        func = self.func
        if not hasattr(self.func, '__call__'):
            raise TypeError("The provided function object is not callable.")

        if type(func) == np.ufunc:
            n_args = func.nin
        else:
            n_args = len(get_func_arg_names(func))
        if n_args < 1:
            raise TypeError("The provided function must have at least one input parameter")

        if n_args - 1 != len(self.additional_bounds):
            raise ValueError("The length of bounds should equal the length of the function input parameters - 1")

    @cache
    def transform(self, *args, retain_lock=False):
        s = self.signal
        n_args = 0
        if s.locked:
            self.r = s()
        else:
            n_args = s.n_additional_args if hasattr(s, 'leaf_signals') and not isinstance(s, CombinedSignal) \
                else s.n_args
            self.r = s(*args[:n_args], retain_lock=retain_lock)

        self.r = self.func(self.r, *args[n_args:], **self.func_kwargs) * 1
        return self.r


class Log(TransformedSignal):
    def __init__(self, signal, name='log', **kwargs):
        super().__init__(signal, np.log, name=name, is_array=True)


class SimpleMovingAverageTransform(TransformedSignal):
    def __init__(self, signal, *args, additional_bounds=((1, 300), ), name='simple_moving_average', **kwargs):
        super().__init__(signal, moving_average, additional_arg_names=('window_size',), additional_arg_types=(int,),
                         additional_bounds=additional_bounds, name=name, is_array=True)
        self.n_t -= self.bounds[0][1]


class ExponentialMovingAverageTransform(TransformedSignal):
    def __init__(self, signal, *args, additional_bounds=((1, 300), ), name='simple_moving_average', **kwargs):
        super().__init__(signal, ema, additional_arg_names=('window_size',), additional_arg_types=(int,),
                         additional_bounds=additional_bounds, name=name, is_array=True)
        self.n_t -= self.bounds[0][1]


class ExpandingTransformedSignal(TransformedSignal):
    def __init__(self, signal, func, additional_bounds=(), additional_arg_names=(),
                 additional_arg_types=(), name='step_transform', warmup=0, **func_kwargs):
        super().__init__(signal, func, additional_bounds=additional_bounds, name=name,
                         additional_arg_names=additional_arg_names, additional_arg_types=additional_arg_types,
                         warmup=warmup, **func_kwargs)
    @cache
    def transform(self, *args, retain_lock=False):
        s = self.signal
        n_args = 0
        if s.locked:
            self.r = s()[-self.n_t:]
        else:
            n_args = s.n_additional_args if hasattr(s, 'leaf_signals') else s.n_args
            self.r = s(*args[:n_args], retain_lock=retain_lock)

        return np.array([self.func(self.r[:n], *args[n_args:], **self.func_kwargs)
                         for n in range(self.warmup, self.n_t + self.warmup)])


class RollingTransformedSignal(TransformedSignal):
    def __init__(self, signal, func, additional_bounds=((1, 300),), additional_arg_names=('window_size',),
                 additional_arg_types=(int,), name='rolling_transform', **func_kwargs):
        super().__init__(signal, func, additional_bounds=additional_bounds, name=name,
                         additional_arg_names=additional_arg_names, additional_arg_types=additional_arg_types,
                         warmup=additional_bounds[-1][-1], **func_kwargs)

    def func_check(self):
        func = self.func
        if not hasattr(self.func, '__call__'):
            raise TypeError("The provided function object is not callable.")

        if type(func) == np.ufunc:
            n_args = func.nin
        else:
            n_args = len(get_func_arg_names(func))
        if n_args < 1:
            raise TypeError("The provided function must have at least one input parameter")

        if n_args != len(self.additional_bounds):
            raise ValueError("The length of bounds should equal the length of the function input parameters - 1")

    def _set_leaf_signals(self):
        s = self.signal
        if hasattr(s, 'leaf_signals'):
            self.leaf_signals.update(s.leaf_signals)
        else:
            self.leaf_signals[s] = None

        self.leaf_signals[self] = None  # This signal still has to be called even though it contains another signal

        self.n_additional_args += 1  # the extra arg is for the window size
        for s in self.leaf_signals.keys():
            if hasattr(s, 'leaf_signals'):
                self.n_args += s.n_additional_args
            else:
                self.n_args += s.n_args

    @cache
    def transform(self, *args, retain_lock=False):
        s = self.signal
        n_args = 0
        if s.locked:
            self.r = s()[-self.n_t:]
        else:
            n_args = s.n_additional_args if hasattr(s, 'leaf_signals') else s.n_args
            self.r = s(*args[:n_args], retain_lock=retain_lock)
        rolling_r = sliding_window_view(self.r, window_shape=args[-1])
        return np.array([self.func(r, *args[n_args:-1], **self.func_kwargs) for r in rolling_r])


SMA = MA = SimpleMovingAverageTransform
EMA = ExponentialMovingAverageTransform


class RollingTrend(RollingTransformedSignal):
    def __init__(self, signal, *args, additional_bounds=((1, 300),), name='rolling_trend', **kwargs):
        super().__init__(signal, trend, additional_bounds=additional_bounds, name=name)


if __name__ == '__main__':
    set_universe(['AMZN'])

    returns = R('AMZN')

    double_trend = RollingTrend(RollingTrend(returns))

    print(double_trend(40))
