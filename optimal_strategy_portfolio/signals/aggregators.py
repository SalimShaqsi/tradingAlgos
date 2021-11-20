from inspect import signature

from optimal_strategy_portfolio.helpers import get_func_arg_names
from optimal_strategy_portfolio.signals import Signal, CombinedSignal
from warnings import warn

import numpy as np


class SignalAggregator(Signal):
    lag = 0
    n_additional_args = 0

    def __init__(self, signals, aggr_func, name='aggregator', additional_bounds=(),
                 additional_arg_names=(), additional_arg_types=(), is_array=True, **aggr_func_kwargs):
        super().__init__(name=name)
        self.signals = signals
        self.signals_check()

        self.index = self.signals[0].index

        for signal in signals:
            if 'price_data' in signal.__dir__():
                self.price_data = signal.price_data
                break
        self.additional_bounds = additional_bounds
        self.additional_arg_types = additional_arg_types
        self.aggr_func = aggr_func
        self.aggr_func_check()
        if type(aggr_func) == np.ufunc:
            self.n_additional_args = aggr_func.nin - 1
            self.additional_arg_names = list(additional_arg_names)
        else:
            arg_names = get_func_arg_names(aggr_func)[1:]
            self.n_additional_args = len(arg_names)
            self.additional_arg_names = arg_names

        self.n_signals = len(signals)

        self.n_t = min([s.n_t - s.lag for s in signals])

        self.aggr_func_kwargs = aggr_func_kwargs

        self.is_array = is_array
        self.index = signals[0].index
        self.leaf_signals = dict()
        self._set_leaf_signals()

        self._set_bounds()
        self._set_arg_types()

        signals_for_test_instance = []
        create_test_instance = False
        for signal in signals:
            if hasattr(signal, 'test_instance'):
                signals_for_test_instance += [signal.test_instance]
                create_test_instance = True
            else:
                signals_for_test_instance += [signal]

        if create_test_instance:
            self.test_instance = SignalAggregator(signals_for_test_instance, aggr_func, name=name,
                                                  additional_bounds=additional_bounds,
                                                  additional_arg_names=additional_arg_names,
                                                  additional_arg_types=additional_arg_types, is_array=is_array,
                                                  **aggr_func_kwargs)

    def __repr__(self):
        a_args = str(self.additional_arg_names).replace('[', '').replace(']', '').replace("'", "")
        if a_args:
            return f"{self.name}({[repr(s) for s in self.signals]}, {a_args})"
        else:
            return f"{self.name}({[repr(s) for s in self.signals]})"

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
        for s in self.signals:
            if hasattr(s, 'leaf_signals'):
                self.leaf_signals.update(s.leaf_signals)
            else:
                self.leaf_signals[s] = None

        self.leaf_signals[self] = None  # This signal still has to be called even though it contains other signal

        for s in self.leaf_signals.keys():
            if hasattr(s, 'leaf_signals'):
                self.n_args += s.n_additional_args
            else:
                self.n_args += s.n_args

    def signals_check(self):
        try:
            [i for i in self.signals]
        except TypeError:
            raise TypeError("Provided object is not an iterable")

        if not hasattr(self.signals, '__getitem__'):
            warn("The provided iterable may not be ordered (such as a set object), "
                 "the resulting signal may not work as expected")

        index = self.signals[0].index
        for s in self.signals:
            if not isinstance(s, Signal):
                raise TypeError(f"{s} is not a Signal. The provided iterable should only contain signals")
            if not s.is_array:
                raise TypeError(f"{s} must be an array signal.")
            if not all(index == s.index):
                raise ValueError(f"Signal data indices must match")

    def aggr_func_check(self):
        if not hasattr(self.aggr_func, '__call__'):
            raise TypeError("The provided function object is not callable.")

    def transform(self, *args, retain_lock=False):
        arr = np.zeros((self.n_signals, self.n_t))
        k = 0
        for i, s in enumerate(self.signals):
            if s.locked:
                arr[i, :] = s()

            else:
                n_args = s.n_additional_args if hasattr(s, 'leaf_signals') and not isinstance(s, CombinedSignal) \
                    else s.n_args
                arr[i, :] = s(*args[k: k + n_args], retain_lock=retain_lock)
                k += n_args

        return self.aggr_func(arr, *args[k:], **self.aggr_func_kwargs)


class HMax(SignalAggregator):
    def __init__(self, signals, **kwargs):
        super().__init__(signals, np.max, name='HMax', axis=0)


class HMin(SignalAggregator):
    name = 'horizontal_min'

    def __init__(self, signals, **kwargs):
        super().__init__(signals, np.min, name='HMin', axis=0)


class HMean(SignalAggregator):
    def __init__(self, signals, **kwargs):
        super().__init__(signals, np.mean, name='HMean', axis=0)


class HStd(SignalAggregator):
    def __init__(self, signals, **kwargs):
        super().__init__(signals, np.std, name='HStd', axis=0)


class HProd(SignalAggregator):
    def __init__(self, signals, **kwargs):
        super().__init__(signals, np.prod, name='HProd', axis=0)


class LinkedArray(SignalAggregator):
    def __init__(self, signals):
        super().__init__(signals, lambda x: x, name='linked_array')
        self._check_signal_args()

    def _check_signal_args(self):
        self.n_args = self.signals[0].n_args
        self.bounds = self.signals[0].bounds

        for s in self.signals:
            if s.n_args != self.n_args:
                raise ValueError(f"")
