from functools import lru_cache, cache
from operator import __add__, __mul__, __eq__, __ge__, __le__, __lt__, __gt__, __sub__, __ne__, \
    __truediv__, __and__, __or__

from collections import deque
import numpy as np

from optimal_strategy_portfolio.helpers import get_func_arg_names
from optimal_strategy_portfolio.strategies import StrategyFromSignal
from optimal_strategy_portfolio.transform_functions import signal_to_positions


class Signal(object):
    n_args = 0
    is_array = False
    locked = False
    r = None  # the return value
    lag = 1  # signal lag. i.e if the signal is for yesterday's data, it should equal 1

    def __init__(self, name=None, bounds=(), arg_types=()):
        self.name = name
        self.bounds = bounds
        self.arg_types = arg_types

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.n_args:
            return f"{self.name}({self.arg_names})".replace('[', '').replace(']', '').replace("'","")
        else:
            return self.name

    @property
    def arg_names(self):
        if hasattr(self, 'leaf_signals'):
            arg_names = []
            for s in self.leaf_signals.keys():
                if hasattr(s, 'leaf_signals'):
                    arg_names += [f'{s.name}__{arg_name}' for arg_name in s.additional_arg_names]
                else:
                    arg_names += [f'{s.name}__{arg_name}' for arg_name in s.arg_names]
            return arg_names
        else:
            return get_func_arg_names(self.transform)

    def transform(self, **kwargs):
        return 1

    def set_lag(self, lag):
        self.lag = lag
        return self

    def test(self, *args, **kwargs):
        return self.test_instance(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.locked:
            return self.r
        r = self.transform(*args, **kwargs)
        if self.is_array and self.lag:
            self.r = r[-self.n_t:-self.lag]
        elif self.is_array:
            self.r = r[-self.n_t:]
        else:
            self.r = r
        return self.r

    def __add__(self, other):
        return CombinedSignal(self, other, __add__)

    def __radd__(self, other):
        return CombinedSignal(other, self, __add__)

    def __eq__(self, other):
        return (self is other) * 1 or CombinedSignal(self, other, __eq__)

    def __ne__(self, other):
        if self is other:
            return 0
        return CombinedSignal(self, other, __ne__)

    def __lt__(self, other):
        if self is other:
            return 0
        return CombinedSignal(self, other, __lt__)

    def __gt__(self, other):
        if self is other:
            return 0
        return CombinedSignal(self, other, __gt__)

    def __le__(self, other):
        return (self is other) * 1 or CombinedSignal(self, other, __le__)

    def __ge__(self, other):
        return (self is other) * 1 or CombinedSignal(self, other, __ge__)

    def __mul__(self, other):
        return CombinedSignal(self, other, __mul__)

    def __rmul__(self, other):
        return CombinedSignal(other, self, __mul__)

    def __sub__(self, other):
        return CombinedSignal(self, other, __sub__)

    def __rsub__(self, other):
        return CombinedSignal(other, self, __sub__)

    def __truediv__(self, other):
        return CombinedSignal(self, other, __truediv__)

    def __rtruediv__(self, other):
        return CombinedSignal(other, self, __truediv__)

    def __abs__(self):
        from optimal_strategy_portfolio.signals.transforms import TransformedSignal
        return TransformedSignal(self, np.abs, name='abs')

    def __and__(self, other):
        return CombinedSignal(self, other, np.logical_and)

    def __rand__(self, other):
        return CombinedSignal(other, self, np.logical_and)

    def __or__(self, other):
        if type(other) in [int, float, bool] and other:
            return 1
        return CombinedSignal(self, other, np.logical_or)

    def __ror__(self, other):
        return CombinedSignal(other, self, np.logical_or)

    def __hash__(self):
        return id(self)

    def __invert__(self):
        from optimal_strategy_portfolio.signals.transforms import TransformedSignal
        return TransformedSignal(self, np.logical_not, name='~')

    def to_strategy(self, price_data=None, transform_function=signal_to_positions, **transform_function_kwargs):
        return StrategyFromSignal(self, transform_function, price_data=price_data, **transform_function_kwargs)


class CombinedSignal(Signal):
    lag = 0
    n_additional_args = 0

    _operator_strings = {
        __add__: '+',
        __sub__: '-',
        __mul__: '*',
        __truediv__: '/',
        __eq__: '==',
        __lt__: '<',
        __gt__: '>',
        __le__: '<=',
        __ge__: '>=',
        np.logical_or: '|',
        np.logical_and: '&'
    }

    def __init__(self, signal, other, operator, name='combined_signal'):
        super().__init__(name=name)
        self.operator = operator
        if type(signal) in [int, float, np.ndarray] and isinstance(other, Signal):
            signal, other = other, signal
        if type(other) in [int, float, np.ndarray]:
            self.other_is_static = True
            self.is_array = signal.is_array or type(other) in [np.ndarray]
            self.n_t = min(signal.n_t - signal.lag, len(other)) if type(other) in [np.ndarray] else signal.n_t - signal.lag
            self.n_args = signal.n_args
            if signal.is_array:
                self.index = signal.index
        elif isinstance(other, Signal):
            if signal.is_array and other.is_array:
                assert all(signal.index == other.index), "Price indices of both signal must match"
                self.n_t = min(signal.n_t - signal.lag, other.n_t - other.lag)
                self.index = signal.index
                self.is_array = True
            elif signal.is_array:
                self.index = signal.index
                self.n_t = signal.n_t - signal.lag
                self.is_array = True
            elif other.is_array:
                self.index = other.index
                self.n_t = other.n_t - other.lag
                self.is_array = True
            self.other_is_static = False
            self.n_args = signal.n_args + other.n_args
        else:
            raise TypeError(f"object of type {type(other)} is not supported.")

        self.signal = signal
        self.other = other
        self.leaf_signals = dict()
        self.pairs = []
        self.operators = []
        self._set_leaf_signals()
        self.value = np.zeros(self.n_t, dtype=float) if self.is_array else 0

        self._set_bounds()
        self._set_arg_types()

        test_signal = test_other = None
        if 'price_data' in signal.__dir__():
            self.price_data = signal.price_data
            if 'test_instance' in signal.__dir__():
                test_signal = signal.test_instance
        elif 'price_data' in other.__dir__():
            self.price_data = other.price_data
        if 'price_data' in other.__dir__() and 'test_instance' in other.__dir__():
            test_other = other.test_instance

        if test_signal or test_other:
            self.test_instance = self.__class__(test_signal or signal, test_other or other, operator, name=name)

    def __repr__(self):
        return f"({repr(self.signal)} {self._operator_strings[self.operator]} {repr(self.other)})"

    def _set_arg_types(self):
        arg_types = ()
        for s in self.leaf_signals.keys():
            arg_types += s.additional_arg_types if hasattr(s, 'additional_arg_types') else s.arg_types
        self.arg_types = arg_types

    def _set_bounds(self):
        bounds = ()
        for s in self.leaf_signals.keys():
            bounds += s.additional_bounds if hasattr(s, 'additional_bounds') else s.bounds
        self.bounds = bounds

    def _set_leaf_signals(self):
        if isinstance(self.signal, Signal) and hasattr(self.signal, 'leaf_signals'):
            self.leaf_signals.update(self.signal.leaf_signals)
        elif isinstance(self.signal, Signal):
            self.leaf_signals[self.signal] = None
        if isinstance(self.other, Signal) and hasattr(self.other, 'leaf_signals'):
            self.leaf_signals.update(self.other.leaf_signals)
        elif isinstance(self.other, Signal):
            self.leaf_signals[self.other] = None

        self.n_args = self.n_additional_args
        for s in self.leaf_signals.keys():
            if hasattr(s, 'leaf_signals'):
                self.n_args += s.n_additional_args
            else:
                self.n_args += s.n_args

    def transform(self, *args, retain_lock=False):
        k = 0
        for s in self.leaf_signals.keys():
            if not s.locked:
                n_args = s.n_additional_args if hasattr(s, 'leaf_signals') else s.n_args
                s(*args[k: n_args + k], retain_lock=True)
                s.locked = True
                k += n_args

        s = self.signal
        o = self.other
        operator = self.operator
        parents = deque()
        s_parents = deque()
        o_parents = deque()
        operators = deque()
        locked_signals = []
        lhs = rhs = None
        while True:
            s_is_combined = isinstance(s, CombinedSignal)
            o_is_combined = isinstance(o, CombinedSignal)
            if s_is_combined and s.locked:
                lhs = s.value[-self.n_t:]
                locked_signals += [s]
            elif s_is_combined:
                parents.append(s)
                s_parents.append(s)
                o_parents.append(o)
                operators.append(operator)
                operator = s.operator
                o = s.other
                s = s.signal
                lhs = rhs = None
            elif isinstance(s, Signal):
                lhs = s()[-self.n_t:] if s.is_array else s()
            else:
                lhs = s[-self.n_t:] if type(s) in [np.ndarray] else s
            if lhs is not None and o_is_combined and o.locked:
                rhs = o.value[-self.n_t:]
                locked_signals += [o]
            elif lhs is not None and o_is_combined:
                parents.append(o)
                s_parents.append(s)
                o_parents.append(o)
                operators.append(operator)
                operator = o.operator
                s = o.signal
                o = o.other
                rhs = lhs = None
            elif lhs is not None and isinstance(o, Signal):
                rhs = o()[-self.n_t:] if o.is_array else o()
            elif lhs is not None:
                rhs = o[-self.n_t:] if type(o) in [np.ndarray] else o
            if lhs is not None and rhs is not None and parents:
                p = parents.pop()
                if p.is_array:
                    p.value[-self.n_t:] = operator(lhs, rhs)
                else:
                    p.value = operator(lhs, rhs)
                p.locked = True
                s = s_parents.pop()
                o = o_parents.pop()
                operator = operators.pop()
            elif lhs is not None and rhs is not None:
                r = operator(lhs, rhs)
                break

        if not retain_lock:
            for s in list(self.leaf_signals.keys()) + locked_signals:
                s.locked = False

        return r * 1

    def old___call__(self, *args):
        if self.other_is_static:
            lhs = self.signal.__call__(*args) if self.signal.is_array else self.signal.__call__(*args)[-self.n_t:]
            rhs = self.other[-self.n_t:] if type(self.other) in [np.ndarray] else self.other
        else:
            lhs = self.signal.__call__(*args[:self.signal.n_args]) if self.signal.is_array \
                else self.signal.__call__(*args[:self.signal.n_args])[-self.n_t:]
            rhs = self.other.__call__(*args[self.signal.n_args:])[-self.n_t:]

        return self.operator(
            lhs, rhs
        ) * 1

    def old__call__2(self, *args, leaf_signals_computed=False, **kwargs):
        if not leaf_signals_computed:
            k = 0
            for s in self.leaf_signals:
                s(*args[k: s.n_args + k])
                s.locked = True
                k += s.n_args

        if self.signal.is_array:
            lhs = self.signal.__call__(leaf_signals_computed=True)
        else:
            lhs = self.signal.__call__(leaf_signals_computed=True)[-self.n_t:]

        if self.other_is_static:
            rhs = self.other[-self.n_t:] if type(self.other) in [np.ndarray] else self.other
        else:
            if self.other.is_array:
                rhs = self.other.__call__(leaf_signals_computed=True)[-self.n_t:]
            else:
                rhs = self.other.__call__(leaf_signals_computed=True)

        if not leaf_signals_computed:
            for s in self.leaf_signals:
                s.locked = False

        return self.operator(
            lhs, rhs
        ) * 1
