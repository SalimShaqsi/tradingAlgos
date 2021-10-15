from operator import __add__, __mul__, __eq__, __ge__, __le__, __lt__, __gt__, __sub__, __and__, __or__

import numpy as np

from optimal_strategy_portfolio.strategies import StrategyFromSignal
from optimal_strategy_portfolio.transforms import signal_to_positions
from shared_utils.data import moving_average4


class Signal(object):
    n_args = 0
    name = 'constant'
    is_array = True
    locked = False
    r = None

    def transform(self, *args):
        return 1

    def __call__(self, *args, **kwargs):
        if self.locked:
            return self.r
        self.r = self.transform(*args)
        return self.r

    def __add__(self, other):
        return CombinedSignal(self, other, __add__)

    def __radd__(self, other):
        return CombinedSignal(other, self, __add__)

    def __eq__(self, other):
        return CombinedSignal(self, other, __eq__)

    def __lt__(self, other):
        return CombinedSignal(self, other, __lt__)

    def __gt__(self, other):
        return CombinedSignal(self, other, __gt__)

    def __le__(self, other):
        return CombinedSignal(self, other, __le__)

    def __ge__(self, other):
        return CombinedSignal(self, other, __ge__)

    def __mul__(self, other):
        return CombinedSignal(self, other, __mul__)

    def __rmul__(self, other):
        return CombinedSignal(other, self, __mul__)

    def __sub__(self, other):
        return CombinedSignal(self, other, __sub__)

    def __rsub__(self, other):
        return CombinedSignal(other, self, __sub__)

    def __abs__(self):
        return TransformedSignal(self, np.abs)

    def __and__(self, other):
        return CombinedSignal(self, other, __and__)

    def __rand__(self, other):
        return CombinedSignal(other, self, __and__)

    def __or__(self, other):
        return CombinedSignal(self, other, __or__)

    def __ror__(self, other):
        return CombinedSignal(other, self, __or__)

    def __hash__(self):
        return id(self)

    def to_strategy(self, transform_function=signal_to_positions):
        return StrategyFromSignal(self, transform_function)


class Variable(Signal):
    def transform(self, x):
        return x


class TransformedSignal(Signal):
    is_array = True

    def __init__(self, signal, transform_function):
        assert not signal.is_array, "Constant signals cannot be transformed"
        self.signal = signal
        self.transform_function = transform_function

    def transform(self, *args):
        return self.transform_function(self.signal.__call__(*args))


class CombinedSignal(Signal):
    def __init__(self, signal, other, operator):
        self.operator = operator
        if type(signal) in [int, float, np.ndarray] and isinstance(other, Signal):
            signal, other = other, signal
        if type(other) in [int, float, np.ndarray]:
            self.other_is_static = True
            self.is_array = signal.is_array or type(other) in [np.ndarray]
            self.n_t = min(signal.n_t, len(other)) if type(other) in [np.ndarray] else signal.n_t
            self.n_args = signal.n_args
            if signal.is_array:
                self.index = signal.index
        elif isinstance(other, Signal):
            if signal.is_array and other.is_array:
                assert all(signal.index == other.index), "Price indices of both signals must match"
                self.n_t = min(signal.n_t, other.n_t)
                self.index = signal.index
                self.is_array = True
            elif signal.is_array:
                self.index = signal.index
                self.n_t = signal.n_t
                self.is_array = True
            elif other.is_array:
                self.index = other.index
                self.n_t = other.n_t
                self.is_array = True
            self.other_is_static = False
            self.n_args = signal.n_args + other.n_args
        else:
            raise TypeError(f"object of type {type(other)} is not supported.")

        self.signal = signal
        self.other = other
        self.leaf_signals = set()
        self._set_leaf_signals()

    def _set_leaf_signals(self):
        if isinstance(self.signal, CombinedSignal):
            self.leaf_signals = self.leaf_signals.union(self.signal.leaf_signals)
        elif isinstance(self.signal, Signal):
            self.leaf_signals.add(self.signal)
        if isinstance(self.other, CombinedSignal):
            self.leaf_signals = self.leaf_signals.union(self.other.leaf_signals)
        elif isinstance(self.other, Signal):
            self.leaf_signals.add(self.other)

        self.n_args = sum([s.n_args for s in self.leaf_signals])

    def __call__(self, *args, leaf_signals_computed=False, **kwargs):
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

