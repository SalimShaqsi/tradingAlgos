from timeit import timeit

import numpy as np
from numba import njit

default_bounds = np.array([-1, 1])


@njit
def signal_to_positions(signal, lower_bound=0, upper_bound=1):
    p = 0
    positions = np.zeros(len(signal))
    for i, s in enumerate(signal):
        if s == 1 and p <= 0:
            positions[i] = p = upper_bound
        elif s == -1 and p >= 0:
            positions[i] = p = lower_bound
        else:
            positions[i] = p

    return positions


def signal_to_positions2(signal, lower_bound=0, upper_bound=1):
    p = 0
    positions = np.zeros(len(signal))
    for i, s in enumerate(signal):
        if s == 1 and p <= 0:
            positions[i] = p = upper_bound
        elif s == -1 and p >= 0:
            positions[i] = p = lower_bound
        else:
            positions[i] = p

    return positions


if __name__ == '__main__':
    sig = np.random.choice([-1, 0, 1], 3000)

    t1 = timeit('signal_to_positions(signal_to_positions(sig))', number=10_000, globals=globals())
    t2 = timeit('signal_to_positions2(signal_to_positions(sig))', number=10_000, globals=globals())
