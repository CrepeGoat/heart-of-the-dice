import functools

import numpy as np


# TODO make this more efficient by convolving together conv-exponent powers of 2
def kdn(k: int, n: int):
    return functools.reduce(np.convolve, [_1dn(n) for _ in range(k)], _0dn()).tolist()


def _1dn(n: int):
    result = np.full(n+1, fill_value=1/n)
    result[0] = 0
    return result

def _0dn():
    return np.ones(1)
