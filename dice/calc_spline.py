import functools
import itertools
import math
from typing import Iterable

import numpy as np

from dice.calc import SequenceWithOffset


def roll_kdn(k: int, n: int) -> SequenceWithOffset:
    return SequenceWithOffset(
        seq=np.array(list(_kdn(k, n)), dtype=np.uint64),
        offset=k,
    )


def _kdn(k: int, n: int) -> Iterable[int]:
    gen = _polynomial_state_machine(init_state=np.zeros(k, dtype=np.int64))
    iter_nck_alt_sign = (
        math.comb(k - 1, i) * (1 if i % 2 == 0 else -1) for i in range(k)
    )
    iter_coeffs = itertools.chain.from_iterable(
        itertools.repeat(nck, n) for nck in iter_nck_alt_sign
    )

    # need to ignore the first iteration, which
    # 1) yields an array of 0's and
    # 2) can only take None as an input, by design of generators
    gen.send(None)

    # Note: there are FEWER return values than coefficients; this is intentional
    for _ in range(k * (n - 1) + 1):
        yield gen.send(next(iter_coeffs))


def roll_kdn_drophigh_km1(k: int, n: int) -> SequenceWithOffset:
    result = roll_kdn_droplow_km1(k, n)
    result.seq = np.flip(result.seq)
    return result


def roll_kdn_droplow_km1(k: int, n: int) -> SequenceWithOffset:
    if k < 1:
        raise ValueError("k >= 1")
    return SequenceWithOffset(
        seq=np.array(list(_kdn_droplow_km1(k, n)), dtype=np.uint64),
        offset=1,
    )


def _kdn_droplow_km1(k: int, n: int) -> Iterable[int]:
    gen = _polynomial_state_machine(init_state=np.zeros(k + 1, dtype=np.int64))
    iter_coeffs = itertools.chain(
        (_eulerian_number(k, i) for i in range(k)),
        itertools.repeat(0),
    )

    # need to ignore the first iteration, which
    # 1) yields an array of 0's and
    # 2) can only take None as an input, by design of generators
    gen.send(None)

    for _ in range(n):
        yield gen.send(next(iter_coeffs))


def roll_kdn_droplow_km2(k: int, n: int) -> SequenceWithOffset:
    if k < 2:
        raise ValueError("k >= 2")
    return SequenceWithOffset(seq=_kdn_droplow_km2(k, n), offset=2)


def _kdn_droplow_km2(k, n):
    if k == 2:
        raise NotImplementedError("TODO implement 2dn drop 2 case")
    scaled_eulers_km1 = k * np.array(
        [_eulerian_number(k - 1, i) for i in range(k - 1)], dtype=np.int64
    )
    eulers_k = np.array([_eulerian_number(k, i) for i in range(k)], dtype=np.int64)

    abs_coeffs_init_raw = np.empty(2 * k - 1, dtype=np.int64)
    abs_coeffs_init_raw[0::2] = eulers_k
    abs_coeffs_init_raw[1::2] = -scaled_eulers_km1

    abs_coeffs_init = np.zeros(2 * n - 1, dtype=np.int64)
    abs_coeffs_init[: len(abs_coeffs_init_raw)] = abs_coeffs_init_raw[
        : len(abs_coeffs_init)
    ]

    abs_coeffs = _poly(abs_coeffs_init, k)
    coeffs = abs_coeffs * (1 - 2 * (np.arange(len(abs_coeffs), dtype=np.int64) % 2))
    coeffs[n : n + k - 1] -= scaled_eulers_km1[: n - 1]
    print(coeffs)
    return _poly(coeffs, k)


def _poly(coeffs, k):
    result = np.copy(coeffs)
    for _ in range(k):
        np.add.accumulate(result, out=result)
    return result


def _poly_inv(values, k):
    result = np.copy(values)
    for _ in range(k):
        result[1:] = np.diff(result)
    return result


def _polynomial_inv_state_machine(init_state: np.array) -> Iterable[int]:
    state = init_state
    del init_state
    while True:
        next_item = yield state[0]
        state[0] = next_item - np.sum(state[1:])
        np.add.accumulate(state, out=state)
        assert next_item == state[-1]


def _polynomial_state_machine(init_state: np.array) -> Iterable[int]:
    state = init_state
    del init_state
    while True:
        next_item = yield state[-1]
        state[0] = next_item
        np.add.accumulate(state, out=state)


@functools.lru_cache(maxsize=None)
def _eulerian_number(n, k):
    """
    Calculate the Eulerian number, A(n, k).

    see https://en.wikipedia.org/wiki/Eulerian_number
    """
    # if k > n // 2:
    #     return _eulerian_number(n, n - k)
    if n < 0:
        raise ValueError("n >= 0")
    if k == 0:
        return 1
    if k == n:
        return 0
    if k < 0 or k >= n:
        raise ValueError("k = 0, or 0 < k < n")

    return _eulerian_number(n - 1, k - 1) * (n - k) + _eulerian_number(n - 1, k) * (
        k + 1
    )
