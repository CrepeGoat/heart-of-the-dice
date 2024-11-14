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


def _polynomial_state_machine(init_state: np.array) -> Iterable[int]:
    state = init_state
    del init_state
    while True:
        next_item = yield state[-1]
        state[0] = next_item
        np.add.accumulate(state, out=state)
