from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, kw_only=True)
class SequenceWithOffset:
    """
    A sequence of numbers, offset from zero by a set amount.

    The offset avoids having to manually offset data by explicitly storing zeros
    in the arrays, and also allows arrays to start before zero.
    """

    seq: np.array
    offset: int

    def _index_end(self):
        return self.offset + len(self.seq)

    def convolve(self, other: SequenceWithOffset) -> SequenceWithOffset:
        if len(self.seq) == 0 or len(other.seq) == 0:
            # make a zero-length array with the correct dtype
            seq = self.seq[:0] + other.seq[:0]
        else:
            seq = np.convolve(self.seq, other.seq)
        return SequenceWithOffset(seq=seq, offset=self.offset + other.offset)

    def consolidate(self, other: SequenceWithOffset) -> SequenceWithOffset:
        if len(self.seq) == 0:
            return other.copy()
        elif len(other.seq) == 0:
            return self.copy()
        index_low = min(self.offset, other.offset)
        index_high = max(self._index_end(), other._index_end())

        seq = np.zeros(
            index_high - index_low,
            dtype=(self.seq[:0] + other.seq[:0]).dtype,
        )
        seq[self.offset - index_low : self._index_end() - index_low] = self.seq
        seq[other.offset - index_low : other._index_end() - index_low] += other.seq

        return SequenceWithOffset(seq=seq, offset=index_low)

    def bias_by(self, bias: int):
        return SequenceWithOffset(seq=np.copy(self.seq), offset=self.offset + bias)

    def __mul__(self, value):
        return SequenceWithOffset(seq=self.seq * value, offset=self.offset)

    def copy(self):
        return SequenceWithOffset(seq=np.copy(self.seq), offset=self.offset)

    def to_labeled(self):
        return dict(
            x=np.arange(self.offset, self._index_end()),
            y=self.seq.tolist(),
        )

    def scaled_to_prob(self):
        """
        Makes a copy of this sequence that is scaled s.t. it sums to 1 like a
        probability distribution.
        """
        if np.all(self.seq == 0):
            raise ValueError("cannot scale array of zeros s.t. it sums to 1")
        return SequenceWithOffset(seq=self.seq / self.seq.sum(), offset=self.offset)


def roll_k_droplow(roll_1: SequenceWithOffset, k: int, drop: int):
    roll_1.seq = np.flip(roll_1.seq)
    result = roll_k_drophigh(roll_1, k, drop)

    roll_1.seq = np.flip(roll_1.seq)
    result.seq = np.flip(result.seq)

    return result


def roll_k_drophigh(roll_1: SequenceWithOffset, k: int, drop: int):
    assert drop > 0
    roll_1_prefices = [
        SequenceWithOffset(seq=roll_1.seq[:n], offset=roll_1.offset)
        for n in range(len(roll_1.seq))
    ]
    kdn_cache = [
        _take_first_n(_roll_k_iter(roll_1_prefix), k)
        for roll_1_prefix in roll_1_prefices
    ]

    @functools.lru_cache(maxsize=None)
    def inner(nsub: int, ksub: int, dsub: int):
        if dsub == 0:
            return kdn_cache[nsub][ksub]

        result = roll_1dn(0)
        for i_fixed in range(nsub):  # i_fixed - the index for the value of fixed dice
            for j in range(1, ksub + 1):  # j - the number of fixed dice
                result = result.consolidate(
                    inner(i_fixed, ksub - j, max(dsub - j, 0)).bias_by(
                        max(j - dsub, 0) * (i_fixed + roll_1.offset)
                    )
                    * math.comb(ksub, j)
                    * (roll_1.seq[i_fixed] ** j)
                )
        return result

    return inner(nsub=len(roll_1.seq), ksub=k, dsub=drop)


def roll_k(roll_1: SequenceWithOffset, k: int):
    # TODO make this more efficient by convolving together conv-exponent powers of 2
    return _take_index_n(_roll_k_iter(roll_1), k)


def _roll_k_iter(roll_1: SequenceWithOffset):
    roll_k = roll_0()
    while True:
        yield roll_k
        roll_k = roll_k.convolve(roll_1)


def roll_1dn(n: int):
    return SequenceWithOffset(seq=np.full(n, fill_value=1, dtype=np.uint64), offset=1)


def roll_0():
    return SequenceWithOffset(seq=np.ones(1, dtype=np.uint64), offset=0)


def _take_index_n(iter, n: int):
    for _ in range(n):
        next(iter)
    return next(iter)


def _take_first_n(iter, n: int):
    return [x for (_, x) in zip(range(n), iter)]
