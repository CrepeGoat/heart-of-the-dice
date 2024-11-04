import pytest
from hypothesis import given, strategies as st

import numpy as np

from . import calc


@given(st.integers(min_value=0, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_sum_n_exp_k(k, n):
    result = calc.kdn(k, n)
    assert sum(result.seq) == pytest.approx(n**k)


@given(st.integers(min_value=1, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_droplow_sum_n_exp_k(k, n):
    result = calc.kdn_droplow(k, n)
    assert sum(result.seq) == pytest.approx(n**k)


@given(st.integers(min_value=1, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_drophigh_sum_n_exp_k(k, n):
    result = calc.kdn_drophigh(k, n)
    assert sum(result.seq) == pytest.approx(n**k)


def test_4d6_droplow():
    result = calc.kdn_droplow(4, 6)
    assert result.offset == 3

    assert result.seq[0] == pytest.approx(1)
    assert result.seq[1] == pytest.approx(4)
    assert result.seq[2] == pytest.approx(10)
    assert result.seq[-1] == pytest.approx(21)


class TestSequenceWithOffset:
    def test_convolve_zero_len(self):
        seq1 = calc.SequenceWithOffset(seq=np.array([]), offset=-3)
        seq2 = calc.SequenceWithOffset(seq=np.array([1, 2, 3]), offset=2)
        result = seq1.convolve(seq2)
        assert np.all(result.seq == [])
        # offset is arbitrary -> don't test

    def test_consolidate_no_overlap(self):
        seq1 = calc.SequenceWithOffset(seq=np.array([1, 2, 3, 4]), offset=-3)
        seq2 = calc.SequenceWithOffset(seq=np.array([500, 600, 700]), offset=2)

        result = seq1.consolidate(seq2)
        assert np.all(result.seq == [1, 2, 3, 4, 0, 500, 600, 700])
        assert result.offset == -3

    def test_consolidate_overlap(self):
        seq1 = calc.SequenceWithOffset(seq=np.array([1, 2, 3, 4]), offset=-1)
        seq2 = calc.SequenceWithOffset(seq=np.array([500, 600, 700]), offset=2)

        result = seq1.consolidate(seq2)
        assert np.all(result.seq == [1, 2, 3, 504, 600, 700])
        assert result.offset == -1

    def test_consolidate_zero_len(self):
        seq1 = calc.SequenceWithOffset(seq=np.array([1, 2, 3, 4]), offset=-1)
        seq2 = calc.SequenceWithOffset(seq=np.array([]), offset=2)

        result = seq1.consolidate(seq2)
        assert np.all(result.seq == seq1.seq)
        assert result.offset == seq1.offset
