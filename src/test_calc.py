import pytest
from hypothesis import given, strategies as st

import numpy as np

from . import calc


@given(st.integers(min_value=0, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_sum1(k, n):
    result = calc.kdn(k, n)
    assert sum(result["y"]) == pytest.approx(1)


@given(st.integers(min_value=1, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_droplow_sum1(k, n):
    result = calc.kdn_droplow(k, n)
    assert sum(result["y"]) == pytest.approx(1)


@given(st.integers(min_value=1, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_drophigh_sum1(k, n):
    result = calc.kdn_drophigh(k, n)
    assert sum(result["y"]) == pytest.approx(1)


def test_4d6_droplow():
    result = calc.kdn_droplow(4, 6)
    print(result)
    assert np.all(result["x"] == np.arange(3, 19))

    total_outcomes = 6**4
    assert result["y"][0] == pytest.approx(1 / total_outcomes)
    assert result["y"][1] == pytest.approx(4 / total_outcomes)
    assert result["y"][2] == pytest.approx(10 / total_outcomes)
    assert result["y"][-1] == pytest.approx(21 / total_outcomes)


class TestSequenceWithOffset:
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
