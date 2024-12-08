import itertools
import math
from collections import defaultdict

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from dice import calc

# Custom test strategies for SequenceWithOffset


@st.composite
def st_sequence_with_offset(
    draw, st_values, len_min=0, len_max=None, offset_min=None, offset_max=None
):
    st_offset = st.integers(min_value=offset_min, max_value=offset_max)
    st_values = st.lists(st_values, min_size=len_min, max_size=len_max).filter(
        lambda x: len(x) == 0 or (x[0] != 0 and x[-1] != 0)
    )

    return calc.SequenceWithOffset(
        seq=np.array(draw(st_values), dtype=np.uint64), offset=draw(st_offset)
    )


@st.composite
def st_default_sequence_with_offset(draw):
    return draw(
        st_sequence_with_offset(
            st_values=st.integers(min_value=0, max_value=3),
            len_min=1,
            len_max=5,
            offset_min=-3,
            offset_max=3,
        )
    )


def generate_outcomes(roll_1: calc.SequenceWithOffset):
    for i, count in enumerate(roll_1.seq, start=roll_1.offset):
        yield from itertools.repeat(i, count)


def collect_outcomes(outcomes) -> dict:
    result = defaultdict(int)
    for o in outcomes:
        result[o] += 1
    return dict(result)


@given(
    st_default_sequence_with_offset(),
    st.integers(min_value=0, max_value=4),
)
def test_roll_k(roll_1, k):
    assert roll_1.seq.dtype.type is np.uint64
    result = calc.roll_k(roll_1, k)
    assert result.seq.dtype.type is np.uint64
    assert result.seq.sum() == roll_1.seq.sum() ** k

    calc_counts = collect_outcomes(generate_outcomes(result))
    expt_outcomes = itertools.product(generate_outcomes(roll_1), repeat=k)
    expt_events = [sum(i) for i in expt_outcomes]
    expt_counts = collect_outcomes(expt_events)
    assert calc_counts == expt_counts


@given(
    st_default_sequence_with_offset(),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=3),
)
def test_roll_k_droplow(roll_1, k, drop):
    assume(drop <= k)
    assert roll_1.seq.dtype.type is np.uint64
    result = calc.roll_k_droplow(roll_1, k, drop)
    assert result.seq.dtype.type is np.uint64
    assert result.seq.sum() == roll_1.seq.sum() ** k

    calc_counts = collect_outcomes(generate_outcomes(result))
    expt_outcomes = itertools.product(generate_outcomes(roll_1), repeat=k)
    expt_events = [sum(sorted(i)[drop:]) for i in expt_outcomes]
    expt_counts = collect_outcomes(expt_events)
    assert calc_counts == expt_counts


@given(
    st_default_sequence_with_offset(),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=3),
)
def test_roll_k_drophigh(roll_1, k, drop):
    assume(drop <= k)
    assert roll_1.seq.dtype.type is np.uint64
    result = calc.roll_k_drophigh(roll_1, k, drop)
    assert result.seq.dtype.type is np.uint64
    assert result.seq.sum() == roll_1.seq.sum() ** k

    calc_counts = collect_outcomes(generate_outcomes(result))
    expt_outcomes = itertools.product(generate_outcomes(roll_1), repeat=k)
    expt_events = [sum(sorted(i)[:-drop]) for i in expt_outcomes]
    expt_counts = collect_outcomes(expt_events)
    assert calc_counts == expt_counts


@given(st.integers(min_value=1, max_value=20))
def test_d2_roll_k(k):
    roll_1 = calc.roll_1dn(2)
    assert roll_1.seq.dtype.type is np.uint64
    result = calc.roll_k(roll_1, k)
    assert result.seq.dtype.type is np.uint64
    assert np.all(result.seq == [math.comb(k, i) for i in range(k + 1)])
    assert result.offset == k


@given(st.integers(min_value=1, max_value=10))
def test_d2_roll_k_droplow_all_but_1(drop):
    roll_1 = calc.roll_1dn(2)
    assert roll_1.seq.dtype.type is np.uint64
    k = drop + 1

    result = calc.roll_k_droplow(roll_1, k, drop)
    assert result.seq.dtype.type is np.uint64
    assert np.all(result.seq == [1, 2**k - 1])
    assert result.offset == 1


def test_4d6_droplow():
    roll_1 = calc.roll_1dn(6)
    assert roll_1.seq.dtype.type is np.uint64
    result = calc.roll_k_droplow(roll_1, 4, 1)
    assert result.seq.dtype.type is np.uint64

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
