import pytest
from hypothesis import given, strategies as st

import numpy as np

from . import calc


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
        seq=np.array(draw(st_values), dtype=np.int64), offset=draw(st_offset)
    )


@st.composite
def st_default_sequence_with_offset(draw):
    return draw(
        st_sequence_with_offset(
            st_values=st.integers(min_value=0, max_value=5),
            len_min=1,
            len_max=7,
            offset_min=-3,
            offset_max=3,
        )
    )


@given(
    st_default_sequence_with_offset(),
    st.integers(min_value=0, max_value=4),
)
def test_roll_k_sum(roll_1, k):
    result = calc.roll_k(roll_1, k)
    assert result.seq.sum() == roll_1.seq.sum() ** k


@given(
    st_default_sequence_with_offset(),
    st.integers(min_value=1, max_value=4),
)
def test_roll_k_droplow_sum(roll_1, k):
    result = calc.roll_k_droplow(roll_1, k)
    assert result.seq.sum() == roll_1.seq.sum() ** k


@given(
    st_default_sequence_with_offset(),
    st.integers(min_value=1, max_value=4),
)
def test_roll_k_drophigh_sum(roll_1, k):
    result = calc.roll_k_drophigh(roll_1, k)
    assert result.seq.sum() == roll_1.seq.sum() ** k


def test_4d6_droplow():
    result = calc.roll_k_droplow(calc.roll_1dn(6), 4)
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
