import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from dice import calc, calc_spline


@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=20),
)
def test_roll_kdn_matches_calc(k, n):
    result = calc_spline.roll_kdn(k, n)
    expt_result = calc.roll_k(calc.roll_1dn(n), k)

    assert np.all(result.seq == expt_result.seq)
    assert result.offset == expt_result.offset