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


@given(
    st.integers(min_value=2, max_value=4),
    st.integers(min_value=1, max_value=20),
)
def test_roll_kdn_droplow_km1_matches_calc(k, n):
    result = calc_spline.roll_kdn_droplow_km1(k, n)
    expt_result = calc.roll_k_droplow(calc.roll_1dn(n), k, k - 1)

    assert np.all(result.seq == expt_result.seq)
    assert result.offset == expt_result.offset


@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=1, max_value=20),
)
def test_roll_kdn_drophigh_km1_matches_calc(k, n):
    result = calc_spline.roll_kdn_drophigh_km1(k, n)
    expt_result = calc.roll_k_drophigh(calc.roll_1dn(n), k, k - 1)

    assert np.all(result.seq == expt_result.seq)
    assert result.offset == expt_result.offset


@given(
    st.integers(min_value=3, max_value=10),
    st.integers(min_value=1, max_value=20),
)
def test_roll_kdn_droplow_km2_matches_calc(k, n):
    result = calc_spline.roll_kdn_droplow_km2(k, n)
    expt_result = calc.roll_k_droplow(calc.roll_1dn(n), k, k - 2)

    assert np.all(result.seq == expt_result.seq)
    assert result.offset == expt_result.offset


@given(
    st.integers(min_value=3, max_value=10),
    st.integers(min_value=1, max_value=20),
)
def test_roll_kdn_drophigh_km2_matches_calc(k, n):
    result = calc_spline.roll_kdn_drophigh_km2(k, n)
    expt_result = calc.roll_k_drophigh(calc.roll_1dn(n), k, k - 2)

    assert np.all(result.seq == expt_result.seq)
    assert result.offset == expt_result.offset


@given(
    st.integers(min_value=1, max_value=5),
    st.lists(st.integers(min_value=-5, max_value=5), min_size=10, max_size=10),
)
def test_poly_inv(degree, values):
    values_convert = calc_spline._poly(values, degree)
    values_unconvert = calc_spline._poly_inv(values_convert, degree)

    assert np.all(values == values_unconvert)


@given(
    st.integers(min_value=1, max_value=5),
    st.lists(st.integers(min_value=-5, max_value=5), min_size=10, max_size=10),
)
def test_poly_state_inv(degree, values):
    state_init = np.zeros(degree + 1, dtype=np.int64)
    gen = calc_spline._polynomial_state_machine(np.copy(state_init))
    next(gen)
    gen_inv = calc_spline._polynomial_inv_state_machine(np.copy(state_init))
    next(gen_inv)

    for value in values:
        value_out = gen.send(value)
        value_inv = gen_inv.send(value_out)
        assert value == value_inv
