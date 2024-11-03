import pytest
from hypothesis import given, strategies as st

import numpy as np

from . import calc


@given(st.integers(min_value=0, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_sum1(k, n):
    assert np.sum(calc.kdn(k, n)["y"]) == pytest.approx(1)


@given(st.integers(min_value=1, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_droplow_sum1(k, n):
    assert np.sum(calc.kdn_droplow(k, n)["y"]) == pytest.approx(1)


@given(st.integers(min_value=1, max_value=4), st.integers(min_value=2, max_value=8))
def test_kdn_drophigh_sum1(k, n):
    assert np.sum(calc.kdn_drophigh(k, n)["y"]) == pytest.approx(1)


def test_4d6_droplow():
    result = calc.kdn_droplow(4, 6)
    print(result)
    assert np.all(result["x"] == np.arange(3, 19))

    total_outcomes = 6**4
    assert result["y"][0] == pytest.approx(1 / total_outcomes)
    assert result["y"][1] == pytest.approx(4 / total_outcomes)
    assert result["y"][2] == pytest.approx(10 / total_outcomes)
    assert result["y"][-1] == pytest.approx(21 / total_outcomes)
