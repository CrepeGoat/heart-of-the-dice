import math
import functools

import numpy as np


def kdn_droplow(k: int, n: int):
    result = kdn_drophigh(k, n)
    result["y"] = np.flip(result["y"])
    return result


def kdn_drophigh(k: int, n: int):
    dist = _kdn_drophigh(k, n) / (n**k)
    dist_wo_leading_zeros = dist[k - 1 :]
    labeled_dist = _to_labeled(dist_wo_leading_zeros)
    labeled_dist["x"] += k - 1
    return labeled_dist


def _kdn_drophigh(k: int, n: int):
    kdn_cache = [[x for (_, x) in zip(range(k), _dn_iter(m))] for m in range(n)]

    return functools.reduce(
        _add_raw,
        (
            _add_bias_raw(kdn_cache[n_fixed - 1][k - j], (j - 1) * n_fixed)
            * math.comb(k, j)
            for j in range(1, k + 1)  # j - the number of fixed dice
            for n_fixed in range(1, n + 1)  # n_fixed - the value for fixed dice
        ),
        np.zeros(0),
    )


def add_bias(labeled_dist, bias: int):
    labeled_dist["x"] += bias
    return labeled_dist


def kdn(k: int, n: int):
    # TODO make this more efficient by convolving together conv-exponent powers of 2
    dist = _take_index_n(_dn_iter(n), k) / (n**k)
    dist_wo_leading_zeros = dist[k:]
    labeled_dist = _to_labeled(dist_wo_leading_zeros)
    labeled_dist["x"] += k
    return labeled_dist


def _to_labeled(array: np.array):
    return dict(
        x=np.arange(len(array)),
        y=array.tolist(),
    )


def _add_bias_raw(dist, bias: int):
    return np.pad(dist, (bias, 0))


def _add_raw(dist1, dist2):
    if len(dist1) > len(dist2):
        dist1, dist2 = dist2, dist1

    len_diff = len(dist2) - len(dist1)
    assert len_diff >= 0

    return dist2 + np.pad(dist1, (0, len_diff))


def _dn_iter(n: int):
    dist_kdn = _0dn()
    dist_1dn = _1dn(n)
    while True:
        yield dist_kdn
        dist_kdn = np.convolve(dist_kdn, dist_1dn)


def _1dn(n: int):
    if n == 0:
        return np.zeros(1)
    result = np.full(n + 1, fill_value=1)
    result[0] = 0
    return result


def _0dn():
    return np.ones(1)


def _take_index_n(iter, n: int):
    for _ in range(n):
        next(iter)
    return next(iter)
