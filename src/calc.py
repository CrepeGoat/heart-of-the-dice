import numpy as np


def kdn_plusb(k: int, n: int, b: int):
    # TODO make this more efficient by convolving together conv-exponent powers of 2
    dist = _take_index_n(_kdn_iter(n), k)
    dist_wo_leading_zeros = dist[k:]
    labeled_dist = _to_labeled(dist_wo_leading_zeros)
    labeled_dist["x"] += k + b
    return labeled_dist

def _to_labeled(array: np.array):
    return dict(
        x=np.arange(len(array)),
        y=array.tolist(),
    )

def _kdn_iter(n: int):
    dist_kdn = _0dn()
    dist_1dn = _1dn(n)
    while True:
        yield dist_kdn
        dist_kdn = np.convolve(dist_kdn, dist_1dn)

def _1dn(n: int):
    result = np.full(n+1, fill_value=1/n)
    result[0] = 0
    return result

def _0dn():
    return np.ones(1)

def _take_index_n(iter, n: int):
    for _ in range(n):
        next(iter)
    return next(iter)
