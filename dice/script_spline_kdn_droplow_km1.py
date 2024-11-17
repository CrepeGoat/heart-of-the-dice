import numpy as np

from dice import calc, calc_spline

for k in range(3, 10):
    drop = k - 2
    print(f"{k} drop {drop}")

    dist = calc.roll_k_droplow(calc.roll_1dn(21), k=k, drop=drop)
    print(f"\tprobs: {dist.seq}")

    coeffs = calc_spline._poly_inv(dist.seq.astype(np.int64), k)
    print(f"\t{k}-degree poly: {coeffs}")

    coeffs_abs = np.abs(coeffs)
    print(f"\t abs: {coeffs_abs}")

    coeffs_abs_coeffs = calc_spline._poly_inv(coeffs_abs[2:], k - 2)
    print(f"\t abs k-th deriv: {coeffs_abs_coeffs}")

    coeffs_est = np.copy(coeffs_abs_coeffs)
    coeffs_est[2 * (k - 3) + 1 :] = 0
    coeffs_est = calc_spline._poly(coeffs_est, k - 2)
    coeffs_est = coeffs_est * (
        np.full_like(coeffs[2:], -1) ** np.arange(len(coeffs) - 2)
    )
    print(f"\t est: {coeffs_est}")

    print(f"\t est error: {coeffs[2:] - coeffs_est}")
