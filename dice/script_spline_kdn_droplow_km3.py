import numpy as np

from dice import calc, calc_spline

for k in range(4, 7):
    drop = k - 3
    print(f"{k} drop {drop}")

    dist = calc.roll_k_droplow(calc.roll_1dn(40), k=k, drop=drop)
    print(f"\tprobs: {dist.seq}")

    for degree in range(1, k + 1):
        coeffs = calc_spline._poly_inv(dist.seq.astype(np.int64), degree)
        print(f"\t{degree}-degree poly: {coeffs}")

    coeffs_abs = np.abs(coeffs)
    print(f"\t abs: {coeffs_abs}")

    for degree in range(1, k + 1):
        coeffs_abs_coeffs = calc_spline._poly_inv(coeffs_abs, degree)
        print(f"\t abs {degree}-th deriv: {coeffs_abs_coeffs}")

    coeffs_abs_coeffs_abs = np.abs(coeffs_abs_coeffs)
    print(f"\t abs abs: {coeffs_abs_coeffs_abs}")

    for degree in range(1, k + 1):
        coeffs_abs_coeffs = calc_spline._poly_inv(coeffs_abs_coeffs_abs, degree)
        print(f"\t abs abs {degree}-th deriv: {coeffs_abs_coeffs}")

    # coeffs_est = np.copy(coeffs_abs_coeffs)
    # coeffs_est[2 * (k - 1) + 1 :] = 0
    # coeffs_est = calc_spline._poly(coeffs_est, k)
    # coeffs_est = coeffs_est * (np.full_like(coeffs, -1) ** np.arange(len(coeffs)))
    # print(f"\t est: {coeffs_est}")

    # print(f"\t est error: {coeffs - coeffs_est}")
