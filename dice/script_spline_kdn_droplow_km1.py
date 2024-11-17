import numpy as np

from dice import calc, calc_spline

############
k = 3
drop = 1
print(f"{k} drop {drop}")

dist = calc.roll_k_droplow(calc.roll_1dn(8), k=k, drop=drop)
print(f"\tprobs: {dist.seq}")

coeffs = []
gen = calc_spline._polynomial_inv_state_machine(np.zeros(k + 1, dtype=np.int64))
next(gen)
for i in dist.seq:
    coeffs.append(gen.send(i))
coeffs = np.array(coeffs, dtype=np.int64)
print(f"\t{k+1}-degree poly: {coeffs}")

coeffs_est = np.array([1, 0] + [(-1) ** i for i in range(13)], dtype=np.int64)
print(f"\tpattern: {coeffs_est}")
print(f"\tdiff from pattern: {coeffs - coeffs_est}")


############
k = 4
drop = 2
print(f"{k} drop {drop}")

dist = calc.roll_k_droplow(calc.roll_1dn(8), k=k, drop=drop)
print(f"\tprobs: {dist.seq}")

coeffs = []
gen = calc_spline._polynomial_inv_state_machine(np.zeros(k + 1, dtype=np.int64))
next(gen)
for i in dist.seq:
    coeffs.append(gen.send(i))
coeffs = np.array(coeffs, dtype=np.int64)
print(f"\t{k+1}-degree poly: {coeffs}")

coeffs_est = np.array(
    [1, 0] + [4 * (i + 1) * ((-1) ** i) for i in range(13)], dtype=np.int64
)
print(f"\tpattern: {coeffs_est}")
print(f"\tdiff from pattern: {coeffs - coeffs_est}")


print("\n\n\n")

for k in range(3, 10):
    drop = k - 2
    print(f"{k} drop {drop}")

    dist = calc.roll_k_droplow(calc.roll_1dn(20), k=k, drop=drop)
    print(f"\tprobs: {dist.seq}")

    coeffs = []
    gen = calc_spline._polynomial_inv_state_machine(np.zeros(k + 1, dtype=np.int64))
    next(gen)
    for i in dist.seq:
        coeffs.append(gen.send(i))
    coeffs = np.array(coeffs, dtype=np.int64)
    print(f"\t{k+1}-degree poly: {coeffs}")

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
