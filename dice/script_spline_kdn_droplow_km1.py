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


for k in range(3, 6):
    drop = k - 2
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

    alt_diff = np.diff(np.abs(coeffs))
    print(f"\t alt diff: {alt_diff}")
    for _ in range(k - 3):
        alt_diff = np.diff(alt_diff)
        print(f"\t\t{alt_diff}")
