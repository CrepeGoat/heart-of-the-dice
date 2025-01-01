import numpy as np

from dice import calc, calc_spline

for k in range(4, 7):
    drop = k - 3
    print(f"{k} drop {drop}")

    dist = calc.roll_k_droplow(calc.roll_1dn(40), k=k, drop=drop)
    print(f"\tprobs: {dist.seq}")

    degree1 = k
    deriv = calc_spline._poly_inv(dist.seq.astype(np.int64), degree1)
    print(f"\t{degree1}-degree poly: {deriv}")

    deriv_0mod3 = deriv[::3]
    deriv_1mod3 = deriv[1::3]
    deriv_2mod3 = deriv[2::3]
    print(f"\t{degree1}-degree poly slice ::3 {deriv_0mod3}")
    print(f"\t{degree1}-degree poly slice 1::3 {deriv_1mod3}")
    print(f"\t{degree1}-degree poly slice 2::3 {deriv_2mod3}")

    degree2 = k - 3
    deriv_0mod3_deriv = calc_spline._poly_inv(deriv_0mod3, degree2)
    deriv_1mod3_deriv = calc_spline._poly_inv(deriv_1mod3, degree2)
    deriv_2mod3_deriv = calc_spline._poly_inv(deriv_2mod3, degree2)
    print(f"\t{degree1}-degree poly slice ::3 {degree2}-degree poly: {deriv_0mod3_deriv}")
    print(f"\t{degree1}-degree poly slice 1::3 {degree2}-degree poly: {deriv_1mod3_deriv}")
    print(f"\t{degree1}-degree poly slice 2::3 {degree2}-degree poly: {deriv_2mod3_deriv}")

