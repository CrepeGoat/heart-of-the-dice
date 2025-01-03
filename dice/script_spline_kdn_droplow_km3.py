import numpy as np

from dice import calc, calc_spline

n = 41

for k in range(4, 10):
    drop = k - 3
    print(f"{k} drop {drop}")

    dist = calc.roll_k_droplow(calc.roll_1dn(n), k=k, drop=drop)
    print(f"\tprobs: {dist.seq}")

    degree1 = k
    deriv = calc_spline._poly_inv(dist.seq.astype(np.int64), degree1)
    print(f"\t{degree1}-degree poly: {deriv}")

    deriv_mods = [deriv[i::3] for i in range(3)]
    for i, deriv_mod in enumerate(deriv_mods):
        print(f"\t{degree1}-degree poly slice {i}::3 {deriv_mod}")

    degree2 = k - 3
    deriv_mods_deriv = [
        calc_spline._poly_inv(deriv_mod, degree2) for deriv_mod in deriv_mods
    ]
    for i, deriv_mod_deriv in enumerate(deriv_mods_deriv):
        print(
            f"\t{degree1}-degree poly slice {i}::3 {degree2}-degree poly"
            f": {deriv_mod_deriv}"
        )

    degree3 = degree2
    deriv_mods_deriv_abs_deriv = [
        calc_spline._poly_inv(np.abs(deriv_mod_deriv), degree3)
        for deriv_mod_deriv in deriv_mods_deriv
    ]
    for i, deriv_mod_deriv_abs_deriv in enumerate(deriv_mods_deriv_abs_deriv):
        print(
            f"\t{degree1}-degree poly slice {i}::3 {degree2}-degree poly"
            f" abs {degree3}-degree poly: {deriv_mod_deriv_abs_deriv}"
        )
