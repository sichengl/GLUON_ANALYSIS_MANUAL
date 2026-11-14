"""
Smearing kernel of the parameter optimization: a sum of the production boosted Gaussians.

    K(d) = sum_j  a_j * G_j(d) * exp(+i 2pi/L k_j d_z),      k_j = mom_frac_j * mom_max

G_j is the Gaussian of coulomb_smearing.py with widths rho_T_j and rho_z_j, normalized to sum 1. In momentum space every
term is a Gaussian of width ~ 1/rho centered at k_j, so the kernel is a sum of Gaussians with different centers and widths.
One row [1, rho_T, rho_z, mom_frac] alone is exactly the production kernel.

The parameters are kept in a table, one row per Gaussian:

    table[j] = [a_j, rho_T_j, rho_z_j, mom_frac_j]

The first weight a_1 stays 1: an overall factor of the kernel does not change the effective energy. The optimizer works
on the parameter vector = the table read row by row without that first weight (4 * n_gauss - 1 numbers).

Plain array functions (numpy or cupy through the xp argument), so they can be tested on a laptop.
"""

import numpy

from coulomb_smearing import boosted_gaussian_kernel  # the kernel of one boosted Gaussian, exactly as in production

COLUMN_NAMES = ["a", "rhoT", "rhoz", "frac"]  # the four columns of the parameter table


# ---------------- table <-> parameter vector ----------------


def table_to_vector(table):
    return numpy.asarray(table, float).reshape(-1)[1:].copy()


def vector_to_table(vector):
    return numpy.concatenate([[1.0], numpy.asarray(vector, float)]).reshape(-1, 4)


def parameter_names(n_gauss):
    """a2, rhoT1, rhoz1, frac1, ... in the order of table_to_vector (the Gaussians are numbered from 1)."""
    names = [f"{column}{j + 1}" for j in range(n_gauss) for column in COLUMN_NAMES]
    return names[1:]


def per_parameter(column_values, n_gauss):
    """One value per column [a, rhoT, rhoz, frac] -> one value per parameter, in the order of table_to_vector."""
    return numpy.tile(numpy.asarray(column_values), n_gauss)[1:]


# ---------------- the kernel and its derivatives ----------------


def kernel_from_table(xp, spatial_size, table, mom_max, sign):
    """
    K[dz, dy, dx] for the displacement d = x - y. sign = +1 boosts every Gaussian by +k_j (prop2, the quark),
    sign = -1 by -k_j (prop1, which is conjugated in the contraction and becomes the antiquark).
    """
    Lx, Ly, Lz = spatial_size
    kernel = xp.zeros((Lz, Ly, Lx), "<c16")
    for a, rho_T, rho_z, frac in table:
        if a != 0:
            kernel += a * boosted_gaussian_kernel(xp, spatial_size, [0, 0, sign * frac * mom_max], rho_T, rho_z)
    return kernel


def kernel_ft_with_derivatives(xp, spatial_size, table, mom_max, sign, h):
    """
    kft[nz, ny, nx]     = 3D FFT of the kernel (what the sink smearing multiplies the propagator with),
    dkft[i, nz, ny, nx] = 3D FFT of d kernel / d parameter i, i in the order of table_to_vector.

    A parameter of Gaussian j changes only Gaussian j:
        d/d a_j                     = the Gaussian itself,
        d/d rho_T_j, rho_z_j, frac_j = a_j * (central difference of that Gaussian with the tiny step h).
    The step h = 1e-5 is on the kernel formula only (no inversion is involved), so the difference is exact to ~1e-10.
    """
    one = lambda rho_T, rho_z, frac: boosted_gaussian_kernel(xp, spatial_size, [0, 0, sign * frac * mom_max], rho_T, rho_z)
    derivatives = []
    for a, rho_T, rho_z, frac in table:
        derivatives.append(one(rho_T, rho_z, frac))
        derivatives.append(a * (one(rho_T + h, rho_z, frac) - one(rho_T - h, rho_z, frac)) / (2 * h))
        derivatives.append(a * (one(rho_T, rho_z + h, frac) - one(rho_T, rho_z - h, frac)) / (2 * h))
        derivatives.append(a * (one(rho_T, rho_z, frac + h) - one(rho_T, rho_z, frac - h)) / (2 * h))
    dkft = xp.fft.fftn(xp.stack(derivatives[1:]), axes=(1, 2, 3))  # [1:]: the first weight is not a parameter
    kft = xp.fft.fftn(kernel_from_table(xp, spatial_size, table, mom_max, sign))
    return kft, dkft


# ---------------- the mom_frac move that changes only the noise ----------------


def tilt_table(table, shift, mom_max, L):
    """
    Move every mom_frac together so that the correlator only gets an overall factor (in expectation): the kernel in
    momentum space is multiplied by exp(alpha * p_z). The quark momenta add up to the hadron momentum P, so this is
    the factor exp(alpha * P) for every state: the effective energy does not change, only the noise does.

    For one Gaussian this is just mom_frac -> mom_frac + shift (the known "boost = rescaling"). For a sum, Gaussian j
    moves by shift * rho_z_1^2 / rho_z_j^2 and its weight is rescaled, so that the shape of the kernel stays the same:
        exp(-rho_z^2 (p - k)^2 / 4) * exp(alpha p) = exp(-rho_z^2 (p - k')^2 / 4) * exp(alpha k + alpha^2 / rho_z^2),
        k' = k + 2 alpha / rho_z^2          (p, k in lattice units, k = 2pi/L * mom_frac * mom_max).
    shift is the change of mom_frac of the first Gaussian.
    """
    table = numpy.array(table, float)
    unit = 2 * numpy.pi / L * mom_max  # mom_frac -> boost in lattice units
    rho_z = table[:, 2]
    k = unit * table[:, 3]
    alpha = shift * unit * rho_z[0] ** 2 / 2
    new = table.copy()
    new[:, 3] += 2 * alpha / rho_z**2 / unit
    new[:, 0] *= numpy.exp(alpha * (k - k[0]) + alpha**2 * (1 / rho_z**2 - 1 / rho_z[0] ** 2))  # relative to Gaussian 1, whose weight stays 1
    return new
