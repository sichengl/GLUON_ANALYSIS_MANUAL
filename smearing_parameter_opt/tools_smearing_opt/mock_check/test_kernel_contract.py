"""
Laptop check (numpy only, tiny lattice, random numbers in place of propagators) of opt_kernel.py and opt_contract.py:
  1. one row [1, rho_T, rho_z, frac] gives exactly the production kernel;
  2. the momentum-space sum equals the production order (smear both propagators at the sink, then contract with the phase);
  3. the derivative weights equal a finite difference of the sink-smeared correlator, for every parameter;
  4. tilt_table multiplies the kernel in momentum space by exp(alpha p_z) (up to the lattice wrap-around).
Checks results only, not speed.   Run:  python3 test_kernel_contract.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "2pt_production", "tools"))

from coulomb_smearing import boosted_gaussian_kernel
from opt_kernel import kernel_from_table, kernel_ft_with_derivatives, table_to_vector, vector_to_table, tilt_table, parameter_names
from opt_contract import pair_weights, resolved_pion

rng = np.random.default_rng(7)
Lx, Ly, Lz, nt = 8, 6, 10, 3
spatial = [Lx, Ly, Lz]
mom_max = 3
pz_list = [0, 1, 2, 3]
x_src = [5, 2, 7]

# gammas as plain random 4x4 matrices: the identity checked here does not depend on what they are
A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
B = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
A2 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))  # a second "gamma" at the sink and at the source
B2 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
A_first, B_first = A, B
shape = (nt, Lz, Ly, Lx, 4, 4, 3, 3)
S1 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
S2 = rng.normal(size=shape) + 1j * rng.normal(size=shape)

table = np.array([[1.0, 2.0, 1.2, 0.8], [0.3, 1.5, 1.5, 0.3], [-0.2, 2.5, 0.9, 1.1], [0.0, 1.8, 1.8, 0.5]])


def direct(table, A=None, B=None):
    A, B = (A_first, B_first) if A is None else (A, B)
    """production order: sink smearing (FFT, multiply, inverse FFT), then the site sum with the momentum phase"""
    out = np.zeros((len(pz_list), nt), complex)
    K1 = kernel_from_table(np, spatial, table, mom_max, -1)
    K2 = kernel_from_table(np, spatial, table, mom_max, +1)
    smear = lambda S, K: np.fft.ifftn(np.fft.fftn(S, axes=(1, 2, 3)) * np.fft.fftn(K)[None, :, :, :, None, None, None, None], axes=(1, 2, 3))
    p1, p2 = smear(S1, K1), smear(S2, K2)
    site = np.einsum("tzyxjiba,jk,tzyxklba,li->tzyx", p1.conj(), A, p2, B)
    z = np.arange(Lz).reshape(Lz, 1, 1)
    for ipz, pz in enumerate(pz_list):
        out[ipz] = (site * np.exp(-2j * np.pi * pz * (z - x_src[2]) / Lz)).sum(axis=(1, 2, 3))
    return out


def resolved(table, pair=None):
    kft1, dkft1 = kernel_ft_with_derivatives(np, spatial, table, mom_max, -1, 1e-5)
    kft2, dkft2 = kernel_ft_with_derivatives(np, spatial, table, mom_max, +1, 1e-5)
    W = pair_weights(np, kft1, kft2, dkft1, dkft2, pz_list)
    out = resolved_pion(np, S1, S2, [A, A2], [B, B2], W, pz_list)  # [gamma_sink, gamma_source, pz, w, t]
    out = out * np.exp(2j * np.pi * np.array(pz_list) * x_src[2] / Lz)[None, None, :, None, None]
    return out[0, 0] if pair is None else out[pair]


# 1. one row = production kernel
one = kernel_from_table(np, spatial, [[1.0, 2.0, 1.2, 0.8]], mom_max, +1)
ref = boosted_gaussian_kernel(np, spatial, [0, 0, 0.8 * mom_max], 2.0, 1.2)
print("1. one-row kernel vs production kernel: max diff", abs(one - ref).max())
assert abs(one - ref).max() == 0

# 2. momentum-space sum = production order
res = resolved(table)
ref = direct(table)
err = abs(res[:, 0] - ref).max() / abs(ref).max()
print("2. momentum-space sum vs sink smearing + contraction: max rel diff", err)
assert err < 1e-12
for pair, (Ax, Bx) in {(0, 1): (A, B2), (1, 0): (A2, B), (1, 1): (A2, B2)}.items():  # the other gamma pairs of the matrix
    ref_pair = direct(table, Ax, Bx)
    err = abs(resolved(table, pair)[:, 0] - ref_pair).max() / abs(ref_pair).max()
    assert err < 1e-12, (pair, err)
print("   the same for the other (sink gamma, source gamma) pairs: ok")

# 3. derivative weights = finite difference of the sink-smeared correlator
vec = table_to_vector(table)
names = parameter_names(len(table))
worst = 0
for i in range(len(vec)):
    h = 1e-5
    up, dn = vec.copy(), vec.copy()
    up[i] += h
    dn[i] -= h
    fd = (direct(vector_to_table(up)) - direct(vector_to_table(dn))) / (2 * h)
    err = abs(res[:, 1 + i] - fd).max() / max(abs(fd).max(), 1e-30) if abs(fd).max() > 0 else abs(res[:, 1 + i]).max()
    worst = max(worst, err)
    assert err < 1e-6, (names[i], err)
print(f"3. derivative weights vs finite difference, {len(vec)} parameters: worst rel diff", worst)
#    a Gaussian with weight 0: its shape derivatives vanish, its weight derivative does not
i_a4 = names.index("a4")
print("   weight-0 Gaussian: |d/da4|", abs(res[:, 1 + i_a4]).max(), " |d/drhoT4|", abs(res[:, 1 + names.index('rhoT4')]).max())
assert abs(res[:, 1 + names.index("rhoT4")]).max() == 0 and abs(res[:, 1 + i_a4]).max() > 0

# 4. tilt = exp(alpha p_z) in momentum space, on a lattice large enough that the kernel does not wrap around
L = 32
big = [L, L, L]
tab = np.array([[1.0, 4.5, 2.25, 0.8], [0.4, 3.0, 1.5, 0.5], [-0.3, 6.0, 3.0, 1.1]])
shift = 0.1
f0 = np.fft.fftn(kernel_from_table(np, big, tab, 6, +1))[:, 0, 0].real
f1 = np.fft.fftn(kernel_from_table(np, big, tilt_table(tab, shift, 6, L), 6, +1))[:, 0, 0].real
nz = np.arange(-8, 16)
nz = nz[f0[nz] > 0.3 * f0.max()]  # where the kernel is large; in its far tails the wide (small rho_z) Gaussians wrap around the Brillouin zone
alpha = shift * (2 * np.pi / L * 6) * tab[0, 2] ** 2 / 2
ratio = f1[nz] / f0[nz] / np.exp(alpha * 2 * np.pi * nz / L)
print(f"4. tilt: f_tilted / (f * exp(alpha p_z)) over n_z = {nz.min()}..{nz.max()}: max/min - 1 =", ratio.max() / ratio.min() - 1)
assert ratio.max() / ratio.min() - 1 < 1e-5
print("all checks passed")
