"""
Pion 2pt of the parameter optimization, with the sink smearing done as weights in momentum space.

The production script smears both solutions at the sink and then contracts:

    C(t, P) = sum_x exp(-2 pi i P.(x - x_src)/L) sum_spin,color  conj(prop1)[j,i,b,a] (G5 Gamma)[j,k] prop2[k,l,b,a] (Gamma_bar G5)[l,i]

Sink smearing multiplies the 3D FFT of a propagator with the FFT of the kernel, so the same number is a sum over the
quark momentum m (the antiquark has P - m):

    C(t, P) = exp(+2 pi i P.x_src/L) / V * sum_m  W(m) * M(t, m)
    M(t, m) = sum_spin,color conj(S1ft[m - P]) (G5 Gamma) S2ft[m] (Gamma_bar G5)      S1ft, S2ft: FFT of the UNSMEARED solutions
    W(m)    = kft2[m] * conj(kft1[m - P])                                              kft1, kft2: FFT of the two sink kernels

The same holds for every pair (Gamma at the sink, Gamma at the source): the gamma matrix of correlators comes from the
same two solutions, with no further inversion.

M does not know the sink kernel. So one FFT and one contraction per source give the correlator for the current kernel
(W built from the kernel) AND its derivative with respect to every kernel parameter (W built from the kernel derivative),
however many parameters there are. That derivative moves only the sink; the derivative of the correlator with the same
kernel at both ends is 2 Re of it (on average over configurations).

Plain array functions (numpy or cupy through the xp argument), so they can be tested on a laptop.
"""

import numpy
from opt_einsum import contract

Ns, Nc = 4, 3


def time_slices(t0, GLt, t_max):
    """Global time of every (fb, tau): fb = 0 forward t0 + tau, fb = 1 backward t0 - tau, tau = 0 .. t_max. Shape (2, t_max + 1)."""
    tau = numpy.arange(t_max + 1)
    return numpy.stack([(t0 + tau) % GLt, (t0 - tau) % GLt])


def source_lexico_from_kernel(xp, local_size, t_offset, kernel, src_pos):
    """
    Smeared source propagator [t, z, y, x, spin, spin, color, color], nonzero only on the source time slice.
    Copy of coulomb_smearing.source_lexico, with the kernel K[dz, dy, dx] given instead of built from one Gaussian.
    """
    Lx, Ly, Lz, Lt = local_size
    x0, y0, z0, t0 = src_pos
    data = xp.zeros((Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), "<c16")
    t_local = t0 - t_offset
    if 0 <= t_local < Lt:
        eta = xp.roll(kernel, shift=(z0, y0, x0), axis=(0, 1, 2))  # eta(x) = K(x - x_src)
        spin_identity = xp.eye(Ns).reshape(Ns, Ns, 1, 1)
        color_identity = xp.eye(Nc).reshape(1, 1, Nc, Nc)
        data[t_local] = eta.reshape(Lz, Ly, Lx, 1, 1, 1, 1) * spin_identity * color_identity
    return data


def pair_weights(xp, kft1, kft2, dkft1, dkft2, pz_list):
    """
    W[ipz, w, nz, ny, nx] for the hadron momentum P = (0, 0, pz):
        w = 0      kft2[m] * conj(kft1[m - P])                                  the kernel itself
        w = 1 + i  d/d parameter i of that product (product rule: quark and antiquark both carry the kernel)
    kft1, dkft1: prop1's kernel (boost -k) and its derivatives; kft2, dkft2: prop2's (boost +k). Arrays [nz, ny, nx].
    """
    W = xp.empty((len(pz_list), 1 + len(dkft2)) + kft2.shape, "<c16")
    for ipz, pz in enumerate(pz_list):
        k1 = xp.roll(kft1, pz, axis=0).conj()  # conj(kft1[m - P]): z is the first axis of [nz, ny, nx]
        dk1 = xp.roll(dkft1, pz, axis=1).conj()
        W[ipz, 0] = kft2 * k1
        W[ipz, 1:] = dkft2 * k1 + kft2 * dk1
    return W


def resolved_pion(xp, S1, S2, G5G_sink_list, Gbar_G5_source_list, W, pz_list):
    """
    S1, S2: the UNSMEARED solutions on some time slices, [nt, z, y, x, spin, spin, color, color] (prop1: source boost -k,
    prop2: source boost +k). One entry per gamma: G5G_sink_list[a] = G5 @ Gamma_a, Gbar_G5_source_list[b] =
    (G4 Gamma_b^dag G4) @ G5, as in production.
    Returns out[a, b, ipz, w, nt] = 1/V sum_m W[ipz, w, m] M_ab(t, m), a = gamma at the sink, b = gamma at the source:
    multiply by exp(+2 pi i pz z_src / L) to get the correlator matrix (w = 0) and its sink derivatives (w >= 1) for a
    source at z_src.
    """
    nt, Lz, Ly, Lx = S1.shape[:4]
    S1ft = xp.fft.fftn(S1, axes=(1, 2, 3))
    S2ft = xp.fft.fftn(S2, axes=(1, 2, 3))
    # both gammas on the quark line, once for all momenta: one array per (sink gamma, source gamma)
    S2_gammas = [[contract("jk,tzyxklba,li->tzyxjiba", A, S2ft, B) for B in Gbar_G5_source_list] for A in G5G_sink_list]
    del S2ft
    out = xp.empty((len(G5G_sink_list), len(Gbar_G5_source_list), len(pz_list), W.shape[1], nt), "<c16")
    for ipz, pz in enumerate(pz_list):
        S1_shifted = xp.roll(S1ft, pz, axis=1).conj()  # conj(S1ft[m - P]): z is axis 1 of [t, z, y, x, ...]
        for a in range(len(G5G_sink_list)):
            for b in range(len(Gbar_G5_source_list)):
                M = (S1_shifted * S2_gammas[a][b]).sum(axis=(4, 5, 6, 7))  # [nt, nz, ny, nx]
                out[a, b, ipz] = W[ipz].reshape(W.shape[1], -1) @ M.reshape(nt, -1).T / (Lz * Ly * Lx)
        del S1_shifted
    return out
