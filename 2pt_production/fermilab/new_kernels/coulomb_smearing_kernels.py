"""
Single-quark smearing kernels in Coulomb gauge for boosted pions: use them only on a Coulomb-gauge-fixed configuration.

This module contains all of coulomb_smearing.py, with bit-for-bit the same results, so a script that imports
    from coulomb_smearing_kernels import coulomb_boosted_source, coulomb_boosted_sink, coulomb_gauge_theta
runs exactly as with coulomb_smearing.py. An sbatch script that copies the smearing file into its run folder must copy this
file instead (SMEAR=coulomb_smearing_kernels.py, with the path where this file is), or python will not find it.
In addition it holds the kernels of the note "Single-Quark Smearing Kernels for
Boosted Pions" (pion_smearing_kernels.pdf, Secs. 1-7; the two-body and hadron-level methods of its Appendix A are not here),
and two functions that smear a source and a sink with any of them:

    coulomb_kernel_source(latt_info, kernel, src_pos)
    coulomb_kernel_sink(latt_info, propagator, kernel)

A kernel is a complex array K[dz, dy, dx] over the displacement d = x - y inside one quark smearing profile, with index
i <-> d = i for i < L/2 and d = i - L otherwise (nearest periodic image), the layout of boosted_gaussian_kernel. The source
is eta(x) = K(x - x_src) on the slice t_src and the sink is prop_smeared(x) = sum_y K(x - y) prop(y), as for the boosted Gaussian.

Conventions shared by all the new kernels (the pion moves along z):
  - lengths (rho_T, rho_z, R_c, R, ...) are in lattice units, amplitude convention exp(-r^2 / rho^2);
  - momenta (k0, delta_mu, delta, P_z, ...) are in units of 2pi/Lz along z, with the phase exp(+i 2pi/Lz k dz) of
    coulomb_smearing.py: k0 here is kvec[2] there. Two exceptions, stated in their functions: v0 of model_WG_kernel is a
    momentum variance in lattice units^-2, and alpha_b of gpd_kernel is in lattice units^2;
  - s = dx^2 + dy^2 is the squared transverse displacement d_perp^2;
  - every kernel has the form of Eq. (40), K = N G_env(s, dz) exp(i mu(s) dz) chi(dz | s): the real envelope G_env is normalized
    to sum 1 over the lattice, as the Gaussian of coulomb_smearing.py, and the momentum factor (chi(0 | s) = 1) multiplies it.
    Two exceptions: polynomial_kernel normalizes its Gaussian before the polynomial multiplies it (the deformed envelope can
    sum to zero), and linear_combination_kernel does not normalize the sum again. The overall normalization does not change
    relative overlaps (note, Sec. 1.3);
  - at the half-box displacement dz = -Lz/2 the two periodic images dz = -Lz/2 and +Lz/2 are averaged, as in
    coulomb_smearing.py. This keeps K(-d) = K(d)^* (Eq. (41)), which makes the sink smearing the adjoint of the source smearing;
  - reversing every momentum gives the complex conjugate kernel: kernel.conj() is the "-k" kernel of the antiquark line.

Kernels, with the equations of the note:
    gaussian_kernel                 Eqs. (6), (8)   boosted (an)isotropic Gaussian, the reference K_G
    two_scale_kernel                Eq. (9)         narrow core plus broad tail
    multi_momentum_kernel           Eq. (10)        discrete momentum superposition
    two_momentum_kernel             Eq. (11)        G exp(i k0 dz) cos(delta dz)
    momentum_superposition_kernel   Eqs. (12)-(14)  continuous momentum superposition, on quadrature nodes
    model_M_kernel                  Eqs. (16)-(18)  transverse-dependent mean momentum
    model_V_kernel                  Eqs. (20)-(21)  transverse-dependent momentum splitting
    model_W_kernel                  Eqs. (23)-(24)  transverse-dependent longitudinal width
    model_WG_kernel                 Eq. (25)        the W family written with a Gaussian characteristic function
    model_MV_kernel                 Eq. (26)        mean and splitting together
    conditional_kernel              Eqs. (27)-(28)  any conditional momentum amplitude w(x | s)
    beta_kernel                     Eqs. (30)-(31)  w(x | s) from a beta distribution
    gpd_kernel                      Eqs. (33)-(34)  w(x | s) from an illustrative impact-parameter GPD model
    stretched_kernel                Eq. (35)        stretched-exponential envelope
    exponential_kernel              Eq. (36)        radial exponential envelope
    power_tail_kernel               Eq. (37)        power-law tail
    polynomial_kernel               Eq. (38)        Gaussian times a low-order polynomial
    linear_combination_kernel       Eq. (39)        sum_a c_a K_a of any of the above

Use in a production script, in place of coulomb_boosted_source / coulomb_boosted_sink (prop1 carries the antiquark, prop2 the quark):
    kernel = model_M_kernel(cupy, latt_info.size[:3], k0, rho_T, rho_z, delta_mu, R_c)     # the "+k" kernel
    prop1_inv = coulomb_kernel_source(latt_info, kernel.conj(), src_pos)                   # the "-k" kernel
    prop2_inv = coulomb_kernel_source(latt_info, kernel, src_pos)
    ... invert both ...
    prop1 = coulomb_kernel_sink(latt_info, prop1_inv, kernel.conj())
    prop2 = coulomb_kernel_sink(latt_info, prop2_inv, kernel)
A kernel does not depend on the source position, so build it once per smearing. With
gaussian_kernel(cupy, latt_info.size[:3], k, rho_T, rho_z) this reproduces coulomb_boosted_source(latt_info, [0, 0, k], ...).
The kernels can also be built with numpy (xp = numpy) to look at them on a laptop; the two PyQUDA functions move them to the GPU.

The boosted Gaussian of coulomb_smearing.py, kept as it was:

    K(d) = exp(-(dx^2 + dy^2) / rho_T^2 - dz^2 / rho_z^2) * exp(+i 2pi/L k.d),    d = x - y (nearest periodic image)

rho_T = rho_z is the isotropic smearing, rho_z = rho_T / 2 halves the radius along z.
k is in units of 2pi/L, with the same meaning as in mom_smearing.py:
  - at rho_T = rho_z = rho this is the free-field limit of source.gaussianSmear(rho, n_steps),
    whose profile is exp(-r^2 / rho^2);
  - the sign of the phase reproduces add_momentum_phase_to_gauge(sign=-1).
Both were verified on FNAL and on Frontier (2026-09-23): the isotropic source overlaps the free-field N40 rho3.25
mom_smearing source at 0.9999 for the same k, and at 0.0761 (k = 3.6) and 0.5212 (k = 1.8) for the opposite k.
coulomb_gauge_theta() measures how well the configuration satisfies the Coulomb condition the smearing relies on.
mean_radius() measures the rms radius of a propagator (or a smeared source) around its source position.
The Gaussian part is normalized to sum 1, as the Wuppertal smearing is at k = 0.

The sink smearing is a 3D FFT on every time slice, so the spatial directions must not be split
over MPI ranks: the process grid must be [1, 1, 1, Gt].
"""

import numpy

try:
    import cupy
except ImportError:  # lets the pure-array functions be tested on a machine without a GPU
    cupy = None

Ns, Nc = 4, 3
BOOST_PHASE_SIGN = +1  # +1 reproduces mom_smearing.py (verified, see above); do not change


def array_module(a):
    if cupy is not None:
        return cupy.get_array_module(a)
    return numpy


# ---------------- layout: PyQUDA even-odd  <->  lexicographic ----------------
# even-odd: data[parity, t, z, y, x // 2, ...] with parity = (x + y + z + t) % 2 in local coordinates
# lexico:   data[t, z, y, x, ...]


def to_lexico(data_evenodd):
    xp = array_module(data_evenodd)
    _, Lt, Lz, Ly, Lx_half = data_evenodd.shape[:5]
    extra = data_evenodd.ndim - 5
    t = xp.arange(Lt).reshape(Lt, 1, 1)
    z = xp.arange(Lz).reshape(1, Lz, 1)
    y = xp.arange(Ly).reshape(1, 1, Ly)
    row_is_even = ((t + z + y) % 2 == 0).reshape(Lt, Lz, Ly, 1, *([1] * extra))  # site x = 0 of this row is even
    data_lexico = xp.empty((Lt, Lz, Ly, 2 * Lx_half, *data_evenodd.shape[5:]), data_evenodd.dtype)
    data_lexico[:, :, :, 0::2] = xp.where(row_is_even, data_evenodd[0], data_evenodd[1])
    data_lexico[:, :, :, 1::2] = xp.where(row_is_even, data_evenodd[1], data_evenodd[0])
    return data_lexico


def to_evenodd(data_lexico):
    xp = array_module(data_lexico)
    Lt, Lz, Ly, Lx = data_lexico.shape[:4]
    extra = data_lexico.ndim - 4
    t = xp.arange(Lt).reshape(Lt, 1, 1)
    z = xp.arange(Lz).reshape(1, Lz, 1)
    y = xp.arange(Ly).reshape(1, 1, Ly)
    row_is_even = ((t + z + y) % 2 == 0).reshape(Lt, Lz, Ly, 1, *([1] * extra))
    data_evenodd = xp.empty((2, Lt, Lz, Ly, Lx // 2, *data_lexico.shape[4:]), data_lexico.dtype)
    data_evenodd[0] = xp.where(row_is_even, data_lexico[:, :, :, 0::2], data_lexico[:, :, :, 1::2])
    data_evenodd[1] = xp.where(row_is_even, data_lexico[:, :, :, 1::2], data_lexico[:, :, :, 0::2])
    return data_evenodd


# ---------------- smearing with any kernel K[z, y, x], on plain arrays ----------------


def kernel_source_lexico(xp, local_size, t_offset, kernel, src_pos):
    """Smeared source propagator [t, z, y, x, spin, spin, color, color] with eta(x) = K(x - x_src), nonzero only on the source time slice."""
    Lx, Ly, Lz, Lt = local_size
    x0, y0, z0, t0 = src_pos
    data = xp.zeros((Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), "<c16")
    t_local = t0 - t_offset
    if 0 <= t_local < Lt:
        eta = xp.roll(kernel, shift=(z0, y0, x0), axis=(0, 1, 2))  # eta(x) = K(x - x_src)
        spin_identity = xp.eye(Ns).reshape(Ns, Ns, 1, 1)
        color_identity = xp.eye(Nc).reshape(1, 1, Nc, Nc)
        data[t_local] = eta.reshape(Lz, Ly, Lx, 1, 1, 1, 1) * spin_identity * color_identity #source is only nonzero in the diagonal dirac-color space
    return data


def kernel_sink_evenodd(data_evenodd, kernel):
    """prop_smeared(x) = sum_y K(x - y) prop(y) on every time slice, done as a product of 3D FFTs."""
    xp = array_module(data_evenodd)
    kernel_ft = xp.fft.fftn(kernel)
    prop_ft = xp.fft.fftn(to_lexico(data_evenodd), axes=(1, 2, 3))
    prop_ft *= kernel_ft[None, :, :, :, None, None, None, None]
    smeared = xp.fft.ifftn(prop_ft, axes=(1, 2, 3))
    del prop_ft
    return to_evenodd(smeared)


# ---------------- the boosted Gaussian kernel of coulomb_smearing.py and its two uses ----------------


def _one_direction(xp, L, k, rho):
    """Gaussian and boost phase along one direction, index i <-> displacement d = i for i < L/2 and d = i - L otherwise."""
    d = (xp.arange(L) + L // 2) % L - L // 2
    gaussian = xp.exp(-(d**2) / rho**2)
    phase = xp.exp(1j * BOOST_PHASE_SIGN * 2 * numpy.pi * k * d / L)
    if L % 2 == 0:
        phase[L // 2] = numpy.cos(2 * numpy.pi * k * (L // 2) / L)  # average of the images d = +L/2 and -L/2: keeps K Hermitian
    return gaussian, phase


def boosted_gaussian_kernel(xp, spatial_size, kvec, rho_T, rho_z):
    """K[dz, dy, dx] for the displacement d = x - y."""
    Lx, Ly, Lz = spatial_size
    gx, px = _one_direction(xp, Lx, kvec[0], rho_T)
    gy, py = _one_direction(xp, Ly, kvec[1], rho_T)
    gz, pz = _one_direction(xp, Lz, kvec[2], rho_z)
    gaussian = gz.reshape(Lz, 1, 1) * gy.reshape(1, Ly, 1) * gx.reshape(1, 1, Lx)
    phase = pz.reshape(Lz, 1, 1) * py.reshape(1, Ly, 1) * px.reshape(1, 1, Lx)
    return gaussian / gaussian.sum() * phase


def source_lexico(xp, local_size, t_offset, kvec, src_pos, rho_T, rho_z):
    """Smeared source propagator [t, z, y, x, spin, spin, color, color], nonzero only on the source time slice."""
    Lx, Ly, Lz, Lt = local_size
    kernel = boosted_gaussian_kernel(xp, [Lx, Ly, Lz], kvec, rho_T, rho_z)
    return kernel_source_lexico(xp, local_size, t_offset, kernel, src_pos)


def sink_smear_evenodd(data_evenodd, spatial_size, kvec, rho_T, rho_z):
    """prop_smeared(x) = sum_y K(x - y) prop(y) on every time slice, done as a product of 3D FFTs."""
    xp = array_module(data_evenodd)
    return kernel_sink_evenodd(data_evenodd, boosted_gaussian_kernel(xp, spatial_size, kvec, rho_T, rho_z))


def squared_distance(xp, spatial_size, src_pos):
    """|d|^2[z, y, x] for d = x - x_src, spatial only, each component taken to the nearest periodic image."""
    Lx, Ly, Lz = spatial_size
    d2 = xp.zeros((Lz, Ly, Lx))
    for L, x0, shape in ((Lx, src_pos[0], (1, 1, Lx)), (Ly, src_pos[1], (1, Ly, 1)), (Lz, src_pos[2], (Lz, 1, 1))):
        d = (xp.arange(L) - x0 + L // 2) % L - L // 2
        d2 = d2 + (d**2).reshape(shape)
    return d2


def radius_moments_lexico(slice_lexico, src_pos):
    """
    Sum of |S|^2 and sum of |d|^2 |S|^2 over the sites of one time slice S[z, y, x, spin, spin, color, color] and over
    every Dirac-color entry, returned as two numbers. d = x - x_src is spatial only.
    """
    xp = array_module(slice_lexico)
    Lz, Ly, Lx = slice_lexico.shape[:3]
    weight = (abs(slice_lexico) ** 2).sum(axis=(3, 4, 5, 6))  # [z, y, x]: |S|^2 summed over every Dirac-color entry
    d2 = squared_distance(xp, [Lx, Ly, Lz], src_pos)
    return float(weight.sum()), float((d2 * weight).sum())


# ---------------- building blocks of the new kernels ----------------
# Every array below is laid out as the kernel, [z, y, x]: a transverse array has shape [1, Ly, Lx] and a longitudinal one
# [Lz, 1, 1], so that the two broadcast to the full kernel.


def displacement(xp, L):
    """Displacement d of every index along one direction: d = i for i < L/2 and d = i - L otherwise (d = -L/2 at i = L/2)."""
    return (xp.arange(L) + L // 2) % L - L // 2


def transverse_s(xp, spatial_size):
    """s = dx^2 + dy^2, the squared transverse displacement, shape [1, Ly, Lx]."""
    Lx, Ly, Lz = spatial_size
    dx = displacement(xp, Lx).reshape(1, 1, Lx)
    dy = displacement(xp, Ly).reshape(1, Ly, 1)
    return 1.0 * (dx**2 + dy**2)


def z_displacement(xp, spatial_size):
    """dz, the longitudinal displacement, shape [Lz, 1, 1]."""
    Lz = spatial_size[2]
    return 1.0 * displacement(xp, Lz).reshape(Lz, 1, 1)


def gaussian_envelope(xp, spatial_size, rho_T, rho_z):
    """G(d; rho_T, rho_z) = exp(-s / rho_T^2 - dz^2 / rho_z^2) of Eq. (6), not normalized (G = 1 at d = 0)."""
    s = transverse_s(xp, spatial_size)
    dz = z_displacement(xp, spatial_size)
    return xp.exp(-s / rho_T**2 - dz**2 / rho_z**2)


def normalized(envelope):
    """The real envelope divided by its sum over the lattice: the factor N of the note, as for boosted_gaussian_kernel."""
    total = float(envelope.sum())
    if total == 0:
        raise ValueError("the envelope sums to zero and cannot be normalized")
    return envelope / total


def momentum_factor(xp, spatial_size, amplitudes, momenta):
    """
    sum_j a_j exp(+i 2pi/Lz kappa_j dz), a coherent sum of longitudinal momentum packets: the factor exp(i mu(s) dz) chi(dz | s)
    of Eq. (40). Every a_j and kappa_j (in units of 2pi/Lz) is a number or a transverse array [1, Ly, Lx] (then it depends on s).
    For even Lz the two images dz = -Lz/2 and +Lz/2 of the half-box index are averaged, as in _one_direction: one packet gives
    cos(pi kappa) there, and real a_j, kappa_j keep K(-d) = K(d)^* on the lattice.
    """
    Lz = spatial_size[2]
    dz_minus = z_displacement(xp, spatial_size)  # dz = -Lz/2 at the half-box index
    dz_plus = dz_minus.copy()
    if Lz % 2 == 0:
        dz_plus[Lz // 2] = Lz // 2  # the other image, dz = +Lz/2
    factor = 0
    for dz in [dz_minus, dz_plus]:
        for a, kappa in zip(amplitudes, momenta):
            factor = factor + 0.5 * a * xp.exp(1j * BOOST_PHASE_SIGN * 2 * numpy.pi * kappa * dz / Lz)
    return factor


def radial_function(s, R_c, power=None):
    """
    The decreasing radial function f(s) of Eq. (15): exp(-s / (2 R_c^2)), or (1 + s / R_c^2)^(-power) when power (n > 0) is given.
    It is 1 at s = 0 and goes to 0 at large s.
    """
    xp = array_module(s)
    if power is None:
        return xp.exp(-s / (2 * R_c**2))
    return (1 + s / R_c**2) ** (-power)


def transverse_mean(s, f, rho_T):
    """
    f_bar of Eq. (18): the mean of f over the transverse plane with the weight |G_T|^2 = exp(-2 s / rho_T^2), as a lattice sum.
    In the continuum it is 1 / (1 + rho_T^2 / (4 R_c^2)) for the Gaussian f.
    """
    xp = array_module(s)
    weight = xp.exp(-2 * s / rho_T**2)
    return float((weight * f).sum() / weight.sum())


def midpoint_nodes(n_x):
    """n_x midpoint nodes x_j = (j + 1/2) / n_x of the interval 0 < x < 1, for the x integral of Eq. (28)."""
    nodes = []
    for j in range(n_x):
        nodes.append((j + 0.5) / n_x)
    return nodes


# ---------------- Secs. 1-3: the Gaussian reference, two scales, momentum superpositions ----------------


def gaussian_kernel(xp, spatial_size, k0, rho_T, rho_z):
    """
    K_G of Eq. (6), K_aniso of Eq. (8): G(d; rho_T, rho_z) exp(i k0 dz), the reference kernel. rho_T = rho_z is isotropic.
    The same as boosted_gaussian_kernel(xp, spatial_size, [0, 0, k0], rho_T, rho_z) up to rounding.
    """
    envelope = gaussian_envelope(xp, spatial_size, rho_T, rho_z)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [1.0], [k0])


def two_scale_kernel(xp, spatial_size, k0, A, rho_T1, rho_z1, B, rho_T2, rho_z2):
    """
    K_2G of Eq. (9): [A G(d; rho_T1, rho_z1) + B G(d; rho_T2, rho_z2)] exp(i k0 dz).
    A and B multiply Gaussians that are 1 at d = 0, as in the note; the sum is then normalized. A, B > 0 give a narrow core and
    a broad tail without a node; opposite signs can make a node.
    """
    envelope = A * gaussian_envelope(xp, spatial_size, rho_T1, rho_z1) + B * gaussian_envelope(xp, spatial_size, rho_T2, rho_z2)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [1.0], [k0])


def multi_momentum_kernel(xp, spatial_size, k_list, rho_T, rho_z, a_list):
    """
    K_MM of Eq. (10): G(d) sum_j a_j exp(i k_j dz). The a_j are divided by their sum, so that sum_j a_j = 1.
    They are amplitude weights, not probabilities; use real a_j to keep K(-d) = K(d)^*.
    """
    total = sum(a_list)
    if total == 0:
        raise ValueError("the amplitudes a_j sum to zero")
    amplitudes = []
    for a in a_list:
        amplitudes.append(a / total)
    envelope = gaussian_envelope(xp, spatial_size, rho_T, rho_z)
    return normalized(envelope) * momentum_factor(xp, spatial_size, amplitudes, k_list)


def two_momentum_kernel(xp, spatial_size, k0, rho_T, rho_z, delta):
    """
    K_2k of Eq. (11): G(d) [exp(i (k0 + delta) dz) + exp(i (k0 - delta) dz)] / 2 = G(d) exp(i k0 dz) cos(delta dz).
    delta = 0 gives gaussian_kernel.
    """
    return multi_momentum_kernel(xp, spatial_size, [k0 + delta, k0 - delta], rho_T, rho_z, [0.5, 0.5])


def momentum_superposition_kernel(xp, spatial_size, k0, rho_T, rho_z, q_list, w_list):
    """
    K_super of Eq. (12): G(d) int dq w(q) exp(i (k0 + q) dz), with the q integral done on the nodes q_list.
    w_list[j] = w(q_j) dq_j; the weights are divided by their sum, so that int dq w(q) = 1.
    A centered Gaussian w(q) only narrows the longitudinal width (Eqs. (13)-(14)): it is not a new shape.
    """
    k_list = []
    for q in q_list:
        k_list.append(k0 + q)
    return multi_momentum_kernel(xp, spatial_size, k_list, rho_T, rho_z, w_list)


# ---------------- Sec. 4: transverse position - momentum correlations ----------------


def model_M_kernel(xp, spatial_size, k0, rho_T, rho_z, delta_mu, R_c, subtract_mean=True, power=None):
    """
    Model M, Eqs. (16)-(18): G(d) exp(i mu(s) dz) with the transverse-dependent mean momentum
        mu(s) = k0 + delta_mu [f(s) - f_bar],    f(s) of radial_function (Eq. (15)).
    f_bar is the |G_T|^2-weighted mean of f (transverse_mean), so the transverse mean of mu is k0; subtract_mean=False sets
    f_bar = 0. delta_mu > 0 for k0 > 0 puts the larger local momentum at small d_perp. Only the phase changes: |K| = |K_G|,
    except on the half-box slice dz = -Lz/2 of an even Lz, where the two images are averaged (negligible for rho_z << Lz/2).
    delta_mu = 0 gives gaussian_kernel.
    """
    s = transverse_s(xp, spatial_size)
    f = radial_function(s, R_c, power)
    f_bar = 0.0
    if subtract_mean:
        f_bar = transverse_mean(s, f, rho_T)
    mu = k0 + delta_mu * (f - f_bar)
    envelope = gaussian_envelope(xp, spatial_size, rho_T, rho_z)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [1.0], [mu])


def model_V_kernel(xp, spatial_size, k0, rho_T, rho_z, delta0, R_c, opposite=False, power=None):
    """
    Model V, Eqs. (20)-(21): G(d) exp(i k0 dz) cos[delta(s) dz] with the transverse-dependent splitting delta(s) = delta0 f(s),
    two packets at k0 +- delta(s). opposite=True uses delta(s) = delta0 [1 - f(s)] instead, the opposite hypothesis of the note
    (split outside, unsplit at the center). delta0 = 0 gives gaussian_kernel.
    """
    s = transverse_s(xp, spatial_size)
    f = radial_function(s, R_c, power)
    if opposite:
        delta = delta0 * (1 - f)
    else:
        delta = delta0 * f
    envelope = gaussian_envelope(xp, spatial_size, rho_T, rho_z)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [0.5, 0.5], [k0 + delta, k0 - delta])


def model_W_kernel(xp, spatial_size, k0, rho_T, rho_z0, alpha, R_c, power=None):
    """
    Model W, Eqs. (23)-(24): exp(-s / rho_T^2 - dz^2 / rho_z(s)^2) exp(i k0 dz) with the transverse-dependent longitudinal radius
        rho_z(s) = rho_z0 [1 + alpha f(s)],    alpha > -1.
    alpha > 0: longer along z (narrower in momentum) at the center; -1 < alpha < 0: shorter. alpha = 0 gives gaussian_kernel.
    Unlike Model M this changes |K| too; its envelope is normalized to sum 1.
    """
    if alpha <= -1:
        raise ValueError("Model W needs alpha > -1")
    s = transverse_s(xp, spatial_size)
    dz = z_displacement(xp, spatial_size)
    rho_z = rho_z0 * (1 + alpha * radial_function(s, R_c, power))
    envelope = xp.exp(-s / rho_T**2 - dz**2 / rho_z**2)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [1.0], [k0])


def model_WG_kernel(xp, spatial_size, k0, rho_T, rho_z, v0, R_c, power=None):
    """
    Model WG, Eq. (25): G(d) exp(i k0 dz) exp(-v(s) dz^2 / 2) with v(s) = v0 f(s) >= 0 (v0 >= 0): a centered Gaussian
    characteristic function, which narrows the longitudinal radius to rho_z(s)^-2 = rho_z^-2 + v(s) / 2.
    v0 is a momentum variance in lattice units^-2 (the unit of 1 / rho_z^2), not in (2pi/Lz)^2: a variance v in units of
    (2pi/Lz)^2 is v0 = v (2pi/Lz)^2.
    The same family as Model W, not a new method. Following Eq. (40), only G is normalized, so with a matching width the two
    kernels differ by a constant factor. v0 = 0 gives gaussian_kernel.
    """
    if v0 < 0:
        raise ValueError("Model WG needs v0 >= 0")
    s = transverse_s(xp, spatial_size)
    dz = z_displacement(xp, spatial_size)
    v = v0 * radial_function(s, R_c, power)
    chi = xp.exp(-v * dz**2 / 2)
    envelope = gaussian_envelope(xp, spatial_size, rho_T, rho_z)
    return normalized(envelope) * chi * momentum_factor(xp, spatial_size, [1.0], [k0])


def model_MV_kernel(xp, spatial_size, k0, rho_T, rho_z, delta_mu, R_c, delta0, R_g, subtract_mean=True, power=None):
    """
    Model MV, Eq. (26): G(d) exp(i {k0 + delta_mu [f(s) - f_bar]} dz) cos[delta0 g(s) dz], the mean of Model M and the
    splitting of Model V together. f = radial_function(s, R_c, power) and g = radial_function(s, R_g, power): the same shape
    with its own length R_g (R_g = R_c is the minimal choice g = f). delta0 = 0 gives Model M, delta_mu = 0 gives Model V,
    both 0 give gaussian_kernel.
    """
    s = transverse_s(xp, spatial_size)
    f = radial_function(s, R_c, power)
    g = radial_function(s, R_g, power)
    f_bar = 0.0
    if subtract_mean:
        f_bar = transverse_mean(s, f, rho_T)
    mu = k0 + delta_mu * (f - f_bar)
    delta = delta0 * g
    envelope = gaussian_envelope(xp, spatial_size, rho_T, rho_z)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [0.5, 0.5], [mu + delta, mu - delta])


# ---------------- Sec. 5: conditional longitudinal-momentum amplitude ----------------


def conditional_kernel(xp, spatial_size, P_z, rho_T, rho_z, x_list, w_list):
    """
    K_cond of Eqs. (27)-(28): G(d) chi(dz | s) with chi(dz | s) = int_0^1 dx w(x | s) exp(i x P_z dz), the x integral done on
    the nodes x_list (for example midpoint_nodes(n_x)). P_z is the pion momentum in units of 2pi/Lz, so node x_j is a packet
    with momentum x_j P_z.
    w_list[j] = w(x_j | s) dx_j is a number, or a transverse array [1, Ly, Lx] built from s = transverse_s(xp, spatial_size).
    At every s the weights are divided by their sum, so int dx w(x | s) = 1 and chi(0 | s) = 1 exactly.
    A node x_j may also be a transverse array (a node that moves with s). Real weights and nodes keep K(-d) = K(d)^*.
    Eq. (29): one node x0 with weight 1 gives gaussian_kernel with k0 = x0 P_z; with P_z = 1 the node [mu(s)] gives Model M,
    and the nodes [k0 + delta(s), k0 - delta(s)] with weights [1/2, 1/2] give Model V.
    """
    total = 0
    for w in w_list:
        total = total + w
    total = xp.asarray(total)
    if float(xp.abs(total).min()) == 0:
        raise ValueError("w(x | s) sums to zero at some s: check the weights")
    amplitudes = []
    momenta = []
    for x, w in zip(x_list, w_list):
        amplitudes.append(w / total)
        momenta.append(x * P_z)
    envelope = gaussian_envelope(xp, spatial_size, rho_T, rho_z)
    kernel = normalized(envelope) * momentum_factor(xp, spatial_size, amplitudes, momenta)
    if not bool(xp.isfinite(kernel).all()):
        raise ValueError("the kernel is not finite: w(x | s) overflows or holds nan at some s, check the weights")
    return kernel


def beta_kernel(xp, spatial_size, P_z, rho_T, rho_z, a0, a1, b0, b1, R_c, n_x=100, power=None):
    """
    K_cond with the beta-family weight of Eqs. (30)-(31):
        w(x | s) = x^(a(s) - 1) (1 - x)^(b(s) - 1) / B[a(s), b(s)],    a(s) = a0 + a1 f(s),  b(s) = b0 + b1 f(s),  a, b > 0.
    The x integral uses n_x cells of width 1/n_x: cell j has the exact weight int over the cell of w(x | s) dx (the regularized
    incomplete beta function, so the weights sum to 1 and stay right for a or b < 1, where w is singular at an end point) and
    its packet at the cell midpoint. For a or b well below 1 raise n_x (a = 0.3: chi is off by 1e-2 at n_x = 100, by 5e-4 at
    n_x = 1000). Needs scipy.
    <x>_s = a / (a + b) in the continuum: a1, b1 move the mean and the width with d_perp; a(s) = b(s) keeps <x>_s = 1/2.
    """
    from scipy.special import betainc

    s = transverse_s(numpy, spatial_size)  # a(s) and b(s) are small transverse arrays: the weights are built on the CPU
    f = radial_function(s, R_c, power)
    a = a0 + a1 * f
    b = b0 + b1 * f
    if float(a.min()) <= 0 or float(b.min()) <= 0:
        raise ValueError("the beta weight needs a(s) > 0 and b(s) > 0 at every s")
    x_list = midpoint_nodes(n_x)
    w_list = []
    for j in range(n_x):
        w = betainc(a, b, (j + 1) / n_x) - betainc(a, b, j / n_x)  # Eq. (30) integrated over cell j
        w_list.append(xp.asarray(w))
    return conditional_kernel(xp, spatial_size, P_z, rho_T, rho_z, x_list, w_list)


def gpd_kernel(xp, spatial_size, P_z, rho_T, rho_z, alpha_b, a_q, b_q, n_x=100):
    """
    K_cond with the GPD-inspired weight of Eqs. (33)-(34), w(x | s) = q(x, b) / int dy q(y, b), with b_perp identified with
    d_perp. That identification is a model assumption, not a consequence of QCD (Sec. 5.3).
    q(x, b) comes from an illustrative Gaussian-profile model of the zero-skewness GPD,
        H(x, 0, -Delta^2) = q(x) exp(-Delta^2 g(x))   ->   q(x, b) = q(x) / (4 pi g(x)) exp(-b^2 / (4 g(x))),
        q(x) = x^a_q (1 - x)^b_q,    g(x) = alpha_b (1 - x) ln(1 / x),
    so the quarks with x -> 1 sit at small b. alpha_b is in lattice units^2: alpha_b = alpha' [GeV^-2] * (0.1973 / a [fm])^2.
    <b^2>(x) = 4 g(x). Replace this model by your own w(x | s) through conditional_kernel.
    Needs a_q > -1 and b_q > 1: near x = 1, g ~ alpha_b (1 - x)^2, so at d_perp = 0 the weight goes as (1 - x)^(b_q - 2) and
    cannot be normalized for b_q <= 1, and near x = 0 it cannot for a_q <= -1. The x integral uses n_x midpoint nodes; for
    a_q < 0 or b_q < 2 the weight is still singular at an end point, so check that doubling n_x leaves the kernel unchanged.
    """
    if a_q <= -1 or b_q <= 1:
        raise ValueError("gpd_kernel needs a_q > -1 and b_q > 1: otherwise w(x | s) cannot be normalized and depends on n_x")
    s = transverse_s(xp, spatial_size)
    x_list = midpoint_nodes(n_x)
    w_list = []
    for x in x_list:
        q = x**a_q * (1 - x) ** b_q
        g = alpha_b * (1 - x) * numpy.log(1 / x)
        w_list.append(q / g * xp.exp(-s / (4 * g)))  # the 4 pi cancels in the normalization
    return conditional_kernel(xp, spatial_size, P_z, rho_T, rho_z, x_list, w_list)


# ---------------- Sec. 6: non-Gaussian spatial envelopes ----------------


def stretched_kernel(xp, spatial_size, k0, rho_T, rho_z, nu_T, nu_z):
    """
    K_stretch of Eq. (35): exp(-(d_perp / rho_T)^nu_T - (|dz| / rho_z)^nu_z) exp(i k0 dz), nu_T, nu_z > 0.
    nu = 2 is the Gaussian, nu = 1 an exponential; nu > 2 is flatter in the core, nu < 2 has longer tails.
    """
    s = transverse_s(xp, spatial_size)
    dz = z_displacement(xp, spatial_size)
    envelope = xp.exp(-(xp.sqrt(s) / rho_T) ** nu_T - (xp.abs(dz) / rho_z) ** nu_z)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [1.0], [k0])


def exponential_kernel(xp, spatial_size, k0, R):
    """K_exp of Eq. (36): exp(-sqrt(s + dz^2) / R) exp(i k0 dz), an isotropic exponential of radius R."""
    s = transverse_s(xp, spatial_size)
    dz = z_displacement(xp, spatial_size)
    envelope = xp.exp(-xp.sqrt(s + dz**2) / R)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [1.0], [k0])


def power_tail_kernel(xp, spatial_size, k0, R_T, R_z, p):
    """K_tail of Eq. (37): (1 + s / R_T^2 + dz^2 / R_z^2)^(-p) exp(i k0 dz). In infinite volume it is normalizable for p > 3/2."""
    s = transverse_s(xp, spatial_size)
    dz = z_displacement(xp, spatial_size)
    envelope = (1 + s / R_T**2 + dz**2 / R_z**2) ** (-p)
    return normalized(envelope) * momentum_factor(xp, spatial_size, [1.0], [k0])


def polynomial_kernel(xp, spatial_size, k0, rho_T, rho_z, c_T, c_z):
    """
    K_poly of Eq. (38): G(d) [1 + c_T (s / rho_T^2 - 1/2) + c_z (dz^2 / rho_z^2 - 1/4)] exp(i k0 dz).
    The subtractions are the |G|^2-weighted means in the continuum. G is normalized before the polynomial multiplies it (the
    deformed envelope can sum to zero), so c_T = c_z = 0 gives gaussian_kernel exactly. Coefficients that make a node may favor
    excited states.
    """
    s = transverse_s(xp, spatial_size)
    dz = z_displacement(xp, spatial_size)
    polynomial = 1 + c_T * (s / rho_T**2 - 0.5) + c_z * (dz**2 / rho_z**2 - 0.25)
    envelope = gaussian_envelope(xp, spatial_size, rho_T, rho_z)
    return normalized(envelope) * polynomial * momentum_factor(xp, spatial_size, [1.0], [k0])


# ---------------- Sec. 7: linear combinations ----------------


def linear_combination_kernel(kernels, coefficients):
    """
    K_lin of Eq. (39): sum_a c_a K_a, still a single-quark kernel. Each K_a keeps its own normalization and the sum is not
    normalized again. Real c_a keep K(-d) = K(d)^*. Smearing both quarks with K_lin gives cross terms in the bilinear: this is
    not the hadron-level variational combination of the appendix.
    """
    total = 0
    for c, kernel in zip(coefficients, kernels):
        total = total + c * kernel
    return total


# ---------------- PyQUDA wrappers used by the production script ----------------


def _check_grid(latt_info):
    if list(latt_info.grid_size[:3]) != [1, 1, 1]:
        raise RuntimeError(f"coulomb_smearing needs an unsplit spatial grid, got grid {latt_info.grid_size}")


def _check_kernel(latt_info, kernel):
    Lx, Ly, Lz = latt_info.size[:3]
    if tuple(kernel.shape) != (Lz, Ly, Lx):
        raise ValueError(f"the kernel has shape {tuple(kernel.shape)}, the lattice needs [z, y, x] = {(Lz, Ly, Lx)}")


def coulomb_boosted_source(latt_info, kvec, src_pos, rho_T, rho_z):
    from pyquda.field import LatticePropagator

    _check_grid(latt_info)
    data = source_lexico(cupy, latt_info.size, latt_info.gt * latt_info.Lt, kvec, src_pos, rho_T, rho_z)
    return LatticePropagator(latt_info, to_evenodd(data))


def coulomb_boosted_sink(latt_info, propagator, kvec, rho_T, rho_z):
    from pyquda.field import LatticePropagator

    _check_grid(latt_info)
    return LatticePropagator(latt_info, sink_smear_evenodd(propagator.data, latt_info.size[:3], kvec, rho_T, rho_z))


def coulomb_kernel_source(latt_info, kernel, src_pos):
    """
    Smeared source eta(x) = K(x - x_src) on the slice t_src, for any kernel K[z, y, x] of this module (numpy or cupy array).
    src_pos = [x0, y0, z0, t0] as for coulomb_boosted_source.
    """
    from pyquda.field import LatticePropagator

    _check_grid(latt_info)
    _check_kernel(latt_info, kernel)
    data = kernel_source_lexico(cupy, latt_info.size, latt_info.gt * latt_info.Lt, cupy.asarray(kernel), src_pos)
    return LatticePropagator(latt_info, to_evenodd(data))


def coulomb_kernel_sink(latt_info, propagator, kernel):
    """Sink smearing prop_smeared(x) = sum_y K(x - y) prop(y) on every time slice, for any kernel K[z, y, x] of this module."""
    from pyquda.field import LatticePropagator

    _check_grid(latt_info)
    _check_kernel(latt_info, kernel)
    return LatticePropagator(latt_info, kernel_sink_evenodd(propagator.data, cupy.asarray(kernel)))


def mean_radius(latt_info, propagator, src_pos):
    """
    Root-mean-square radius of a propagator around its source on the source time slice t0, with every Dirac-color entry
    pooled into one sum before the division:
        r^2 = sum_abcd sum_x |x - x_src|^2 |S_abcd(x)|^2 / sum_abcd sum_x |S_abcd(x)|^2,    x on the slice t = t0
    Entries that vanish (the off-diagonal Dirac entries of a smeared source) add nothing to either sum. Summing over both
    color indices makes r exactly gauge invariant, since only Tr S^dag S enters, so a gauge-covariant (Wuppertal) source
    gives the same r in any gauge.
    src_pos = [x0, y0, z0, t0] as for coulomb_boosted_source; the distance is spatial only (nearest periodic image). The
    plane-wave boost drops out of |S|^2: the boosted Gaussian source has r = rho sqrt(3) / 2 at rho_T = rho_z = rho,
    sqrt((2 rho_T^2 + rho_z^2) / 4) in general. For a solved propagator this is its radius on the slice t0 only.
    A smeared source lives on the slice t0 alone, so only the GPU that holds t0 computes, on that slice in lexicographic
    order. The one number is then shared so that every rank, including rank 0 that prints the log, returns it: every rank
    must call this function.
    """
    _check_grid(latt_info)
    t_local = src_pos[3] - latt_info.gt * latt_info.Lt
    r = 0.0
    if 0 <= t_local < latt_info.Lt:
        slice_evenodd = propagator.data[:, t_local : t_local + 1]
        if t_local % 2 == 1:
            slice_evenodd = slice_evenodd[::-1]  # to_lexico takes this one slice as t = 0, so an odd slice has its parities swapped
        norm, moment = radius_moments_lexico(to_lexico(slice_evenodd)[0], src_pos)
        r = (moment / norm) ** 0.5
    return latt_info.mpi_comm.allreduce(r)  # r on the GPU that holds t0, 0 on the others


def coulomb_gauge_theta(latt_info, gauge):
    """
    Violation of the Coulomb gauge condition, normalized like QUDA's theta (compare with the last theta QUDA prints):
        theta = sum_x Re Tr[Delta(x) Delta(x)^dag] / (3 V),   Delta = traceless part of D - D^dag,
        D(x)  = sum_{i = x, y, z} [U_i(x - i) - U_i(x)].
    It is zero in exact Coulomb gauge and of order 1 on an unfixed configuration. The plaquette cannot show this,
    because it is gauge invariant. Only spatial links enter and space is not split over ranks, so no halo is needed.
    Every rank must call it: the sum over time slices is an allreduce.
    """
    _check_grid(latt_info)
    xp = array_module(gauge.data)
    D = 0
    for i in range(3):
        U_i = to_lexico(gauge.data[i])  # [t, z, y, x, color, color]; direction i = x, y, z lives on array axis 3, 2, 1
        D = D + xp.roll(U_i, 1, axis=3 - i) - U_i  # U_i(x - i) - U_i(x)
    Delta = D - D.conj().swapaxes(-1, -2)
    Delta = Delta - (xp.trace(Delta, axis1=-2, axis2=-1) / Nc)[..., None, None] * xp.eye(Nc)
    local = float((abs(Delta) ** 2).sum().item())
    return latt_info.mpi_comm.allreduce(local) / (Nc * latt_info.global_volume)
