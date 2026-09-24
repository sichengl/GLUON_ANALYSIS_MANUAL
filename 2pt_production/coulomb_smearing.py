"""
Boosted Gaussian smearing in Coulomb gauge: use it only on a Coulomb-gauge-fixed configuration.

Source and sink are smeared with an explicit Gaussian times a plane wave, with no gauge links:

    K(d) = exp(-(dx^2 + dy^2) / rho_T^2 - dz^2 / rho_z^2) * exp(+i 2pi/L k.d),    d = x - y (nearest periodic image)

rho_T = rho_z is the isotropic smearing, rho_z = rho_T / 2 halves the radius along z.
k is in units of 2pi/L, with the same meaning as in mom_smearing.py:
  - at rho_T = rho_z = rho this is the free-field limit of source.gaussianSmear(rho, n_steps),
    whose profile is exp(-r^2 / rho^2);
  - the sign of the phase reproduces add_momentum_phase_to_gauge(sign=-1).
Both were verified on FNAL and on Frontier (2026-09-23): the isotropic source overlaps the free-field N40 rho3.25
mom_smearing source at 0.9999 for the same k, and at 0.0761 (k = 3.6) and 0.5212 (k = 1.8) for the opposite k.
coulomb_gauge_theta() measures how well the configuration satisfies the Coulomb condition the smearing relies on.
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


# ---------------- the smearing kernel and its two uses, on plain arrays ----------------


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
    x0, y0, z0, t0 = src_pos
    data = xp.zeros((Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), "<c16")
    t_local = t0 - t_offset
    if 0 <= t_local < Lt:
        kernel = boosted_gaussian_kernel(xp, [Lx, Ly, Lz], kvec, rho_T, rho_z)
        eta = xp.roll(kernel, shift=(z0, y0, x0), axis=(0, 1, 2))  # eta(x) = K(x - x_src)
        spin_identity = xp.eye(Ns).reshape(Ns, Ns, 1, 1)
        color_identity = xp.eye(Nc).reshape(1, 1, Nc, Nc)
        data[t_local] = eta.reshape(Lz, Ly, Lx, 1, 1, 1, 1) * spin_identity * color_identity
    return data


def sink_smear_evenodd(data_evenodd, spatial_size, kvec, rho_T, rho_z):
    """prop_smeared(x) = sum_y K(x - y) prop(y) on every time slice, done as a product of 3D FFTs."""
    xp = array_module(data_evenodd)
    kernel_ft = xp.fft.fftn(boosted_gaussian_kernel(xp, spatial_size, kvec, rho_T, rho_z))
    prop_ft = xp.fft.fftn(to_lexico(data_evenodd), axes=(1, 2, 3))
    prop_ft *= kernel_ft.reshape(1, *kernel_ft.shape, *([1] * (prop_ft.ndim - 4)))
    smeared = xp.fft.ifftn(prop_ft, axes=(1, 2, 3))
    del prop_ft
    return to_evenodd(smeared)


# ---------------- PyQUDA wrappers used by the production script ----------------


def _check_grid(latt_info):
    if list(latt_info.grid_size[:3]) != [1, 1, 1]:
        raise RuntimeError(f"coulomb_smearing needs an unsplit spatial grid, got grid {latt_info.grid_size}")


def coulomb_boosted_source(latt_info, kvec, src_pos, rho_T, rho_z):
    from pyquda.field import LatticePropagator

    _check_grid(latt_info)
    data = source_lexico(cupy, latt_info.size, latt_info.gt * latt_info.Lt, kvec, src_pos, rho_T, rho_z)
    return LatticePropagator(latt_info, to_evenodd(data))


def coulomb_boosted_sink(latt_info, propagator, kvec, rho_T, rho_z):
    from pyquda.field import LatticePropagator

    _check_grid(latt_info)
    return LatticePropagator(latt_info, sink_smear_evenodd(propagator.data, latt_info.size[:3], kvec, rho_T, rho_z))


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

