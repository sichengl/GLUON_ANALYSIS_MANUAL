"""
PYQUDA FFT version: a copy of the Delta direction-x production smearing (a09m310/delta_2pt_production_a09m310_stream_e_direction_x/
coulomb_smearing_boost_direction.py, md5 86169bf4) that works on any MPI process grid, also with x, y or z split over the ranks.
Every changed part is marked PYQUDA FFT; the kernel, its normalization and the boost phase are not changed. The changes:
  - sink smearing: the 3D FFTs are PyQUDA's pyquda_utils.fft.fft / ifft (fft3d=True), which give every rank whole time slices before the
    FFT, instead of cupy's FFT of the rank's own array. The FFT of the kernel is made on the whole spatial box (one small 3D array) and every
    rank takes its block of momenta;
  - source: the kernel is made on the whole spatial box, moved to the source position, and every rank keeps its block of sites;
  - coulomb_gauge_theta: the neighbour link U_i(x - i) comes from PyQUDA's shift, which brings the links at a rank boundary from the
    neighbouring rank, instead of numpy/cupy roll inside the rank's own array;
  - mean_radius (not used by the production script) is unchanged and still needs an unsplit spatial grid.
On a grid split along t only, the same numbers go through the same operations as in the production file, so the result should be bit for bit
the same. PyQUDA's FFT copies the propagator through host memory and MPI on the way, which costs time.

The production file was: a copy of 2pt_production/tools/coulomb_smearing.py with one more argument, boost_direction ("x", "y" or "z",
default "z"): the lattice axis that gets the width rho_z. The other two axes get rho_T. With "z" everything is as in the original file.
Its second change is one line in coulomb_gauge_theta, marked there.

Boosted Gaussian smearing in Coulomb gauge: use it only on a Coulomb-gauge-fixed configuration.

Source and sink are smeared with an explicit Gaussian times a plane wave, with no gauge links:

    K(d) = exp(-(dx^2 + dy^2) / rho_T^2 - dz^2 / rho_z^2) * exp(+i 2pi/L k.d),    d = x - y (nearest periodic image)

rho_T = rho_z is the isotropic smearing, rho_z = rho_T / 2 halves the radius along z.
The formula is written for boost_direction "z". For "x" or "y" the width rho_z is along x or y and rho_T along the other two axes.
k is always given on the lattice axes (x, y, z).
k is in units of 2pi/L, with the same meaning as in mom_smearing.py:
  - at rho_T = rho_z = rho this is the free-field limit of source.gaussianSmear(rho, n_steps),
    whose profile is exp(-r^2 / rho^2);
  - the sign of the phase reproduces add_momentum_phase_to_gauge(sign=-1).
Both were verified on FNAL and on Frontier (2026-09-23): the isotropic source overlaps the free-field N40 rho3.25
mom_smearing source at 0.9999 for the same k, and at 0.0761 (k = 3.6) and 0.5212 (k = 1.8) for the opposite k.
coulomb_gauge_theta() measures how well the configuration satisfies the Coulomb condition the smearing relies on.
mean_radius() measures the rms radius of a propagator (or a smeared source) around its source position.
The Gaussian part is normalized to sum 1, as the Wuppertal smearing is at k = 0.

The sink smearing is a 3D FFT on every time slice. PYQUDA FFT: PyQUDA's FFT gathers whole time slices on every rank first, so the process
grid may be split in any direction (the production file needed [1, 1, 1, Gt]).
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


# ---------------- the smearing kernel and its two uses, on plain arrays ----------------


def _one_direction(xp, L, k, rho):
    """Gaussian and boost phase along one direction, index i <-> displacement d = i for i < L/2 and d = i - L otherwise."""
    d = (xp.arange(L) + L // 2) % L - L // 2
    gaussian = xp.exp(-(d**2) / rho**2)
    phase = xp.exp(1j * BOOST_PHASE_SIGN * 2 * numpy.pi * k * d / L)
    if L % 2 == 0:
        phase[L // 2] = numpy.cos(2 * numpy.pi * k * (L // 2) / L)  # average of the images d = +L/2 and -L/2: keeps K Hermitian
    return gaussian, phase


def boosted_gaussian_kernel(xp, spatial_size, kvec, rho_T, rho_z, boost_direction="z"):
    """K[dz, dy, dx] for the displacement d = x - y. rho_z is the width along boost_direction, rho_T the width along the other two axes."""
    Lx, Ly, Lz = spatial_size
    if boost_direction == "x":
        gx, px = _one_direction(xp, Lx, kvec[0], rho_z)
        gy, py = _one_direction(xp, Ly, kvec[1], rho_T)
        gz, pz = _one_direction(xp, Lz, kvec[2], rho_T)
    if boost_direction == "y":
        gx, px = _one_direction(xp, Lx, kvec[0], rho_T)
        gy, py = _one_direction(xp, Ly, kvec[1], rho_z)
        gz, pz = _one_direction(xp, Lz, kvec[2], rho_T)
    if boost_direction == "z":
        gx, px = _one_direction(xp, Lx, kvec[0], rho_T)
        gy, py = _one_direction(xp, Ly, kvec[1], rho_T)
        gz, pz = _one_direction(xp, Lz, kvec[2], rho_z)
    gaussian = gz.reshape(Lz, 1, 1) * gy.reshape(1, Ly, 1) * gx.reshape(1, 1, Lx)
    phase = pz.reshape(Lz, 1, 1) * py.reshape(1, Ly, 1) * px.reshape(1, 1, Lx)
    return gaussian / gaussian.sum() * phase


def source_lexico(xp, local_size, offset, global_size, kvec, src_pos, rho_T, rho_z, boost_direction="z"):
    """
    Smeared source propagator [t, z, y, x, spin, spin, color, color] on this rank's sites, nonzero only on the source time slice.
    PYQUDA FFT: offset = [x, y, z, t] of this rank's first site and global_size = [GLx, GLy, GLz, GLt]; the production file had only t_offset.
    """
    Lx, Ly, Lz, Lt = local_size
    ox, oy, oz, ot = offset
    x0, y0, z0, t0 = src_pos
    data = xp.zeros((Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), "<c16")
    t_local = t0 - ot
    if 0 <= t_local < Lt:
        kernel = boosted_gaussian_kernel(xp, global_size[:3], kvec, rho_T, rho_z, boost_direction)  # PYQUDA FFT: on the whole spatial box [GLz, GLy, GLx]
        eta = xp.roll(kernel, shift=(z0, y0, x0), axis=(0, 1, 2))  # eta(x) = K(x - x_src)
        eta = eta[oz : oz + Lz, oy : oy + Ly, ox : ox + Lx]  # PYQUDA FFT: this rank's block of sites [Lz, Ly, Lx]
        spin_identity = xp.eye(Ns).reshape(Ns, Ns, 1, 1)
        color_identity = xp.eye(Nc).reshape(1, 1, Nc, Nc)
        data[t_local] = eta.reshape(Lz, Ly, Lx, 1, 1, 1, 1) * spin_identity * color_identity #source is only nonzero in the diagonal dirac-color space
    return data


def sink_smear_evenodd(latt_info, data_evenodd, kvec, rho_T, rho_z, boost_direction="z"):
    """
    prop_smeared(x) = sum_y K(x - y) prop(y) on every time slice, done as a product of 3D FFTs.
    PYQUDA FFT: the FFTs are PyQUDA's distributed 3D FFTs, run where the propagator lives (on the GPU for a cupy propagator). After fft() the
    field is in PyQUDA's even-odd layout with the momentum p = (grid_coord * size + local index) in place of the site, so the kernel's FFT is
    put in the same layout before the product (the production file multiplied in the lexicographic layout).
    """
    from pyquda.field import LatticePropagator
    from pyquda_utils.fft import fft, ifft

    xp = array_module(data_evenodd)
    Lx, Ly, Lz, Lt = latt_info.size
    gx, gy, gz, gt = latt_info.grid_coord
    kernel_ft = xp.fft.fftn(boosted_gaussian_kernel(xp, latt_info.global_size[:3], kvec, rho_T, rho_z, boost_direction))  # [GLz, GLy, GLx]: whole box
    kernel_ft = kernel_ft[gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx]  # [Lz, Ly, Lx]: this rank's momenta
    kernel_ft = xp.ascontiguousarray(xp.broadcast_to(kernel_ft, (Lt, Lz, Ly, Lx)))  # [Lt, Lz, Ly, Lx]: the same on every time slice
    kernel_ft = latt_info.evenodd(kernel_ft, False, backend="cupy")  # [2, Lt, Lz, Ly, Lx//2]: the layout of the propagator
    propagator = LatticePropagator(latt_info, data_evenodd)
    prop_ft = fft(propagator, True, propagator.location)  # fft3d=True: x, y, z only
    prop_ft.data *= kernel_ft[:, :, :, :, :, None, None, None, None]
    smeared = ifft(prop_ft, True, propagator.location)
    del prop_ft
    return smeared.data


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


# ---------------- PyQUDA wrappers used by the production script ----------------


def _check_grid(latt_info):
    if list(latt_info.grid_size[:3]) != [1, 1, 1]:
        raise RuntimeError(f"coulomb_smearing needs an unsplit spatial grid, got grid {latt_info.grid_size}")


def coulomb_boosted_source(latt_info, kvec, src_pos, rho_T, rho_z, boost_direction="z"):
    from pyquda.field import LatticePropagator

    offset = [g * L for g, L in zip(latt_info.grid_coord, latt_info.size)]  # PYQUDA FFT: [x, y, z, t] of this rank's first site; no grid check
    data = source_lexico(cupy, latt_info.size, offset, latt_info.global_size, kvec, src_pos, rho_T, rho_z, boost_direction)
    return LatticePropagator(latt_info, latt_info.evenodd(data, False, backend="cupy"))


def coulomb_boosted_sink(latt_info, propagator, kvec, rho_T, rho_z, boost_direction="z"):
    from pyquda.field import LatticePropagator

    return LatticePropagator(latt_info, sink_smear_evenodd(latt_info, propagator.data, kvec, rho_T, rho_z, boost_direction))  # PYQUDA FFT: no grid check


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
        slice_lexico = latt_info.lexico(propagator.data, False, backend="cupy")[t_local]  # [z, y, x, spin, spin, color, color]
        norm, moment = radius_moments_lexico(slice_lexico, src_pos)
        r = (moment / norm) ** 0.5
    return latt_info.mpi_comm.allreduce(r)  # r on the GPU that holds t0, 0 on the others


def coulomb_gauge_theta(latt_info, gauge):
    """
    Violation of the Coulomb gauge condition, normalized like QUDA's theta (compare with the last theta QUDA prints):
        theta = sum_x Re Tr[Delta(x) Delta(x)^dag] / (3 V),   Delta = traceless part of D - D^dag,
        D(x)  = sum_{i = x, y, z} [U_i(x - i) - U_i(x)].
    It is zero in exact Coulomb gauge and of order 1 on an unfixed configuration. The plaquette cannot show this,
    because it is gauge invariant. Only spatial links enter. PYQUDA FFT: U_i(x - i) comes from PyQUDA's shift, which brings the links at a
    rank boundary from the neighbouring rank, so space may be split.
    Every rank must call it: the sum over the sites is an allreduce.
    """
    D = 0
    for i in range(3):
        U_i = latt_info.lexico(gauge.data[i], False, backend="cupy")  # [t, z, y, x, color, color]; direction i = x, y, z lives on array axis 3, 2, 1
        U_i_back = latt_info.lexico(gauge[i].shift(1, i + 4).data, False, backend="cupy")  # PYQUDA FFT: U_i(x - i); mu = i + 4 is the backward direction i
        xp = array_module(U_i)  # CHANGED from "xp = array_module(gauge.data)" before the loop: gauge.data is a numpy array on the host, U_i a cupy array, and the xp.eye below must be of U_i's kind
        D = D + U_i_back - U_i  # U_i(x - i) - U_i(x)   (PYQUDA FFT: was xp.roll(U_i, 1, axis=3 - i))
    Delta = D - D.conj().swapaxes(-1, -2)
    Delta = Delta - (xp.trace(Delta, axis1=-2, axis2=-1) / Nc)[..., None, None] * xp.eye(Nc)
    local = float((abs(Delta) ** 2).sum().item())
    return latt_info.mpi_comm.allreduce(local) / (Nc * latt_info.global_volume)

