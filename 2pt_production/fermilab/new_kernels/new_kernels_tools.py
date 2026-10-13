#Tools of the new-kernel test, shared by the job script (on the GPUs) and kernel_radii.py (on the CPU):
#    build_kernel(xp, spatial_size, row)       the kernel of one row [name, function, parameters] of operator_list
#    kernel_moments(xp, spatial_size, kernel)  its rms radii and the mean and width of its momentum along z
#    check_kernels(latt_info, operator_list, kernels, src_pos)
#                                              checks of every kernel on the GPUs of the job, before any inversion
#cupy, mpi4py and pyquda are imported inside check_kernels only, so that kernel_radii.py runs on a laptop.
import numpy as np
import coulomb_smearing_kernels as csk


def build_kernel(xp, spatial_size, row):
    """K[z, y, x] of one row [name, function, parameters] of operator_list: function of coulomb_smearing_kernels called with xp, spatial_size and the parameters."""
    name, function, parameters = row
    return getattr(csk, function)(xp, spatial_size, **parameters)


def kernel_moments(xp, spatial_size, kernel):
    """
    rms_T, rms_z: rms transverse and longitudinal radii of |K|^2, sqrt(sum s |K|^2 / sum |K|^2) and sqrt(sum dz^2 |K|^2 / sum |K|^2);
    mean_k, width_k: mean and width of the momentum intensity along z, |sum_dz exp(-i 2pi/Lz k dz) K|^2 summed over the transverse plane,
    k in units of 2pi/Lz (the boost of gaussian_kernel shows up as mean_k = k0). The lattice momenta are periodic, so the mean is the
    circular mean and the width is taken over the Lz momenta closest to it (a plain mean over -Lz/2 .. Lz/2 - 1 would pull the mean of a
    broad kernel toward 0).
    """
    Lz = spatial_size[2]
    weight = abs(kernel) ** 2
    s = csk.transverse_s(xp, spatial_size)
    dz = csk.z_displacement(xp, spatial_size)
    rms_T = float(xp.sqrt((weight * s).sum() / weight.sum()))
    rms_z = float(xp.sqrt((weight * dz**2).sum() / weight.sum()))
    intensity = (abs(xp.fft.fft(kernel, axis=0)) ** 2).sum(axis=(1, 2))
    k = xp.fft.fftfreq(Lz) * Lz
    mean_k = float(xp.angle((intensity * xp.exp(2j * np.pi * k / Lz)).sum())) * Lz / (2 * np.pi)
    k_near = (k - mean_k + Lz / 2) % Lz - Lz / 2 + mean_k  # every momentum moved by a multiple of Lz to within Lz/2 of mean_k
    width_k = float(xp.sqrt((intensity * (k_near - mean_k) ** 2).sum() / intensity.sum()))
    return rms_T, rms_z, mean_k, width_k


def relative_difference(latt_info, a, b):
    """max |a - b| / max |b| over the whole lattice (every rank must call it)."""
    from mpi4py import MPI

    diff = latt_info.mpi_comm.allreduce(float(abs(a - b).max()), op=MPI.MAX)
    scale = latt_info.mpi_comm.allreduce(float(abs(b).max()), op=MPI.MAX)
    return diff / scale


def check_kernels(latt_info, operator_list, kernels, src_pos):
    """
    Checks of the kernels built on the GPUs, before any inversion. Every rank must call it (mean_radius and the maxima are allreduce).
    For every operator:
      1. the kernel has the shape [Lz, Ly, Lx], is complex128 and finite;
      2. the same kernel built with numpy on the CPU agrees (relative 1e-12);
      3. K(-d) = K(d)^* (Eq. (41) of the note), which makes the sink smearing the adjoint of the source smearing (relative 1e-12);
      4. the source coulomb_kernel_source(K) at src_pos has the radius (mean_radius) of |K|^2 computed from the kernel itself
         (relative 1e-10): this checks where the source sits on the lattice and on the rank that holds t_src.
    For every gaussian_kernel operator, also:
      5. coulomb_kernel_source with K and K^* equals coulomb_boosted_source with +k and -k, and coulomb_kernel_sink equals
         coulomb_boosted_sink on that source (relative 1e-12): the new functions give the correlators of the smearing tuning.
    A failed check raises RuntimeError on every rank (python -m mpi4py then stops the job). Returns one row
    [rms_T, rms_z, source radius, mean_k, width_k] per operator, for the .h5 files.
    """
    import cupy as cp
    from pyquda_utils import core

    spatial_size = list(latt_info.size[:3])
    Lx, Ly, Lz = spatial_size
    moments = []
    for row, kernel in zip(operator_list, kernels):
        name = row[0]
        if tuple(kernel.shape) != (Lz, Ly, Lx) or kernel.dtype != cp.complex128 or not bool(cp.isfinite(kernel).all()):
            raise RuntimeError(f"kernel {name}: shape {kernel.shape} and dtype {kernel.dtype}, or not finite")
        scale = float(abs(kernel).max())
        cpu_gpu = float(abs(kernel - cp.asarray(build_kernel(np, spatial_size, row))).max()) / scale
        negated = cp.roll(kernel[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))  # K(-d): index i -> (-i) mod L
        hermitian = float(abs(negated - kernel.conj()).max()) / scale
        rms_T, rms_z, mean_k, width_k = kernel_moments(cp, spatial_size, kernel)
        r_kernel = (rms_T**2 + rms_z**2) ** 0.5
        source = csk.coulomb_kernel_source(latt_info, kernel, src_pos)
        r_source = csk.mean_radius(latt_info, source, src_pos)
        del source
        core.getLogger().info(f"KERNEL CHECK {name}: cpu-gpu {cpu_gpu:.1e}, K(-d) - K(d)^* {hermitian:.1e}, rms_T {rms_T:.4f}, rms_z {rms_z:.4f}, "
                              f"source radius {r_source:.6f} (from the kernel {r_kernel:.6f}), momentum {mean_k:.3f} +- {width_k:.3f}")
        if cpu_gpu > 1e-12 or hermitian > 1e-12 or abs(r_source - r_kernel) > 1e-10 * r_kernel:
            raise RuntimeError(f"kernel {name} failed: cpu-gpu {cpu_gpu:.1e}, K(-d) - K(d)^* {hermitian:.1e}, radius {r_source} vs {r_kernel}")
        if row[1] == "gaussian_kernel":
            check_against_boosted_gaussian(latt_info, row, kernel, src_pos)
        moments.append([rms_T, rms_z, r_source, mean_k, width_k])
    return moments


def check_against_boosted_gaussian(latt_info, row, kernel, src_pos):
    """Check 5 of check_kernels: the new source and sink functions with gaussian_kernel against coulomb_boosted_source / sink."""
    from pyquda_utils import core

    name, function, parameters = row
    rho_T, rho_z = parameters["rho_T"], parameters["rho_z"]
    worst = 0.0
    for kvec, K in (([0, 0, parameters["k0"]], kernel), ([0, 0, -parameters["k0"]], kernel.conj())):
        new_source = csk.coulomb_kernel_source(latt_info, K, src_pos)
        old_source = csk.coulomb_boosted_source(latt_info, kvec, src_pos, rho_T, rho_z)
        worst = max(worst, relative_difference(latt_info, new_source.data, old_source.data))
        new_sink = csk.coulomb_kernel_sink(latt_info, new_source, K)
        old_sink = csk.coulomb_boosted_sink(latt_info, old_source, kvec, rho_T, rho_z)
        worst = max(worst, relative_difference(latt_info, new_sink.data, old_sink.data))
        del new_source, old_source, new_sink, old_sink
    core.getLogger().info(f"KERNEL CHECK {name}: new source and sink vs coulomb_boosted_source / sink at +-k: {worst:.1e}")
    if worst > 1e-12:
        raise RuntimeError(f"kernel {name}: the new functions differ from coulomb_boosted_source / sink by {worst:.1e}")
