"""
Tests of coulomb_smearing_kernels.py on the CPU, with numpy and scipy only (no GPU, no PyQUDA): run with
    python3 test_coulomb_smearing_kernels.py
from this folder (2pt_production/fermilab/new_kernels); the old package it is compared with is 2pt_production/tools/coulomb_smearing.py.
They check correctness only: the old API against a copy of coulomb_smearing.py, the lattice symmetries of every new kernel,
the limiting cases of the note (including Eq. (29)), an independent re-implementation of every kernel formula (site by site,
or by scipy quadrature for beta_kernel and gpd_kernel), and the analytic momentum moments of Eqs. (7), (14), (19), (22), (24).
"""

import importlib.util
import math
import os
import sys
import types

import numpy
from scipy import integrate, special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coulomb_smearing_kernels as cs  # noqa: E402

spec = importlib.util.spec_from_file_location("coulomb_smearing_original", os.path.join(HERE, "..", "..", "tools", "coulomb_smearing.py"))  # the old package
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

rng = numpy.random.default_rng(12345)
SIZES = [[6, 8, 10], [8, 8, 9], [5, 7, 12]]  # [Lx, Ly, Lz]: non-cubic, odd and even


# ---------------- helpers ----------------


def negate_index(K):
    """K[-d]: index i -> (-i) mod L along every axis."""
    return numpy.roll(K[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))


def assert_close(a, b, tol, what):
    err = float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))
    scale = max(float(numpy.max(numpy.abs(b))), 1e-300)
    if err > tol * scale:
        raise AssertionError(f"{what}: max |diff| = {err:.3e} > {tol:.1e} * {scale:.3e}")


def all_kernels(size):
    """(name, kernel, kernel with every momentum reversed) for one parameter set of every new kernel."""
    xp = numpy
    out = []
    out.append(("gaussian", cs.gaussian_kernel(xp, size, 1.7, 2.1, 1.3), cs.gaussian_kernel(xp, size, -1.7, 2.1, 1.3)))
    out.append(("two_scale", cs.two_scale_kernel(xp, size, 1.3, 1.0, 1.2, 0.8, 0.4, 3.0, 2.0), cs.two_scale_kernel(xp, size, -1.3, 1.0, 1.2, 0.8, 0.4, 3.0, 2.0)))
    out.append(("multi_momentum", cs.multi_momentum_kernel(xp, size, [0.7, 1.9, 2.6], 2.0, 1.5, [0.2, 0.5, 0.3]),
                cs.multi_momentum_kernel(xp, size, [-0.7, -1.9, -2.6], 2.0, 1.5, [0.2, 0.5, 0.3])))
    out.append(("two_momentum", cs.two_momentum_kernel(xp, size, 2.2, 2.0, 1.5, 0.9), cs.two_momentum_kernel(xp, size, -2.2, 2.0, 1.5, 0.9)))
    out.append(("superposition", cs.momentum_superposition_kernel(xp, size, 1.1, 2.0, 1.5, [-0.5, 0.0, 0.5], [1.0, 2.0, 1.0]),
                cs.momentum_superposition_kernel(xp, size, -1.1, 2.0, 1.5, [0.5, 0.0, -0.5], [1.0, 2.0, 1.0])))
    out.append(("model_M", cs.model_M_kernel(xp, size, 2.3, 2.0, 1.4, 1.1, 1.5), cs.model_M_kernel(xp, size, -2.3, 2.0, 1.4, -1.1, 1.5)))
    out.append(("model_M_power", cs.model_M_kernel(xp, size, 2.3, 2.0, 1.4, 1.1, 1.5, power=1.5),
                cs.model_M_kernel(xp, size, -2.3, 2.0, 1.4, -1.1, 1.5, power=1.5)))
    out.append(("model_V", cs.model_V_kernel(xp, size, 2.3, 2.0, 1.4, 0.8, 1.5), cs.model_V_kernel(xp, size, -2.3, 2.0, 1.4, 0.8, 1.5)))
    out.append(("model_V_opposite", cs.model_V_kernel(xp, size, 2.3, 2.0, 1.4, 0.8, 1.5, opposite=True),
                cs.model_V_kernel(xp, size, -2.3, 2.0, 1.4, 0.8, 1.5, opposite=True)))
    out.append(("model_W", cs.model_W_kernel(xp, size, 2.3, 2.0, 1.4, 0.6, 1.5), cs.model_W_kernel(xp, size, -2.3, 2.0, 1.4, 0.6, 1.5)))
    out.append(("model_WG", cs.model_WG_kernel(xp, size, 2.3, 2.0, 1.4, 0.3, 1.5), cs.model_WG_kernel(xp, size, -2.3, 2.0, 1.4, 0.3, 1.5)))
    out.append(("model_MV", cs.model_MV_kernel(xp, size, 2.3, 2.0, 1.4, 1.1, 1.5, 0.7, 2.5),
                cs.model_MV_kernel(xp, size, -2.3, 2.0, 1.4, -1.1, 1.5, 0.7, 2.5)))
    s = cs.transverse_s(xp, size)
    x_list = cs.midpoint_nodes(16)
    w_list = [numpy.exp(-((x - 0.6) ** 2) / 0.02) * (1 + s / 10.0) ** (-x) for x in x_list]
    out.append(("conditional", cs.conditional_kernel(xp, size, 4.0, 2.0, 1.4, x_list, w_list),
                cs.conditional_kernel(xp, size, -4.0, 2.0, 1.4, x_list, w_list)))
    out.append(("beta", cs.beta_kernel(xp, size, 4.0, 2.0, 1.4, 2.0, 1.5, 2.5, -1.0, 1.5, n_x=40),
                cs.beta_kernel(xp, size, -4.0, 2.0, 1.4, 2.0, 1.5, 2.5, -1.0, 1.5, n_x=40)))
    out.append(("gpd", cs.gpd_kernel(xp, size, 4.0, 2.0, 1.4, 3.0, -0.3, 1.5, n_x=40), cs.gpd_kernel(xp, size, -4.0, 2.0, 1.4, 3.0, -0.3, 1.5, n_x=40)))
    out.append(("stretched", cs.stretched_kernel(xp, size, 1.6, 2.0, 1.4, 1.3, 2.7), cs.stretched_kernel(xp, size, -1.6, 2.0, 1.4, 1.3, 2.7)))
    out.append(("exponential", cs.exponential_kernel(xp, size, 1.6, 1.8), cs.exponential_kernel(xp, size, -1.6, 1.8)))
    out.append(("power_tail", cs.power_tail_kernel(xp, size, 1.6, 1.5, 1.2, 2.0), cs.power_tail_kernel(xp, size, -1.6, 1.5, 1.2, 2.0)))
    out.append(("polynomial", cs.polynomial_kernel(xp, size, 1.6, 2.0, 1.4, 0.4, -0.3), cs.polynomial_kernel(xp, size, -1.6, 2.0, 1.4, 0.4, -0.3)))
    k_a = cs.gaussian_kernel(xp, size, 1.0, 2.0, 1.4)
    k_b = cs.model_V_kernel(xp, size, 2.0, 1.5, 1.0, 0.5, 2.0)
    k_a_neg = cs.gaussian_kernel(xp, size, -1.0, 2.0, 1.4)
    k_b_neg = cs.model_V_kernel(xp, size, -2.0, 1.5, 1.0, 0.5, 2.0)
    out.append(("linear_combination", cs.linear_combination_kernel([k_a, k_b], [0.7, -0.4]), cs.linear_combination_kernel([k_a_neg, k_b_neg], [0.7, -0.4])))
    return out


# ---------------- old API: identical to coulomb_smearing.py ----------------


def test_old_array_functions_identical():
    for Lx, Ly, Lz in SIZES:
        if Lx % 2 == 1:
            continue  # the even-odd layout needs an even Lx
        for kvec in ([0, 0, 2.4], [0.5, -1.0, 3.6], [0, 0, 0]):
            for rho_T, rho_z in ((3.25, 3.25), (2.0, 1.0)):
                K_new = cs.boosted_gaussian_kernel(numpy, [Lx, Ly, Lz], kvec, rho_T, rho_z)
                K_old = old.boosted_gaussian_kernel(numpy, [Lx, Ly, Lz], kvec, rho_T, rho_z)
                assert numpy.array_equal(K_new, K_old)
                for t_offset, src in ((0, [1, 2, 3, 1]), (4, [0, 5, 7, 6]), (4, [0, 5, 7, 1])):
                    a = cs.source_lexico(numpy, [Lx, Ly, Lz, 4], t_offset, kvec, src, rho_T, rho_z)
                    b = old.source_lexico(numpy, [Lx, Ly, Lz, 4], t_offset, kvec, src, rho_T, rho_z)
                    assert numpy.array_equal(a, b)
                prop = rng.normal(size=(2, 2, Lz, Ly, Lx // 2, 4, 4, 3, 3)) + 1j * rng.normal(size=(2, 2, Lz, Ly, Lx // 2, 4, 4, 3, 3))
                a = cs.sink_smear_evenodd(prop, [Lx, Ly, Lz], kvec, rho_T, rho_z)
                b = old.sink_smear_evenodd(prop, [Lx, Ly, Lz], kvec, rho_T, rho_z)
                assert numpy.array_equal(a, b)
                assert numpy.array_equal(cs.to_lexico(prop), old.to_lexico(prop))
                lex = cs.to_lexico(prop)
                assert numpy.array_equal(cs.to_evenodd(lex), old.to_evenodd(lex))
                assert numpy.array_equal(cs.to_evenodd(lex), prop)
                assert cs.radius_moments_lexico(lex[0], [1, 2, 3, 0]) == old.radius_moments_lexico(lex[0], [1, 2, 3, 0])
    for name in ("Ns", "Nc", "BOOST_PHASE_SIGN", "coulomb_boosted_source", "coulomb_boosted_sink", "coulomb_gauge_theta", "mean_radius",
                 "squared_distance", "radius_moments_lexico", "to_lexico", "to_evenodd", "array_module", "_check_grid", "_one_direction"):
        assert hasattr(cs, name), name


# ---------------- PyQUDA wrappers, with a mock PyQUDA and numpy in place of cupy ----------------


class FakePropagator:
    def __init__(self, latt_info, data):
        self.latt_info = latt_info
        self.data = data


class FakeComm:
    def allreduce(self, value):
        return value


class FakeLattInfo:
    def __init__(self, size, gt, Lt_global):
        self.size = size  # local [Lx, Ly, Lz, Lt]
        self.Lt = size[3]
        self.gt = gt
        self.grid_size = [1, 1, 1, Lt_global // size[3]]
        self.mpi_comm = FakeComm()
        self.global_volume = size[0] * size[1] * size[2] * Lt_global


class NumpyAsCupy:
    """numpy dressed as cupy: the functions the module calls on the cupy namespace."""

    def __getattr__(self, name):
        return getattr(numpy, name)

    def get_array_module(self, a):
        return numpy


def install_mocks():
    pyquda = types.ModuleType("pyquda")
    field = types.ModuleType("pyquda.field")
    field.LatticePropagator = FakePropagator
    pyquda.field = field
    sys.modules["pyquda"] = pyquda
    sys.modules["pyquda.field"] = field
    cs.cupy = NumpyAsCupy()
    old.cupy = NumpyAsCupy()


def test_pyquda_wrappers():
    install_mocks()
    Lx, Ly, Lz, Lt = 6, 8, 10, 4
    for gt, t0 in ((0, 2), (1, 5), (1, 2)):
        latt = FakeLattInfo([Lx, Ly, Lz, Lt], gt, 8)
        src = [1, 6, 9, t0]
        for kvec in ([0, 0, 2.4], [1, 0, -3.6]):
            a = cs.coulomb_boosted_source(latt, kvec, src, 2.5, 1.25)
            b = old.coulomb_boosted_source(latt, kvec, src, 2.5, 1.25)
            assert numpy.array_equal(a.data, b.data)
            prop = rng.normal(size=(2, Lt, Lz, Ly, Lx // 2, 4, 4, 3, 3)) + 1j * rng.normal(size=(2, Lt, Lz, Ly, Lx // 2, 4, 4, 3, 3))
            a = cs.coulomb_boosted_sink(latt, FakePropagator(latt, prop), kvec, 2.5, 1.25)
            b = old.coulomb_boosted_sink(latt, FakePropagator(latt, prop), kvec, 2.5, 1.25)
            assert numpy.array_equal(a.data, b.data)
        # the generic wrappers with gaussian_kernel reproduce the old wrappers for k along z
        kernel = cs.gaussian_kernel(numpy, [Lx, Ly, Lz], 2.4, 2.5, 1.25)
        a = cs.coulomb_kernel_source(latt, kernel, src)
        b = old.coulomb_boosted_source(latt, [0, 0, 2.4], src, 2.5, 1.25)
        assert_close(a.data, b.data, 1e-14, "kernel source vs boosted source")
        a = cs.coulomb_kernel_source(latt, kernel.conj(), src)
        b = old.coulomb_boosted_source(latt, [0, 0, -2.4], src, 2.5, 1.25)
        assert_close(a.data, b.data, 1e-14, "conjugate kernel source vs boosted source at -k")
        prop = rng.normal(size=(2, Lt, Lz, Ly, Lx // 2, 4, 4, 3, 3)) + 1j * rng.normal(size=(2, Lt, Lz, Ly, Lx // 2, 4, 4, 3, 3))
        a = cs.coulomb_kernel_sink(latt, FakePropagator(latt, prop), kernel)
        b = old.coulomb_boosted_sink(latt, FakePropagator(latt, prop), [0, 0, 2.4], 2.5, 1.25)
        assert_close(a.data, b.data, 1e-13, "kernel sink vs boosted sink")
        src_prop = cs.coulomb_kernel_source(latt, kernel, src)
        assert cs.mean_radius(latt, src_prop, src) == old.mean_radius(latt, src_prop, src)
    # a wrong kernel shape is refused
    latt = FakeLattInfo([Lx, Ly, Lz, Lt], 0, 8)
    try:
        cs.coulomb_kernel_source(latt, numpy.zeros((Lx, Ly, Lz)), [0, 0, 0, 0])
        raise AssertionError("a kernel with the wrong shape was accepted")
    except ValueError:
        pass
    # coulomb_gauge_theta on random links, against the original
    gauge = types.SimpleNamespace(data=rng.normal(size=(4, 2, Lt, Lz, Ly, Lx // 2, 3, 3)) + 1j * rng.normal(size=(4, 2, Lt, Lz, Ly, Lx // 2, 3, 3)))
    assert cs.coulomb_gauge_theta(latt, gauge) == old.coulomb_gauge_theta(latt, gauge)


def test_source_and_sink_agree_for_any_kernel():
    """Sink smearing of a point propagator gives the smeared source, and the sink equals a direct convolution sum."""
    Lx, Ly, Lz, Lt = 6, 4, 5, 2
    kernel = rng.normal(size=(Lz, Ly, Lx)) + 1j * rng.normal(size=(Lz, Ly, Lx))
    src = [2, 3, 1, 1]
    point = numpy.zeros((Lt, Lz, Ly, Lx, 4, 4, 3, 3), complex)
    point[1, 1, 3, 2] = numpy.eye(4).reshape(4, 4, 1, 1) * numpy.eye(3).reshape(1, 1, 3, 3)
    smeared = cs.to_lexico(cs.kernel_sink_evenodd(cs.to_evenodd(point), kernel))
    source = cs.kernel_source_lexico(numpy, [Lx, Ly, Lz, Lt], 0, kernel, src)
    assert_close(smeared, source, 1e-13, "sink of a point source vs source")
    prop = rng.normal(size=(Lt, Lz, Ly, Lx, 4, 4, 3, 3)) + 1j * rng.normal(size=(Lt, Lz, Ly, Lx, 4, 4, 3, 3))
    smeared = cs.to_lexico(cs.kernel_sink_evenodd(cs.to_evenodd(prop), kernel))
    direct = numpy.zeros_like(prop)
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                for zz in range(Lz):
                    for yy in range(Ly):
                        for xx in range(Lx):
                            direct[:, z, y, x] += kernel[(z - zz) % Lz, (y - yy) % Ly, (x - xx) % Lx] * prop[:, zz, yy, xx]
    assert_close(smeared, direct, 1e-12, "FFT sink vs direct convolution")


# ---------------- symmetries of every new kernel ----------------


def test_hermitian_and_momentum_reversal():
    for size in SIZES:
        Lx, Ly, Lz = size
        for name, K, K_reversed in all_kernels(size):
            assert K.shape == (Lz, Ly, Lx), name
            assert K.dtype == numpy.complex128, name
            assert numpy.all(numpy.isfinite(K)), name
            assert_close(negate_index(K), K.conj(), 1e-14, f"{name} {size}: K(-d) = K(d)^*")
            assert_close(K_reversed, K.conj(), 1e-14, f"{name} {size}: reversed momenta = conjugate")


def test_envelope_normalization():
    """The real envelope sums to 1: at k = 0 and no deformation the kernel sums to 1, and chi(0 | s) = 1 keeps K(dz = 0) = G_hat."""
    size = [6, 8, 10]
    G_hat = cs.normalized(cs.gaussian_envelope(numpy, size, 2.0, 1.4))
    assert abs(cs.gaussian_kernel(numpy, size, 0.0, 2.0, 1.4).sum() - 1) < 1e-14
    for name, K, _ in all_kernels(size):
        if name in ("gaussian", "multi_momentum", "two_momentum", "superposition", "model_M", "model_M_power", "model_V", "model_V_opposite",
                    "model_WG", "model_MV", "conditional", "beta", "gpd"):
            rho_T, rho_z = (2.1, 1.3) if name == "gaussian" else ((2.0, 1.5) if name in ("multi_momentum", "two_momentum", "superposition") else (2.0, 1.4))
            G_ref = cs.normalized(cs.gaussian_envelope(numpy, size, rho_T, rho_z))
            assert_close(K[0], G_ref[0], 1e-14, f"{name}: K at dz = 0 is the normalized Gaussian")
    for K in (cs.model_W_kernel(numpy, size, 0.0, 2.0, 1.4, 0.6, 1.5), cs.stretched_kernel(numpy, size, 0.0, 2.0, 1.4, 1.3, 2.7),
              cs.exponential_kernel(numpy, size, 0.0, 1.8), cs.power_tail_kernel(numpy, size, 0.0, 1.5, 1.2, 2.0),
              cs.two_scale_kernel(numpy, size, 0.0, 1.0, 1.2, 0.8, 0.4, 3.0, 2.0)):
        assert abs(K.sum() - 1) < 1e-14
    assert_close(G_hat.sum(), 1.0, 1e-14, "normalized")


# ---------------- limiting cases named in the note ----------------


def test_limits_give_the_gaussian():
    for size in SIZES:
        xp = numpy
        G = cs.gaussian_kernel(xp, size, 2.3, 2.0, 1.4)
        assert_close(G, cs.boosted_gaussian_kernel(xp, size, [0, 0, 2.3], 2.0, 1.4), 1e-14, "gaussian vs boosted_gaussian")
        assert_close(cs.model_M_kernel(xp, size, 2.3, 2.0, 1.4, 0.0, 1.5), G, 1e-14, "M at delta_mu = 0")
        assert_close(cs.model_V_kernel(xp, size, 2.3, 2.0, 1.4, 0.0, 1.5), G, 1e-14, "V at delta0 = 0")
        assert_close(cs.model_W_kernel(xp, size, 2.3, 2.0, 1.4, 0.0, 1.5), G, 1e-14, "W at alpha = 0")
        assert_close(cs.model_WG_kernel(xp, size, 2.3, 2.0, 1.4, 0.0, 1.5), G, 1e-14, "WG at v0 = 0")
        assert_close(cs.model_MV_kernel(xp, size, 2.3, 2.0, 1.4, 0.0, 1.5, 0.0, 2.0), G, 1e-14, "MV at 0, 0")
        assert_close(cs.two_momentum_kernel(xp, size, 2.3, 2.0, 1.4, 0.0), G, 1e-14, "2k at delta = 0")
        assert_close(cs.multi_momentum_kernel(xp, size, [2.3], 2.0, 1.4, [3.0]), G, 1e-14, "MM with one packet")
        assert_close(cs.momentum_superposition_kernel(xp, size, 2.3, 2.0, 1.4, [0.0], [1.0]), G, 1e-14, "super with one node")
        assert_close(cs.stretched_kernel(xp, size, 2.3, 2.0, 1.4, 2.0, 2.0), G, 1e-14, "stretched at nu = 2")
        assert_close(cs.polynomial_kernel(xp, size, 2.3, 2.0, 1.4, 0.0, 0.0), G, 1e-14, "poly at c = 0")
        assert_close(cs.two_scale_kernel(xp, size, 2.3, 1.0, 2.0, 1.4, 0.0, 5.0, 3.0), G, 1e-14, "2G with B = 0")
        assert_close(cs.two_scale_kernel(xp, size, 2.3, 0.3, 2.0, 1.4, 0.7, 2.0, 1.4), G, 1e-14, "2G with equal radii")
        assert_close(cs.conditional_kernel(xp, size, 4.6, 2.0, 1.4, [0.5], [1.0]), G, 1e-14, "cond with one node at x = 1/2")
        x_list = cs.midpoint_nodes(10)
        w_list = [0.0] * 10
        w_list[7] = 1.0
        assert_close(cs.conditional_kernel(xp, size, 2.3 / x_list[7], 2.0, 1.4, x_list, w_list), G, 1e-13, "cond with a delta weight (Eq. (29))")
        assert_close(cs.linear_combination_kernel([G], [1.0]), G, 0, "linear combination of one kernel")
        # Model MV contains M and V
        M = cs.model_M_kernel(xp, size, 2.3, 2.0, 1.4, 1.1, 1.5)
        V = cs.model_V_kernel(xp, size, 2.3, 2.0, 1.4, 0.8, 1.5)
        assert_close(cs.model_MV_kernel(xp, size, 2.3, 2.0, 1.4, 1.1, 1.5, 0.0, 1.5), M, 1e-14, "MV at delta0 = 0 is M")
        assert_close(cs.model_MV_kernel(xp, size, 2.3, 2.0, 1.4, 0.0, 1.5, 0.8, 1.5), V, 1e-14, "MV at delta_mu = 0 is V")
        # Model V with constant f is the two-momentum kernel: R_c -> infinity
        assert_close(cs.model_V_kernel(xp, size, 2.3, 2.0, 1.4, 0.8, 1e9), cs.two_momentum_kernel(xp, size, 2.3, 2.0, 1.4, 0.8), 1e-12, "V with f = 1")
        # Eq. (29): the conditional kernel with delta weights at s-dependent nodes gives Model M and Model V (P_z = 1: the node is the momentum)
        s = cs.transverse_s(xp, size)
        f = numpy.exp(-s / (2 * 1.5**2))
        w = numpy.exp(-2 * s / 2.0**2)
        mu = 2.3 + 1.1 * (f - (w * f).sum() / w.sum())
        assert_close(cs.conditional_kernel(xp, size, 1.0, 2.0, 1.4, [mu], [1.0]), M, 1e-13, "Eq. (29): delta(k - mu(s)) is Model M")
        assert_close(cs.conditional_kernel(xp, size, 1.0, 2.0, 1.4, [2.3 + 0.8 * f, 2.3 - 0.8 * f], [0.5, 0.5]), V, 1e-13,
                     "Eq. (29): two deltas at k0 +- delta(s) is Model V")


# ---------------- independent scalar re-implementation of every formula ----------------


def nearest(i, L):
    return (i + L // 2) % L - L // 2


def lattice_from_formula(size, envelope_fn, factor_fn, normalize_envelope=True):
    """
    Build K[z, y, x] site by site from scalar formulas: envelope_fn(s, dz) real, factor_fn(s, dz) complex (the momentum part),
    averaging factor_fn over dz = -Lz/2 and +Lz/2 at the half-box index, and dividing the envelope by its sum.
    """
    Lx, Ly, Lz = size
    env = numpy.zeros((Lz, Ly, Lx))
    fac = numpy.zeros((Lz, Ly, Lx), complex)
    for iz in range(Lz):
        dz = nearest(iz, Lz)
        for iy in range(Ly):
            for ix in range(Lx):
                s = nearest(ix, Lx) ** 2 + nearest(iy, Ly) ** 2
                env[iz, iy, ix] = envelope_fn(s, dz)
                if Lz % 2 == 0 and iz == Lz // 2:
                    fac[iz, iy, ix] = 0.5 * (factor_fn(s, -Lz // 2) + factor_fn(s, Lz // 2))
                else:
                    fac[iz, iy, ix] = factor_fn(s, dz)
    if normalize_envelope:
        env = env / env.sum()
    return env * fac


def test_formulas_site_by_site():
    for size in ([6, 8, 10], [5, 7, 9]):
        Lz = size[2]
        two_pi = 2 * math.pi / Lz

        def gauss(rho_T, rho_z):
            return lambda s, dz: math.exp(-s / rho_T**2 - dz**2 / rho_z**2)

        def phase(k):
            return lambda s, dz: complex(math.cos(two_pi * k * dz), math.sin(two_pi * k * dz))

        def f(s, R_c, power=None):
            return math.exp(-s / (2 * R_c**2)) if power is None else (1 + s / R_c**2) ** (-power)

        # f_bar by an explicit transverse sum
        def f_bar(R_c, rho_T, power=None):
            num = den = 0.0
            for iy in range(size[1]):
                for ix in range(size[0]):
                    s = nearest(ix, size[0]) ** 2 + nearest(iy, size[1]) ** 2
                    num += math.exp(-2 * s / rho_T**2) * f(s, R_c, power)
                    den += math.exp(-2 * s / rho_T**2)
            return num / den

        checks = []
        checks.append(("gaussian", cs.gaussian_kernel(numpy, size, 1.7, 2.1, 1.3), lattice_from_formula(size, gauss(2.1, 1.3), phase(1.7))))
        checks.append(("two_scale", cs.two_scale_kernel(numpy, size, 1.3, 1.0, 1.2, 0.8, 0.4, 3.0, 2.0),
                       lattice_from_formula(size, lambda s, dz: 1.0 * gauss(1.2, 0.8)(s, dz) + 0.4 * gauss(3.0, 2.0)(s, dz), phase(1.3))))
        checks.append(("multi_momentum", cs.multi_momentum_kernel(numpy, size, [0.7, 1.9], 2.0, 1.5, [1.0, 3.0]),
                       lattice_from_formula(size, gauss(2.0, 1.5), lambda s, dz: 0.25 * phase(0.7)(s, dz) + 0.75 * phase(1.9)(s, dz))))
        checks.append(("two_momentum", cs.two_momentum_kernel(numpy, size, 2.2, 2.0, 1.5, 0.9),
                       lattice_from_formula(size, gauss(2.0, 1.5), lambda s, dz: phase(2.2)(s, dz) * math.cos(two_pi * 0.9 * dz))))
        fb = f_bar(1.5, 2.0)
        checks.append(("model_M", cs.model_M_kernel(numpy, size, 2.3, 2.0, 1.4, 1.1, 1.5),
                       lattice_from_formula(size, gauss(2.0, 1.4), lambda s, dz: phase(2.3 + 1.1 * (f(s, 1.5) - fb))(s, dz))))
        checks.append(("model_M_nosub", cs.model_M_kernel(numpy, size, 2.3, 2.0, 1.4, 1.1, 1.5, subtract_mean=False),
                       lattice_from_formula(size, gauss(2.0, 1.4), lambda s, dz: phase(2.3 + 1.1 * f(s, 1.5))(s, dz))))
        fbp = f_bar(1.5, 2.0, 2.0)
        checks.append(("model_M_power", cs.model_M_kernel(numpy, size, 2.3, 2.0, 1.4, 1.1, 1.5, power=2.0),
                       lattice_from_formula(size, gauss(2.0, 1.4), lambda s, dz: phase(2.3 + 1.1 * (f(s, 1.5, 2.0) - fbp))(s, dz))))
        checks.append(("model_V", cs.model_V_kernel(numpy, size, 2.3, 2.0, 1.4, 0.8, 1.5),
                       lattice_from_formula(size, gauss(2.0, 1.4), lambda s, dz: phase(2.3)(s, dz) * math.cos(two_pi * 0.8 * f(s, 1.5) * dz))))
        checks.append(("model_V_opposite", cs.model_V_kernel(numpy, size, 2.3, 2.0, 1.4, 0.8, 1.5, opposite=True),
                       lattice_from_formula(size, gauss(2.0, 1.4), lambda s, dz: phase(2.3)(s, dz) * math.cos(two_pi * 0.8 * (1 - f(s, 1.5)) * dz))))
        checks.append(("model_W", cs.model_W_kernel(numpy, size, 2.3, 2.0, 1.4, 0.6, 1.5),
                       lattice_from_formula(size, lambda s, dz: math.exp(-s / 4.0 - dz**2 / (1.4 * (1 + 0.6 * f(s, 1.5))) ** 2), phase(2.3))))
        # WG: only G is normalized, chi multiplies afterwards
        G_sum = sum(gauss(2.0, 1.4)(nearest(ix, size[0]) ** 2 + nearest(iy, size[1]) ** 2, nearest(iz, Lz))
                    for iz in range(Lz) for iy in range(size[1]) for ix in range(size[0]))
        checks.append(("model_WG", cs.model_WG_kernel(numpy, size, 2.3, 2.0, 1.4, 0.3, 1.5),
                       lattice_from_formula(size, lambda s, dz: gauss(2.0, 1.4)(s, dz) * math.exp(-0.3 * f(s, 1.5) * dz**2 / 2) / G_sum,
                                            phase(2.3), normalize_envelope=False)))
        checks.append(("model_MV", cs.model_MV_kernel(numpy, size, 2.3, 2.0, 1.4, 1.1, 1.5, 0.7, 2.5),
                       lattice_from_formula(size, gauss(2.0, 1.4),
                                            lambda s, dz: phase(2.3 + 1.1 * (f(s, 1.5) - fb))(s, dz) * math.cos(two_pi * 0.7 * f(s, 2.5) * dz))))
        checks.append(("model_V_power", cs.model_V_kernel(numpy, size, 2.3, 2.0, 1.4, 0.8, 1.5, power=2.0),
                       lattice_from_formula(size, gauss(2.0, 1.4), lambda s, dz: phase(2.3)(s, dz) * math.cos(two_pi * 0.8 * f(s, 1.5, 2.0) * dz))))
        checks.append(("model_W_power", cs.model_W_kernel(numpy, size, 2.3, 2.0, 1.4, -0.4, 1.5, power=1.0),
                       lattice_from_formula(size, lambda s, dz: math.exp(-s / 4.0 - dz**2 / (1.4 * (1 - 0.4 * f(s, 1.5, 1.0))) ** 2), phase(2.3))))
        checks.append(("model_WG_power", cs.model_WG_kernel(numpy, size, 2.3, 2.0, 1.4, 0.3, 1.5, power=1.0),
                       lattice_from_formula(size, lambda s, dz: gauss(2.0, 1.4)(s, dz) * math.exp(-0.3 * f(s, 1.5, 1.0) * dz**2 / 2) / G_sum,
                                            phase(2.3), normalize_envelope=False)))
        fbp2 = f_bar(1.5, 2.0, 1.0)
        checks.append(("model_MV_power_nosub", cs.model_MV_kernel(numpy, size, 2.3, 2.0, 1.4, 1.1, 1.5, 0.7, 2.5, subtract_mean=False, power=1.0),
                       lattice_from_formula(size, gauss(2.0, 1.4),
                                            lambda s, dz: phase(2.3 + 1.1 * f(s, 1.5, 1.0))(s, dz) * math.cos(two_pi * 0.7 * f(s, 2.5, 1.0) * dz))))
        checks.append(("model_MV_power", cs.model_MV_kernel(numpy, size, 2.3, 2.0, 1.4, 1.1, 1.5, 0.7, 2.5, power=1.0),
                       lattice_from_formula(size, gauss(2.0, 1.4),
                                            lambda s, dz: phase(2.3 + 1.1 * (f(s, 1.5, 1.0) - fbp2))(s, dz) * math.cos(two_pi * 0.7 * f(s, 2.5, 1.0) * dz))))
        checks.append(("superposition", cs.momentum_superposition_kernel(numpy, size, 1.1, 2.0, 1.5, [-0.5, 0.0, 0.7], [1.0, 2.0, 1.0]),
                       lattice_from_formula(size, gauss(2.0, 1.5),
                                            lambda s, dz: 0.25 * phase(0.6)(s, dz) + 0.5 * phase(1.1)(s, dz) + 0.25 * phase(1.8)(s, dz))))
        # conditional: w(x | s) = exp(-(x - 0.6)^2 / 0.02) (1 + s/10)^-x on 3 nodes, normalized at each s
        x_nodes = [0.2, 0.5, 0.9]

        def w_cond(x, s):
            return math.exp(-((x - 0.6) ** 2) / 0.02) * (1 + s / 10.0) ** (-x)

        def chi_cond(s, dz):
            total = sum(w_cond(x, s) for x in x_nodes)
            return sum(w_cond(x, s) / total * phase(4.0 * x)(s, dz) for x in x_nodes)

        s_arr = cs.transverse_s(numpy, size)
        checks.append(("conditional", cs.conditional_kernel(numpy, size, 4.0, 2.0, 1.4, x_nodes, [w_cond(x, s_arr) for x in x_nodes]),
                       lattice_from_formula(size, gauss(2.0, 1.4), chi_cond)))
        K1 = lattice_from_formula(size, gauss(2.0, 1.4), phase(1.0))
        K2 = lattice_from_formula(size, lambda s, dz: math.exp(-math.sqrt(s + dz**2) / 1.8), phase(-0.5))
        checks.append(("linear_combination", cs.linear_combination_kernel([cs.gaussian_kernel(numpy, size, 1.0, 2.0, 1.4), cs.exponential_kernel(numpy, size, -0.5, 1.8)],
                                                                          [0.7, -0.4]), 0.7 * K1 - 0.4 * K2))
        checks.append(("stretched", cs.stretched_kernel(numpy, size, 1.6, 2.0, 1.4, 1.3, 2.7),
                       lattice_from_formula(size, lambda s, dz: math.exp(-((math.sqrt(s) / 2.0) ** 1.3) - (abs(dz) / 1.4) ** 2.7), phase(1.6))))
        checks.append(("exponential", cs.exponential_kernel(numpy, size, 1.6, 1.8),
                       lattice_from_formula(size, lambda s, dz: math.exp(-math.sqrt(s + dz**2) / 1.8), phase(1.6))))
        checks.append(("power_tail", cs.power_tail_kernel(numpy, size, 1.6, 1.5, 1.2, 2.0),
                       lattice_from_formula(size, lambda s, dz: (1 + s / 1.5**2 + dz**2 / 1.2**2) ** (-2.0), phase(1.6))))
        checks.append(("polynomial", cs.polynomial_kernel(numpy, size, 1.6, 2.0, 1.4, 0.4, -0.3),
                       lattice_from_formula(size, lambda s, dz: gauss(2.0, 1.4)(s, dz) * (1 + 0.4 * (s / 4.0 - 0.5) - 0.3 * (dz**2 / 1.96 - 0.25)) / G_sum,
                                            phase(1.6), normalize_envelope=False)))
        for name, got, want in checks:
            assert_close(got, want, 1e-13, f"{name} {size} vs the formula")


def test_conditional_kernels_against_quadrature():
    """beta_kernel and gpd_kernel against the x integral of Eq. (28) done by scipy quad, at a few displacements."""
    size = [8, 8, 16]
    Lz = size[2]
    P_z = 3.0
    n_x = 4000
    K_beta = cs.beta_kernel(numpy, size, P_z, 2.0, 1.4, 2.0, 1.5, 2.5, -1.0, 1.5, n_x=n_x)
    K_gpd = cs.gpd_kernel(numpy, size, P_z, 2.0, 1.4, 3.0, 0.5, 3.0, n_x=n_x)
    G_hat = cs.normalized(cs.gaussian_envelope(numpy, size, 2.0, 1.4))
    for iz, iy, ix in ((1, 0, 0), (3, 1, 2), (2, 3, 3), (5, 2, 1), (8, 1, 1)):
        dz = nearest(iz, Lz)
        s = nearest(ix, size[0]) ** 2 + nearest(iy, size[1]) ** 2
        dz_images = [dz] if iz != Lz // 2 else [-Lz // 2, Lz // 2]
        # beta
        a = 2.0 + 1.5 * math.exp(-s / (2 * 1.5**2))
        b = 2.5 - 1.0 * math.exp(-s / (2 * 1.5**2))
        chi = 0
        for d in dz_images:
            re = integrate.quad(lambda x: x ** (a - 1) * (1 - x) ** (b - 1) * math.cos(2 * math.pi / Lz * x * P_z * d), 0, 1, limit=200)[0]
            im = integrate.quad(lambda x: x ** (a - 1) * (1 - x) ** (b - 1) * math.sin(2 * math.pi / Lz * x * P_z * d), 0, 1, limit=200)[0]
            chi += (re + 1j * im) / special.beta(a, b) / len(dz_images)
        assert abs(K_beta[iz, iy, ix] - G_hat[iz, iy, ix] * chi) < 1e-6 * abs(G_hat[iz, iy, ix]), ("beta", iz, iy, ix)
        # gpd
        def q_xb(x):
            g = 3.0 * (1 - x) * math.log(1 / x)
            return x**0.5 * (1 - x) ** 3.0 / g * math.exp(-s / (4 * g))
        norm = integrate.quad(q_xb, 0, 1, limit=400)[0]
        chi = 0
        for d in dz_images:
            re = integrate.quad(lambda x: q_xb(x) * math.cos(2 * math.pi / Lz * x * P_z * d), 0, 1, limit=400)[0]
            im = integrate.quad(lambda x: q_xb(x) * math.sin(2 * math.pi / Lz * x * P_z * d), 0, 1, limit=400)[0]
            chi += (re + 1j * im) / norm / len(dz_images)
        assert abs(K_gpd[iz, iy, ix] - G_hat[iz, iy, ix] * chi) < 1e-5 * abs(G_hat[iz, iy, ix]), ("gpd", iz, iy, ix)


def test_beta_kernel_with_singular_end_points():
    """a < 1 or b < 1: the cell weights keep beta_kernel close to Eq. (28) with Eq. (30) (scipy quad with the exact x^(a-1)(1-x)^(b-1))."""
    size = [8, 8, 32]
    Lz, P_z = 32, 6.0
    G_hat = cs.normalized(cs.gaussian_envelope(numpy, size, 3.25, 1.625))
    for a, b in ((0.3, 3.0), (0.5, 2.0), (2.0, 0.5)):
        errors = []
        for n_x in (100, 1000):
            K = cs.beta_kernel(numpy, size, P_z, 3.25, 1.625, a, 0.0, b, 0.0, 1.5, n_x=n_x)
            worst = 0.0
            for dz in range(1, 8):
                kappa = 2 * math.pi / Lz * P_z * dz
                re = integrate.quad(lambda x: math.cos(kappa * x), 0, 1, weight="alg", wvar=(a - 1, b - 1))[0]
                im = integrate.quad(lambda x: math.sin(kappa * x), 0, 1, weight="alg", wvar=(a - 1, b - 1))[0]
                chi = (re + 1j * im) / special.beta(a, b)
                worst = max(worst, abs(K[dz, 0, 0] / G_hat[dz, 0, 0] - chi))
            errors.append(worst)
        assert errors[0] < 1e-2 and errors[1] < 1e-3, (a, b, errors)  # the midpoint values were off by 0.03 - 0.13 at n_x = 100


# ---------------- analytic statements of the note ----------------


def local_momentum_moments(K_column, Lz):
    """Mean and variance of the momentum intensity |K(k)|^2 along z of one column K[dz], k in radians per lattice unit."""
    Kk = numpy.fft.fft(K_column)  # sum_dz exp(-i k dz) K(dz), Eq. (7) with k = 2pi n / Lz
    k = 2 * numpy.pi * numpy.fft.fftfreq(Lz)
    w = numpy.abs(Kk) ** 2
    mean = float((w * k).sum() / w.sum())
    var = float((w * (k - mean) ** 2).sum() / w.sum())
    return mean, var


def test_momentum_moments_of_the_note():
    """Eqs. (7), (19), (22), (24) in a long box, where the lattice is close to the continuum along z."""
    Lz = 512
    size = [8, 8, Lz]
    unit = 2 * numpy.pi / Lz  # one unit of k in radians
    k0, rho_T, rho_z = 20.0, 2.0, 4.0  # k0 = 0.245 rad, width 1/rho_z = 0.25 rad: well inside the Brillouin zone
    # Eq. (7): Gaussian, mean k0, variance 1/rho_z^2
    K = cs.gaussian_kernel(numpy, size, k0, rho_T, rho_z)
    mean, var = local_momentum_moments(K[:, 0, 0], Lz)
    assert abs(mean - k0 * unit) < 1e-6 and abs(var - 1 / rho_z**2) < 1e-6, (mean, k0 * unit, var, 1 / rho_z**2)
    # Eq. (19): Model M, mean mu(s), variance 1/rho_z^2 at every s
    K = cs.model_M_kernel(numpy, size, k0, rho_T, rho_z, 8.0, 1.5)
    s = cs.transverse_s(numpy, size)
    f = cs.radial_function(s, 1.5)
    mu = k0 + 8.0 * (f - cs.transverse_mean(s, f, rho_T))
    for iy, ix in ((0, 0), (1, 2), (3, 3)):
        mean, var = local_momentum_moments(K[:, iy, ix], Lz)
        assert abs(mean - mu[0, iy, ix] * unit) < 1e-6 and abs(var - 1 / rho_z**2) < 1e-6, ("M", iy, ix)
    # the transverse |G_T|^2 mean of mu is k0 (Eq. (18))
    w = numpy.exp(-2 * s / rho_T**2)
    assert abs(float((w * mu).sum() / w.sum()) - k0) < 1e-12
    # Eq. (18) in the continuum: a wide transverse plane
    s_wide = cs.transverse_s(numpy, [64, 64, 4])
    fb = cs.transverse_mean(s_wide, cs.radial_function(s_wide, 3.0), 4.0)
    assert abs(fb - 1 / (1 + 4.0**2 / (4 * 3.0**2))) < 1e-6, fb
    # Eq. (22): Model V, variance 1/rho_z^2 + delta^2 / (1 + exp(-rho_z^2 delta^2 / 2)), mean k0
    K = cs.model_V_kernel(numpy, size, k0, rho_T, rho_z, 10.0, 1.5)
    delta = 10.0 * f * unit
    for iy, ix in ((0, 0), (1, 1), (2, 3)):
        mean, var = local_momentum_moments(K[:, iy, ix], Lz)
        d = delta[0, iy, ix]
        assert abs(mean - k0 * unit) < 1e-6, ("V mean", iy, ix)
        assert abs(var - (1 / rho_z**2 + d**2 / (1 + numpy.exp(-(rho_z**2) * d**2 / 2)))) < 1e-6, ("V var", iy, ix)
    # Eq. (24): Model W, variance 1/rho_z(s)^2
    K = cs.model_W_kernel(numpy, size, k0, rho_T, rho_z, 0.5, 1.5)
    rho_s = rho_z * (1 + 0.5 * f)
    for iy, ix in ((0, 0), (1, 2), (3, 3)):
        mean, var = local_momentum_moments(K[:, iy, ix], Lz)
        assert abs(var - 1 / rho_s[0, iy, ix] ** 2) < 1e-6, ("W var", iy, ix)
    # Eq. (25): WG has rho_z(s)^-2 = rho_z^-2 + v(s)/2
    K = cs.model_WG_kernel(numpy, size, k0, rho_T, rho_z, 0.04, 1.5)
    for iy, ix in ((0, 0), (1, 2)):
        mean, var = local_momentum_moments(K[:, iy, ix], Lz)
        assert abs(var - (1 / rho_z**2 + 0.04 * f[0, iy, ix] / 2)) < 1e-6, ("WG var", iy, ix)
    # Eq. (14): a centered Gaussian momentum superposition is a Gaussian with rho_z_eff^-2 = rho_z^-2 + v/2 (shape, up to N)
    v = 0.02  # radians^2
    q_rad = numpy.linspace(-8 * numpy.sqrt(v), 8 * numpy.sqrt(v), 801)
    weights = numpy.exp(-(q_rad**2) / (2 * v))
    K = cs.momentum_superposition_kernel(numpy, size, k0, rho_T, rho_z, list(q_rad / unit), list(weights))
    rho_eff = (rho_z**-2 + v / 2) ** -0.5
    K_eff = cs.gaussian_kernel(numpy, size, k0, rho_T, rho_eff)
    inside = numpy.r_[0:100, Lz - 100:Lz]  # away from the half box, where the periodic images differ
    assert_close(K[inside] / K[0, 0, 0], K_eff[inside] / K_eff[0, 0, 0], 1e-9, "Eq. (14)")


def test_model_M_is_phase_only():
    size = [8, 8, 12]
    G = cs.gaussian_kernel(numpy, size, 2.0, 2.0, 1.4)
    M = cs.model_M_kernel(numpy, size, 2.0, 2.0, 1.4, 1.3, 1.5)
    interior = [i for i in range(12) if i != 6]
    assert_close(numpy.abs(M[interior]), numpy.abs(G[interior]), 1e-14, "|K_M| = |K_G| off the half box")


def test_beta_mean():
    """<x>_s of the discrete beta weights is a(s) / (a(s) + b(s)) (Eq. (32)), up to the midpoint error near a singular endpoint."""
    x_list = numpy.array(cs.midpoint_nodes(2000))
    for a, b in ((2.0, 3.0), (3.5, 1.5), (1.2, 1.2)):
        w = x_list ** (a - 1) * (1 - x_list) ** (b - 1)
        assert abs((w * x_list).sum() / w.sum() - a / (a + b)) < 1e-5


def test_invalid_parameters_are_refused():
    size = [6, 8, 10]
    for call in (lambda: cs.model_W_kernel(numpy, size, 1.0, 2.0, 1.4, -1.0, 1.5),
                 lambda: cs.model_WG_kernel(numpy, size, 1.0, 2.0, 1.4, -0.1, 1.5),
                 lambda: cs.beta_kernel(numpy, size, 3.0, 2.0, 1.4, 0.5, -1.0, 2.0, 0.0, 1.5),
                 lambda: cs.gpd_kernel(numpy, size, 3.0, 2.0, 1.4, 3.0, -0.5, 1.0),
                 lambda: cs.gpd_kernel(numpy, size, 3.0, 2.0, 1.4, 3.0, -1.0, 2.0),
                 lambda: cs.multi_momentum_kernel(numpy, size, [1.0, 2.0], 2.0, 1.4, [1.0, -1.0]),
                 lambda: cs.two_scale_kernel(numpy, size, 1.0, 1.0, 2.0, 1.4, -1.0, 2.0, 1.4),
                 lambda: cs.conditional_kernel(numpy, size, 3.0, 2.0, 1.4, [0.2, 0.8], [0.0, 0.0])):
        try:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                call()
            raise AssertionError("invalid parameters were accepted")
        except ValueError:
            pass


if __name__ == "__main__":
    names = [name for name in sorted(globals()) if name.startswith("test_")]
    failed = 0
    for name in names:
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as error:  # report every test, then fail
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(names) - failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)
