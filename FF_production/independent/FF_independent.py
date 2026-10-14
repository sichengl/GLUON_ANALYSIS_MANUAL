#Gluon bilocal operator from an independent F_munu, for a list of Wilson-line lengths and momenta
#
#  F_munu(x)       clover: [Q(x) - Q(x)^dag] / 8i, made traceless, Q = sum of the four mu-nu plaquettes touching x,
#                  each one as a closed loop starting and ending at x (F here is a^2 g F_munu)
#  W(x, x+n mu)    = U_mu(x) U_mu(x+mu) ... U_mu(x+(n-1)mu),     W(x, x-n mu) = U_mu(x-mu)^dag ... U_mu(x-n mu)^dag
#  C_ab(z, q, t)   = sum_{x,y,z} e^{i q.x} Tr[ F_a(x) W(x, x+z mu) F_b(x+z mu) W(x+z mu, x) ]
#                  computed as Tr[A_a B_b] with A_a = F_a(x) W(x, x+z mu), B_b = F_b(x+z mu) W(x, x+z mu)^dag
#
#pyquda is used only to read the gauge field, HYP smear it and build the Fourier phase e^{i q.x}.
#The plaquettes, F_munu, the Wilson lines and all shifts are done here on lexicographic arrays with periodic
#xp.roll, so the script must run on ONE process (grid [1,1,1,1]); a roll does not cross process boundaries.
#
#usage: python FF_independent.py --cfg 204 [--num_cfg 1 --cfg_step 6]

#==========================
#For direction, xyzt
#For data structure (index), tzyx
#==========================
import argparse
import os
from time import perf_counter

import h5py
import numpy as np

try:
    import cupy as xp
except ImportError:                     # CPU fallback, only for testing the functions on a laptop
    xp = np

X, Y, Z, T = 0, 1, 2, 3
AXIS = {X: -3, Y: -4, Z: -5, T: -6}     # array axis of each direction in (..., t, z, y, x, color, color)
PLANES = [(X, Y), (X, Z), (X, T), (Y, Z), (Y, T), (Z, T)]
PLANE_NAMES = ["xy", "xz", "xt", "yz", "yt", "zt"]

#======== run parameters ========
GAUGE_FILE = "/lustre/orion/lgt132/world-shared/DATA/MILC/a09m310/gauge/l3296f211b630m0074m037m440e.{cfg}"   # stream e
OUT_DIR = "/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/FF_production/FF_independent_data"
QUDA_RESOURCE_PATH = "/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/.cache/quda"

HYP_LIST = list(range(0, 16))           # measure after this many HYP steps (ascending)
HYP_ALPHA = (0.75, 0.6, 0.3)
HYP_DIR_IGNORE = -1                     # -1: smear links in all four directions

WL_DIR = Z                              # Wilson-line direction
Z_LIST = list(range(0, 16))             # signed lengths: z > 0 puts F_b at x + z, z < 0 at x - |z|
Q_LIST = ([[qx, qy, qz] for qx in [0, -1, 1] for qy in [0, -1, 1] for qz in [0, -1, 1]]
          + [[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0]])      # integer momenta, q_i in units of 2 pi / L_i


#======== lattice algebra ========
def dag(m):
    return m.conj().swapaxes(-1, -2)


def shift(f, mu, n):
    """f(x + n mu), periodic"""
    return xp.roll(f, -n, axis=AXIS[mu])


def unit_matrix_like(m):
    one = xp.zeros_like(m)
    for i in range(m.shape[-1]):
        one[..., i, i] = 1
    return one


def evenodd_to_lexico(eo, n_inner):
    """QUDA even-odd (..., 2, Lt, Lz, Ly, Lx/2, *inner) -> lexicographic (..., Lt, Lz, Ly, Lx, *inner).
    Site (x, y, z, t) has parity (x+y+z+t) % 2 and sits at x // 2; n_inner = number of trailing axes (2 for links)."""
    k = eo.ndim - n_inner                                         # eo axes k-5 .. k-1 are parity, t, z, y, x/2
    Lt, Lz, Ly, Lxh = eo.shape[k - 4:k]
    inner = (slice(None),) * n_inner
    even = eo[(Ellipsis, 0) + (slice(None),) * 4 + inner]
    odd = eo[(Ellipsis, 1) + (slice(None),) * 4 + inner]
    row = xp.arange(Lt)[:, None, None] + xp.arange(Lz)[None, :, None] + xp.arange(Ly)[None, None, :]
    odd_row = (row % 2 == 1).reshape((Lt, Lz, Ly, 1) + (1,) * n_inner)   # x even holds parity 1 on these rows
    out = xp.empty(eo.shape[:k - 5] + (Lt, Lz, Ly, 2 * Lxh) + eo.shape[k:], eo.dtype)
    out[(Ellipsis, slice(0, None, 2)) + inner] = xp.where(odd_row, odd, even)
    out[(Ellipsis, slice(1, None, 2)) + inner] = xp.where(odd_row, even, odd)
    return out


def plaquette(U, mu, nu):
    """P_munu(x) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag"""
    return U[mu] @ shift(U[nu], mu, 1) @ dag(shift(U[mu], nu, 1)) @ dag(U[nu])


def mean_plaquette(U):
    """Re Tr P / 3 averaged over sites and the six planes"""
    return float(np.mean([xp.trace(plaquette(U, mu, nu), axis1=-2, axis2=-1).real.mean().item() / 3 for mu, nu in PLANES]))


def field_strength(U, mu, nu):
    """Clover F_munu(x). The plaquette with corner y = x, x-mu, x-nu or x-mu-nu touches x; its loop is moved to start
    and end at x by V^dag P(y) V, V = links from y to x along its edges, and then the result is shifted from y to x."""
    P = plaquette(U, mu, nu)
    V_mu = U[mu]                                     # y -> y+mu
    V_nu = U[nu]                                     # y -> y+nu
    V_munu = U[mu] @ shift(U[nu], mu, 1)             # y -> y+mu -> y+mu+nu
    Q = (P                                                                   # corner x
         + shift(dag(V_mu) @ P @ V_mu, mu, -1)                               # corner x - mu
         + shift(dag(V_nu) @ P @ V_nu, nu, -1)                               # corner x - nu
         + shift(shift(dag(V_munu) @ P @ V_munu, mu, -1), nu, -1))           # corner x - mu - nu
    F = (Q - dag(Q)) / 8j
    tr = (F[..., 0, 0] + F[..., 1, 1] + F[..., 2, 2]) / 3
    for i in range(3):
        F[..., i, i] -= tr
    return F


def project(F, W, Fb, phase):
    """out[a, b, q, t] = sum_{x,y,z} phase[q, t, z, y, x] Tr[ F_a(x) W(x) Fb_b(x) W(x)^dag ]"""
    A = F @ W                                        # F_a(x) W(x, x+z)
    B = Fb @ dag(W)                                  # F_b(x+z) W(x+z, x)
    out = xp.empty((len(F), len(F), phase.shape[0], F.shape[1]), xp.complex128)
    for a in range(len(F)):
        tr = xp.einsum("tzyxij,btzyxji->btzyx", A[a], B)
        out[a] = xp.einsum("qtzyx,btzyx->bqt", phase, tr)
    return out


def bilocal_correlator(F, U, phase, z_list, mu):
    """C[a, b, iz, q, t] for every signed length z_list[iz] along mu; F: (nF, Lt, Lz, Ly, Lx, 3, 3), phase: (Nq, Lt, Lz, Ly, Lx)"""
    C = xp.zeros((len(F), len(F), len(z_list), phase.shape[0], F.shape[1]), xp.complex128)
    for sign in (+1, -1):
        lengths = sorted({abs(z) for z in z_list if z * sign > 0 or (z == 0 and sign > 0)})
        W = unit_matrix_like(U[mu])                  # W(x, x + n sign mu), n = 0
        Fb = F                                       # F_b(x + n sign mu)
        n = 0
        for length in lengths:
            while n < length:
                if sign > 0:
                    W = U[mu] @ shift(W, mu, 1)                  # W(x, x+(n+1)mu) = U_mu(x) W(x+mu, x+(n+1)mu)
                else:
                    W = shift(dag(U[mu]) @ W, mu, -1)            # W(x, x-(n+1)mu) = U_mu(x-mu)^dag W(x-mu, x-(n+1)mu)
                Fb = shift(Fb, mu, sign)
                n += 1
            C[:, :, z_list.index(sign * length)] = project(F, W, Fb, phase)
    return C


def to_host(a):
    return a.get() if hasattr(a, "get") else a


def sync():
    if xp is not np:
        xp.cuda.Device().synchronize()


#======== production ========
def main():
    from pyquda_utils import core, io, phase

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=int, required=True)
    parser.add_argument("--num_cfg", type=int, default=1)
    parser.add_argument("--cfg_step", type=int, default=6)
    args = parser.parse_args()

    assert HYP_LIST == sorted(HYP_LIST) and len(set(Z_LIST)) == len(Z_LIST)
    core.init([1, 1, 1, 1], resource_path=QUDA_RESOURCE_PATH)   # one process: the rolls above need the whole lattice
    log = core.getLogger()
    os.makedirs(OUT_DIR, exist_ok=True)

    phase_lex = None
    for cfg in [args.cfg + args.cfg_step * i for i in range(args.num_cfg)]:
        gauge_file = GAUGE_FILE.format(cfg=cfg)
        gauge = io.readMILCGauge(gauge_file)
        Lt = gauge.latt_info.Lt
        if phase_lex is None:
            phase_eo = xp.asarray(phase.MomentumPhase(gauge.latt_info).getPhases(Q_LIST))   # (Nq, 2, Lt, Lz, Ly, Lx/2)
            phase_lex = evenodd_to_lexico(phase_eo, 0)                                      # (Nq, Lt, Lz, Ly, Lx)
            del phase_eo

        corr = xp.zeros((len(HYP_LIST), len(PLANES), len(PLANES), len(Z_LIST), len(Q_LIST), Lt), xp.complex128)
        n_hyp_done = 0
        for i_hyp, n_hyp in enumerate(HYP_LIST):
            s = perf_counter()
            if n_hyp > n_hyp_done:
                gauge.hypSmear(n_hyp - n_hyp_done, *HYP_ALPHA, HYP_DIR_IGNORE, True, False)
                n_hyp_done = n_hyp
            U = evenodd_to_lexico(xp.asarray(gauge.data), 2)                                # (4, Lt, Lz, Ly, Lx, 3, 3)
            F = xp.stack([field_strength(U, mu, nu) for mu, nu in PLANES])                  # (6, Lt, Lz, Ly, Lx, 3, 3)
            corr[i_hyp] = bilocal_correlator(F, U, phase_lex, Z_LIST, WL_DIR)
            sync()
            log.info(f"cfg {cfg} hyp {n_hyp}: plaquette {mean_plaquette(U):.12f}, {perf_counter() - s:.1f} secs")
            del U, F

        filename = (f"{OUT_DIR}/FF_independent_hyp{HYP_LIST[0]}-{HYP_LIST[-1]}"
                    f"_z{min(Z_LIST)}-{max(Z_LIST)}{'xyzt'[WL_DIR]}_cfg{cfg}.h5")
        with h5py.File(filename, "w") as f:
            d = f.create_dataset("corr", data=to_host(corr))
            d.attrs["dim_spec"] = "hyp, F_a, F_b, z, q, t"
            f.create_dataset("planes", data=np.array(PLANE_NAMES, dtype="S2"))
            f.create_dataset("hyp_list", data=np.asarray(HYP_LIST, dtype=np.int64))
            f.create_dataset("z_list", data=np.asarray(Z_LIST, dtype=np.int64))
            f.create_dataset("q_list", data=np.asarray(Q_LIST, dtype=np.int64))
            f.attrs["operator"] = "sum_{x,y,z} e^{i q.x} Tr[F_a(x) W(x, x+z) F_b(x+z) W(x+z, x)], phase at the site of F_a"
            f.attrs["phase"] = "pyquda_utils.phase.MomentumPhase(latt_info).getPhases(q_list)"
            f.attrs["F_definition"] = "clover (Q - Q^dag)/8i, traceless, Q = four plaquettes touching x; F = a^2 g F_munu"
            f.attrs["wilson_line_direction"] = "xyzt"[WL_DIR]
            f.attrs["hyp_alpha"] = HYP_ALPHA
            f.attrs["hyp_dir_ignore"] = HYP_DIR_IGNORE
            f.attrs["gauge_file"] = gauge_file
        log.info(f"saved {filename}")


if __name__ == "__main__":
    main()
