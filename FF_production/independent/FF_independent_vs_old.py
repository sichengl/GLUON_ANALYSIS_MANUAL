#Runs the old FF method and FF_independent.py on the same HYP-smeared gauge field and logs their difference at every HYP step.
#
#old method = the per-HYP-step block of FF_covshift_oddeven_trimag_negative_direction_hyp.py: gauge.loop clover, F_b moved
#             toward the source by the even-odd covariant shift along -z, contract with the pyquda phase (even-odd layout)
#new method = FF_independent.py: own plaquette clover and explicit Wilson line on lexicographic arrays
#
#The old method only has the -z direction and the plane order xy, yz, xz, xt, yt, zt (PERM maps it to the new order), so
#  z = -w : old[a, b, w] = new[PERM[a], PERM[b], -w]                                        (direct)
#  z = +w : old[a, b, w] = phase(x + w z) / phase(x) * new[PERM[b], PERM[a], +w]            (x -> x - w z in the site sum)
#Deviations are max |old - new| over (a, b, q, t) divided by max |old| at the same w.
#
#usage: python FF_independent_vs_old.py --cfg 204 [--num_cfg 1 --cfg_step 6]     (same settings as FF_independent.py)

#==========================
#For direction, xyzt
#For data structure (index), tzyx
#==========================
import argparse
import os
from time import perf_counter

import h5py
import numpy as np
from opt_einsum import contract

import FF_independent as new
from FF_independent import xp

N_W = 16                                         # old method: w = 0 .. N_W-1 along -z
Z_LIST = list(range(-(N_W - 1), N_W))            # new method: both directions, -15 .. 15
OLD_PLANE_NAMES = ["xy", "yz", "xz", "xt", "yt", "zt"]
PERM = [new.PLANE_NAMES.index(name) for name in OLD_PLANE_NAMES]
TOL = 1e-10                                      # relative deviation above this is logged as a warning


def old_method(gauge, latt_info, phase_eo, n_w):
    """corr[a, b, w, q, t] and the six F (even-odd), as in FF_covshift_oddeven_trimag_negative_direction_hyp.py"""
    from pyquda_utils.core import X, Y, Z, T, LatticeGauge
    Qij = gauge.loop(
        [
            [[X, Y, -X, -Y], [Y, -X, -Y, X], [-X, -Y, X, Y], [-Y, X, Y, -X]],
            [[Y, Z, -Y, -Z], [Z, -Y, -Z, Y], [-Y, -Z, Y, Z], [-Z, Y, Z, -Y]],
            [[X, Z, -X, -Z], [Z, -X, -Z, X], [-X, -Z, X, Z], [-Z, X, Z, -X]],
            [[T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T]],
        ],
        [1, 1, 1, 1],
    )
    Qi4 = gauge.loop(
        [
            [[X, T, -X, -T], [T, -X, -T, X], [-X, -T, X, T], [-T, X, T, -X]],
            [[Y, T, -Y, -T], [T, -Y, -T, Y], [-Y, -T, Y, T], [-T, Y, T, -Y]],
            [[Z, T, -Z, -T], [T, -Z, -T, Z], [-Z, -T, Z, T], [-T, Z, T, -Z]],
            [[T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T]],
        ],
        [1, 1, 1, 1],
    )
    Qij_dagger = LatticeGauge(latt_info)
    Qi4_dagger = LatticeGauge(latt_info)
    Qij_dagger.data[:] = Qij.data.conj().swapaxes(-1, -2)
    Qi4_dagger.data[:] = Qi4.data.conj().swapaxes(-1, -2)
    Fij = -1j / 8 * (Qij - Qij_dagger)
    Fi4 = -1j / 8 * (Qi4 - Qi4_dagger)

    eye3 = xp.eye(3, dtype=Fij.data.dtype)
    tr_Fij = xp.trace(Fij.data, axis1=-2, axis2=-1) / 3.0
    tr_Fi4 = xp.trace(Fi4.data, axis1=-2, axis2=-1) / 3.0
    Fij.data[:] = Fij.data - tr_Fij[..., None, None] * eye3
    Fi4.data[:] = Fi4.data - tr_Fi4[..., None, None] * eye3

    gauge_local = xp.asarray(gauge.data[2, :])
    gauge_local_conj = gauge_local.conj().swapaxes(-1, -2)
    Fmunu = [xp.asarray(Fij.data[0, :]), xp.asarray(Fij.data[1, :]), xp.asarray(Fij.data[2, :]),
             xp.asarray(Fi4.data[0, :]), xp.asarray(Fi4.data[1, :]), xp.asarray(Fi4.data[2, :])]
    Fmunu_shift = [arr.copy() for arr in Fmunu]

    corr = xp.zeros((6, 6, n_w, phase_eo.shape[0], latt_info.Lt), xp.complex128)
    for i_W in range(n_w):
        if i_W != 0:
            Fmunu_shift = [xp.roll((gauge_local_conj @ arr @ gauge_local)[::-1], shift=1, axis=2) for arr in Fmunu_shift]
        for i_src in range(6):
            for i_sink in range(6):
                corr[i_src, i_sink, i_W] = contract("pwtzyx,wtzyxij,wtzyxji->pt", phase_eo, Fmunu[i_src], Fmunu_shift[i_sink])
    return corr, Fmunu


def main():
    from pyquda_utils import core, io, phase

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=int, required=True)
    parser.add_argument("--num_cfg", type=int, default=1)
    parser.add_argument("--cfg_step", type=int, default=6)
    args = parser.parse_args()

    core.init([1, 1, 1, 1], resource_path=new.QUDA_RESOURCE_PATH)
    log = core.getLogger()
    os.makedirs(new.OUT_DIR, exist_ok=True)
    i_neg = [Z_LIST.index(-w) for w in range(N_W)]
    i_pos = [Z_LIST.index(w) for w in range(N_W)]

    phase_eo = None
    for cfg in [args.cfg + args.cfg_step * i for i in range(args.num_cfg)]:
        gauge_file = new.GAUGE_FILE.format(cfg=cfg)
        gauge = io.readMILCGauge(gauge_file)
        latt_info = gauge.latt_info
        if phase_eo is None:
            phase_eo = xp.asarray(phase.MomentumPhase(latt_info).getPhases(new.Q_LIST))     # (Nq, 2, Lt, Lz, Ly, Lx/2)
            phase_lex = new.evenodd_to_lexico(phase_eo, 0)                                  # (Nq, Lt, Lz, Ly, Lx)
            Lz = phase_lex.shape[2]
            hop = xp.stack([phase_lex[:, 0, w % Lz, 0, 0] / phase_lex[:, 0, 0, 0, 0] for w in range(N_W)])   # phase(x+wz)/phase(x), (w, q)

        dev_F = np.zeros(len(new.HYP_LIST))
        dev_neg = np.zeros((len(new.HYP_LIST), N_W))
        dev_pos = np.zeros((len(new.HYP_LIST), N_W))
        t_old = np.zeros(len(new.HYP_LIST))
        t_new = np.zeros(len(new.HYP_LIST))
        n_hyp_done = 0
        for i_hyp, n_hyp in enumerate(new.HYP_LIST):
            if n_hyp > n_hyp_done:
                gauge.hypSmear(n_hyp - n_hyp_done, *new.HYP_ALPHA, new.HYP_DIR_IGNORE, True, False)
                n_hyp_done = n_hyp

            new.sync()
            s = perf_counter()
            C_old, F_old = old_method(gauge, latt_info, phase_eo, N_W)
            new.sync()
            t_old[i_hyp] = perf_counter() - s

            s = perf_counter()
            U = new.evenodd_to_lexico(xp.asarray(gauge.data), 2)
            F = xp.stack([new.field_strength(U, mu, nu) for mu, nu in new.PLANES])
            C_new = new.bilocal_correlator(F, U, phase_lex, Z_LIST, new.WL_DIR)
            new.sync()
            t_new[i_hyp] = perf_counter() - s

            dev_F[i_hyp] = max(xp.abs(F[PERM[a]] - new.evenodd_to_lexico(F_old[a], 2)).max().item() for a in range(6)) / xp.abs(F).max().item()
            C = C_new[PERM][:, PERM]                                                 # new, in the old plane order
            C_neg = C[:, :, i_neg]
            C_pos = C[:, :, i_pos].swapaxes(0, 1) * hop[None, None, :, :, None]
            scale = xp.abs(C_old).max(axis=(0, 1, 3, 4))                             # per w
            dev_neg[i_hyp] = new.to_host(xp.abs(C_old - C_neg).max(axis=(0, 1, 3, 4)) / scale)
            dev_pos[i_hyp] = new.to_host(xp.abs(C_old - C_pos).max(axis=(0, 1, 3, 4)) / scale)

            msg = (f"cfg {cfg} hyp {n_hyp}: plaquette {new.mean_plaquette(U):.12f} | old vs new: F {dev_F[i_hyp]:.1e}, "
                   f"C z<0 {dev_neg[i_hyp].max():.1e} (worst w={dev_neg[i_hyp].argmax()}), "
                   f"C z>0 {dev_pos[i_hyp].max():.1e} (worst w={dev_pos[i_hyp].argmax()}) | "
                   f"old {t_old[i_hyp]:.1f} s, new {t_new[i_hyp]:.1f} s")
            if max(dev_F[i_hyp], dev_neg[i_hyp].max(), dev_pos[i_hyp].max()) > TOL:
                log.warning("MISMATCH " + msg)
            else:
                log.info(msg)
            del U, F, F_old, C_new, C, C_neg, C_pos

        filename = f"{new.OUT_DIR}/FF_compare_old_new_hyp{new.HYP_LIST[0]}-{new.HYP_LIST[-1]}_w0-{N_W - 1}_cfg{cfg}.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("dev_F", data=dev_F).attrs["dim_spec"] = "hyp"
            f.create_dataset("dev_C_negative_z", data=dev_neg).attrs["dim_spec"] = "hyp, w"
            f.create_dataset("dev_C_positive_z", data=dev_pos).attrs["dim_spec"] = "hyp, w"
            f.create_dataset("time_old", data=t_old)
            f.create_dataset("time_new", data=t_new)
            f.create_dataset("hyp_list", data=np.asarray(new.HYP_LIST, dtype=np.int64))
            f.create_dataset("q_list", data=np.asarray(new.Q_LIST, dtype=np.int64))
            f.attrs["deviation"] = "max_{a,b,q,t} |old - new| / max_{a,b,q,t} |old| at each w; F: max |dF| / max |F|"
            f.attrs["gauge_file"] = gauge_file
        log.info(f"saved {filename}")


if __name__ == "__main__":
    main()
