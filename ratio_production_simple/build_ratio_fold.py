import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

# must match the frame the three-point production ran with
frame = "symmetric"

INPUT_3PT = (SCRIPT_DIR.parent / "3pt_production_simple"
             / f"gpd_3pt_{frame}")
FOLD_C2 = True   # average C2(t) with C2(Gt - t); reads both C2 from TWOPT_PATH
OUT_DIR = SCRIPT_DIR / (f"ratio_jk_{frame}" + ("_foldc2" if FOLD_C2 else ""))
PLOT_DIR = SCRIPT_DIR / ("ratio_plots" + ("_foldc2" if FOLD_C2 else ""))
TWOPT_PATH = SCRIPT_DIR.parent / "2PT_simple" / "twopt_source_averaged.h5"

Ls = 32
Gt = 96

# which produced files to turn into ratios; every combination below must
# exist in INPUT_3PT or the run stops before doing any work
tgf_list = [5, 10, 15, 20, 25, 30, 35, 40]
pf_list = [(0, 0, pz) for pz in range(0, 7)]
q_list = [(0, 0, 0)]

tsep_list = [4, 5, 6, 7, 8, 9, 10]
w_plot_list = [0, 2, 4, 6, 8]

N_WORKERS = 25


def jk_mean_err(samples):
    """Mean and jackknife error over axis 0 of a set of jackknife samples."""
    n = samples.shape[0]
    mean = samples.mean(axis=0)
    err = np.sqrt((n - 1.0) / n * np.sum((samples - mean) ** 2, axis=0))
    return mean, err


def load_c2(cfg_list, mom):
    """Jackknife samples of C2 at one momentum, [jk, Gt], folded if asked.

    Reads the full-length source-averaged correlator, so t and Gt - t are
    both available; the 3pt file's pt2_f_jk is truncated at tsep_max and
    cannot be folded.
    """
    mom = tuple(int(v) for v in mom)
    with h5py.File(TWOPT_PATH, "r") as f:
        moms = [tuple(p) for p in f["momentum_list"][:].tolist()]
        cfgs = f["cfg_list"][:]
        if mom not in moms:
            raise ValueError(f"{mom} is not in {TWOPT_PATH}")
        raw = f["correlator_cfg"][:, moms.index(mom), :].real
    if not np.array_equal(cfgs, cfg_list):
        raise ValueError(
            f"{TWOPT_PATH} holds a different cfg_list than the 3pt file")
    if FOLD_C2:
        raw = 0.5 * (raw + np.roll(raw[:, ::-1], 1, axis=1))
    n = raw.shape[0]
    return (raw.sum(axis=0) - raw) / (n - 1)


def ratio_one_file(path):
    with h5py.File(path, "r") as f:
        pt3_jk = f["pt3_jk"][:]              # [jk, op, z_WL, tsep, tins]
        c2_f_jk = f["pt2_f_jk"][:]           # [jk, tsep]
        cfg_list = f["cfg_list"][:]
        w_list = f["w_list"][:]
        op_names = [s.decode() for s in f["operator_names"][:]]
        tgf = int(f.attrs["tgf"])
        q = np.array(f.attrs["q"], dtype=np.int64)
        pf = np.array(f.attrs["pf"], dtype=np.int64)

    n_jk, n_op, n_z = pt3_jk.shape[:3]
    n_tsep = len(tsep_list)
    tau_max = max(tsep_list)

    pi = tuple(int(pf[k] + q[k]) for k in range(3))
    if FOLD_C2:
        c2_f = load_c2(cfg_list, pf)
        c2_i = c2_f if np.all(q == 0) else load_c2(cfg_list, pi)
        d = np.abs(c2_f[:, :c2_f_jk.shape[1]] - c2_f_jk.real)
        rel = np.max(d / np.abs(c2_f_jk.real))
        print(f"  {path.name}: maximum percentage of change after folding :folded C2_f - pt2_f_jk| / C2 max = {rel:.3e}",
              flush=True)
    else:
        c2_f = c2_f_jk.real
        c2_i = c2_f if np.all(q == 0) else load_c2(cfg_list, pi)

    # exp(+i pi qz w / Ls): the centering phase of the bilocal operator,
    # exactly 1 in the forward limit
    centering = np.exp(1j * np.pi * float(q[2])
                       * np.asarray(w_list, dtype=float) / Ls)

    ratio_jk = np.full((n_jk, n_op, n_z, n_tsep, tau_max + 1),
                       np.nan + 1j * np.nan, dtype="<c16")

    for itsep, tsep in enumerate(tsep_list):
        tau = np.arange(tsep + 1)
        # [jk, tau] two-point factors
        f_T = c2_f[:, tsep][:, None]
        i_T = c2_i[:, tsep][:, None]
        f_tau, i_tau = c2_f[:, tau], c2_i[:, tau]
        f_Tmt, i_Tmt = c2_f[:, tsep - tau], c2_i[:, tsep - tau]
        factor = np.sqrt((f_T * i_Tmt * f_tau) / (i_T * f_Tmt * i_tau))
        # [jk, op, z_WL, tau]
        num = pt3_jk[:, :, :, tsep, : tsep + 1] * centering[None, None, :, None]
        ratio_jk[:, :, :, itsep, : tsep + 1] = (
            num / f_T[:, None, None, :] * factor[:, None, None, :])

    bad = sum(int(np.sum(~np.isfinite(ratio_jk[:, :, :, i, : t + 1])))
              for i, t in enumerate(tsep_list))
    if bad:
        print(f"  WARNING {path.name}: {bad} non-finite ratio entries "
              f"(negative argument under the square root?)", flush=True)

    out = (OUT_DIR / f"ratio_{frame}_tgf{tgf}_pf{pf[0]}_{pf[1]}_{pf[2]}"
           f"_q{q[0]}_{q[1]}_{q[2]}.h5")
    with h5py.File(out, "w") as f:
        f.create_dataset("ratio_jk", data=ratio_jk)
        f.create_dataset("tsep_list", data=np.array(tsep_list, dtype=np.int64))
        f.create_dataset("w_list", data=w_list)
        f.create_dataset("cfg_list", data=cfg_list)
        f.create_dataset("operator_names",
                         data=np.array(op_names, dtype="S"))
        f.attrs["dim_ratio_jk"] = "jk,op,z_WL,tsep_index,tau"
        f.attrs["frame"] = frame
        f.attrs["tgf"] = tgf
        f.attrs["pf"] = pf
        f.attrs["pz"] = int(pf[2])
        f.attrs["q"] = q
        f.attrs["n_jk"] = n_jk
        f.attrs["ratio_definition"] = (
            "R = C3(tsep,tau)/C2_f(tsep) * "
            "sqrt(C2_f(tsep) C2_i(tsep-tau) C2_f(tau) / "
            "(C2_i(tsep) C2_f(tsep-tau) C2_i(tau))), "
            "times exp(i pi qz w / Ls); equals C3/C2 at q = 0")
        f.attrs["source"] = str(path)
        f.attrs["fold_c2"] = FOLD_C2

    plot_ratio(ratio_jk, op_names, w_list, tgf, pf, q)
    return out


def plot_ratio(ratio_jk, op_names, w_list, tgf, pf, q):
    ws = [w for w in w_plot_list if w in list(w_list)]
    for iop, op_name in enumerate(op_names):
        fig, axes = plt.subplots(2, len(ws), figsize=(4.0 * len(ws), 7.0),
                                 squeeze=False, sharex=True)
        for icol, w in enumerate(ws):
            iw = list(w_list).index(w)
            for itsep, tsep in enumerate(tsep_list):
                tau = np.arange(tsep + 1)
                x = tau - 0.5 * tsep
                block = ratio_jk[:, iop, iw, itsep, : tsep + 1]
                for irow, part in enumerate((block.real, block.imag)):
                    mean, err = jk_mean_err(part)
                    axes[irow][icol].errorbar(
                        x, mean, yerr=err, capsize=3, marker="o",
                        linestyle="-", label=f"tsep={tsep}")
            axes[0][icol].set_title(f"w = {w}")
            axes[1][icol].set_xlabel(r"$\tau - t_{\rm sep}/2$")
        axes[0][0].set_ylabel(r"Re $R$")
        axes[1][0].set_ylabel(r"Im $R$")
        for row in axes:
            for ax in row:
                ax.axvline(0, color="k", linestyle="--", lw=1, alpha=0.4)
        axes[0][-1].legend(fontsize=8, frameon=True)
        fig.suptitle(f"{op_name}   pf={tuple(int(v) for v in pf)}   "
                     f"q={tuple(int(v) for v in q)}   "
                     f"flow={tgf * 0.1:.1f}", fontsize=13)
        fig.tight_layout()
        name = (f"ratio_{frame}_{op_name}_tgf{tgf}_"
                f"pf{pf[0]}_{pf[1]}_{pf[2]}_q{q[0]}_{q[1]}_{q[2]}.png")
        fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = [
            INPUT_3PT / (f"3pt_tgf{tgf}_pf{pf[0]}_{pf[1]}_{pf[2]}"
                         f"_q{q[0]}_{q[1]}_{q[2]}.h5")
            for tgf in tgf_list for pf in pf_list for q in q_list
        ]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(paths)} input files are missing, "
                f"e.g. {missing[0]}")
    print(f"{len(paths)} input files", flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(ratio_one_file, p): p for p in paths}
        for future in as_completed(futures):
            path = futures[future]
            out = future.result()
            print(f"  {path.name} -> {out.name}", flush=True)
    print("finished", flush=True)
