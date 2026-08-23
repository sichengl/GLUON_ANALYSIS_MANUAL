import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
frame = "symmetric"
operator = "TXTXpTYTYm2XYXY"

INPUT_RATIO = (SCRIPT_DIR.parent / "ratio_production_simple"
               / f"ratio_jk_{frame}")
PLOT_DIR = SCRIPT_DIR / "corr_eigen_plots"

# ---- which data points to use: exactly the ratio fit's selection ----------
tgf_list = [20, 25, 30, 35, 40]
pf_list = [(0, 0, pz) for pz in range(0, 7)]
q_list = [(0, 0, 0)]

w_fit_list = [4,5,6]
tsep_fit_list = [4, 5, 6, 7, 8, 9, 10]
tau_skip = 1

SVDCUT = 1e-7          # only drawn as a reference line
LOO_STRIDE = 10        # use every Nth leave-one-out set; 1 = all of them
N_WORKERS = 30
# ---------------------------------------------------------------------------


def build_samples(tgf, pf, q):
    """The ratio fit's data vector: jackknife samples [jk, n_pt]."""
    name = (f"ratio_{frame}_tgf{tgf}_pf{pf[0]}_{pf[1]}_{pf[2]}"
            f"_q{q[0]}_{q[1]}_{q[2]}.h5")
    with h5py.File(INPUT_RATIO / name, "r") as f:
        w_all = list(f["w_list"][:])
        tsep_all = list(f["tsep_list"][:])
        iop = [s.decode() for s in f["operator_names"][:]].index(operator)
        ratio_jk = f["ratio_jk"][:, iop]      # only the operator we need

    vector_data = []
    for w in w_fit_list:
        for tsep in tsep_fit_list:
            for tau in range(tau_skip, tsep - tau_skip + 1):
                vector_data.append(ratio_jk[:, w_all.index(w),
                                            tsep_all.index(tsep), tau].real)
    return np.stack(vector_data, axis=1)


def leave_one_out_spectra(samples):
    """Correlation eigenvalues from each choice of n-1 measurements.

    samples are the delete-one replicates, theta_k = (sum C - C_k)/(n-1),
    so the measurements come back as C_k = sum(theta) - (n-1) theta_k.
    """
    n, n_pt = samples.shape
    meas = samples.sum(axis=0) - (n - 1.0) * samples
    corr_all = np.corrcoef(meas, rowvar=False)
    ev_all = np.linalg.eigvalsh(corr_all)
    left_out = np.arange(0, n, LOO_STRIDE)
    ev_loo = np.zeros((len(left_out), n_pt))
    for i, k in enumerate(left_out):
        sub = np.delete(meas, k, axis=0)
        corr = np.corrcoef(sub, rowvar=False)
        ev_loo[i] = np.linalg.eigvalsh(corr)
    return ev_loo, ev_all, np.trace(corr_all), n


def plot_spectra(ev_loo, ev_all, n, tgf, pf, q):
    n_curve, n_pt = ev_loo.shape
    index = np.arange(1, n_pt + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))

    ax = axes[0]
    ax.plot(ev_all, index, color="k", lw=2.5, zorder=1,
            label=f"all {n} measurements")
    for k in range(n_curve):
        ax.plot(ev_loo[k], index, color="C0", lw=0.5, alpha=0.4, zorder=2)
    ax.plot(ev_loo.min(axis=0), index, color="C1", lw=1.0, ls="--", zorder=3,
            label=f"{n_curve} x leave-one-out, min / max")
    ax.plot(ev_loo.max(axis=0), index, color="C1", lw=1.0, ls="--", zorder=3)
    n_below = int(np.sum(ev_all < SVDCUT * ev_all.max()))
    ax.axvline(SVDCUT * ev_all.max(), color="r", ls="--", lw=1, zorder=4,
               label=f"svdcut {SVDCUT:.0e} x max  ({n_below} below)")
    ax.set_xscale("log")
    ax.set_xlabel("eigenvalue of the correlation matrix")
    ax.set_ylabel("index (ascending)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ratio = ev_loo / ev_all[None, :]
    for k in range(n_curve):
        ax.plot(index, ratio[k], color="C0", lw=0.5, alpha=0.4)
    ax.plot(index, ratio.min(axis=0), color="C1", lw=1.0, ls="--",
            label="leave-one-out min / max")
    ax.plot(index, ratio.max(axis=0), color="C1", lw=1.0, ls="--")
    ax.axhline(1.0, color="k", lw=1.5)
    ax.axvline(n_below + 0.5, color="r", ls="--", lw=1,
               label="modes below svdcut")
    ax.set_yscale("log")
    ax.set_xlabel("index (ascending)")
    ax.set_ylabel("eigenvalue / full-sample eigenvalue")
    ax.legend(fontsize=8)

    fig.suptitle(f"{operator}  {frame}  tgf{tgf}  pf{pf}  q{q}   "
                 f"{n_pt} points", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    name = (f"correig_{operator}_{frame}_tgf{tgf}"
            f"_pf{pf[0]}_{pf[1]}_{pf[2]}_q{q[0]}_{q[1]}_{q[2]}.png")
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name, n_below


def run_one(tgf, pf, q):
    samples = build_samples(tgf, pf, q)
    ev_loo, ev_all, trace, n = leave_one_out_spectra(samples)
    name, n_below = plot_spectra(ev_loo, ev_all, n, tgf, pf, q)
    spread = ev_loo.max(axis=0) / ev_loo.min(axis=0)
    return (f"tgf{tgf} pf{pf} q{q}: {samples.shape[1]} points, {n} measurements, "
            f"trace {trace:.4f}, eigenvalues {ev_all.min():.3e} .. "
            f"{ev_all.max():.3e}, ratio {ev_all.min() / ev_all.max():.2e}, "
            f"{n_below} modes below svdcut, "
            f"largest leave-one-out spread {spread.max():.3f}x -> {name}")


if __name__ == "__main__":
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(tgf, pf, q) for tgf in tgf_list for pf in pf_list
             for q in q_list]
    print(f"{len(tasks)} spectra, N_WORKERS = {N_WORKERS}, "
          f"LOO_STRIDE = {LOO_STRIDE}", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_one, *t) for t in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    print("finished", flush=True)
