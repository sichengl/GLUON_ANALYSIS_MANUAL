import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gvar as gv
import h5py
import lsqfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

frame = "symmetric"

INPUT_RATIO = (SCRIPT_DIR.parent / "ratio_production_simple"
               / f"ratio_jk_{frame}")
OUTPUT_DIR = SCRIPT_DIR / f"bare_matrix_element_{frame}"
PLOT_DIR = SCRIPT_DIR / "ratio_fit_plots"

# which ratio files to fit
tgf_list = [ 30]
pf_list = [(0, 0, pz) for pz in range(0, 7)]
q_list = [(0, 0, 0)]
operator_list = ["TXTXpTYTYm2XYXY"]

# every Wilson line in this list is fitted SEPARATELY, each with its own
# M00, Ai, Af, Afi, dEi and dEf
w_fit_list = list(range(0, 10))
tsep_fit_list = [ 6, 7, 8, 9, 10]
tau_skip = 1                     # drop this many tau points at each end
w_plot_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# only the real part is fitted
M00_PRIOR_WIDTH = 5.0            # in units of max |R| of that Wilson line
A_PRIOR_WIDTH = 5.0              # same units
DE_PRIOR = gv.gvar(0.6, 3)       # excited-state gap, log-normal

SVDCUT = 1e-7
MAXIT = 20000

# labels that go into every output name, so a file states the fit range and
# the regulator it was produced with
tsep_tag = (
    f"tsep{tsep_fit_list[0]}-{tsep_fit_list[-1]}"
    if tsep_fit_list == list(range(tsep_fit_list[0], tsep_fit_list[-1] + 1))
    else "tsep" + "_".join(str(t) for t in tsep_fit_list)
)
svd_tag = f"svd{SVDCUT:.0e}_sepw"
N_WORKERS = int(os.environ.get("N_WORKERS", "8"))
PROGRESS_EVERY = 200


def ratio_model(x, p):
    """R(w, tsep, tau), all parameters carrying their own w index.

    In the forward case the prior omits Af and dEf: the initial and final
    states are identical, so one amplitude and one gap serve both ends.
    """
    w = x["w_index"]
    tau = x["tau"]
    t_minus_tau = x["T"] - x["tau"]
    dEi = p["dEi"][w]
    dEf = p["dEf"][w] if "dEf" in p else dEi
    Ai = p["Ai"][w]
    Af = p["Af"][w] if "Af" in p else Ai
    return (
        p["M00"][w]
        + Ai * gv.exp(-dEi * tau)
        + Af * gv.exp(-dEf * t_minus_tau)
        + p["Afi"][w] * gv.exp(-dEi * tau - dEf * t_minus_tau)
    )


def make_prior(n_w, scale, forward):
    zero = np.zeros(n_w)
    prior = gv.BufferDict()
    prior["M00"] = gv.gvar(zero, np.full(n_w, M00_PRIOR_WIDTH * scale))
    prior["Ai"] = gv.gvar(zero, np.full(n_w, A_PRIOR_WIDTH * scale))
    prior["Afi"] = gv.gvar(zero, np.full(n_w, A_PRIOR_WIDTH * scale))
    prior["log(dEi)"] = gv.log(gv.gvar(np.full(n_w, gv.mean(DE_PRIOR)),
                                       np.full(n_w, gv.sdev(DE_PRIOR))))
    if not forward:
        prior["Af"] = gv.gvar(zero, np.full(n_w, A_PRIOR_WIDTH * scale))
        prior["log(dEf)"] = gv.log(gv.gvar(np.full(n_w, gv.mean(DE_PRIOR)),
                                           np.full(n_w, gv.sdev(DE_PRIOR))))
    return prior


def jackknife_mean_err(values):
    """Mean and jackknife error over axis 0 of delete-one samples."""
    n = values.shape[0]
    mean = np.nanmean(values, axis=0)
    err = np.sqrt((n - 1.0) / n
                  * np.nansum((values - mean) ** 2, axis=0))
    return mean, err


def model_band(params_jk, iw, tsep, tau_fine):
    """The fitted curve on every jackknife sample, [jk, tau].

    Evaluated from each sample's own fitted parameters, so the band below
    is the jackknife spread of the curve -- the same estimator as the
    quoted M00 error, not lsqfit's propagated covariance.
    """
    m00 = params_jk["M00"][:, iw][:, None]
    ai = params_jk["Ai"][:, iw][:, None]
    afi = params_jk["Afi"][:, iw][:, None]
    dei = params_jk["dEi"][:, iw][:, None]
    af = params_jk["Af"][:, iw][:, None] if "Af" in params_jk else ai
    def_ = params_jk["dEf"][:, iw][:, None] if "dEf" in params_jk else dei
    tau = tau_fine[None, :]
    tmt = tsep - tau
    return (m00 + ai * np.exp(-dei * tau) + af * np.exp(-def_ * tmt)
            + afi * np.exp(-dei * tau - def_ * tmt))


def plot_fit(x, samples, params_jk, chi2dof, q_value, operator,
             tgf, pf, q):
    """Ratio data with the fitted band, one panel per plotted Wilson line."""
    mean, err = jackknife_mean_err(samples)
    m00_mean, m00_err = jackknife_mean_err(params_jk["M00"])

    ws = [w for w in w_plot_list if w in w_fit_list]
    fig, axes = plt.subplots(1, len(ws), figsize=(4.2 * len(ws), 4.6),
                             squeeze=False)
    for icol, w in enumerate(ws):
        ax = axes[0][icol]
        iw_fit = w_fit_list.index(w)
        for itsep, tsep in enumerate(tsep_fit_list):
            color = f"C{itsep}"
            sel = (x["w_index"] == iw_fit) & (x["T"] == tsep)
            ax.errorbar(x["tau"][sel] - 0.5 * tsep, mean[sel], yerr=err[sel],
                        marker="o", ls="none", capsize=3, ms=4, color=color,
                        label=f"tsep={tsep}")
            # the fitted band: jackknife spread of the curve over samples
            tau_fine = np.linspace(tau_skip, tsep - tau_skip, 60)
            b_mean, b_err = jackknife_mean_err(
                model_band(params_jk, iw_fit, float(tsep), tau_fine))
            ax.fill_between(tau_fine - 0.5 * tsep, b_mean - b_err,
                            b_mean + b_err, color=color, alpha=0.25, lw=0)
        ax.axhspan(m00_mean[iw_fit] - m00_err[iw_fit],
                   m00_mean[iw_fit] + m00_err[iw_fit], color="k", alpha=0.15)
        ax.axhline(m00_mean[iw_fit], color="k", ls="--", lw=1)
        ax.set_title(f"w = {w}   M00 = "
                     f"{gv.gvar(m00_mean[iw_fit], m00_err[iw_fit])}",
                     fontsize=10)
        ax.set_xlabel(r"$\tau - t_{\rm sep}/2$")
    axes[0][0].set_ylabel(r"$R$")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"{operator}   pf={tuple(int(v) for v in pf)}   "
                 f"q={tuple(int(v) for v in q)}   flow={tgf * 0.1:.1f}   "
                 f"<chi2/dof>={chi2dof:.2f}   <Q>={q_value:.2f}",
                 fontsize=13)
    fig.tight_layout()
    name = (f"fit_{operator}_{frame}_tgf{tgf}_pf{pf[0]}_{pf[1]}_{pf[2]}"
            f"_q{q[0]}_{q[1]}_{q[2]}_{tsep_tag}_{svd_tag}.png")
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def jackknife_covariance(samples):
    """Covariance of the mean from delete-one samples, [jk, npt].

    (n-1)/n sum_k (x_k - xbar)(x_k - xbar)^T; np.cov(ddof=1) is that sum
    divided by (n-1), hence the (n-1)^2/n factor.
    """
    n = samples.shape[0]
    cov = np.cov(samples, rowvar=False, ddof=1) * (n - 1.0) ** 2 / n
    return 0.5 * (cov + cov.T)


def build_data(path, operator):
    """Flatten (w, tsep, tau) of one ratio file into a fit data vector."""
    with h5py.File(path, "r") as f:
        ratio_jk = f["ratio_jk"][:]
        w_list = list(f["w_list"][:])
        tsep_list = list(f["tsep_list"][:])
        op_names = [s.decode() for s in f["operator_names"][:]]
        meta = {
            "tgf": int(f.attrs["tgf"]),
            "pf": np.array(f.attrs["pf"], dtype=np.int64),
            "q": np.array(f.attrs["q"], dtype=np.int64),
            "cfg_list": f["cfg_list"][:],
        }
    iop = op_names.index(operator)

    w_index, tau_all, T_all, columns = [], [], [], []
    for iw_fit, w in enumerate(w_fit_list):
        iw = w_list.index(w)
        for tsep in tsep_fit_list:
            itsep = tsep_list.index(tsep)
            for tau in range(tau_skip, int(tsep) - tau_skip + 1):
                block = ratio_jk[:, iop, iw, itsep, tau].real
                if not np.all(np.isfinite(block)):
                    continue
                w_index.append(iw_fit)
                tau_all.append(float(tau))
                T_all.append(float(tsep))
                columns.append(block)

    x = {"w_index": np.array(w_index, dtype=np.int64),
         "tau": np.array(tau_all), "T": np.array(T_all)}
    return x, np.stack(columns, axis=1), meta


def fit_one_point(operator, tgf, pf, q):
    name = (f"ratio_{frame}_tgf{tgf}_pf{pf[0]}_{pf[1]}_{pf[2]}"
            f"_q{q[0]}_{q[1]}_{q[2]}.h5")
    label = f"{operator} tgf{tgf} pf{pf} q{q}"
    x, samples, meta = build_data(INPUT_RATIO / name, operator)

    n_jk, n_pt = samples.shape
    n_w = len(w_fit_list)
    forward = bool(np.all(meta["q"] == 0))
    print(f"{label}: {n_pt} points over {n_w} Wilson lines "
          f"({n_pt // n_w} per fit), {n_jk} samples", flush=True)
    if n_pt // n_w > n_jk // 4:
        print(f"  WARNING {label}: {n_pt // n_w} points per fit from "
              f"{n_jk} samples -- the covariance is poorly determined",
              flush=True)

    names = ["M00", "Ai", "Afi", "dEi"] + ([] if forward else ["Af", "dEf"])
    params_jk = {name: np.full((n_jk, n_w), np.nan) for name in names}
    chi2dof_jk = np.full((n_jk, n_w), np.nan)
    Q_jk = np.full((n_jk, n_w), np.nan)
    status = np.zeros((n_jk, n_w), dtype=np.int64)
    svdn_w = np.zeros(n_w, dtype=np.int64)

    # every Wilson line is its own fit: n_pt/n_w points against 799 samples,
    # so each covariance is estimable instead of one joint 420-point block
    for iw_fit, w in enumerate(w_fit_list):
        sel = x["w_index"] == iw_fit
        x_w = {"w_index": np.zeros(int(sel.sum()), dtype=np.int64),
               "tau": x["tau"][sel], "T": x["T"][sel]}
        samples_w = samples[:, sel]
        mean_w = samples_w.mean(axis=0)
        cov_w = jackknife_covariance(samples_w)
        # the amplitude priors scale to THIS w's data, not the whole block
        prior = make_prior(1, float(np.max(np.abs(mean_w))), forward)

        central = lsqfit.nonlinear_fit(data=(x_w, gv.gvar(mean_w, cov_w)),
                                       fcn=ratio_model, prior=prior,
                                       svdcut=SVDCUT, maxit=MAXIT)
        svdn_w[iw_fit] = int(getattr(central, "svdn", 0))
        if svdn_w[iw_fit]:
            print(f"  WARNING {label} w={w}: svdcut modified "
                  f"{svdn_w[iw_fit]} of {sel.sum()} modes", flush=True)
        p0 = {k: central.pmean[k] for k in prior}

        for ijk in range(n_jk):
            try:
                fit = lsqfit.nonlinear_fit(
                    data=(x_w, gv.gvar(samples_w[ijk], cov_w)),
                    fcn=ratio_model, prior=prior, svdcut=SVDCUT,
                    maxit=MAXIT, p0=p0)
                for name in names:
                    params_jk[name][ijk, iw_fit] = gv.mean(fit.p[name])[0]
                chi2dof_jk[ijk, iw_fit] = fit.chi2 / fit.dof
                Q_jk[ijk, iw_fit] = fit.Q
            except Exception as err:
                status[ijk, iw_fit] = 1
                print(f"  fit failed {label} w={w} sample {ijk}: {err}",
                      flush=True)
        print(f"  {label} w={w}: {int(sel.sum())} points, "
              f"<chi2/dof> {np.nanmean(chi2dof_jk[:, iw_fit]):.2f}",
              flush=True)

    M00_mean, M00_err = jackknife_mean_err(params_jk["M00"])

    out = OUTPUT_DIR / (f"bareM_{operator}_{frame}_tgf{tgf}"
                        f"_pf{pf[0]}_{pf[1]}_{pf[2]}"
                        f"_q{q[0]}_{q[1]}_{q[2]}"
                        f"_{tsep_tag}_{svd_tag}.h5")
    with h5py.File(out, "w") as f:
        f.create_dataset("bare_matrix_element_jk", data=params_jk["M00"])
        for name in names:
            f.create_dataset(f"{name}_jk", data=params_jk[name])
        f.create_dataset("chi2dof_jk", data=chi2dof_jk)
        f.create_dataset("Q_jk", data=Q_jk)
        f.create_dataset("status", data=status)
        f.create_dataset("M00_jk_mean", data=M00_mean)
        f.create_dataset("M00_jk_err", data=M00_err)
        f.create_dataset("w_list", data=np.array(w_fit_list, dtype=np.int64))
        f.create_dataset("tsep_fit_list",
                         data=np.array(tsep_fit_list, dtype=np.int64))
        f.create_dataset("cfg_list", data=meta["cfg_list"])
        f.create_dataset("fit_w_index", data=x["w_index"])
        f.create_dataset("fit_T", data=x["T"].astype(np.int64))
        f.create_dataset("fit_tau", data=x["tau"].astype(np.int64))
        f.create_dataset("svdn_per_w", data=svdn_w)
        f.attrs["dim_bare_matrix_element_jk"] = "jk,w_index"
        f.attrs["operator"] = operator
        f.attrs["frame"] = frame
        f.attrs["forward"] = forward
        f.attrs["tgf"] = tgf
        f.attrs["pf"] = np.array(pf, dtype=np.int64)
        f.attrs["q"] = np.array(q, dtype=np.int64)
        f.attrs["fit_part"] = "real"
        f.attrs["fit_model"] = (
            "R = M00[w] + Ai[w] exp(-dEi[w] tau) + Af[w] exp(-dEf[w] (T-tau))"
            " + Afi[w] exp(-dEi[w] tau - dEf[w] (T-tau))")
        f.attrs["fit_strategy"] = (
            "per-w dEi and dEf; Ai=Af and dEi=dEf when q=0; one fit per "
            "jackknife sample, frozen covariance, seeded from the central "
            "fit; EVERY WILSON LINE FITTED SEPARATELY")
        f.attrs["tau_skip"] = tau_skip
        f.attrs["svdcut"] = SVDCUT
        f.attrs["n_points"] = n_pt
        f.attrs["n_points_per_w"] = n_pt // n_w
        f.attrs["chi2dof_jk_mean"] = float(np.nanmean(chi2dof_jk))
        f.attrs["Q_jk_mean"] = float(np.nanmean(Q_jk))
        f.attrs["n_failed"] = int(status.sum())
    plot_fit(x, samples, params_jk, float(np.nanmean(chi2dof_jk)),
             float(np.nanmean(Q_jk)), operator, tgf, pf, q)
    print(f"  saved {out.name}  (<chi2/dof> "
          f"{np.nanmean(chi2dof_jk):.2f}, <Q> {np.nanmean(Q_jk):.2f}, "
          f"{int(status.sum())} failed)", flush=True)
    return out


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(op, tgf, pf, q)
             for op in operator_list for tgf in tgf_list
             for pf in pf_list for q in q_list]
    missing = [t for t in tasks
               if not (INPUT_RATIO / (f"ratio_{frame}_tgf{t[1]}"
                                      f"_pf{t[2][0]}_{t[2][1]}_{t[2][2]}"
                                      f"_q{t[3][0]}_{t[3][1]}_{t[3][2]}.h5")
                       ).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(tasks)} ratio files are "
                         f"missing, e.g. {missing[0]}")
    print(f"{len(tasks)} fits, N_WORKERS = {N_WORKERS}", flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(fit_one_point, *t) for t in tasks]
        for future in as_completed(futures):
            future.result()
    print("finished", flush=True)
