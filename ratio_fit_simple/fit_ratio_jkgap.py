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
operator = "TXTXpTYTYm2XYXY"

INPUT_RATIO = (SCRIPT_DIR.parent / "ratio_production_simple"
               / f"ratio_jk_{frame}")
TWOPT_FIT = SCRIPT_DIR.parent / "2PT_simple" / "twopt_fit_results.h5"
TWOPT_TAG = "nstate3_t4-15_svd1e-12"     # must match `tag` in 2pt_fit.py
OUTPUT_DIR = SCRIPT_DIR / f"bare_matrix_element_{frame}"
PLOT_DIR = SCRIPT_DIR / "ratio_fit_plots"

tgf_list = [20,25,30,35,40]
pf_list = [(0, 0, pz) for pz in range(0, 7)]
q_list = [(0, 0, 0)]

w_fit_list = list(range(0, 10))
tsep_fit_list = [4,5,6, 7, 8, 9, 10]
tau_skip = 1
w_plot_list = [0, 2, 4, 6, 8]

DE_WIDTH_FACTOR = 5.0          
A_PRIOR_WIDTH = 2
M00_PRIOR_WIDTH = 2.0
SVDCUT = 1e-7
MAXIT = 20000
N_WORKERS = 30

tag = (f"tsep{tsep_fit_list[0]}-{tsep_fit_list[-1]}_svd{SVDCUT:.0e}"
       f"_dEf{DE_WIDTH_FACTOR:g}".replace(".", "p"))


def jk(values, jack_axis=0):
    """Jackknife mean and error over jack_axis."""
    v = np.moveaxis(values, jack_axis, 0)
    n = v.shape[0]
    mean = v.mean(axis=0)
    return mean, np.sqrt((n - 1.0) / n * np.sum((v - mean) ** 2, axis=0))

def ratio_model(x, p):
    """input data-poins x = (tsep,tau,w) and prior; output value of that data-point"""
    w, tau, tsep_tau = x["w_index"], x["tau"], x["tsep"] - x["tau"]
    dEi = p["dEi"][w]
    dEf = p["dEf"][w] if "log(dEf)" in p else dEi
    Ai = p["Ai"][w]
    Af = p["Af"][w] if "Af" in p else Ai
    return (p["M00"][w] + Ai * gv.exp(-dEi * tau) + Af * gv.exp(-dEf * tsep_tau)
            + p["Afi"][w] * gv.exp(-dEi * tau - dEf * tsep_tau))


def make_prior(n_w, forward, gap_i=0.6,width_i=1,gap_f=0.6, width_f=1):
   
    """n_w : the number of z's
       forward : controls if we have Af
       gap : central value of dE from two-point fit
       width :  width of dE from two-point fit
    """


    prior = gv.BufferDict()
    prior["M00"] = gv.gvar([0.0] * n_w, [M00_PRIOR_WIDTH] * n_w) 
    prior["Ai"]  = gv.gvar([0.0] * n_w, [A_PRIOR_WIDTH] * n_w)
    prior["Afi"] = gv.gvar([0.0] * n_w, [A_PRIOR_WIDTH] * n_w)
    prior["log(dEi)"] = gv.log(gv.gvar([gap_i] * n_w, [width_i*DE_WIDTH_FACTOR] * n_w))
    if not forward:
        prior["Af"] = gv.gvar([0.0] * n_w, [A_PRIOR_WIDTH] * n_w)
        prior["log(dEf)"] = gv.log(gv.gvar([gap_f] * n_w, [width_f*DE_WIDTH_FACTOR] * n_w))
    return prior


def fit_one_point(tgf, pf, q):
    label = f"tgf{tgf} pf{pf} q{q}"
    name = (f"ratio_{frame}_tgf{tgf}_pf{pf[0]}_{pf[1]}_{pf[2]}"
            f"_q{q[0]}_{q[1]}_{q[2]}.h5")
    with h5py.File(INPUT_RATIO / name, "r") as f:
        ratio_jk = f["ratio_jk"][:] #shape: jk=799,op=4,w=10,tsep=7,tau=11
        w_all = list(f["w_list"][:])
        tsep_all = list(f["tsep_list"][:])
        operator_str_list = [s.decode() for s in f["operator_names"][:]]
        iop = operator_str_list.index(operator)  #get the index of the needed operator

    w_index, tau_all, T_all, vector_data = [], [], [], []
    
    #The following loops build the data-points to be used
    for iw_fit, w in enumerate(w_fit_list):
        for tsep in tsep_fit_list:
            for tau in range(tau_skip, tsep - tau_skip + 1):
                w_index.append(iw_fit)
                tau_all.append(float(tau))
                T_all.append(float(tsep))
                vector_data.append(ratio_jk[:, iop, w_all.index(w),tsep_all.index(tsep), tau].real)
    
    #x is the dictionary of coordinates of used data-points 
    x = {"w_index": np.array(w_index), "tau": np.array(tau_all),"tsep": np.array(T_all)}
    
    samples = np.stack(vector_data, axis=1) #(jackknife_index,data_point_index)
    n_jk, n_pt = samples.shape
    n_w = len(w_fit_list)
    
    # gap priors from the two-point fit at pf and at pi = pf + q, per jackknife sample
    forward = all(c == 0 for c in q)
    pi = tuple(int(pf[k] + q[k]) for k in range(3))
    with h5py.File(TWOPT_FIT, "r") as f:
        g = f[TWOPT_TAG]
        gap_f_jk = g[f"p{pf[0]}_{pf[1]}_{pf[2]}"]["dE_jk"][:, 1]
        if forward:
            gap_i_jk = gap_f_jk
        else:
            gap_i_jk = g[f"p{pi[0]}_{pi[1]}_{pi[2]}"]["dE_jk"][:, 1]
    gap_f_mean, gap_f_err = jk(gap_f_jk)
    gap_i_mean, gap_i_err = jk(gap_i_jk)
    if len(gap_f_jk) != n_jk:
        print(f"  WARNING {label}: 2pt has {len(gap_f_jk)} samples, "
              f"ratio has {n_jk}", flush=True)
    print(f"{label}: {n_pt} points, {n_jk} samples, 2pt gap "
          f"pi{pi} {gap_i_mean:.3f} +- {gap_i_err:.3f}, "
          f"pf{tuple(pf)} {gap_f_mean:.3f} +- {gap_f_err:.3f}, "
          f"prior widths {DE_WIDTH_FACTOR * gap_i_err:.3f} / "
          f"{DE_WIDTH_FACTOR * gap_f_err:.3f}", flush=True)

    # prepare the central value of data points and covariance matrix
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False, ddof=1) * (n_jk - 1.0) ** 2 / n_jk #with ddof=1, already a n_jk-1 factor is in the denominator

    central_fit = lsqfit.nonlinear_fit(
        data=(x, gv.gvar(mean, cov)), fcn=ratio_model, svdcut=SVDCUT,
        maxit=MAXIT, prior=make_prior(n_w,  forward, gap_i_mean, gap_i_err,gap_f_mean,gap_f_err))
    
    if getattr(central_fit, "svdn", 0):
        print(f"  WARNING {label}: svdcut modified {central_fit.svdn} of {n_pt} "
                f"modes", flush=True)
    p0 = {k: central_fit.pmean[k] for k in central_fit.prior} # build a dictionary of central values of fit parameters

    names = ["M00", "Ai", "Afi", "dEi"] + ([] if forward else ["Af", "dEf"])
    params = {n: np.zeros((n_jk, n_w)) for n in names}
    chi2dof, Q = np.zeros(n_jk), np.zeros(n_jk)
    for i in range(n_jk):
        jackknife_fit = lsqfit.nonlinear_fit(
            data=(x, gv.gvar(samples[i], cov)), fcn=ratio_model, p0=p0,
            svdcut=SVDCUT, maxit=MAXIT,
            prior=make_prior(n_w, forward, gap_i_jk[i], gap_i_err,gap_f_jk[i], gap_f_err))
        for n in names:
            params[n][i] = gv.mean(jackknife_fit.p[n]) #gv.mean(array of gvar objects) = array of central values
        chi2dof[i], Q[i] = jackknife_fit.chi2 / jackknife_fit.dof, jackknife_fit.Q

    M00_mean, M00_err = jk(params["M00"])
    out = OUTPUT_DIR / (f"bareM_{operator}_{frame}_tgf{tgf}"
                        f"_pf{pf[0]}_{pf[1]}_{pf[2]}"
                        f"_q{q[0]}_{q[1]}_{q[2]}_{tag}.h5")
    with h5py.File(out, "w") as f:
        f.create_dataset("bare_matrix_element_jk", data=params["M00"])
        for n in names:
            f.create_dataset(f"{n}_jk", data=params[n])
        f.create_dataset("M00_jk_mean", data=M00_mean)
        f.create_dataset("M00_jk_err", data=M00_err)
        f.create_dataset("w_list", data=np.array(w_fit_list, dtype=np.int64))
        f.create_dataset("gap_i_prior_center_jk", data=gap_i_jk)
        f.create_dataset("gap_f_prior_center_jk", data=gap_f_jk)
        f.attrs["dim_bare_matrix_element_jk"] = "jk,w_index"
        f.attrs["operator"] = operator
        f.attrs["twopt_tag"] = TWOPT_TAG
        f.attrs["frame"] = frame
        f.attrs["tgf"] = tgf
        f.attrs["pf"] = np.array(pf, dtype=np.int64)
        f.attrs["q"] = np.array(q, dtype=np.int64)
        f.attrs["gap_prior"] = (f"2pt E1-E0 per sample, width "
                                f"{DE_WIDTH_FACTOR} x its jackknife error")
        f.attrs["gap_i_prior_width"] = DE_WIDTH_FACTOR * gap_i_err
        f.attrs["gap_f_prior_width"] = DE_WIDTH_FACTOR * gap_f_err
        f.attrs["pi"] = np.array(pi, dtype=np.int64)
        f.attrs["tsep_fit_list"] = np.array(tsep_fit_list, dtype=np.int64)
        f.attrs["tau_skip"] = tau_skip
        f.attrs["svdcut"] = SVDCUT
        f.attrs["svdn"] = int(getattr(central_fit, "svdn", 0))
        f.attrs["chi2dof"] = float(chi2dof.mean())
        f.attrs["Q"] = float(Q.mean())

    plot_fit(x, samples, params, chi2dof.mean(), Q.mean(), tgf, pf, q)
    print(f"  saved {out.name}  <chi2/dof> {chi2dof.mean():.2f}", flush=True)


def plot_fit(x, samples, params, chi2dof, q_value, tgf, pf, q):
    data_mean, data_err = jk(samples)
    m00_mean, m00_err = jk(params["M00"])
    ws = [w for w in w_plot_list if w in w_fit_list]
    fig, axes = plt.subplots(1, len(ws), figsize=(4.2 * len(ws), 4.6),
                             squeeze=False)
    for icol, w in enumerate(ws):
        ax, iw = axes[0][icol], w_fit_list.index(w)
        for itsep, tsep in enumerate(tsep_fit_list):
            sel = (x["w_index"] == iw) & (x["tsep"] == tsep)
            ax.errorbar(x["tau"][sel] - 0.5 * tsep, data_mean[sel],
                        yerr=data_err[sel], marker="o", ls="none", ms=4,
                        capsize=3, color=f"C{itsep}", label=f"tsep={tsep}")
            # the band: the fitted curve on every jackknife sample
            tau = np.linspace(tau_skip, tsep - tau_skip, 60)[None, :]
            ai = params["Ai"][:, iw][:, None]
            de = params["dEi"][:, iw][:, None]
            af = params.get("Af", params["Ai"])[:, iw][:, None]
            df = params.get("dEf", params["dEi"])[:, iw][:, None]
            curve = (params["M00"][:, iw][:, None] + ai * np.exp(-de * tau)
                     + af * np.exp(-df * (tsep - tau))
                     + params["Afi"][:, iw][:, None]
                     * np.exp(-de * tau - df * (tsep - tau)))
            b_mean, b_err = jk(curve)
            ax.fill_between(tau[0] - 0.5 * tsep, b_mean - b_err,
                            b_mean + b_err, color=f"C{itsep}", alpha=0.25,
                            lw=0)
        ax.axhspan(m00_mean[iw] - m00_err[iw], m00_mean[iw] + m00_err[iw],
                   color="k", alpha=0.15)
        ax.set_title(f"w = {w}   M00 = {gv.gvar(m00_mean[iw], m00_err[iw])}",
                     fontsize=10)
        ax.set_xlabel(r"$\tau - t_{\rm sep}/2$")
    axes[0][0].set_ylabel("R")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"{operator}  pf={pf}  q={q}  flow={tgf * 0.1:.1f}  "
                 f"<chi2/dof>={chi2dof:.2f}  <Q>={q_value:.2f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / (f"fit_{operator}_{frame}_tgf{tgf}"
                            f"_pf{pf[0]}_{pf[1]}_{pf[2]}"
                            f"_q{q[0]}_{q[1]}_{q[2]}_{tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(tgf, pf, q) for tgf in tgf_list for pf in pf_list
             for q in q_list]
    print(f"{len(tasks)} fits, N_WORKERS = {N_WORKERS}", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(fit_one_point, *t) for t in tasks]
        for future in as_completed(futures):
            future.result()
    print("finished", flush=True)
