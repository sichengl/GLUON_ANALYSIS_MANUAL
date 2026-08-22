from pathlib import Path
import h5py
import numpy as np
import gvar as gv
import lsqfit

PT2_PATH = Path("./twopt_source_averaged.h5")
FIT_PATH = Path("./twopt_fit_results.h5")
Gs, Gt = 32, 96

MOM_LIST = [(0, 0, pz) for pz in range(7)]
TMIN, TMAX = 4, 15
TSEP_LIST = [4, 5, 6]
N_STATES = 3
SVDCUT = 1e-12

M0 = 0.14
E0_WIDTH = 0.10
DE1_PRIOR = gv.gvar("0.60(60)")
DE2_PRIOR = gv.gvar("0.60(60)")
C_CENTER = 0.0
C_WIDTH = 1e4

PROGRESS_EVERY = 100


def twopt_model(t, p):
    corr = 0.0
    e = 0.0
    for n in range(len(p["dE"])):
        e = e + p["dE"][n]
        corr = corr + p["c"][n] * (gv.exp(-e * t) + gv.exp(-e * (Gt - t)))
    return corr


def make_prior(n_states, mom):
    p_lat = 2.0 * np.pi * np.array(mom, dtype=float) / Gs
    e0 = 2.0 * np.arcsinh(np.sqrt(np.sinh(M0 / 2.0) ** 2
                                  + np.sum(np.sin(p_lat / 2.0) ** 2)))
    gaps = [DE1_PRIOR, DE2_PRIOR][:n_states - 1]
    dE = gv.gvar([e0] + [gv.mean(g) for g in gaps],
                 [E0_WIDTH] + [gv.sdev(g) for g in gaps])
    c = gv.gvar([C_CENTER] * n_states, [C_WIDTH] * n_states)
    return {"c": c, "log(dE)": gv.log(dE)}


def cov_of_mean(samples):
    return np.cov(samples, rowvar=False, ddof=1) / samples.shape[0]


with h5py.File(PT2_PATH, "r") as f:
    name = "correlator_cfg" if "correlator_cfg" in f else "pion_45"
    pion = f[name][:]
    if "momentum_list" in f:
        moms = f["momentum_list"][:]
    else:
        moms = f[name].attrs["momentums"]

mom_to_idx = {tuple(p): i for i, p in enumerate(moms.tolist())}
t = np.arange(TMIN, TMAX + 1)
n_mom = len(MOM_LIST)
n_jk = pion.shape[0]

E_central = np.zeros((n_mom, N_STATES))
E_central_sdev = np.zeros((n_mom, N_STATES))
C_central = np.zeros((n_mom, N_STATES))
C_central_sdev = np.zeros((n_mom, N_STATES))
chi2dof_central = np.zeros(n_mom)
Q_central = np.zeros(n_mom)

E_jk = np.zeros((n_mom, n_jk, N_STATES))
C_jk = np.zeros((n_mom, n_jk, N_STATES))
chi2dof_jk = np.zeros((n_mom, n_jk))
Q_jk = np.zeros((n_mom, n_jk))

for imom, pf in enumerate(MOM_LIST):
    ipf = mom_to_idx[pf]
    C2_pf = pion[..., ipf, :].real

    samples = C2_pf[:, TMIN:TMAX + 1]
    mean = samples.mean(axis=0)
    cov = cov_of_mean(samples)
    jk = (samples.sum(axis=0) - samples) / (n_jk - 1)

    prior = make_prior(N_STATES, pf)

    fit = lsqfit.nonlinear_fit(data=(t, gv.gvar(mean, cov)), fcn=twopt_model,
                               prior=prior, svdcut=SVDCUT)
    E = np.cumsum(fit.p["dE"])
    E_central[imom] = gv.mean(E)
    E_central_sdev[imom] = gv.sdev(E)
    C_central[imom] = gv.mean(fit.p["c"])
    C_central_sdev[imom] = gv.sdev(fit.p["c"])
    chi2dof_central[imom] = fit.chi2 / fit.dof
    Q_central[imom] = fit.Q

    print(f"\npf = {pf}   n_states = {N_STATES}   t = {TMIN}..{TMAX}")
    print(fit.format(maxline=True))

    ratio = (fit.p["c"][1] / fit.p["c"][0]
             * gv.exp(-fit.p["dE"][1] * np.array(TSEP_LIST)))
    for tsep, r in zip(TSEP_LIST, ratio):
        print(f"  C1 e^(-E1 tsep) / C0 e^(-E0 tsep) at tsep = {tsep}: {r}")

    p0 = {k: fit.pmean[k] for k in prior}
    for i in range(n_jk):
        fit_jk = lsqfit.nonlinear_fit(data=(t, gv.gvar(jk[i], cov)),
                                      fcn=twopt_model, prior=prior,
                                      svdcut=SVDCUT, p0=p0)
        E_jk[imom, i] = np.cumsum(gv.mean(fit_jk.p["dE"]))
        C_jk[imom, i] = gv.mean(fit_jk.p["c"])
        chi2dof_jk[imom, i] = fit_jk.chi2 / fit_jk.dof
        Q_jk[imom, i] = fit_jk.Q
        if PROGRESS_EVERY and (i + 1) % PROGRESS_EVERY == 0:
            print(f"  jackknife {i + 1}/{n_jk}", flush=True)

    E_jk_err = np.sqrt((n_jk - 1) / n_jk
                       * np.sum((E_jk[imom] - E_jk[imom].mean(axis=0)) ** 2,
                                axis=0))
    print(f"E0 central = {E_central[imom, 0]:.6f} "
          f"+- {E_central_sdev[imom, 0]:.6f} (fit)   "
          f"+- {E_jk_err[0]:.6f} (jackknife)   "
          f"chi2/dof = {chi2dof_central[imom]:.2f}")

with h5py.File(FIT_PATH, "w") as f:
    f.create_dataset("momentum_list", data=np.array(MOM_LIST, dtype=np.int64))
    f.create_dataset("E_central", data=E_central)
    f.create_dataset("E_central_sdev", data=E_central_sdev)
    f.create_dataset("C_central", data=C_central)
    f.create_dataset("C_central_sdev", data=C_central_sdev)
    f.create_dataset("chi2dof_central", data=chi2dof_central)
    f.create_dataset("Q_central", data=Q_central)
    f.create_dataset("E_jk", data=E_jk)
    f.create_dataset("C_jk", data=C_jk)
    f.create_dataset("chi2dof_jk", data=chi2dof_jk)
    f.create_dataset("Q_jk", data=Q_jk)
    f.attrs["tmin"] = TMIN
    f.attrs["tmax"] = TMAX
    f.attrs["n_states"] = N_STATES
    f.attrs["svdcut"] = SVDCUT
    f.attrs["n_jk"] = n_jk

print(f"\nwritten: {FIT_PATH}", flush=True)
