from pathlib import Path
import h5py
import numpy as np
import gvar as gv
import lsqfit

"""Reads twopt_source_averaged.h5 which has all the source-averaged 2pts"""
"""Shape of input 2pt: [n_cfg,n_mom,tsink]"""

PT2_PATH = Path("./twopt_source_averaged.h5")
FIT_PATH = Path("./twopt_fit_results.h5")
Gs, Gt = 32, 96

MOM_LIST = [(0, 0, pz) for pz in range(7)]
TMIN, TMAX = 4, 14
TSEP_LIST = [4, 5, 6,7]  #(c1*e^-E1tsep) / (c0*e^-E0tsep) will be printed out
N_STATES = 3
SVDCUT = 1e-12
FOLD = True     # average C(t) with C(Gt - t)

MASS = 0.141
E0_WIDTH = 0.10                  # default width, used when mom is not in E0_PRIOR

# per-momentum overrides: mom -> gv.gvar(center, width).
# anything not listed uses the lattice dispersion relation with E0_WIDTH.
E0_PRIOR = {
}

DE1_PRIOR = gv.gvar("0.60(60)")
DE2_PRIOR = gv.gvar("0.60(60)")
C_CENTER = 0.0
C_WIDTH = 100

PROGRESS_EVERY = 100

tag = (f"nstate{N_STATES}_t{TMIN}-{TMAX}_svd{SVDCUT:.0e}"
       + ("_fold" if FOLD else ""))


def twopt_model(t, p):
    corr = 0.0
    e = 0.0
    for n in range(len(p["dE"])):
        e = e + p["dE"][n]
        corr = corr + p["c"][n] * (gv.exp(-e * t) + gv.exp(-e * (Gt - t)))
    return corr


def make_prior(n_states, mom):
    if mom in E0_PRIOR:
        e0, e0_width = gv.mean(E0_PRIOR[mom]), gv.sdev(E0_PRIOR[mom])
    else:
        p_lat = 2.0 * np.pi * np.array(mom, dtype=float) / Gs
        e0 = 2.0 * np.arcsinh(np.sqrt(np.sinh(MASS / 2.0) ** 2
                                      + np.sum(np.sin(p_lat / 2.0) ** 2)))
        e0_width = E0_WIDTH
    gaps = [DE1_PRIOR, DE2_PRIOR][:n_states - 1]

    #The 0th gap is E0
    dE = gv.gvar([e0] + [gv.mean(g) for g in gaps],
                 [e0_width] + [gv.sdev(g) for g in gaps])
    c = gv.gvar([C_CENTER] * n_states, [C_WIDTH] * n_states)
    return {"c": c, "log(dE)": gv.log(dE)}


def cov_of_mean(samples):
    return np.cov(samples, rowvar=False, ddof=1) / samples.shape[0]


def jk_stats(values):
    """Jackknife mean and error over axis 0 of delete-one samples."""
    n = values.shape[0]
    mean = values.mean(axis=0)
    return mean, np.sqrt((n - 1.0) / n * np.sum((values - mean) ** 2, axis=0))


with h5py.File(PT2_PATH, "r") as f:
    name = "correlator_cfg" if "correlator_cfg" in f else "pion_45"
    pion = f[name][:]
    if "momentum_list" in f:
        moms = f["momentum_list"][:]
    else:
        moms = f[name].attrs["momentums"]

mom_to_idx = {tuple(p): i for i, p in enumerate(moms.tolist())}
t = np.arange(TMIN, TMAX + 1)
n_jk = pion.shape[0]


for pf in MOM_LIST:
    ipf = mom_to_idx[pf]
    C2_pf = pion[..., ipf, :].real
    if FOLD:
        raw = C2_pf
        C2_pf = 0.5 * (C2_pf + np.roll(C2_pf[:, ::-1], 1, axis=1))
        a = raw[:, TMIN:TMAX + 1]
        b = raw[:, Gt - TMAX:Gt - TMIN + 1][:, ::-1]
        fold_err = np.std(C2_pf[:, TMIN:TMAX + 1], axis=0, ddof=1)
        raw_err = np.std(a, axis=0, ddof=1)
        print(f"\n  fold {pf}: relative error gain "
              f"{np.mean(raw_err / fold_err):.3f}x (sqrt(2) = 1.414),  "
              f"max |C(t)-C(Gt-t)| / sigma = "
              f"{np.max(np.abs(a.mean(0) - b.mean(0)) / (raw_err / np.sqrt(a.shape[0]))):.2f}",
              flush=True)
    samples = C2_pf[:, TMIN:TMAX + 1]
    mean = samples.mean(axis=0)
    cov = cov_of_mean(samples)
    jk = (samples.sum(axis=0) - samples) / (n_jk - 1)

    prior = make_prior(N_STATES, pf)

    central_fit = lsqfit.nonlinear_fit(data=(t, gv.gvar(mean, cov)),
                                       fcn=twopt_model, prior=prior,
                                       svdcut=SVDCUT)
    E = np.cumsum(central_fit.p["dE"])  #[E0, dE1, dE2] -> [E0, E1, E2]
    E_central = gv.mean(E)
    E_central_sdev = gv.sdev(E)
    C_central = gv.mean(central_fit.p["c"])
    C_central_sdev = gv.sdev(central_fit.p["c"])
    chi2dof_central = central_fit.chi2 / central_fit.dof
    Q_central = central_fit.Q

    print(f"\n====== CENTRAL FIT RESULT   pf = {pf} ======")
    print(f"  central: n_states = {N_STATES}   t = {TMIN}..{TMAX}")
    for n in range(N_STATES):
        print(f"  central: E{n} = {E[n]:<14} c{n} = {central_fit.p['c'][n]}")
    print(f"  central: chi2/dof = {chi2dof_central:.2f}   "
          f"Q = {Q_central:.3f}   svdn = {getattr(central_fit, 'svdn', 0)}")
    print(central_fit.format(maxline=True))

    ratio = (central_fit.p["c"][1] / central_fit.p["c"][0]
             * gv.exp(-central_fit.p["dE"][1] * np.array(TSEP_LIST)))
    for tsep, r in zip(TSEP_LIST, ratio):
        print(f"  central: C1 e^(-E1 tsep) / C0 e^(-E0 tsep) "
              f"at tsep = {tsep}: {r}")

    E_jk = np.zeros((n_jk, N_STATES))
    dE_jk = np.zeros((n_jk, N_STATES))
    C_jk = np.zeros((n_jk, N_STATES))
    chi2dof_jk = np.zeros(n_jk)
    Q_jk = np.zeros(n_jk)

    p0 = {k: central_fit.pmean[k] for k in prior}
    for i in range(n_jk):
        fit_jk = lsqfit.nonlinear_fit(data=(t, gv.gvar(jk[i], cov)),
                                      fcn=twopt_model, prior=prior,
                                      svdcut=SVDCUT, p0=p0)
        dE_jk[i] = gv.mean(fit_jk.p["dE"])
        E_jk[i] = np.cumsum(dE_jk[i])
        C_jk[i] = gv.mean(fit_jk.p["c"])
        chi2dof_jk[i] = fit_jk.chi2 / fit_jk.dof
        Q_jk[i] = fit_jk.Q
        if PROGRESS_EVERY and (i + 1) % PROGRESS_EVERY == 0:
            print(f"  jackknife {i + 1}/{n_jk}", flush=True)

    dE_jk_mean, dE_jk_err = jk_stats(dE_jk)
    E_jk_mean, E_jk_err = jk_stats(E_jk)
    C_jk_mean, C_jk_err = jk_stats(C_jk)
    chi2_jk_mean, chi2_jk_err = jk_stats(chi2dof_jk)
    Q_jk_mean, Q_jk_err = jk_stats(Q_jk)

    print(f"\n====== JACKKNIFE RESULT   pf = {pf} ======")
    print(f"  E0 : jk_mean = {dE_jk_mean[0]:.6f}   jk_err = {dE_jk_err[0]:.6f}")
    for n in range(1, N_STATES):
        print(f"  dE{n}: jk_mean = {dE_jk_mean[n]:.6f}   "
              f"jk_err = {dE_jk_err[n]:.6f}")
    for n in range(N_STATES):
        print(f"  c{n} : jk_mean = {C_jk_mean[n]:.6e}   "
              f"jk_err = {C_jk_err[n]:.6e}")
    print(f"  chi2/dof: jk_mean = {chi2_jk_mean:.3f}   "
          f"jk_err = {chi2_jk_err:.3f}")
    print(f"  Q       : jk_mean = {Q_jk_mean:.3f}   jk_err = {Q_jk_err:.3f}")

    with h5py.File(FIT_PATH, "a") as f:
        g = f.require_group(tag)  #get a group by fit parameter
        g.attrs["tmin"] = TMIN
        g.attrs["tmax"] = TMAX
        g.attrs["n_states"] = N_STATES
        g.attrs["svdcut"] = SVDCUT
        g.attrs["n_jk"] = n_jk
        g.attrs["fold"] = FOLD
        mom_name = f"p{pf[0]}_{pf[1]}_{pf[2]}"
        if mom_name in g:
            del g[mom_name]
        gm = g.create_group(mom_name)  #get a subgroup by momentum
        gm.create_dataset("momentum", data=np.array(pf, dtype=np.int64))
        gm.create_dataset("E_central", data=E_central)
        gm.create_dataset("E_central_sdev", data=E_central_sdev)
        gm.create_dataset("C_central", data=C_central)
        gm.create_dataset("C_central_sdev", data=C_central_sdev)
        gm.create_dataset("chi2dof_central", data=chi2dof_central)
        gm.create_dataset("Q_central", data=Q_central)
        gm.create_dataset("E_jk", data=E_jk)
        gm.create_dataset("dE_jk", data=dE_jk)
        gm.create_dataset("C_jk", data=C_jk)
        gm.create_dataset("chi2dof_jk", data=chi2dof_jk)
        gm.create_dataset("Q_jk", data=Q_jk)
        gm.create_dataset("E_jk_mean", data=E_jk_mean)
        gm.create_dataset("E_jk_err", data=E_jk_err)
        gm.create_dataset("dE_jk_mean", data=dE_jk_mean)
        gm.create_dataset("dE_jk_err", data=dE_jk_err)
        gm.attrs["dim_E_jk"] = "jk,state"
        gm.attrs["dim_dE_jk"] = "jk,state (slot 0 is E0, slots 1+ are gaps)"
    print(f"  written: {FIT_PATH}:{tag}/{mom_name}", flush=True)

print("\nfinished", flush=True)
