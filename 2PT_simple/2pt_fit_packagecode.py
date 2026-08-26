from pathlib import Path
import h5py
import numpy as np
import gvar as gv
import lsqfit

"""Reads twopt_source_averaged.h5 which has all the source-averaged 2pts"""
"""Shape of input 2pt: [n_cfg,n_mom,tsink]"""
"""Fit method matches the gluon_gpd stage-1 two-point fit:
   - E0 is its own Gaussian parameter (not slot 0 of log(dE))
   - E0 prior center: continuum dispersion sqrt(m_rest^2 + (2 pi/Gs)^2 |p|^2)
     with m_rest MEASURED (acosh effective mass at t = TREF, rest frame)
   - E0 prior width: this momentum's own jackknife error of the effective
     mass at t = TREF
   - gaps log-normal 0.50(40), amplitudes Gaussian 0(1e4)"""

PT2_PATH = Path("./twopt_source_averaged.h5")
FIT_PATH = Path("./twopt_fit_results.h5")
Gs, Gt = 32, 96

MOM_LIST = [(0, 0, pz) for pz in range(7)]
TMIN, TMAX = 4, 15
TSEP_LIST = [4, 5, 6, 7]  #(c1*e^-E1tsep) / (c0*e^-E0tsep) will be printed out
N_STATES = 3
SVDCUT = 1e-4
TREF = 10                 # E0 prior is read off the effective mass at this t
MAXIT = 20000

DE1_PRIOR = gv.gvar("0.50(0.40)")
DE2_PRIOR = gv.gvar("0.50(0.40)")
C_CENTER = 0.0
C_WIDTH = 1e4

PROGRESS_EVERY = 100

tag = f"nstate{N_STATES}_t{TMIN}-{TMAX}_svd{SVDCUT:.0e}_tref{TREF}"


def twopt_model(t, p):
    e = p["E0"]
    corr = p["c"][0] * (gv.exp(-e * t) + gv.exp(-e * (Gt - t)))
    for n in range(1, len(p["c"])):
        e = e + p["dE"][n - 1]
        corr = corr + p["c"][n] * (gv.exp(-e * t) + gv.exp(-e * (Gt - t)))
    return corr


def make_prior(n_states, e0_center, e0_width):
    """E0 Gaussian (center and width measured, see effmass_at_tref);
    gaps log-normal so they stay positive; amplitudes loose Gaussians."""
    gaps = [DE1_PRIOR, DE2_PRIOR][:n_states - 1]
    prior = {
        "E0": gv.gvar(e0_center, e0_width),
        "c": gv.gvar([C_CENTER] * n_states, [C_WIDTH] * n_states),
    }
    if n_states > 1:
        prior["log(dE)"] = gv.log(gv.gvar([gv.mean(g) for g in gaps],
                                          [gv.sdev(g) for g in gaps]))
    return prior


def cov_of_mean(samples):
    return np.cov(samples, rowvar=False, ddof=1) / samples.shape[0]


def jk_stats(values):
    """Jackknife mean and error over axis 0 of delete-one samples."""
    n = values.shape[0]
    mean = values.mean(axis=0)
    return mean, np.sqrt((n - 1.0) / n * np.sum((values - mean) ** 2, axis=0))


def effmass_at_tref(raw):
    """acosh[(C(t+1) + C(t-1)) / (2 C(t))] at t = TREF, on the delete-one
    jackknife replicates of raw [n_cfg, t].  Returns (jk_mean, jk_err)."""
    n = raw.shape[0]
    reps = (raw.sum(axis=0) - raw) / (n - 1.0)
    em = np.arccosh((reps[:, TREF + 1] + reps[:, TREF - 1])
                    / (2.0 * reps[:, TREF]))
    return jk_stats(em)


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

# the measured rest mass that feeds every momentum's dispersion prior center
m_rest, _ = effmass_at_tref(pion[..., mom_to_idx[(0, 0, 0)], :].real)
print(f"rest mass from effective mass at t = {TREF}: {m_rest:.6f}", flush=True)


for pf in MOM_LIST:
    ipf = mom_to_idx[pf]
    C2_pf = pion[..., ipf, :].real
    samples = C2_pf[:, TMIN:TMAX + 1]
    mean = samples.mean(axis=0)
    cov = cov_of_mean(samples)
    jk = (samples.sum(axis=0) - samples) / (n_jk - 1)

    # E0 prior: dispersion center from the measured rest mass, width from
    # THIS momentum's effective-mass jackknife error at t = TREF
    e0_center = np.sqrt(m_rest ** 2
                        + (2.0 * np.pi / Gs) ** 2 * sum(c ** 2 for c in pf))
    _, e0_width = effmass_at_tref(C2_pf)
    prior = make_prior(N_STATES, e0_center, e0_width)

    central_fit = lsqfit.nonlinear_fit(data=(t, gv.gvar(mean, cov)),
                                       fcn=twopt_model, prior=prior,
                                       svdcut=SVDCUT, maxit=MAXIT)
    #[E0, dE1, dE2] -> [E0, E1, E2]; cumsum on the gvars keeps correlations
    dE_full = np.concatenate([np.atleast_1d(central_fit.p["E0"]),
                              central_fit.p["dE"]])
    E = np.cumsum(dE_full)
    E_central = gv.mean(E)
    E_central_sdev = gv.sdev(E)
    C_central = gv.mean(central_fit.p["c"])
    C_central_sdev = gv.sdev(central_fit.p["c"])
    chi2dof_central = central_fit.chi2 / central_fit.dof
    Q_central = central_fit.Q

    print(f"\n====== CENTRAL FIT RESULT   pf = {pf} ======")
    print(f"  central: n_states = {N_STATES}   t = {TMIN}..{TMAX}")
    print(f"  central: E0 prior = {gv.gvar(e0_center, e0_width)}   "
          f"(dispersion center, effmass width at t = {TREF})")
    for n in range(N_STATES):
        print(f"  central: E{n} = {E[n]:<14} c{n} = {central_fit.p['c'][n]}")
    print(f"  central: chi2/dof = {chi2dof_central:.2f}   "
          f"Q = {Q_central:.3f}   svdn = {getattr(central_fit, 'svdn', 0)}")
    print(central_fit.format(maxline=True))

    ratio = (central_fit.p["c"][1] / central_fit.p["c"][0]
             * gv.exp(-central_fit.p["dE"][0] * np.array(TSEP_LIST)))
    for tsep, r in zip(TSEP_LIST, ratio):
        print(f"  central: C1 e^(-E1 tsep) / C0 e^(-E0 tsep) "
              f"at tsep = {tsep}: {r}")

    #per-jackknife central values (slot 0 of dE_jk is E0, slots 1+ are gaps,
    #the SAME layout as before -- the ratio fit reads dE_jk[:, 1] unchanged)
    E_jk = np.zeros((n_jk, N_STATES))
    dE_jk = np.zeros((n_jk, N_STATES))
    C_jk = np.zeros((n_jk, N_STATES))

    #per-jackknife sdev: each replicate's own fit error, not the scatter across them
    E_sdev_jk = np.zeros((n_jk, N_STATES))
    dE_sdev_jk = np.zeros((n_jk, N_STATES))
    C_sdev_jk = np.zeros((n_jk, N_STATES))

    #per-jackknife fit quality
    chi2dof_jk = np.zeros(n_jk)
    Q_jk = np.zeros(n_jk)

    p0 = {k: central_fit.pmean[k] for k in prior}
    for i in range(n_jk):
        fit_jk = lsqfit.nonlinear_fit(data=(t, gv.gvar(jk[i], cov)),
                                      fcn=twopt_model, prior=prior,
                                      svdcut=SVDCUT, p0=p0, maxit=MAXIT)
        dE_full = np.concatenate([np.atleast_1d(fit_jk.p["E0"]),
                                  fit_jk.p["dE"]])
        E_full = np.cumsum(dE_full)   # gvar cumsum: keeps the correlations

        #save central values
        dE_jk[i] = gv.mean(dE_full)
        E_jk[i] = gv.mean(E_full)
        C_jk[i] = gv.mean(fit_jk.p["c"])

        #save sdevs
        E_sdev_jk[i]  = gv.sdev(E_full)
        dE_sdev_jk[i] = gv.sdev(dE_full)
        C_sdev_jk[i]  = gv.sdev(fit_jk.p["c"])

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
        g.attrs["tref"] = TREF
        g.attrs["E0_prior_scheme"] = ("center sqrt(m_rest^2 + p^2), m_rest = "
                                      f"effmass(t={TREF}) at rest; width = "
                                      f"this momentum's effmass jk error")
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
        gm.create_dataset("E_sdev_jk", data=E_sdev_jk)
        gm.create_dataset("dE_sdev_jk", data=dE_sdev_jk)
        gm.create_dataset("C_sdev_jk", data=C_sdev_jk)
        gm.create_dataset("chi2dof_jk", data=chi2dof_jk)
        gm.create_dataset("Q_jk", data=Q_jk)
        gm.create_dataset("E_jk_mean", data=E_jk_mean)
        gm.create_dataset("E_jk_err", data=E_jk_err)
        gm.create_dataset("dE_jk_mean", data=dE_jk_mean)
        gm.create_dataset("dE_jk_err", data=dE_jk_err)
        gm.attrs["E0_prior_center"] = e0_center
        gm.attrs["E0_prior_width"] = e0_width
        gm.attrs["dim_E_jk"] = "jk,state"
        gm.attrs["dim_dE_jk"] = "jk,state (slot 0 is E0, slots 1+ are gaps)"
        gm.attrs["dim_dE_sdev_jk"] = ("jk,state -- lsqfit's uncertainty on each "
                                      "replicate's own fit, NOT the jackknife "
                                      "scatter of dE_jk across replicates")
    print(f"  written: {FIT_PATH}:{tag}/{mom_name}", flush=True)

print("\nfinished", flush=True)
