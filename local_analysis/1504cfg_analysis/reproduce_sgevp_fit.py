# reproduce the sGEVP fit with the default (fixed) fit windows from the two files alone, at one GEVP time t0 and one svdcut
#   python reproduce_sgevp_fit.py pt2_6x6_jk_...h5 pt3_summed_6x6_jk_...h5 t0 svdcut out.npz
# needs numpy, h5py, gvar, lsqfit;  one BLAS thread makes the many small fits faster:  VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
import sys, time
import numpy as np, h5py, gvar as gv, lsqfit

pt2_file, pt3_file, t0, svdcut, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
WINDOW = {0: (2, 12), 1: (2, 12), 2: (2, 9), 3: (2, 8), 4: (2, 7), 5: (2, 6), 6: (2, 6)}     # default fit window (first t, last t) of each pz
MINUS_A_PRIOR, B_PRIOR, DE_PRIOR = (1.0, 10.0), (0.0, 10.0), (0.5, 0.5)                    # log(-A) = log(1(10)) = 0(10);  B = 0(10);  log(dE) = log(0.5(5))
DE_STARTS, FREE_A_PRIOR, MINUS_A_MIN = [0.15, 0.3, 0.5, 0.8, 1.2], (0.0, 10.0), 1e-3
MAXIT, TOL, FITTER = 20000, 1e-8, "scipy_least_squares"
if not gv.BufferDict.has_distribution("neglog"):
    gv.BufferDict.add_distribution("neglog", lambda x: -gv.exp(x))     # the fit varies neglog(A);  p["A"] = -exp(neglog(A)) < 0


def jk_mean_err(x):
    n = len(x)
    m = x.mean(axis=0)
    return m, np.sqrt((n - 1) / n * ((x - m) ** 2).sum(axis=0))


# ---- step 1: M_eff(t) of every jackknife sample, pz, w
f2, f3 = h5py.File(pt2_file, "r"), h5py.File(pt3_file, "r")
assert np.array_equal(f2["cfg_list"][()], f3["cfg_list"][()]), "the two files are from different cfgs"
R = 0
for half in ("forward", "backward"):
    C = f2[f"pt2_jk_{half}"][()]                    # (jk, pz, t, a, b)
    P3 = f3[f"pt3_summed_jk_{half}"][()]            # (jk, w, pz, t, a, b)
    n_jk, n_pz, n_t = C.shape[:3]
    n_w = P3.shape[1]
    R_half = np.full((n_jk, n_pz, n_w, n_t), np.nan)
    for p in range(n_pz):
        L_inv = np.linalg.inv(np.linalg.cholesky(C[:, p, t0]))           # C(t0) = L L^dag
        L_inv_dag = np.conj(np.swapaxes(L_inv, -1, -2))
        R_half[:, p, :, t0] = 0
        for t in range(1, n_t):
            if t == t0:
                continue
            M = L_inv @ C[:, p, t] @ L_inv_dag
            lam, v = np.linalg.eigh(0.5 * (M + np.conj(np.swapaxes(M, -1, -2))))      # eigenvalues in increasing order
            k = -1 if t > t0 else 0                                       # ground state: largest above t0, smallest below
            u = (L_inv_dag @ v[:, :, k][:, :, None])[:, :, 0]             # u^dag C(t0) u = 1
            u_pt3_u = np.einsum("ja,jwab,jb->jw", u.conj(), P3[:, :, p, t], u).real
            u_pt3_t0_u = np.einsum("ja,jwab,jb->jw", u.conj(), P3[:, :, p, t0], u).real
            R_half[:, p, :, t] = u_pt3_u / lam[:, k][:, None] - u_pt3_t0_u
    R = R + 0.5 * R_half                                                  # the two halves averaged
meff = R[..., 1:] - R[..., :-1]                                           # (jk, pz, w, t): M_eff(t) = R(t+1) - R(t), t = 0 .. n_t-2


# ---- step 2: the fit of each pz, every jackknife sample
def fit_pz(y_jk, t_points):
    n_jk, n_w, n_t = y_jk.shape
    t = t_points.astype(float)
    x = {"t": np.tile(t, n_w), "iw": np.repeat(np.arange(n_w), n_t)}

    def model(x, p):
        return p["A"][x["iw"]] + p["B"][x["iw"]] * x["t"] * gv.exp(-p["dE"][0] * x["t"])

    def prior():
        p = gv.BufferDict()
        p["neglog(A)"] = gv.log(gv.gvar(np.full(n_w, MINUS_A_PRIOR[0]), np.full(n_w, MINUS_A_PRIOR[1])))
        p["B"] = gv.gvar(np.full(n_w, B_PRIOR[0]), np.full(n_w, B_PRIOR[1]))
        p["log(dE)"] = gv.log(gv.gvar([DE_PRIOR[0]], [DE_PRIOR[1]]))
        return p

    def prior_free():
        p = gv.BufferDict()
        p["A"] = gv.gvar(np.full(n_w, FREE_A_PRIOR[0]), np.full(n_w, FREE_A_PRIOR[1]))
        p["B"] = gv.gvar(np.full(n_w, B_PRIOR[0]), np.full(n_w, B_PRIOR[1]))
        p["log(dE)"] = gv.log(gv.gvar([DE_PRIOR[0]], [DE_PRIOR[1]]))
        return p

    kw = dict(fcn=model, svdcut=svdcut, maxit=MAXIT, tol=TOL, fitter=FITTER)
    Y = y_jk.reshape(n_jk, n_w * n_t)                                     # (jk, point), w after w
    y_mean = Y.mean(axis=0)
    D = Y - y_mean
    cov_all = (n_jk - 1) / n_jk * (D.T @ D)                               # covariance of the mean, all cfgs: only for the fit of the mean
    y_w = y_mean.reshape(n_w, n_t)
    starts = [{"neglog(A)": np.log(np.maximum(-y_w[:, -1], MINUS_A_MIN)), "B": (y_w[:, 0] - y_w[:, -1]) / (t[0] * np.exp(-g * t[0])), "log(dE)": np.log([g])}
              for g in DE_STARTS]
    free, chi2_free = None, np.nan                                        # start 6: the sign-free fit, A clipped to <= -MINUS_A_MIN
    for p0 in starts:
        gv.switch_gvar()
        f = lsqfit.nonlinear_fit(data=(x, gv.gvar(y_mean, cov_all)), prior=prior_free(), p0={"A": y_w[:, -1], "B": p0["B"], "log(dE)": p0["log(dE)"]}, **kw)
        if free is None or f.chi2 < chi2_free:
            chi2_free, free = float(f.chi2), {k: np.array(f.pmean[k]) for k in ("A", "B", "log(dE)")}
        gv.restore_gvar()
    starts.append({"neglog(A)": np.log(np.maximum(-free["A"], MINUS_A_MIN)), "B": free["B"], "log(dE)": free["log(dE)"]})
    best = None
    for p0 in starts:                                                     # the fit of the mean: lowest chi2 of the six starts
        gv.switch_gvar()
        f = lsqfit.nonlinear_fit(data=(x, gv.gvar(y_mean, cov_all)), prior=prior(), p0=p0, **kw)
        if best is None or f.chi2 < best[0]:
            best = (float(f.chi2), {k: np.array(f.pmean[k]) for k in ("neglog(A)", "B", "log(dE)")})
        gv.restore_gvar()
    y_cfg = n_jk * y_mean - (n_jk - 1) * Y                               # per-cfg values made back from the jackknife samples
    y_c = y_cfg - y_cfg.mean(axis=0)
    Q = y_c.T @ y_c
    m = n_jk - 1
    A_jk, dE_jk, chi2_jk, n_bad = np.zeros((n_jk, n_w)), np.zeros(n_jk), np.zeros(n_jk), 0
    for j in range(n_jk):                                                 # sample j: the covariance of the other cfgs, start from the fit of the mean
        cov_j = (Q - (1 + 1 / m) * np.outer(y_c[j], y_c[j])) / (m - 1) / (n_jk - 1)
        gv.switch_gvar()
        f = lsqfit.nonlinear_fit(data=(x, gv.gvar(Y[j], cov_j)), prior=prior(), p0=best[1], **kw)
        A_jk[j], dE_jk[j], chi2_jk[j] = f.pmean["A"], f.pmean["dE"][0], f.chi2
        n_bad += int(f.stopping_criterion == 0)
        gv.restore_gvar()
    return A_jk, dE_jk, chi2_jk, n_bad


t_start = time.time()
M = np.zeros((meff.shape[0], meff.shape[2], meff.shape[1]))             # (jk, w, pz)
dE = np.zeros((meff.shape[0], meff.shape[1]))                            # (jk, pz)
for p in range(meff.shape[1]):
    first, last = WINDOW[p]
    tt = np.arange(first, last + 1)
    M[:, :, p], dE[:, p], chi2, n_bad = fit_pz(meff[:, p][:, :, tt], tt)
    print(f"pz {p}: window {first}-{last}, fits at maxit {n_bad}, {time.time() - t_start:.0f} s", flush=True)
# ---- step 3: the double ratio
Dr = (M / M[:, :1]) / (M[:, :, :1] / M[:, :1, :1])                       # (jk, w, pz)
np.savez(out, M_jk=M, dE_jk=dE, D_jk=Dr, t0=t0, svdcut=svdcut)
for name, X in (("M(w 0)", M[:, 0]), ("dE", dE), ("D(w 4)", Dr[:, 4]), ("D(w 8)", Dr[:, 8])):
    m_, e_ = jk_mean_err(X)
    print(f"{name:>7s} pz 0-6: " + " ".join(str(gv.gvar(a, b)) for a, b in zip(m_, e_)))
