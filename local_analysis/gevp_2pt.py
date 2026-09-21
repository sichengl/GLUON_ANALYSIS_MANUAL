import h5py
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def jackknife(x):
    n_cfg = x.shape[0]
    return (x.sum(axis=0) - x) / (n_cfg - 1)

def jk(data, jk_axis=0):
    n = data.shape[jk_axis]
    data = np.moveaxis(data, jk_axis, 0)
    mean = np.mean(data, axis=0)
    err = np.sqrt((n - 1.0) / n * np.sum((data - mean) ** 2, axis=0))
    return mean, err

# ---------------- settings ----------------
local = "/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/local_analysis"
hadron = "proton"                                   # "proton" or "pion"
pt2_path = f"{local}/2pt_N40_rho3.25_GEVP_ez_momfrac0p6_{hadron}_tsrc_q5_fb_tsep18_ncfg10.h5"
out_path = f"{local}/gevp_{hadron}_N40_rho3.25_momfrac0p6_ncfg10.h5"
t0 = 3                                              # reference time of the GEVP
t_ref = 8                                           # time at which the eigenvectors are fixed for the projected correlators
use_backward = True                                 # average forward and backward correlators
n_state = 3

# ---------------- read the packed 2pt, plain source average ----------------
with h5py.File(pt2_path, "r") as f:
    pt2_forward = f["pt2_forward"][:, :, :, :, 0]              # [cfg, gs, gsrc, tsrc, pf, tsep]   q = 0 entry
    pt2_backward = f["pt2_backward"][:, :, :, :, 0]
    pf_list = f["pf_list"][:]
    cfg_list = f["cfg_list"][:]
    gamma_list = [g.decode() if isinstance(g, bytes) else str(g) for g in f.attrs["gamma_list"]]
C_cfg = pt2_forward.mean(axis=3)                                # [cfg, gs, gsrc, pf, tsep]
if use_backward:
    C_cfg = 0.5 * (C_cfg + pt2_backward.mean(axis=3))
n_cfg, _, _, n_pf, tsep_max = C_cfg.shape
print(f"C_cfg shape {C_cfg.shape}: cfg, gamma_sink, gamma_source, pf, tsep")

# hermiticity: C_ij and C_ji^* estimate the same number; keep the Hermitian part, record the size of the rest
C_herm = 0.5 * (C_cfg + np.conj(np.swapaxes(C_cfg, 1, 2)))
antiherm_fraction = np.linalg.norm(C_cfg - C_herm) / np.linalg.norm(C_cfg)
print(f"anti-Hermitian fraction of C (all cfg, pf, t): {antiherm_fraction:.2e}")

# jackknife samples plus the central value as the last "sample"
C_jk = jackknife(C_herm)                                        # [jk, gs, gsrc, pf, tsep]
C_all = np.concatenate([C_jk, C_herm.mean(axis=0)[None]], axis=0)   # [jk + 1, gs, gsrc, pf, tsep]
n_samp = C_all.shape[0]

# ---------------- solve the GEVP:  C(t) v = lambda C(t0) v ----------------
t_list = np.arange(t0 + 1, tsep_max)                            # times where the GEVP is solved
lam = np.full((n_samp, n_pf, n_state, len(t_list)), np.nan)                     # [samp, pf, n, t]
vec = np.full((n_samp, n_pf, n_state, len(t_list), n_state), np.nan, "<c16")    # [samp, pf, n, t, i]   v_n(t, t0)_i
failed = np.zeros((n_pf, len(t_list)), int)
for ipf in range(n_pf):
    for isamp in range(n_samp):
        for it, t in enumerate(t_list):
            try:
                w, v = eigh(C_all[isamp, :, :, ipf, t], C_all[isamp, :, :, ipf, t0])   # ascending w, columns v with v^dag C(t0) v = 1
            except np.linalg.LinAlgError:
                failed[ipf, it] += 1
                continue
            order = np.argsort(w)[::-1]                          # largest eigenvalue = ground state
            for n in range(n_state):
                v_n = v[:, order[n]]
                k = np.argmax(np.abs(v_n))
                v_n = v_n * np.conj(v_n[k]) / np.abs(v_n[k])    # phase convention: largest component real and positive
                lam[isamp, ipf, n, it] = w[order[n]]
                vec[isamp, ipf, n, it] = v_n
print("GEVP failures per (pf, t):"); print(failed)

# effective energies of the eigenvalues, E_n(t) = ln lambda_n(t) / lambda_n(t+1)
E_eff = np.log(lam[..., :-1] / lam[..., 1:])                    # [samp, pf, n, t]  for t in t_list[:-1]

# diagonal effective masses for comparison
C_diag = np.stack([C_all[:, i, i] for i in range(n_state)], axis=1)              # [samp, i, pf, tsep]
E_eff_diag = np.log(C_diag[..., :-1] / C_diag[..., 1:]).real                    # [samp, i, pf, tsep - 1]

# projected correlators with the eigenvectors fixed at (t_ref, t0):  C_n(t) = v_n^dag C(t) v_n
it_ref = list(t_list).index(t_ref)
v_ref = vec[:, :, :, it_ref]                                                    # [samp, pf, n, i]
C_proj = np.einsum("spni,sijpt,spnj->spnt", np.conj(v_ref), C_all, v_ref)      # [samp, pf, n, tsep]
E_eff_proj = np.log(C_proj[..., :-1] / C_proj[..., 1:]).real

# diagnostics on the central value
eig_Ct0 = np.array([np.linalg.eigvalsh(C_all[-1, :, :, ipf, t0]) for ipf in range(n_pf)])   # [pf, 3] ascending
cond_Ct0 = eig_Ct0[:, -1] / eig_Ct0[:, 0]
overlap = np.abs(np.einsum("spnti,sijp,spntj->spnt", np.conj(vec[..., :-1, :]), C_all[..., t0], vec[..., 1:, :]))   # [samp, pf, n, t]  |v_n(t)^dag C(t0) v_n(t+1)|
for ipf in range(n_pf):
    print(f"pf {pf_list[ipf]}: eig C(t0) = {eig_Ct0[ipf]}, condition number {cond_Ct0[ipf]:.1f}")

# ---------------- save ----------------
with h5py.File(out_path, "w") as f:
    f.create_dataset("C_jk", data=C_all[:-1]);           f.create_dataset("C_central", data=C_all[-1])            # [jk, gs, gsrc, pf, tsep]
    f.create_dataset("lambda_jk", data=lam[:-1]);        f.create_dataset("lambda_central", data=lam[-1])         # [jk, pf, n, t]  t = t_list
    f.create_dataset("E_eff_jk", data=E_eff[:-1]);       f.create_dataset("E_eff_central", data=E_eff[-1])        # [jk, pf, n, t]  t = t_list[:-1]
    f.create_dataset("v_jk", data=vec[:-1]);             f.create_dataset("v_central", data=vec[-1])              # [jk, pf, n, t, i]
    f.create_dataset("v_ref_jk", data=v_ref[:-1]);       f.create_dataset("v_ref_central", data=v_ref[-1])        # [jk, pf, n, i]  at (t_ref, t0)
    f.create_dataset("C_proj_jk", data=C_proj[:-1]);     f.create_dataset("C_proj_central", data=C_proj[-1])      # [jk, pf, n, tsep]
    f.create_dataset("E_eff_proj_jk", data=E_eff_proj[:-1]); f.create_dataset("E_eff_proj_central", data=E_eff_proj[-1])
    f.create_dataset("E_eff_diag_jk", data=E_eff_diag[:-1]); f.create_dataset("E_eff_diag_central", data=E_eff_diag[-1])   # [jk, i, pf, tsep - 1]
    f.create_dataset("overlap_central", data=overlap[-1])                                                          # [pf, n, t]  |v_n(t)^dag C(t0) v_n(t+1)|
    f.create_dataset("eig_Ct0_central", data=eig_Ct0);   f.create_dataset("cond_Ct0_central", data=cond_Ct0)
    f.create_dataset("failed", data=failed)
    f.create_dataset("t_list", data=t_list);             f.create_dataset("pf_list", data=pf_list);   f.create_dataset("cfg_list", data=cfg_list)
    f.attrs["hadron"] = hadron
    f.attrs["t0"] = t0;  f.attrs["t_ref"] = t_ref;  f.attrs["use_backward"] = use_backward
    f.attrs["gamma_list"] = gamma_list
    f.attrs["antiherm_fraction"] = antiherm_fraction
    f.attrs["source_file"] = pt2_path
    f.attrs["gevp"] = "C(t) v = lambda C(t0) v on the Hermitian part of C, per delete-one jackknife sample; eigenvalues descending, n = 0 ground state"
    f.attrs["normalization"] = "v^dag C(t0) v = 1, largest component real positive"
    f.attrs["E_eff"] = "ln lambda(t)/lambda(t+1); C_proj(t) = v_ref^dag C(t) v_ref with v_ref = v(t_ref, t0)"
    f.attrs["dim_lambda"] = "jk, pf, n, t";  f.attrs["dim_v"] = "jk, pf, n, t, i";  f.attrs["dim_C_proj"] = "jk, pf, n, tsep"
print("saved", out_path)

# ---------------- plot: GEVP effective energies against the diagonal ones ----------------
E_mean, E_err = jk(E_eff[:-1])                     # [pf, n, t]
Ed_mean, Ed_err = jk(E_eff_diag[:-1])              # [i, pf, tsep - 1]
Ep_mean, Ep_err = jk(E_eff_proj[:-1])              # [pf, n, tsep - 1]
fig, axes = plt.subplots(1, n_pf, figsize=(3.2 * n_pf, 3.4), sharey=False)
for ipf in range(n_pf):
    ax = axes[ipf] if n_pf > 1 else axes
    for i in range(n_state):
        ax.errorbar(np.arange(tsep_max - 1) + 0.1 * i - 0.1, Ed_mean[i, ipf], Ed_err[i, ipf], fmt="s", ms=3, mfc="none", alpha=0.5, label=f"C_{gamma_list[i]}{gamma_list[i]}")
    for n in range(n_state):
        ax.errorbar(t_list[:-1] + 0.1 * n, E_mean[ipf, n], E_err[ipf, n], fmt="o", ms=4, label=f"GEVP n={n}")
    ax.errorbar(np.arange(tsep_max - 1), Ep_mean[ipf, 0], Ep_err[ipf, 0], fmt="^", ms=4, color="k", label="projected n=0")
    ax.set_title(f"{hadron}  pf={pf_list[ipf].tolist()}  t0={t0}")
    ax.set_xlabel("t"); ax.set_ylim(0, 2.0)
    if ipf == 0:
        ax.set_ylabel("E_eff"); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{local}/gevp_{hadron}_t0{t0}_tref{t_ref}.png", dpi=150)
plt.show()
