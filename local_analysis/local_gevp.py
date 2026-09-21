import h5py
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# ---------------- settings ----------------
local = "/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/local_analysis"
hadron = "proton"
smear_tag = "N16rho2.0_N40rho3.25_N96rho5.0"
pt2_path = f"{local}/2pt_{smear_tag}_GEVP_ez_momfrac0p6_{hadron}_tsrc_q5_fb_tsep18_ncfg10.h5"
t0 = 3
out_path = f"{local}/gevp_{hadron}_9x9_t0{t0}_ncfg10.h5"

# ---------------- read: q = 0 entry, average over source times and over forward/backward ----------------
with h5py.File(pt2_path, "r") as f:
    pt2_forward = f["pt2_forward"][:, :, :, :, :, :, 0]         # [cfg, smear_sink, smear_src, gamma_sink, gamma_src, tsrc, pf, tsep]
    pt2_backward = f["pt2_backward"][:, :, :, :, :, :, 0]
    pf_list = f["pf_list"][:]
C_full = 0.5 * (pt2_forward.mean(axis=5) + pt2_backward.mean(axis=5))     # [cfg, ss, sr, gs, gsrc, pf, tsep]
n_cfg, n_smear, _, n_gamma, _, n_pf, tsep_max = C_full.shape
n_op = n_smear * n_gamma                                                    # operator index a = smear * n_gamma + gamma

# ---------------- 9x9 matrix C[cfg, a, b, pf, t], a = sink operator, b = source operator ----------------
C = C_full.transpose(0, 1, 3, 2, 4, 5, 6).reshape(n_cfg, n_op, n_op, n_pf, tsep_max)
C = 0.5 * (C + np.conj(np.swapaxes(C, 1, 2)))                              # Hermitian part
sign = np.sign(np.trace(C.mean(axis=0)[:, :, 0, t0]).real)                  # C(t0) must be positive definite
C = sign * C
scale = np.sqrt(np.abs(np.einsum("aapt->apt", C.mean(axis=0))[:, :, t0]))   # [op, pf]: normalize each operator to C_aa(t0) = 1
C = C / scale[None, :, None, :, None] / scale[None, None, :, :, None]

# ---------------- jackknife samples, central value last ----------------
C_jk = (C.sum(axis=0) - C) / (n_cfg - 1)
C_all = np.concatenate([C_jk, C.mean(axis=0)[None]], axis=0)                # [n_cfg + 1, op, op, pf, tsep]
n_samp = n_cfg + 1

# ---------------- GEVP: C(t) v = lambda C(t0) v ----------------
t_list = np.arange(t0 + 1, tsep_max)
lam = np.full((n_samp, n_pf, n_op, len(t_list)), np.nan)                    # [samp, pf, n, t]
vec = np.full((n_samp, n_pf, n_op, len(t_list), n_op), np.nan, "<c16")      # [samp, pf, n, t, op]
for isamp in range(n_samp):
    for ipf in range(n_pf):
        for it, t in enumerate(t_list):
            try:
                w, v = eigh(C_all[isamp, :, :, ipf, t], C_all[isamp, :, :, ipf, t0])
            except np.linalg.LinAlgError:
                continue
            order = np.argsort(w)[::-1]                                     # largest eigenvalue = ground state
            lam[isamp, ipf, :, it] = w[order]
            vec[isamp, ipf, :, it] = v[:, order].T / scale[:, ipf]          # back to the original operators
with np.errstate(invalid="ignore", divide="ignore"):
    E_eff = np.log(lam[..., :-1] / lam[..., 1:])                            # [samp, pf, n, t], t = t_list[:-1]
    C_ref = C_all[:, 4, 4]                                                   # diagonal N40 G45 for comparison (rescaling cancels in the ratio)
    E_eff_ref = np.log(C_ref[..., :-1] / C_ref[..., 1:]).real               # [samp, pf, tsep - 1]

def jk_err(x):
    n = x.shape[0]
    return np.sqrt((n - 1.0) / n * np.nansum((x - np.nanmean(x, axis=0)) ** 2, axis=0))

with h5py.File(out_path, "w") as f:
    f.create_dataset("lambda_jk", data=lam[:-1]);   f.create_dataset("lambda_central", data=lam[-1])
    f.create_dataset("E_eff_jk", data=E_eff[:-1]);  f.create_dataset("E_eff_central", data=E_eff[-1])
    f.create_dataset("v_jk", data=vec[:-1]);        f.create_dataset("v_central", data=vec[-1])
    f.create_dataset("E_eff_ref_jk", data=E_eff_ref[:-1]);  f.create_dataset("E_eff_ref_central", data=E_eff_ref[-1])
    f.create_dataset("t_list", data=t_list);        f.create_dataset("pf_list", data=pf_list)
    f.attrs["t0"] = t0
    f.attrs["operator"] = "a = smear * 3 + gamma, smear 0=N16 1=N40 2=N96, gamma 0=G5 1=G45 2=G35"
    f.attrs["v"] = "v[samp, pf, n, t, op]: eigenvector of state n at (t, t0) in the original operator normalization"
print("saved", out_path)

# ---------------- plot: ground state against the N40 G45 diagonal ----------------
fig, axes = plt.subplots(1, n_pf, figsize=(3.2 * n_pf, 3.6))
for ipf in range(n_pf):
    ax = axes[ipf]
    ax.errorbar(np.arange(tsep_max - 1), E_eff_ref[-1, ipf], jk_err(E_eff_ref[:-1, ipf]), fmt="ko-", ms=3, lw=1, capsize=2, label="diag N40 G45")
    ax.errorbar(t_list[:-1] + 0.15, E_eff[-1, ipf, 0], jk_err(E_eff[:-1, ipf, 0]), fmt="rs--", ms=3, lw=1, capsize=2, mfc="none", label="GEVP 9x9 n=0")
    c = E_eff_ref[-1, ipf, 8]
    ax.set_ylim(c - 0.25, c + 0.45); ax.set_xlim(0, 12); ax.grid(alpha=0.3)
    ax.set_title(f"{hadron}  pz={pf_list[ipf][2]}  t0={t0}", fontsize=9); ax.set_xlabel("t")
    if ipf == 0:
        ax.set_ylabel("E_eff"); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{local}/gevp_{hadron}_9x9_t0{t0}.png", dpi=150)
plt.show()