#Pion effective energies of the stream-e test (Coulomb-gauge boosted smearing, *_streame_test.py) vs the old stream-e data (Wuppertal N40 rho3.25, momfrac 0.6, G5, 799 cfgs).
#Reads every cfg<cfg>/ folder downloaded into new_dir, so the same script runs on 1 cfg or on all 10:
#  1 cfg:  errors are jackknife over its 8 t_src
#  >1 cfg: errors are jackknife over the cfgs (each cfg averaged over its t_src)
#The old data is compared twice: on the same cfgs as the new data (same gauge fields), and on all 799 cfgs (jackknife bins of bin_old cfgs).
#E_eff(t) = log(C(t)/C(t+1)) of the real part of C = (forward + backward)/2, averaged over the sources.
import os, glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

local = "/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/local_analysis"
new_dir = f"{local}/streame_test_delta"                                   # cfg<cfg>/pion_..._forward_..._cfg<cfg>.h5, as downloaded from the cluster
new_name = "pion_coulomb_rhoT3.25_iso-aniso2-aniso3_frac0p6-0p3_GEVP_sepq_forward_phyp_fb_tsep20_cfg{cfg}.h5"
old_path = f"{local}/799cfgs/2pt_N40_rho3.25_G5_ez_momfrac0p6_fb_tsep18_ncfg799.h5"
old_cache = f"{new_dir}/old_799_G5_srcavg.npz"                            # the old data averaged over the x,y,z sources, built once (the old file is 3.3 GB)
plot_dir = new_dir
L = 32
t0 = 2              # GEVP reference time
n_keep = 3          # GEVP in the subspace of the n_keep largest eigenvectors of the normalized C(t0): the 6 smearings have about 3 independent directions
bin_old = 10        # jackknife bins of the 799 old cfgs
#new single operator: shape 0 = iso (rho_z = rho_T), frac 0 = 0.6, gamma 0 = G5, at sink and source (the closest to the old G5 operator)
i_shape, i_frac, i_gamma = 0, 0, 0


def meff(c):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(c[..., :-1] / c[..., 1:])

def jack(fn, data, nbin=1):
    #central value of fn on the full average; error from the delete-one jackknife over axis 0 in bins of nbin (a remainder that does not fill a bin is left out of the error only)
    n = data.shape[0] // nbin
    d = data[: n * nbin].reshape(n, nbin, *data.shape[1:]).mean(axis=1)
    jk = np.array([fn((d.sum(0) - d[i]) / (n - 1)) for i in range(n)])
    return fn(data.mean(0)), np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))


#======== new data: (cfg, shape_snk, shape_src, frac_snk, frac_src, g_snk, g_src, tsrc, pz, tsep) ========
new_cfgs = sorted(int(os.path.basename(d)[3:]) for d in glob.glob(f"{new_dir}/cfg*") if os.path.exists(f"{d}/{new_name.format(cfg=os.path.basename(d)[3:])}"))
new = []
for cfg in new_cfgs:
    with h5py.File(f"{new_dir}/cfg{cfg}/{new_name.format(cfg=cfg)}", "r") as f:
        dim_spec = list(f["pt2_forward"].attrs["dim_spec"])
        assert dim_spec[6:] == ["t_src_list", "q_list", "pz", "tsep"] and f["q_list"].shape[0] == 1, dim_spec
        pz_new = f["momentum_list"][0, :, 2]
        new.append(0.5 * (f["pt2_forward"][...] + f["pt2_backward"][...])[..., 0, :, :])    # fold, drop the single q (= 0)
        if cfg == new_cfgs[0]:
            print("new:", [str(s) for s in f["pt2_forward"].attrs["shape_names"]], "fracs", f["mom_frac_list"][...], [str(g) for g in f["pt2_forward"].attrs["gamma_list"]])
new = -np.array(new)                     # the pion correlators come out negative in this contraction convention (the old ones are positive); E_eff does not depend on the sign
n_tsrc, n_pz, n_tsep = new.shape[-3:]
print(f"new cfgs: {new_cfgs}, {n_tsrc} t_src, pz {list(pz_new)}, tsep 0..{n_tsep - 1}")

if len(new_cfgs) == 1:
    new_samples, nbin_new, new_err = np.moveaxis(new[0], 6, 0), 1, "jackknife over the t_src"   # samples = t_src, moved to the front
else:
    new_samples, nbin_new, new_err = new.mean(axis=7), 1, "jackknife over the cfgs"  # samples = cfgs, averaged over t_src
iso = new_samples[:, i_shape, i_shape, i_frac, i_frac, i_gamma, i_gamma].real        # (sample, pz, tsep)

#GEVP over the 6 smearings (3 shapes x 2 fracs) with gamma G5 at sink and source; principal correlator of the ground state
M = new_samples[:, :, :, :, :, i_gamma, i_gamma]                                      # (sample, ssnk, ssrc, fsnk, fsrc, pz, tsep)
M = M.transpose(0, 5, 6, 1, 3, 2, 4).reshape(M.shape[0], n_pz, n_tsep, 6, 6)         # (sample, pz, tsep, i = (shape, frac) sink, j source)
basis = []                                                                            # fixed from the full average, so that no jackknife sample swaps eigenvectors
for p in range(n_pz):
    A = M.mean(0)[p, t0]
    A = 0.5 * (A + A.conj().T)
    norm = np.sqrt(A.diagonal().real)
    w, v = np.linalg.eigh(A / np.outer(norm, norm))
    basis.append((norm, v[:, ::-1][:, :n_keep]))

def gevp_meff(m):                                                                     # m: (pz, tsep, 6, 6)
    out = np.full((n_pz, n_tsep - 1), np.nan)
    for p in range(n_pz):
        norm, U = basis[p]
        B = np.einsum("ia,tij,jb->tab", U.conj(), m[p] / np.outer(norm, norm), U)
        B = 0.5 * (B + np.conj(np.swapaxes(B, -1, -2)))
        try:
            L_inv = np.linalg.inv(np.linalg.cholesky(B[t0]))
        except np.linalg.LinAlgError:
            continue
        lam = np.array([np.linalg.eigvalsh(L_inv @ B[t] @ L_inv.conj().T).max() for t in range(n_tsep)])
        out[p] = meff(lam)
    return out


#======== old data: (cfg, tsrc, pz, tsep), averaged over the x,y,z sources ========
if not os.path.exists(old_cache):
    with h5py.File(old_path, "r") as f:
        out = {"cfg_list": f["cfg_list"][...], "pf_list": f["pf_list"][...]}
        for name in ["pt2_forward", "pt2_backward"]:
            d = f[name]
            acc = np.empty((d.shape[0], d.shape[1], d.shape[5], d.shape[6]), complex)
            for c0 in range(0, d.shape[0], 50):
                acc[c0:c0 + 50] = d[c0:c0 + 50].mean(axis=(2, 3, 4))
            out[name] = acc
    np.savez(old_cache, **out)
    print(f"saved {old_cache}")
old = np.load(old_cache)
old_cfgs = old["cfg_list"]
assert np.array_equal(old["pf_list"][:, 2], pz_new) and np.all(old["pf_list"][:, :2] == 0), old["pf_list"]
old_c = (0.5 * (old["pt2_forward"] + old["pt2_backward"])).real                     # (cfg, tsrc, pz, tsep)
same = [int(np.where(old_cfgs == cfg)[0][0]) for cfg in new_cfgs if cfg in old_cfgs]
old_same = old_c[same[0]] if len(same) == 1 else old_c[same].mean(axis=1)            # same sampling as the new data: t_src for 1 cfg, cfgs otherwise


res = {
    "new_iso":  jack(meff, iso, nbin_new),
    "new_gevp": jack(gevp_meff, M, nbin_new),
    "old_same": jack(meff, old_same),
    "old_all":  jack(meff, old_c.mean(axis=1), bin_old),
}
m_pi = res["old_all"][0][0, 12:17].mean()
E_disp = np.sqrt(m_pi ** 2 + (2 * np.pi * pz_new / L) ** 2)

print(f"\nam_pi = {m_pi:.4f} (old, {len(old_cfgs)} cfgs, pz = 0, t = 12-16); E_disp = sqrt(m_pi^2 + (2 pi pz / L)^2)")
print(f"new errors: {new_err}; old on the same cfgs: same sampling; old all cfgs: jackknife bins of {bin_old}")
print(" pz | E_disp |  t | new iso G5       | new GEVP         | old same cfgs    | old all cfgs")
for p in range(n_pz):
    for t in ([4, 8, 12] if pz_new[p] < 3 else [3, 5, 7]):
        row = [f"{res[k][0][p, t]:8.4f}({res[k][1][p, t] * 1e4:5.0f})" for k in ["new_iso", "new_gevp", "old_same", "old_all"]]
        print(f" {pz_new[p]:2d} | {E_disp[p]:.4f} | {t:2d} | " + " | ".join(row))


#======== plots: one panel per pz ========
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF})
cfg_label = f"cfg {new_cfgs[0]}" if len(new_cfgs) == 1 else f"{len(new_cfgs)} cfgs"

def figure(series, title, fname):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), constrained_layout=True)
    for p in range(n_pz):
        ax = axes.flat[p]
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axhline(E_disp[p], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        lo, hi = E_disp[p] - 0.04 - 0.035 * pz_new[p], E_disp[p] + 0.15 + 0.03 * pz_new[p]
        for k, (key, label, col, mk, dx) in enumerate(series):
            c, e = res[key]
            t = np.arange(c.shape[1])
            ok = np.isfinite(c[p]) & np.isfinite(e[p]) & (e[p] < 0.08) & (c[p] > lo) & (c[p] < hi) & (t >= 1)
            ax.errorbar(t[ok] + dx, c[p][ok], e[p][ok], fmt=mk, ms=4.5, color=col, mec=SURF, mew=0.8, elinewidth=1.2, capsize=0, label=label, zorder=3 + k)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, n_tsep - 1.5)
        ax.set_title(f"$p_z$ = {pz_new[p]}  (2$\\pi$/L)", color=INK, fontsize=10, loc="left")
        if p % 4 == 0:
            ax.set_ylabel("$aE_{\\mathrm{eff}}(t)$")
        if p >= 3:
            ax.set_xlabel("$t/a$")
    leg = axes.flat[7]
    leg.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3))))
    l.append(f"$\\sqrt{{m_\\pi^2+p^2}}$, $am_\\pi$ = {m_pi:.4f}\n(old {len(old_cfgs)} cfgs, $p_z$=0, t=12-16)")
    leg.legend(h, l, loc="center left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(f"{plot_dir}/{fname}", dpi=160)
    plt.close(fig)
    print(f"wrote {plot_dir}/{fname}")

figure([("new_iso", f"new, {cfg_label}: Coulomb iso, frac 0.6, G5-G5", C1, "o", -0.15),
        ("old_same", f"old, {cfg_label}: Wuppertal N40 rho 3.25, G5", C2, "s", 0.0),
        ("old_all", f"old, {len(old_cfgs)} cfgs (jackknife, bins of {bin_old})", C3, "^", 0.15)],
       f"Pion effective energy, stream e: new Coulomb-boosted smearing ({cfg_label}) vs old Wuppertal data", "pion_Eeff_new_vs_old.png")
figure([("new_iso", f"new, {cfg_label}: iso, frac 0.6 (single operator)", C1, "o", -0.15),
        ("new_gevp", f"new, {cfg_label}: 6x6 GEVP (3 shapes x 2 fracs, t0={t0}, {n_keep}-dim subspace)", C2, "s", 0.0),
        ("old_all", f"old, {len(old_cfgs)} cfgs", C3, "^", 0.15)],
       f"Pion effective energy, {cfg_label}: single operator vs smearing GEVP", "pion_Eeff_gevp.png")
