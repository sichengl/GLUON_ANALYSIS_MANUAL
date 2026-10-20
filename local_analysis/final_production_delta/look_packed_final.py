#First look at the packed final Delta 2pt (output of delta_package_final.py): pion and proton at one q of the pack (q = 0 by default),
#with the 2 smearings x 3 gammas = 6 operators. For each particle:
#  1. per cfg, the mean over the 24 t_src; the proton projected to positive parity (forward tr[P+ M], backward -tr[P- M]); forward and backward kept apart
#  2. effective mass of the 6 diagonal correlators, forward and backward averaged (on the diagonal s_a s_a = +1, so no sign is needed)
#  3. GEVP of each basis in gevp_bases, solved for forward and backward separately: E(t) = ln lambda0(t)/lambda0(t+1), then the two averaged
#Jackknife errors over cfgs. One page per figure, the 7 pz in a 4 x 2 grid as in compare_3rhoz_pion.py; the figures go to plot_dir.
import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

pack_dir = "/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/local_analysis/final_production_delta"
plot_dir = f"{pack_dir}/look_plots"
iq = 0                                          # index in the packed q_list (the default pack has only q = 0)
t0 = 2                                          # GEVP reference time
c_sign = {"pion": -1, "proton": +1}             # the pion comes out negative (the fermion-loop sign is not in the code), the projected proton positive
m_disp = {"pion": 0.141, "proton": None}        # mass of the dispersion line; None: from the first GEVP basis at pz = 0, mean over t_plateau
t_plateau = {"pion": (10, 15), "proton": (8, 12)}
gevp_bases = {                                  # name: (smearings, gammas); smear 0 = [4.5, 2.25, 0.8], 1 = [1.5, 1.5, 0]; gamma 0 = G5, 1 = G5G4, 2 = G3G5
    "6x6: 2 smear x 3 gamma": ([0, 1], [0, 1, 2]),
    "2x2: 2 smear x G5G4":    ([0, 1], [1]),
    "3x3: smear 0 x 3 gamma": ([0], [0, 1, 2]),
}
Ls = 32
os.makedirs(plot_dir, exist_ok=True)


def jk(data, jk_axis=0):
    n = data.shape[jk_axis]
    data = np.moveaxis(data, jk_axis, 0)
    mean = np.mean(data, axis=0)
    err = np.sqrt((n - 1.0) / n * np.sum((data - mean) ** 2, axis=0))
    return mean, err


def text(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def newest(particle):
    #the pack of this particle with the most cfgs (the proton without projection is preferred)
    files = glob.glob(f"{pack_dir}/2pt_*_{particle}.h5") or glob.glob(f"{pack_dir}/2pt_*_{particle}_projected.h5")
    assert files, f"no packed {particle} file in {pack_dir}"
    return max(files, key=lambda name: int(name.split("_ncfg")[1].split("_")[0]))


def load(particle):
    """per cfg, mean over t_src: C[half] (cfg, op_snk, op_src, pz, tsep) with op = 3 * smear + gamma, and the sign of c_sign"""
    name = newest(particle)
    with h5py.File(name, "r") as f:
        cfgs, pf, q = f["cfg_list"][:], f["pf_list"][iq], f["q_list"][iq]
        smear_names, gamma_list = text(f["smear_names"][:]), text(f.attrs["gamma_list"])
        print(f"\n==== {particle}: {os.path.basename(name)}")
        print(f"pt2_forward {f['pt2_forward'].shape}   ({f.attrs['dim_pt2']})")
        print(f"{len(cfgs)} cfgs, {cfgs.min()} ... {cfgs.max()};  q = {q.tolist()};  p_f = {pf[:, 2].tolist()} along z")
        open_dirac = particle == "proton" and not name.endswith("_projected.h5")
        C = {}
        for half in ["pt2_forward", "pt2_backward"]:
            n_pz, tsep = f[half].shape[-2:]
            out = np.zeros((len(cfgs), 2, 3, 2, 3, n_pz, tsep), "<c16")                 # (cfg, smear_snk, g_snk, smear_src, g_src, pz, tsep)
            for ic in range(len(cfgs)):                                                  # one cfg at a time: the proton pack is several GB
                x = f[half][ic]                                                          # (smear_snk, smear_src, g_snk, g_src, [dirac, dirac,] t_src, q, pz, tsep)
                if open_dirac:
                    proj = f["parity_p"][:] if half == "pt2_forward" else -f["parity_m"][:]
                    x = np.einsum("kl,abcdlktqpT->abcdtqpT", proj, x)                     # forward sum_kl P+[k,l] M[l,k], backward -sum_kl P-[k,l] M[l,k]
                x = x[..., iq, :, :].mean(axis=4)                                        # mean over t_src: (smear_snk, smear_src, g_snk, g_src, pz, tsep)
                out[ic] = x.transpose(0, 2, 1, 3, 4, 5)
            C[half] = c_sign[particle] * out.reshape(len(cfgs), 6, 6, n_pz, tsep)
    labels = [f"{smear_names[s]} {gamma_list[g]}" for s in range(2) for g in range(3)]
    return cfgs, pf, C, labels


def gevp_energy(Cm, ops):
    """Cm (op, op, pz, tsep) -> E0(t) = ln lambda0(t)/lambda0(t+1) for t = t0+1 .. tsep-2, (pz, t); NaN where C(t0) is not positive definite"""
    C_sub = Cm[np.ix_(ops, ops)]
    n_pz, tsep = C_sub.shape[2:]
    lam = np.full((n_pz, tsep), np.nan)
    for ip in range(n_pz):
        try:
            L_inv = np.linalg.inv(np.linalg.cholesky(C_sub[:, :, ip, t0]))
        except np.linalg.LinAlgError:
            continue
        for t in range(t0 + 1, tsep):
            lam[ip, t] = np.linalg.eigvalsh(L_inv @ C_sub[:, :, ip, t] @ L_inv.conj().T).max()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(lam[:, :-1] / lam[:, 1:])


def dispersion(m, pz):
    p_lat = 2 * np.pi * pz / Ls
    return 2 * np.arcsinh(np.sqrt(np.sinh(m / 2) ** 2 + np.sin(p_lat / 2) ** 2))


def page(particle, pf, curves, title, ylabel, filename, m):
    """curves: list of (label, E[pz, t] mean, error) on one page, the 7 pz in a 4 x 2 grid"""
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    markers = ["o", "s", "^", "v", "D", "P"]
    ylim = {"pion": (-0.2, 0.35), "proton": (-0.3, 0.6)}[particle]
    plt.figure(figsize=(6.5, 8.5))                          # one page in LaTeX: 6.5 in wide, 8.5 in tall
    for ip in range(len(pf)):
        plt.subplot(4, 2, ip + 1)
        E_disp = dispersion(m, pf[ip][2])
        plt.axhline(E_disp, color="gray", lw=1, ls="--", label=f"lattice dispersion, m = {m:.3f}")
        for i, (label, mean, err) in enumerate(curves):
            t = np.arange(mean.shape[1]) + 0.1 * (i - (len(curves) - 1) / 2)
            plt.errorbar(t, mean[ip], yerr=err[ip], fmt=markers[i % 6] + "-", color=colors[i % 6], capsize=2, ms=3, mfc="none", lw=0.8, label=label)
        plt.ylim(E_disp + ylim[0], E_disp + ylim[1])
        plt.xlim(0, 16)
        plt.xticks(np.arange(0, 17, 2), fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(alpha=0.3)
        plt.title(f"pz = {pf[ip][2]}", fontsize=9)
        if ip >= len(pf) - 2:
            plt.xlabel("t", fontsize=9)
        if ip % 2 == 0:
            plt.ylabel(ylabel, fontsize=9)
        if ip == 0:
            handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.subplot(4, 2, 8)                                    # the empty 8th slot holds the legend
    plt.axis("off")
    plt.legend(handles, legend_labels, loc="center", fontsize=7, title=title, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{filename}")
    plt.show()


for particle in ["pion", "proton"]:
    cfgs, pf, C, labels = load(particle)
    n_cfg = len(cfgs)
    for half in C:                                          # hermitian in the operator indices on average: keep the hermitian part
        C[half] = 0.5 * (C[half] + np.conj(np.swapaxes(C[half], 1, 2)))
    C_mean = {half: C[half].mean(axis=0) for half in C}                                   # (op, op, pz, tsep)
    C_jk = {half: (C[half].sum(axis=0)[None] - C[half]) / (n_cfg - 1) for half in C}      # (sample, op, op, pz, tsep)

    #1. the 6 diagonal correlators, forward and backward averaged
    diag_curves = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for op in range(6):
            D_jk = 0.5 * (C_jk["pt2_forward"][:, op, op] + C_jk["pt2_backward"][:, op, op]).real     # (sample, pz, tsep)
            D = 0.5 * (C_mean["pt2_forward"][op, op] + C_mean["pt2_backward"][op, op]).real
            diag_curves.append((labels[op], np.log(D[:, :-1] / D[:, 1:]), jk(np.log(D_jk[:, :, :-1] / D_jk[:, :, 1:]))[1]))

    #2. GEVP per basis: forward and backward solved separately, then averaged
    gevp_curves = []
    for name, (smears, gammas) in gevp_bases.items():
        ops = [3 * s + g for s in smears for g in gammas]
        E = 0.5 * (gevp_energy(C_mean["pt2_forward"], ops) + gevp_energy(C_mean["pt2_backward"], ops))
        E_jk = np.array([0.5 * (gevp_energy(C_jk["pt2_forward"][i], ops) + gevp_energy(C_jk["pt2_backward"][i], ops)) for i in range(n_cfg)])
        bad = [int(pf[ip][2]) for ip in range(len(pf)) if np.isnan(E[ip]).all()]
        if bad:
            print(f"{name}: C(t0 = {t0}) not positive definite at pz = {bad}, no GEVP there")
        gevp_curves.append((name, E, jk(E_jk)[1]))

    m = m_disp[particle]
    if m is None:                                           # from the first basis at pz = 0
        lo, hi = t_plateau[particle]
        m = float(np.nanmean(gevp_curves[0][1][0, lo:hi]))
        print(f"dispersion mass from '{gevp_curves[0][0]}' at pz = 0, mean over t = {lo}..{hi - 1}: {m:.4f}")

    print(f"E_eff from '{gevp_curves[0][0]}' (forward/backward averaged), t = 4, 6, 8, 10, 12:")
    for ip in range(len(pf)):
        mean, err = gevp_curves[0][1][ip], gevp_curves[0][2][ip]
        print(f"  pz {pf[ip][2]}: " + "   ".join(f"{mean[t]:.4f}({err[t]:.4f})" for t in [4, 6, 8, 10, 12]) + f"    dispersion {dispersion(m, pf[ip][2]):.4f}")

    page(particle, pf, diag_curves, f"{particle}, {n_cfg} cfgs, single operators", "E_eff (single operator)", f"meff_single_{particle}_ncfg{n_cfg}.pdf", m)
    page(particle, pf, gevp_curves, f"{particle}, {n_cfg} cfgs, GEVP t0 = {t0}", "E_eff (GEVP, ln lambda0 ratio)", f"meff_gevp_{particle}_t0{t0}_ncfg{n_cfg}.pdf", m)
