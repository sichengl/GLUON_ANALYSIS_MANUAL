import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def jk(data, jk_axis=0):
    n = data.shape[jk_axis]
    data = np.moveaxis(data, jk_axis, 0)
    mean = np.mean(data, axis=0)
    err = np.sqrt((n - 1.0) / n * np.sum((data - mean) ** 2, axis=0))
    return mean, err

def solve_gevp(C_matrix):
    eigen_val = np.full((n_pf, n_op, len(t_list)), np.nan)
    eigen_vec = np.full((n_pf, n_op, len(t_list), n_op), np.nan, "<c16")
    for ipf in range(n_pf):
        L = np.linalg.cholesky(C_matrix[:, :, ipf, t0])
        L_inv = np.linalg.inv(L)
        for it, t in enumerate(t_list):
            M = L_inv @ C_matrix[:, :, ipf, t] @ L_inv.conj().T
            w, u = np.linalg.eigh(M)
            v = L_inv.conj().T @ u
            order = np.argsort(w)[::-1]
            eigen_val[ipf, :, it] = w[order]
            eigen_vec[ipf, :, it] = v[:, order].T
    return eigen_val, eigen_vec

local = "/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/local_analysis/test_2pts"
hadron = "pion"
smear_tag = "coulomb_rhoT3.25_iso-rz1o2-rz1o3_frac0p6"
pt2_path = f"{local}/2pt_{smear_tag}_GEVP_qsrc_physp_{hadron}_tsrc_q1_fb_tsep20_64src_ncfg40.h5"
plot_dir = f"{local}/compare_3rhoz"
os.makedirs(plot_dir, exist_ok=True)

t0 = 2
t_vec = 4
#shape 0 = iso (rho_z = rho_T), 1 = rz1o2 (rho_z = rho_T/2), 2 = rz1o3 (rho_z = rho_T/3);  frac 0 = boost 0.6 (only one);  gamma 0 = G5, 1 = G45
shape_use = [0, 1, 2]
frac_use = [0]
gamma_use = [0, 1]

with h5py.File(pt2_path, "r") as f:
    pt2_forward  = f["pt2_forward"][:]
    pt2_backward = f["pt2_backward"][:]
    pf_list  = f["pf_list"][:]
    shape_list = f["shape_list"][:]
    mom_frac_list = f["mom_frac_list"][:]
    shape_names = f.attrs["shape_names"]
    gamma_list = f.attrs["gamma_list"]

print(f"shape of pt2 is {pt2_forward.shape}")
print("cfg,shape_snk,shape_src,frac_snk,frac_src,g_snk,g_src,tsrc,q,pf,tsep")

pt2_forward  = pt2_forward[:, shape_use][:, :, shape_use][:, :, :, frac_use][:, :, :, :, frac_use][:, :, :, :, :, gamma_use][:, :, :, :, :, :, gamma_use]
pt2_backward = pt2_backward[:, shape_use][:, :, shape_use][:, :, :, frac_use][:, :, :, :, frac_use][:, :, :, :, :, gamma_use][:, :, :, :, :, :, gamma_use]

print(f"shape of pt2 after selection is {pt2_forward.shape}")
print("shapes used:", [f"{shape_names[s]} rho_T {shape_list[s][0]} rho_z {shape_list[s][1]:.4g}" for s in shape_use])
print("boosts used:", [f"frac {mom_frac_list[fr]}" for fr in frac_use])
print("gammas used:", [str(gamma_list[g]) for g in gamma_use])

n_cfg, n_shape, _, n_frac, _, n_gamma, _, n_tsrc, n_q, n_pf, tsep_max = pt2_forward.shape
n_op = n_shape * n_frac * n_gamma

#cfg, shape_sink, frac_sink, gamma_sink, shape_src, frac_src, gamma_src
pt2_forward  = pt2_forward.transpose(0, 1, 3, 5, 2, 4, 6, 7, 8, 9, 10).reshape(n_cfg, n_op, n_op, n_tsrc, n_q, n_pf, tsep_max)
pt2_backward = pt2_backward.transpose(0, 1, 3, 5, 2, 4, 6, 7, 8, 9, 10).reshape(n_cfg, n_op, n_op, n_tsrc, n_q, n_pf, tsep_max)

# check: merged pt2_forward[:, a, b] must equal the stored element for the shape/frac/gamma triple that op_labels names
op_triples = [(s, fr, g) for s in shape_use for fr in frac_use for g in gamma_use]          # same loop order as op_labels
with h5py.File(pt2_path, "r") as f:
    for a, (s_snk, fr_snk, g_snk) in enumerate(op_triples):
        for b, (s_src, fr_src, g_src) in enumerate(op_triples):
            stored = f["pt2_forward"][:, s_snk, s_src, fr_snk, fr_src, g_snk, g_src]     # file order: cfg, shape_snk, shape_src, frac_snk, frac_src, g_snk, g_src, ...
            assert np.array_equal(pt2_forward[:, a, b], stored), f"operator pair ({a}, {b}) does not match the file"
print(f"checked: all {len(op_triples) ** 2} operator pairs match the file, so the labels below are true")

print(f"shape of pt2 after merging is {pt2_forward.shape}   (cfg, op_snk, op_src, tsrc, q, pf, tsep)")
op_labels = [f"{shape_names[s]} frac{mom_frac_list[fr]} {gamma_list[g]}" for s in shape_use for fr in frac_use for g in gamma_use]
for a, label in enumerate(op_labels):
    print(f"operator {a}: {label}")


#take the q=0 elements
C = {}
C["forward"]  = pt2_forward[:, :, :, :, 0].mean(axis=3)       # (cfg, op_snk, op_src, pf, tsep)
C["backward"] = pt2_backward[:, :, :, :, 0].mean(axis=3)      # same; tsep counts backwards from the source
for half in ["forward", "backward"]:
    C[half] = 0.5 * (C[half] + np.conj(np.swapaxes(C[half], 1, 2)))
c_sign = -1
for half in ["forward", "backward"]:
    C[half] = c_sign * C[half]
#build jackknife lists
C_mean, C_jk = {}, {}
for half in ["forward", "backward"]:
    C_mean[half] = C[half].mean(axis=0)                                               # (op, op, pf, tsep)
    C_jk[half] = (C[half].sum(axis=0)[None] - C[half]) / (n_cfg - 1)                  # (cfg, op, op, pf, tsep)
print(f"jackknife samples: {C_jk['forward'].shape}   (sample, op_snk, op_src, pf, tsep)")

t_list = np.arange(t0 + 1, tsep_max)
eigen_val_central, eigen_vec_central = {}, {}
for half in ["forward", "backward"]:
    eigen_val_central[half], eigen_vec_central[half] = solve_gevp(C_mean[half])

it_vec = list(t_list).index(t_vec)
v0 = {}
for half in ["forward", "backward"]:
    v0[half] = eigen_vec_central[half][:, 0, it_vec, :]

pt2_rec_mean = {}
for half in ["forward", "backward"]:
    pt2_rec_mean[half] = np.einsum("pa,abpT,pb->pT", np.conj(v0[half]), C_mean[half], v0[half])

for half in ["forward", "backward"]:
    deviation = np.abs(pt2_rec_mean[half][:, t0] - 1).max()
    print(f"{half}: max over pf of |C_rec(t0) - 1| = {deviation:.1e}")
    assert deviation < 1e-10, f"{half}: v0 is not normalized to v0^dag C(t0) v0 = 1"
pt2_rec = {}
pt2_rec["forward"] = pt2_rec_mean["forward"].real
pt2_rec["backward"] = pt2_rec_mean["backward"].real
pt2_rec["average"] = 0.5 * (pt2_rec["forward"] + pt2_rec["backward"])

meff = {}
with np.errstate(invalid="ignore", divide="ignore"):
    for half in ["forward", "backward", "average"]:
        meff[half] = np.log(pt2_rec[half][:, :-1] / pt2_rec[half][:, 1:])

for ipf in range(n_pf):
    print(f"pz {pf_list[ipf][2]}: meff t = 1..8  " + " ".join(f"{meff['average'][ipf, t]:.4f}" for t in range(1, 9)))

E_lambda = {}
with np.errstate(invalid="ignore", divide="ignore"):
    for half in ["forward", "backward"]:
        E_lambda[half] = np.log(eigen_val_central[half][:, 0, :-1] / eigen_val_central[half][:, 0, 1:])
E_lambda["average"] = 0.5 * (E_lambda["forward"] + E_lambda["backward"])
i_single = op_labels.index(f"iso frac{mom_frac_list[frac_use[0]]} G45")
pt2_single = {}
for half in ["forward", "backward"]:
    pt2_single[half] = C_mean[half][i_single, i_single].real
pt2_single["average"] = 0.5 * (pt2_single["forward"] + pt2_single["backward"])
with np.errstate(invalid="ignore", divide="ignore"):
    meff_single = np.log(pt2_single["average"][:, :-1] / pt2_single["average"][:, 1:])
meff_err = {}
pt2_rec_jk = {}
for half in ["forward", "backward"]:
    pt2_rec_jk[half] = np.einsum("pa,cabpT,pb->cpT", np.conj(v0[half]), C_jk[half], v0[half]).real
pt2_rec_jk["average"] = 0.5 * (pt2_rec_jk["forward"] + pt2_rec_jk["backward"])
with np.errstate(invalid="ignore", divide="ignore"):
    meff_jk = np.log(pt2_rec_jk["average"][:, :, :-1] / pt2_rec_jk["average"][:, :, 1:])
meff_err["rec"] = jk(meff_jk)[1]

eigen_val_jk = {}
for half in ["forward", "backward"]:
    eigen_val_jk[half] = np.array([solve_gevp(C_jk[half][icfg])[0] for icfg in range(n_cfg)])
with np.errstate(invalid="ignore", divide="ignore"):
    E_lambda_jk = 0.5 * (np.log(eigen_val_jk["forward"][:, :, 0, :-1] / eigen_val_jk["forward"][:, :, 0, 1:])
                       + np.log(eigen_val_jk["backward"][:, :, 0, :-1] / eigen_val_jk["backward"][:, :, 0, 1:]))
meff_err["lambda"] = jk(E_lambda_jk)[1]

pt2_single_jk = 0.5 * (C_jk["forward"][:, i_single, i_single].real + C_jk["backward"][:, i_single, i_single].real)
with np.errstate(invalid="ignore", divide="ignore"):
    meff_single_jk = np.log(pt2_single_jk[:, :, :-1] / pt2_single_jk[:, :, 1:])
meff_err["single"] = jk(meff_single_jk)[1]
m_pi = 0.141
Ls = 32
t_meff = np.arange(tsep_max - 1)
plt.figure(figsize=(6.5, 8.5))                              # one page in LaTeX: 6.5 in wide, 8.5 in tall
for ipf in range(n_pf):
    plt.subplot(4, 2, ipf + 1)                              # 2 + 2 + 2 + 1 panels, filled row by row
    p_lat = 2 * np.pi * pf_list[ipf][2] / Ls
    E_disp = 2 * np.arcsinh(np.sqrt(np.sinh(m_pi / 2) ** 2 + np.sin(p_lat / 2) ** 2))
    plt.axhline(E_disp, color="gray", lw=1, ls="--", label=f"lattice dispersion, m = {m_pi}")
    plt.errorbar(t_meff, meff["average"][ipf], yerr=meff_err["rec"][ipf], fmt="rs-", capsize=2, ms=3, mfc="none", lw=0.8, label=f"reconstructed 2pt, t_vec = {t_vec}")
    plt.errorbar(t_meff+0.25, meff_single[ipf], yerr=meff_err["single"][ipf], fmt="bo-", capsize=2, ms=3, mfc="none", lw=0.8, label=f"single {op_labels[i_single]}")
    plt.errorbar(t_list[:-1]+0.5, E_lambda["average"][ipf], yerr=meff_err["lambda"][ipf], fmt="g^:", capsize=2, ms=3, mfc="none", lw=0.8, label="ln lambda(t)/lambda(t+1)")
    plt.ylim(E_disp - 0.2, E_disp + 0.35)
    plt.xlim(0, 13)
    plt.xticks(np.arange(0, 14, 2), fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(alpha=0.3)
    plt.title(f"pz = {pf_list[ipf][2]}", fontsize=9)
    if ipf >= n_pf - 2:
        plt.xlabel("t", fontsize=9)
    if ipf % 2 == 0:
        plt.ylabel("E_eff", fontsize=9)
    if ipf == 0:
        handles, labels = plt.gca().get_legend_handles_labels()
plt.subplot(4, 2, 8)                                        # the empty 8th slot holds the legend
plt.axis("off")
plt.legend(handles, labels, loc="center", fontsize=8, title=f"{hadron},  {n_op}x{n_op},  t0 = {t0}", title_fontsize=9)
plt.tight_layout()
plt.savefig(f"{plot_dir}/meff_{hadron}_{smear_tag}_t0{t0}_tvec{t_vec}.svg")
plt.show()


#shape 0 = iso, 1 = rz1o2, 2 = rz1o3;  frac 0 = boost 0.6;  gamma 0 = G5, 1 = G45
basis_list_all = [{"{iso} x {0.6} x {G5, G45}":                  [[0], [0], [0, 1]],          # figure 0: each rho_z alone, then all three together
                   "{rz1o2} x {0.6} x {G5, G45}":                [[1], [0], [0, 1]],
                   "{rz1o3} x {0.6} x {G5, G45}":                [[2], [0], [0, 1]],
                   "{iso, rz1o2, rz1o3} x {0.6} x {G5, G45}":    [[0, 1, 2], [0], [0, 1]],
                  },
                  {"{iso} x {0.6} x {G45}":                      [[0], [0], [1]],             # figure 1: single G45 operator for each rho_z (1x1, no GEVP mixing)
                   "{rz1o2} x {0.6} x {G45}":                    [[1], [0], [1]],
                   "{rz1o3} x {0.6} x {G45}":                    [[2], [0], [1]],
                  },
                  {"{iso, rz1o2} x {0.6} x {G5, G45}":           [[0, 1], [0], [0, 1]],       # figure 2: pairs of rho_z, then all three
                   "{iso, rz1o3} x {0.6} x {G5, G45}":           [[0, 2], [0], [0, 1]],
                   "{rz1o2, rz1o3} x {0.6} x {G5, G45}":         [[1, 2], [0], [0, 1]],
                   "{iso, rz1o2, rz1o3} x {0.6} x {G5, G45}":    [[0, 1, 2], [0], [0, 1]],
                  }]          # each dictionary is one figure; name: [shape_use, frac_use, gamma_use]

with h5py.File(pt2_path, "r") as f:
    pt2_forward_file  = f["pt2_forward"][:]
    pt2_backward_file = f["pt2_backward"][:]

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "black", "tab:pink", "tab:olive", "tab:cyan"]
markers = ["o", "s", "^", "v", "D", "P", "*", "X", "<", ">"]

for i_fig, basis_list in enumerate(basis_list_all):
    meff_basis, meff_basis_err = {}, {}
    for name, (shape_use, frac_use, gamma_use) in basis_list.items():
        # same steps as above, for this basis
        pt2_forward  = pt2_forward_file[:, shape_use][:, :, shape_use][:, :, :, frac_use][:, :, :, :, frac_use][:, :, :, :, :, gamma_use][:, :, :, :, :, :, gamma_use]
        pt2_backward = pt2_backward_file[:, shape_use][:, :, shape_use][:, :, :, frac_use][:, :, :, :, frac_use][:, :, :, :, :, gamma_use][:, :, :, :, :, :, gamma_use]
        n_cfg, n_shape, _, n_frac, _, n_gamma, _, n_tsrc, n_q, n_pf, tsep_max = pt2_forward.shape
        n_op = n_shape * n_frac * n_gamma
        pt2_forward  = pt2_forward.transpose(0, 1, 3, 5, 2, 4, 6, 7, 8, 9, 10).reshape(n_cfg, n_op, n_op, n_tsrc, n_q, n_pf, tsep_max)
        pt2_backward = pt2_backward.transpose(0, 1, 3, 5, 2, 4, 6, 7, 8, 9, 10).reshape(n_cfg, n_op, n_op, n_tsrc, n_q, n_pf, tsep_max)

        C = {}
        C["forward"]  = pt2_forward[:, :, :, :, 0].mean(axis=3)
        C["backward"] = pt2_backward[:, :, :, :, 0].mean(axis=3)
        for half in ["forward", "backward"]:
            C[half] = 0.5 * (C[half] + np.conj(np.swapaxes(C[half], 1, 2)))
            C[half] = c_sign * C[half]
        C_mean, C_jk = {}, {}
        for half in ["forward", "backward"]:
            C_mean[half] = C[half].mean(axis=0)
            C_jk[half] = (C[half].sum(axis=0)[None] - C[half]) / (n_cfg - 1)

        eigen_val_central, eigen_vec_central = {}, {}
        for half in ["forward", "backward"]:
            eigen_val_central[half], eigen_vec_central[half] = solve_gevp(C_mean[half])
        v0 = {}
        for half in ["forward", "backward"]:
            v0[half] = eigen_vec_central[half][:, 0, it_vec, :]

        pt2_rec_mean, pt2_rec_jk = {}, {}
        for half in ["forward", "backward"]:
            pt2_rec_mean[half] = np.einsum("pa,abpT,pb->pT", np.conj(v0[half]), C_mean[half], v0[half]).real
            pt2_rec_jk[half] = np.einsum("pa,cabpT,pb->cpT", np.conj(v0[half]), C_jk[half], v0[half]).real
        pt2_rec_mean["average"] = 0.5 * (pt2_rec_mean["forward"] + pt2_rec_mean["backward"])
        pt2_rec_jk["average"] = 0.5 * (pt2_rec_jk["forward"] + pt2_rec_jk["backward"])
        with np.errstate(invalid="ignore", divide="ignore"):
            meff_basis[name] = np.log(pt2_rec_mean["average"][:, :-1] / pt2_rec_mean["average"][:, 1:])
            meff_basis_err[name] = jk(np.log(pt2_rec_jk["average"][:, :, :-1] / pt2_rec_jk["average"][:, :, 1:]))[1]

    plt.figure(figsize=(6.5, 8.5))                          # one page in LaTeX: 6.5 in wide, 8.5 in tall
    n_basis = len(basis_list)
    for ipf in range(n_pf):
        plt.subplot(4, 2, ipf + 1)                          # 2 + 2 + 2 + 1 panels, filled row by row
        p_lat = 2 * np.pi * pf_list[ipf][2] / Ls
        E_disp = 2 * np.arcsinh(np.sqrt(np.sinh(m_pi / 2) ** 2 + np.sin(p_lat / 2) ** 2))
        plt.axhline(E_disp, color="gray", lw=1, ls="--", label=f"lattice dispersion, m = {m_pi}")
        for i, name in enumerate(basis_list):
            shift = 0.1 * (i - (n_basis - 1) / 2)
            plt.errorbar(t_meff + shift, meff_basis[name][ipf], yerr=meff_basis_err[name][ipf], fmt=markers[i % 10] + "-",
                         color=colors[i % 10], capsize=2, ms=3, mfc="none", lw=0.8, label=name)
        plt.ylim(E_disp - 0.2, E_disp + 0.35)
        plt.xlim(0, 13)
        plt.xticks(np.arange(0, 14, 2), fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(alpha=0.3)
        plt.title(f"pz = {pf_list[ipf][2]}", fontsize=9)
        if ipf >= n_pf - 2:
            plt.xlabel("t", fontsize=9)
        if ipf % 2 == 0:
            plt.ylabel("E_eff (reconstructed 2pt)", fontsize=9)
        if ipf == 0:
            handles, labels = plt.gca().get_legend_handles_labels()
    plt.subplot(4, 2, 8)                                    # the empty 8th slot holds the legend
    plt.axis("off")
    plt.legend(handles, labels, loc="center", fontsize=8, title=f"{hadron},  t0 = {t0},  t_vec = {t_vec}", title_fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/meff_compare{i_fig}_{hadron}_{smear_tag}_t0{t0}_tvec{t_vec}_compare3rhoz.pdf")
    plt.show()
