#Effective mass of the GEVP-reconstructed 2pt of every operator of a packed new-kernel file (fermilab_package_new_kernels.py), pion or proton.
#For each operator (one smearing kernel at source and sink) the 3 x 3 matrix of the gammas G5, G5G4, G3G5 is diagonalized as in
#local_analysis/build_local_GEVP_pion.ipynb:
#    C[half][cfg, a, b, pf, t]  the t_src mean of pt2_forward and of pt2_backward, each half on its own (the GEVP absorbs the time-reflection signs);
#                               C -> (C + C^dag) / 2; multiplied by the sign that makes C(t0) positive (-1 for the pion in this convention)
#    GEVP C(t) v = lambda C(t0) v of the cfg mean (Cholesky of C(t0)); v0[half][pf] = the ground-state vector at t_ref
#    reconstructed 2pt  C_rec[half][cfg, pf, t] = sign v0^dag C[half][cfg] v0, then C_rec = (forward + backward) / 2 for t >= 1
#    E_eff(t) = log(C_rec(t) / C_rec(t+1)), jackknife over cfgs (v0 fixed from the cfg mean)
#and every operator a is compared with G_ref on its reconstructed 2pt:
#    R      = [C_a(t2) / C_a(t1)] / [C_ref(t2) / C_ref(t1)]    > 1: less excited-state contamination than G_ref between t1 and t2
#    noise  = err(E_eff_a(t_n)) / err(E_eff_ref(t_n))          < 1: a more precise effective energy than G_ref at t_n
#    pull   = (E_eff_a(t_n) - E_eff_ref(t_n)) / its error      the same pion or proton: a large |pull| that shrinks at later t_n is excited states
#Writes a table (printed and saved as .txt) and a .pdf: R and noise against the strength of the Model M operators (spread of the local boost over
#the profile, in units of the test's M_plus; one line per R_c), then one page per pz with E_eff(t) of every operator against G_ref.
#    python3 gevp_look_new_kernels.py 2pt_model_M_scan1_13ops_pion_tsrc_fb_tsep20_32src_ncfg40.h5
#    python3 gevp_look_new_kernels.py 2pt_model_M_scan1_13ops_proton_tsrc_fb_tsep20_32src_ncfg40.h5 --t0 2 --t_ref 4
import os
import sys
import json
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coulomb_smearing_kernels as csk

parser = argparse.ArgumentParser()
parser.add_argument("packed_file")
parser.add_argument("--t0", type=int, default=2)            # GEVP C(t) v = lambda C(t0) v
parser.add_argument("--t_ref", type=int, default=3)         # time of the ground-state vector v0
parser.add_argument("--t1", type=int, default=3)            # R = [C(t2) / C(t1)] / [same for G_ref]
parser.add_argument("--t2", type=int, default=7)
parser.add_argument("--t_n", type=int, default=10)          # time of noise and pull
args = parser.parse_args()

# ======== data ========
with h5py.File(args.packed_file, "r") as f:
    data = {"forward": f["pt2_forward"][:], "backward": f["pt2_backward"][:]}     # (cfg, operator, g_snk, g_src, t_src, pf, tsep)
    pf_list = f["pf_list"][:]
    operator_names = [name.decode() if isinstance(name, bytes) else str(name) for name in f["operator_names"][:]]
    operator_list = json.loads(f.attrs["operator_list_json"])
    gamma_list = [str(g) for g in f.attrs["gamma_list"]]
    particle = "proton" if "parity_p" in f else "pion"
assert operator_names[0] == "G_ref", "the first operator must be the reference G_ref"
n_cfg, n_op, n_gamma, _, n_tsrc, n_pf, n_t = data["forward"].shape
pz_list = [int(pf[2]) for pf in pf_list]
assert 0 <= args.t0 < args.t_ref < n_t, f"need t0 < t_ref < {n_t} (the ground state is the largest eigenvalue only after t0)"
assert args.t1 < args.t2 < n_t and args.t_n < n_t - 1, f"need t1 < t2 < {n_t} and t_n < {n_t - 1}"
t_list = np.arange(args.t0 + 1, n_t)
print(f"{args.packed_file}: {particle}, {n_cfg} cfgs, {n_tsrc} t_src, {n_op} operators, {n_gamma} x {n_gamma} gammas {gamma_list}, "
      f"t0 {args.t0}, t_ref {args.t_ref}")
if n_cfg < 10:
    print(f"WARNING: only {n_cfg} cfgs, the jackknife errors are rough")


def jackknife_samples(x):
    """Leave-one-out means of x over its first axis (cfg)."""
    return (x.sum(axis=0)[None] - x) / (x.shape[0] - 1)

def jackknife_error(values):
    n = values.shape[0]
    return np.sqrt((n - 1) / n * ((values - values.mean(axis=0)) ** 2).sum(axis=0))


def ground_state_vector(C_mean):
    """
    v0[pf, a]: the ground-state GEVP vector at t_ref of the matrix C_mean[a, b, pf, t] (Cholesky of C(t0), largest eigenvalue first),
    as solve_gevp of build_local_GEVP_pion.ipynb; nan where C(t0) is not positive definite.
    """
    v0 = np.full((n_pf, n_gamma), np.nan, "<c16")
    for ipf in range(n_pf):
        try:
            L = np.linalg.cholesky(C_mean[:, :, ipf, args.t0])
        except np.linalg.LinAlgError:
            continue
        L_inv = np.linalg.inv(L)
        M = L_inv @ C_mean[:, :, ipf, args.t_ref] @ L_inv.conj().T
        w, u = np.linalg.eigh(M)
        v = L_inv.conj().T @ u
        v0[ipf] = v[:, np.argmax(w)]
    return v0


# ======== reconstructed 2pt of every operator: rec[cfg, operator, pf, t] ========
rec = np.zeros((n_cfg, n_op, n_pf, n_t))
for i_op in range(n_op):
    halves = {}
    for half in ["forward", "backward"]:
        C = data[half][:, i_op].mean(axis=3)                        # (cfg, a, b, pf, t): mean over t_src
        C = 0.5 * (C + np.conj(np.swapaxes(C, 1, 2)))               # average with C^dagger
        sign = np.sign(np.trace(C.mean(axis=0)[:, :, :, args.t0].sum(axis=2)).real)
        C = sign * C
        v0 = ground_state_vector(C.mean(axis=0))
        halves[half] = np.einsum("pa,cabpt,pb->cpt", np.conj(v0), C, v0).real
        if i_op == 0:
            print(f"G_ref {half}: sign {sign:+.0f}, ground-state vector at pz {pz_list[-1]}: {np.round(v0[-1], 3)}")
    rec[:, i_op] = halves["forward"]
    rec[:, i_op, :, 1:] = 0.5 * (halves["forward"][..., 1:] + halves["backward"][..., 1:])

rec_jk = jackknife_samples(rec)
rec_mean = rec.mean(axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    meff_jk = np.log(rec_jk[..., :-1] / rec_jk[..., 1:])            # (sample, operator, pf, t)
    meff = np.log(rec_mean[..., :-1] / rec_mean[..., 1:])
    meff_err = jackknife_error(meff_jk)
    ratio_jk = rec_jk[..., args.t2] / rec_jk[..., args.t1]           # (sample, operator, pf)
    R_jk = ratio_jk / ratio_jk[:, :1]
    R = (rec_mean[..., args.t2] / rec_mean[..., args.t1]) / (rec_mean[:1, :, args.t2] / rec_mean[:1, :, args.t1])
    R_err = jackknife_error(R_jk)
    noise = meff_err[..., args.t_n] / meff_err[:1, :, args.t_n]
    pull = (meff[..., args.t_n] - meff[:1, :, args.t_n]) / jackknife_error(meff_jk[..., args.t_n] - meff_jk[:, :1, :, args.t_n])
pull[0] = 0.0


# ======== strength of the Model M operators: spread of the local boost over the profile, in units of the test's M_plus ========
def model_M_spread(parameters):
    s = csk.transverse_s(np, [32, 32, 32])
    w = np.exp(-2 * s / parameters["rho_T"] ** 2)
    w = w / w.sum()
    f = csk.radial_function(s, parameters["R_c"])
    return parameters["delta_mu"] * np.sqrt((w * (f - (w * f).sum()) ** 2).sum()) / 0.849

spread = np.full(n_op, np.nan)
R_c = np.full(n_op, np.nan)
for i_op, (name, function, parameters) in enumerate(operator_list):
    if function == "model_M_kernel":
        spread[i_op] = model_M_spread(parameters)
        R_c[i_op] = parameters["R_c"]
    elif function == "gaussian_kernel" and name == "G_ref":
        spread[i_op] = 0.0


# ======== table ========
lines = [f"{particle}, GEVP {n_gamma}x{n_gamma} ({', '.join(gamma_list)}) per operator, t0 {args.t0}, t_ref {args.t_ref}, {n_cfg} cfgs (jackknife); "
         f"R = [C({args.t2})/C({args.t1})] / G_ref (> 1 better), noise = err E_eff({args.t_n}) / G_ref (< 1 better), pull = E_eff({args.t_n}) - G_ref in sigma"]
lines.append(f"{'operator':14s} {'spread':>6s} {'R_c':>5s} | " + " | ".join(f"pz={pz}: R, noise, pull, E_eff({args.t_n})".ljust(44) for pz in pz_list))
for i_op, name in enumerate(operator_names):
    cells = []
    for i_pf in range(n_pf):
        cells.append(f"{R[i_op, i_pf]:.3f}({R_err[i_op, i_pf]:.3f}) {noise[i_op, i_pf]:5.2f} {pull[i_op, i_pf]:+5.1f} "
                     f"{meff[i_op, i_pf, args.t_n]:.4f}({meff_err[i_op, i_pf, args.t_n]:.4f})".ljust(44))
    lines.append(f"{name:14s} {spread[i_op]:+6.2f} {R_c[i_op]:5.2f} | " + " | ".join(cells))
table = "\n".join(lines)
print(table)
stem = os.path.splitext(os.path.basename(args.packed_file))[0] + f"_gevp_t0{args.t0}_tref{args.t_ref}"
with open(f"{stem}.txt", "w") as f:
    f.write(table + "\n")


# ======== plots ========
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]                  # one color per R_c, in a fixed order
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

with PdfPages(f"{stem}.pdf") as pdf:
    # page 1: R and noise of the reconstructed 2pt against the Model M strength, one line per R_c, every pz
    R_c_values = sorted({float(r) for r in R_c if np.isfinite(r)})
    if R_c_values:
        fig, axes = plt.subplots(2, n_pf, figsize=(2.0 * n_pf + 1.5, 5.6), sharex=True, squeeze=False)
        for i_pf, pz in enumerate(pz_list):
            for row, (values, errors, label) in enumerate(((R, R_err, "R  (> 1 better)"), (noise, None, "noise  (< 1 better)"))):
                ax = axes[row, i_pf]
                ax.axhline(1.0, color=GRAY, lw=1, ls="--")
                for i_rc, rc in enumerate(R_c_values):
                    members = [0] + [i for i in range(n_op) if R_c[i] == rc]            # G_ref (spread 0) belongs to every line
                    members = sorted(members, key=lambda i: spread[i])
                    x = spread[members]
                    y = values[members, i_pf]
                    if errors is None:
                        ax.plot(x, y, "o-", color=SERIES[i_rc % 3], ms=3.5, lw=1.4, label=f"R_c {rc:g}")
                    else:
                        ax.errorbar(x, y, errors[members, i_pf], fmt="o-", color=SERIES[i_rc % 3], ms=3.5, lw=1.4, capsize=2, label=f"R_c {rc:g}")
                ax.grid(color=GRID, lw=0.5)
                if i_pf == 0:
                    ax.set_ylabel(label)
                if row == 0:
                    ax.set_title(f"pz = {pz}", fontsize=9, color=INK)
                else:
                    ax.set_xlabel("spread (x M_plus)")
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.suptitle(f"{particle}: Model M against G_ref on the GEVP-reconstructed 2pt (t0 {args.t0}, t_ref {args.t_ref}, "
                     f"R from t = {args.t1}, {args.t2}, noise at t = {args.t_n}, {n_cfg} cfgs)", fontsize=9, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

    # one page per pz: E_eff(t) of the reconstructed 2pt, every operator (blue) against G_ref (gray)
    t = np.arange(n_t - 1)
    n_col = 4
    n_row = max(1, int(np.ceil((n_op - 1) / n_col)))
    for i_pf, pz in enumerate(pz_list):
        if n_op == 1:
            break
        fig, axes = plt.subplots(n_row, n_col, figsize=(13, 2.4 * n_row), sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()
        good = np.isfinite(meff[0, i_pf]) & np.isfinite(meff_err[0, i_pf]) & (t >= args.t1)
        clean = good & (meff_err[0, i_pf] < 0.2)
        if clean.any():
            good = clean
        for i_op in range(1, n_op):
            ax = axes[i_op - 1]
            ax.errorbar(t - 0.1, meff[0, i_pf], meff_err[0, i_pf], fmt="o", ms=3, lw=1, color=GRAY)
            ax.errorbar(t + 0.1, meff[i_op, i_pf], meff_err[i_op, i_pf], fmt="o", ms=3, lw=1, color=SERIES[0])
            ax.set_title(f"{operator_names[i_op]}: R {R[i_op, i_pf]:.3f}, noise {noise[i_op, i_pf]:.2f}", fontsize=8, color=INK)
            ax.grid(color=GRID, lw=0.5)
            if good.any():
                low = np.min((meff[0, i_pf] - meff_err[0, i_pf])[good])
                high = np.max((meff[0, i_pf] + meff_err[0, i_pf])[good])
                ax.set_ylim(low - 0.3 * (high - low), high + 0.3 * (high - low))
        for ax in axes[n_op - 1:]:
            ax.set_visible(False)
        for ax in axes[max(0, n_op - 1 - n_col):n_op - 1]:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("t")
        handles = [plt.Line2D([], [], color=GRAY, marker="o", ls=""), plt.Line2D([], [], color=SERIES[0], marker="o", ls="")]
        fig.legend(handles, ["G_ref", "operator of the panel"], loc="upper right", frameon=False)
        fig.suptitle(f"{particle}: E_eff(t) of the GEVP-reconstructed 2pt, pz = {pz}, t0 {args.t0}, t_ref {args.t_ref}", fontsize=10, color=INK)
        fig.tight_layout(rect=[0, 0, 0.92, 0.96])
        pdf.savefig(fig)
        plt.close(fig)
print(f"saved {stem}.txt and {stem}.pdf")
