#Quick look at the new-kernel test: is every operator working, and does any of them give a better pion signal than the reference G_ref?
#Reads the packed file of fermilab_package_new_kernels.py (read only) and, for one diagonal gamma pair, compares every operator a with G_ref:
#    R      = [C_a(t2) / C_a(t1)] / [C_ref(t2) / C_ref(t1)]   > 1: less excited-state contamination than G_ref between t1 and t2
#    noise  = err(E_eff_a(t_n)) / err(E_eff_ref(t_n))         < 1: a more precise effective energy than G_ref at t_n
#    pull   = (E_eff_a(t_n) - E_eff_ref(t_n)) / its error     every operator creates the same pion, but E_eff(t_n) still holds the excited
#                                                             states of each operator: a large |pull| from a working operator shrinks
#                                                             toward late t, a broken kernel never meets G_ref (see its E_eff page)
#    imag   = |mean Im C_a(t_n)| / err(Im C_a(t_n))            K(-d) = K(d)^* at source and sink: C is real up to noise
#(R and noise are the measures of the smearing tuning.) pull and imag are Student t variables with n_sample - 1 degrees of freedom; the checks
#flag them above a Bonferroni cut over all cells (5 % chance that noise alone flags any cell), and flag any value that is not finite.
#Errors are jackknife over cfgs, or over (cfg, t_src) with --per_tsrc (more samples for a test with few cfgs; the t_src are 24 apart).
#C(t) is the mean of forward and backward (the time-reflection sign is +1 on the diagonal).
#The controlled pairs of the note (Sec. 8.2) are also compared directly: M_plus / M_minus (the sign of the transverse-momentum correlation),
#two_momentum / G_rz1.5 (sidebands at a fixed width), V / two_momentum (transverse-dependent against constant splitting).
#Writes a table (printed and saved as .txt) and a .pdf: a summary page of R, noise and pull for every operator and pz, then one page per pz
#with E_eff(t) of every operator against G_ref.
#With --tune1 FILE (the packed pion file of the Fermilab smearing tuning, same cfgs and sources) it also checks every gaussian_kernel operator
#(G_ref, G_rz1.5) against the tuning's smearing of the same [rho_T, rho_z, k] cfg by cfg: they must agree to solver precision.
#    python3 quick_look_new_kernels.py 2pt_new_kernels_test1_14ops_pion_tsrc_fb_tsep20_32src_ncfg4.h5 --per_tsrc
import os
import json
import argparse
import numpy as np
import h5py
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser()
parser.add_argument("packed_file")
parser.add_argument("--gamma", default="G5G4")              # diagonal gamma pair: G5, G5G4 or G3G5
parser.add_argument("--t1", type=int, default=3)            # R = [C(t2) / C(t1)] / [same for the reference]
parser.add_argument("--t2", type=int, default=7)
parser.add_argument("--t_n", type=int, default=10)          # time of noise, pull and imag
parser.add_argument("--per_tsrc", action="store_true")      # jackknife over (cfg, t_src) instead of cfg
parser.add_argument("--tune1", default=None)                # packed pion file of the smearing tuning, for the check of the Gaussian operators
args = parser.parse_args()

# ======== data ========
with h5py.File(args.packed_file, "r") as f:
    forward = f["pt2_forward"][:]                           # (cfg, operator, g_snk, g_src, t_src, pf, tsep)
    backward = f["pt2_backward"][:]
    pf_list = f["pf_list"][:]
    cfg_list = f["cfg_list"][:]
    t_src_list = f["t_src_list"][:]
    operator_names = [name.decode() if isinstance(name, bytes) else str(name) for name in f["operator_names"][:]]
    kernel_moments = f["kernel_moments"][:]                 # rows [rms_T, rms_z, r_source, mean_k, width_k]
    gamma_list = [str(g) for g in f.attrs["gamma_list"]]
    operator_list = json.loads(f.attrs["operator_list_json"])
assert operator_names[0] == "G_ref", "the first operator must be the reference G_ref"
ig = gamma_list.index(args.gamma)
n_cfg, n_op, n_tsrc, n_pf, n_t = forward.shape[0], forward.shape[1], forward.shape[4], forward.shape[5], forward.shape[6]
pz_list = [int(pf[2]) for pf in pf_list]

# C(t) on the diagonal gamma pair: forward and backward averaged for t >= 1
corr = forward[:, :, ig, ig].copy()                         # (cfg, operator, t_src, pf, tsep)
corr[..., 1:] = 0.5 * (forward[:, :, ig, ig, ..., 1:] + backward[:, :, ig, ig, ..., 1:])
if args.per_tsrc:
    samples = corr.transpose(0, 2, 1, 3, 4).reshape(n_cfg * n_tsrc, n_op, n_pf, n_t)     # (cfg x t_src, operator, pf, tsep)
else:
    samples = corr.mean(axis=2)                                                            # (cfg, operator, pf, tsep)
n_sample = samples.shape[0]
print(f"{args.packed_file}: {n_cfg} cfgs, {n_tsrc} t_src, {n_op} operators, gamma {args.gamma}-{args.gamma}, "
      f"jackknife over {n_sample} {'(cfg, t_src)' if args.per_tsrc else 'cfgs'}")
if n_sample < 10:
    print(f"WARNING: only {n_sample} jackknife samples, the errors are rough")


# ======== jackknife ========
def jackknife_samples(x):
    """Leave-one-out means of x over its first axis."""
    return (x.sum(axis=0)[None] - x) / (x.shape[0] - 1)

def jackknife_error(values):
    """Error of a quantity from its leave-one-out values (first axis)."""
    n = values.shape[0]
    return np.sqrt((n - 1) / n * ((values - values.mean(axis=0)) ** 2).sum(axis=0))

real_jk = jackknife_samples(samples.real)                   # (sample, operator, pf, tsep)
imag_jk = jackknife_samples(samples.imag)
real_mean = samples.real.mean(axis=0)                       # (operator, pf, tsep)
with np.errstate(divide="ignore", invalid="ignore"):
    meff_jk = np.log(real_jk[..., :-1] / real_jk[..., 1:])   # E_eff(t) = log(C(t) / C(t+1)), t = 0 .. n_t - 2
    meff = np.log(real_mean[..., :-1] / real_mean[..., 1:])
    meff_err = jackknife_error(meff_jk)
    imag = abs(samples.imag.mean(axis=0)[..., args.t_n]) / jackknife_error(imag_jk[..., args.t_n])     # (operator, pf)


def compare(i_a, i_b):
    """R, its error, noise and pull of operator i_a against operator i_b, every pz (jackknife, correlated through the shared samples)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_a_jk = real_jk[:, i_a, :, args.t2] / real_jk[:, i_a, :, args.t1]                      # (sample, pf)
        ratio_b_jk = real_jk[:, i_b, :, args.t2] / real_jk[:, i_b, :, args.t1]
        R_jk = ratio_a_jk / ratio_b_jk
        R =(real_mean[i_a, :, args.t2] / real_mean[i_a, :, args.t1]) / (real_mean[i_b, :, args.t2] / real_mean[i_b, :, args.t1])
        noise = meff_err[i_a, :, args.t_n] / meff_err[i_b, :, args.t_n]
        pull = (meff[i_a, :, args.t_n] - meff[i_b, :, args.t_n]) / jackknife_error(meff_jk[:, i_a, :, args.t_n] - meff_jk[:, i_b, :, args.t_n])
    return R, jackknife_error(R_jk), noise, pull

R = np.zeros((n_op, n_pf))
R_err = np.zeros((n_op, n_pf))
noise = np.zeros((n_op, n_pf))
pull = np.zeros((n_op, n_pf))
for i_op in range(1, n_op):
    R[i_op], R_err[i_op], noise[i_op], pull[i_op] = compare(i_op, 0)
R[0], noise[0] = 1.0, 1.0                                   # G_ref against itself: R 1 (error 0), noise 1, pull 0
ref_ok = np.isfinite(meff[0, :, args.t_n]) & np.isfinite(meff_err[0, :, args.t_n])    # pz where G_ref has an E_eff(t_n) to compare with
noise[0, ~ref_ok] = np.nan
cut =student_t.isf(0.05 / (2 * 2 * n_op * n_pf), n_sample - 1)    # two-sided, over the pull and imag cells


# ======== table ========
lines = [f"gamma {args.gamma}-{args.gamma}, {n_sample} jackknife samples; R = [C({args.t2})/C({args.t1})] / G_ref (> 1 better), "
         f"noise = err E_eff({args.t_n}) / G_ref (< 1 better), pull = E_eff({args.t_n}) - G_ref in sigma, imag = |Im C({args.t_n})| in sigma; "
         f"checks flag |pull| or imag above {cut:.1f} (Bonferroni over {2 * n_op * n_pf} cells) or not finite"]
lines.append(f"{'operator':14s} {'rms_T':>6s} {'rms_z':>6s} {'k_mean':>6s} | " + " | ".join(f"pz={pz}: R, noise, pull, imag".ljust(38) for pz in pz_list))
for i_op, name in enumerate(operator_names):
    rms_T, rms_z, r_source, mean_k, width_k = kernel_moments[i_op]
    cells = []
    for i_pf in range(n_pf):
        cells.append(f"{R[i_op, i_pf]:.3f}({R_err[i_op, i_pf]:.3f}) {noise[i_op, i_pf]:5.2f} {pull[i_op, i_pf]:+5.1f} {imag[i_op, i_pf]:4.1f}".ljust(38))
    lines.append(f"{name:14s} {rms_T:6.3f} {rms_z:6.3f} {mean_k:6.2f} | " + " | ".join(cells))

pairs = []
for name_a, name_b, question in (("M_plus", "M_minus", "sign of the transverse-momentum correlation"),
                                 ("two_momentum", "G_rz1.5", "sidebands at the same z width"),
                                 ("V", "two_momentum", "transverse-dependent against constant splitting")):
    if name_a in operator_names and name_b in operator_names:
        pair_R, pair_R_err, pair_noise, pair_pull = compare(operator_names.index(name_a), operator_names.index(name_b))
        cells = [f"pz={pz}: {pair_R[i]:.3f}({pair_R_err[i]:.3f}) {pair_noise[i]:5.2f}" for i, pz in enumerate(pz_list)]
        pairs.append(f"  {name_a} / {name_b} ({question}): " + ", ".join(cells))
if pairs:
    lines.append("controlled pairs (Sec. 8.2 of the note): R(error) and noise of the first against the second")
    lines += pairs

flagged = []
for i_pf, pz in enumerate(pz_list):
    if not ref_ok[i_pf]:
        flagged.append(f"G_ref pz={pz}: E_eff({args.t_n}) is not finite, so no pull or noise at this pz (try a smaller --t_n)")
for i_op, name in enumerate(operator_names):
    for i_pf, pz in enumerate(pz_list):
        bad_pull = ref_ok[i_pf] and not abs(pull[i_op, i_pf]) <= cut          # a nan pull is flagged too, where G_ref itself is finite
        if bad_pull or not imag[i_op, i_pf] <= cut or not np.isfinite(R[i_op, i_pf]):
            flagged.append(f"{name} pz={pz}: pull {pull[i_op, i_pf]:+.1f}, imag {imag[i_op, i_pf]:.1f}, R {R[i_op, i_pf]:.3f}")
if not flagged:
    lines.append(f"checks: every operator has a finite |pull| <= {cut:.1f}, imag <= {cut:.1f} and a finite R")
else:
    lines.append(f"checks: FLAGGED (look at the E_eff page of each; a pull that shrinks at later --t_n is excited states, not a broken kernel):\n  "
                 + "\n  ".join(flagged))


# ======== optional: the Gaussian operators against the smearing tuning ========
#worst: max |new - old| / max |old| over a whole operator (the verdict); per t: the same with max |old| taken per (pf, tsep), for information
#(late t at high pz holds the smallest numbers, where two runs with solver tolerance 1e-10 can differ the most)
if args.tune1 is not None:
    with h5py.File(args.tune1, "r") as f:
        tune_smear_list = f["smear_list"][:]                # rows [rho_T, rho_z, mom_frac], k = mom_frac * 6
        tune_cfgs = list(f["cfg_list"][:])
        tune_tsrc = f["t_src_list"][:]
        assert np.array_equal(f["pf_list"][:], pf_list), "the two files keep different sink momenta"
        common = [int(cfg) for cfg in cfg_list if cfg in tune_cfgs]
        for i_op, (name, function, parameters) in enumerate(operator_list):
            if function != "gaussian_kernel":
                continue
            match = np.where(np.isclose(tune_smear_list[:, 0], parameters["rho_T"]) & np.isclose(tune_smear_list[:, 1], parameters["rho_z"])
                             & np.isclose(tune_smear_list[:, 2] * 6, parameters["k0"]))[0]
            if len(match) != 1 or not common:
                lines.append(f"{name} vs the tuning: NOT CHECKED ({len(match)} matching smearings in the tuning file, {len(common)} common cfgs)")
                continue
            worst, worst_per_t, t_src_differ = 0.0, 0.0, []
            for cfg in common:
                i_new, i_old = list(cfg_list).index(cfg), tune_cfgs.index(cfg)
                if not np.array_equal(t_src_list[i_new], tune_tsrc[i_old]):
                    t_src_differ.append(cfg)
                    continue
                for pt2, new_data in (("pt2_forward", forward), ("pt2_backward", backward)):
                    old = f[pt2][i_old, match[0]]           # (g_snk, g_src, t_src, pf, tsep)
                    diff = abs(new_data[i_new, i_op] - old)
                    worst = max(worst, float(diff.max() / abs(old).max()))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        worst_per_t = max(worst_per_t, float(np.nanmax(diff / abs(old).max(axis=(0, 1, 2), keepdims=True))))
                    if not np.isfinite(diff).all():
                        worst = np.inf                      # a nan in either file is a difference
            same = worst < 1e-6 and not t_src_differ
            lines.append(f"{name} vs the tuning's {tune_smear_list[match[0]].tolist()} on {len(common)} of {n_cfg} cfgs: max relative difference "
                         f"{worst:.2e} (per t {worst_per_t:.2e}) " + ("(the same correlators)" if same else "DIFFERENT: check the sources and the kernel")
                         + (f", different t_src on cfgs {t_src_differ}" if t_src_differ else ""))

table = "\n".join(lines)
print(table)
stem = os.path.splitext(os.path.basename(args.packed_file))[0] + f"_quicklook_{args.gamma}" + ("_per_tsrc" if args.per_tsrc else "")
with open(f"{stem}.txt", "w") as f:
    f.write(table + "\n")


# ======== plots ========
GRAY, BLUE, RED, INK, MUTED = "#8a8985", "#2a78d6", "#e34948", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

def diverging(better_low):
    """blue - gray - red map centered on the no-change value: blue is the better side."""
    colors = [BLUE, "#f0efec", RED] if better_low else [RED, "#f0efec", BLUE]
    return LinearSegmentedColormap.from_list("diverging", colors)

with PdfPages(f"{stem}.pdf") as pdf:
    # summary page: R, noise and pull for every operator and pz
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * n_op + 2.2))
    panels = [("R  (> 1: less contamination)", R, diverging(False), 1.0, 0.25),
              ("noise  (< 1: more precise)", noise, diverging(True), 1.0, 0.5),
              (f"pull of E_eff({args.t_n}) vs G_ref  (includes excited states)", pull, diverging(True), 0.0, cut)]
    for ax, (title, values, cmap, center, span) in zip(axes, panels):
        ax.imshow(np.clip(values, center - span, center + span), cmap=cmap, vmin=center - span, vmax=center + span, aspect="auto")
        for i_op in range(n_op):
            for i_pf in range(n_pf):
                ax.text(i_pf, i_op, f"{values[i_op, i_pf]:.2f}", ha="center", va="center", fontsize=6, color=INK)
        ax.set_xticks(range(n_pf), [str(pz) for pz in pz_list])
        ax.set_yticks(range(n_op), operator_names)
        ax.set_xlabel("pz [2pi/L]")
        ax.set_title(title, fontsize=9, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"{os.path.basename(args.packed_file)}, {args.gamma}-{args.gamma}, t1 {args.t1}, t2 {args.t2}, t_n {args.t_n}, "
                 f"{n_sample} jackknife samples", fontsize=9, color=INK)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # one page per pz: E_eff(t) of every operator (blue) against G_ref (gray); the y range follows the clean points of G_ref at t >= t1
    t = np.arange(n_t - 1)
    n_col = 4
    n_row = int(np.ceil((n_op - 1) / n_col))
    for i_pf, pz in enumerate(pz_list):
        if n_op == 1:
            break                                           # G_ref alone: nothing to compare
        fig, axes = plt.subplots(n_row, n_col, figsize=(13, 2.4 * n_row), sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()
        good = np.isfinite(meff[0, i_pf]) & np.isfinite(meff_err[0, i_pf]) & (t >= args.t1)
        clean = good & (meff_err[0, i_pf] < 0.2)
        if clean.any():
            good = clean
        for i_op in range(1, n_op):
            ax = axes[i_op - 1]
            ax.errorbar(t - 0.1, meff[0, i_pf], meff_err[0, i_pf], fmt="o", ms=3, lw=1, color=GRAY, label="G_ref")
            ax.errorbar(t + 0.1, meff[i_op, i_pf], meff_err[i_op, i_pf], fmt="o", ms=3, lw=1, color=BLUE, label="operator")
            ax.set_title(f"{operator_names[i_op]}: R {R[i_op, i_pf]:.2f}, noise {noise[i_op, i_pf]:.2f}", fontsize=8, color=INK)
            ax.grid(color="#e4e3df", lw=0.5)
            if good.any():                                  # otherwise G_ref has no finite E_eff here: matplotlib picks the range
                low = np.min((meff[0, i_pf] - meff_err[0, i_pf])[good])
                high = np.max((meff[0, i_pf] + meff_err[0, i_pf])[good])
                ax.set_ylim(low - 0.3 * (high - low), high + 0.3 * (high - low))
        for ax in axes[n_op - 1:]:
            ax.set_visible(False)
        for ax in axes[max(0, n_op - 1 - n_col):n_op - 1]:      # the lowest visible panel of every column carries the t axis
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("t")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, ["G_ref", "operator of the panel"], loc="upper right", frameon=False)
        fig.suptitle(f"E_eff(t), pz = {pz}, {args.gamma}-{args.gamma}", fontsize=10, color=INK)
        fig.tight_layout(rect=[0, 0, 0.92, 0.96])
        pdf.savefig(fig)
        plt.close(fig)
print(f"saved {stem}.txt and {stem}.pdf")
