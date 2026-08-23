from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

frame = "symmetric"
operator = "TXTXpTYTYm2XYXY"
INPUT_BAREM = (SCRIPT_DIR.parent / "ratio_fit_simple"  / f"bare_matrix_element_{frame}")
TWOPT_FIT = SCRIPT_DIR.parent / "2PT_simple" / "twopt_fit_results.h5"
TWOPT_TAG = "nstate3_t3-15_svd1e-12"     # must match `tag` in 2pt_fit.py
OUTPUT_DIR = SCRIPT_DIR / f"itd_{frame}"
PLOT_DIR = SCRIPT_DIR / "itd_plots"

tgf_list = [20,25,30,35,40]
pf_list = [(-1, 0, pz) for pz in range(0, 7)]
q = (2, 0, 0)
bareM_tag = "tsep5-10_svd1e-07_dEf5"        # the tag on the bare matrix element files

Ls = 32
PF_REF = (-1, 0, 0)     # the reference momentum both ratios divide by
W_REF = 0


def jk_mean_err(values):
    n = values.shape[0]
    mean = values.mean(axis=0)
    err = np.sqrt((n - 1.0) / n * np.sum((values - mean) ** 2, axis=0))
    return mean, err


def mom_name(p):
    """The group / filename tag of a momentum, e.g. (-2,0,3) -> p-2_0_3."""
    return f"p{p[0]}_{p[1]}_{p[2]}"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ground-state energy of every pf, [jk, pf]
with h5py.File(TWOPT_FIT, "r") as f:
    g = f[TWOPT_TAG]
    E = np.stack([g[mom_name(pf)]["E_jk"][:, 0] for pf in pf_list], axis=1)

for flow in tgf_list:
    # bare matrix elements, [jk, pf, w]
    blocks = []
    for pf in pf_list:
        name = (f"bareM_{operator}_{frame}_tgf{flow}"
                f"_pf{pf[0]}_{pf[1]}_{pf[2]}"
                f"_q{q[0]}_{q[1]}_{q[2]}_{bareM_tag}.h5")
        with h5py.File(INPUT_BAREM / name, "r") as f:
            blocks.append(f["bare_matrix_element_jk"][:].real)
            w_list = f["w_list"][:]
    M = np.stack(blocks, axis=1)
    if E.shape[0] != M.shape[0]:
        print(f"WARNING: 2pt has {E.shape[0]} samples, bareM has {M.shape[0]}",
              flush=True)

    ipf = pf_list.index(PF_REF)
    iw = list(w_list).index(W_REF)
    # the Wilson line runs along z, so only the z component enters nu = p.z
    pz_of_pf = np.array([p[2] for p in pf_list], dtype=float)
    nu = (2.0 * np.pi / Ls) * pz_of_pf[:, None] * w_list[None, :]

    # every ratio formed sample by sample
    kinematic = E[:, ipf][:, None] / E                       # E(pf_ref) / E(pf)
    single = M / M[:, ipf:ipf + 1, :] * kinematic[:, :, None]
    double = ((M / M[:, :, iw:iw + 1])
              / (M[:, ipf:ipf + 1, :] / M[:, ipf:ipf + 1, iw:iw + 1]))

    single_mean, single_err = jk_mean_err(single)
    double_mean, double_err = jk_mean_err(double)

    stem = f"itd_{operator}_{frame}_tgf{flow}_{bareM_tag}"
    with h5py.File(OUTPUT_DIR / f"{stem}.h5", "w") as f:
        f.create_dataset("single_ratio_jk", data=single)
        f.create_dataset("double_ratio_jk", data=double)
        f.create_dataset("single_mean", data=single_mean)
        f.create_dataset("single_err", data=single_err)
        f.create_dataset("double_mean", data=double_mean)
        f.create_dataset("double_err", data=double_err)
        f.create_dataset("nu", data=nu)
        f.create_dataset("pf_list", data=np.array(pf_list, dtype=np.int64))
        f.create_dataset("w_list", data=w_list)
        f.attrs["dim_ratio_jk"] = "jk,pf_index,w_index"
        f.attrs["single_ratio"] = "[M(pf,w)/M(pf_ref,w)] * E(pf_ref)/E(pf)"
        f.attrs["double_ratio"] = (
            "[M(pf,w)/M(pf,w_ref)] / [M(pf_ref,w)/M(pf_ref,w_ref)]")
        f.attrs["nu"] = "2 pi pf_z w / Ls"
        f.attrs["part"] = "real"
        f.attrs["flow"] = flow
        f.attrs["pf_ref"] = np.array(PF_REF, dtype=np.int64)
        f.attrs["w_ref"] = W_REF
        f.attrs["q"] = np.array(q, dtype=np.int64)
        f.attrs["twopt_tag"] = TWOPT_TAG
        f.attrs["bareM_tag"] = bareM_tag

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for i, (mean, err, title) in enumerate(
            [(single_mean, single_err, r"single ratio $\times\, E_{\rm ref}/E$"),
             (double_mean, double_err, "double ratio")]):
        for ipf_plot, pf in enumerate(pf_list):
            axes[i].errorbar(nu[ipf_plot], mean[ipf_plot], yerr=err[ipf_plot],
                             marker="o", ls="-", lw=1, ms=4, capsize=3,
                             label=f"pf={pf}")
        axes[i].axhline(1.0, color="k", ls=":", lw=1, alpha=0.5)
        axes[i].set_xlabel(r"$\nu = 2\pi\, p_z w / L_s$")
        axes[i].set_title(title)
    axes[0].set_ylabel("Re")
    axes[1].legend(fontsize=8, ncol=2)
    axes[0].set_ylim(-0.2, 1.6)
    axes[1].set_ylim(-0.2, 1.2)
    fig.suptitle(f"{operator}   {frame}   flow={flow * 0.1:.1f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"flow {flow}: saved {stem}", flush=True)
print("finished", flush=True)
