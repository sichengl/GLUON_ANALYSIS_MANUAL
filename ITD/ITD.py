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
OUTPUT_DIR = SCRIPT_DIR / f"itd_{frame}"
PLOT_DIR = SCRIPT_DIR / "itd_plots"

tgf_list = [30]
pz_list = [0, 1, 2, 3, 4, 5, 6]
q = (0, 0, 0)
bareM_tag = "tsep6-10_svd1e-07_dEf5_sepw"        # the tag on the bare matrix element files

Ls = 32
PZ_REF = 0
W_REF = 0


def jk_mean_err(values):
    n = values.shape[0]
    mean = values.mean(axis=0)
    err = np.sqrt((n - 1.0) / n * np.sum((values - mean) ** 2, axis=0))
    return mean, err


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ground-state energy of every pz, [jk, pz]
with h5py.File(TWOPT_FIT, "r") as f:
    E = f["E_jk"][:, :, 0].T

for flow in tgf_list:
    # bare matrix elements, [jk, pz, w]
    blocks = []
    for pz in pz_list:
        name = (f"bareM_{operator}_{frame}_tgf{flow}_pf0_0_{pz}"
                f"_q{q[0]}_{q[1]}_{q[2]}_{bareM_tag}.h5")
        with h5py.File(INPUT_BAREM / name, "r") as f:
            blocks.append(f["bare_matrix_element_jk"][:].real)
            w_list = f["w_list"][:]
    M = np.stack(blocks, axis=1)

    ipz = pz_list.index(PZ_REF)
    iw = list(w_list).index(W_REF)
    nu = (2.0 * np.pi / Ls) * np.array(pz_list)[:, None] * w_list[None, :]

    # every ratio formed sample by sample
    kinematic = E[:, ipz][:, None] / E                       # m / E(pz)
    single = M / M[:, ipz:ipz + 1, :] * kinematic[:, :, None]
    double = ((M / M[:, :, iw:iw + 1])
              / (M[:, ipz:ipz + 1, :] / M[:, ipz:ipz + 1, iw:iw + 1]))

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
        f.create_dataset("pz_list", data=np.array(pz_list, dtype=np.int64))
        f.create_dataset("w_list", data=w_list)
        f.attrs["dim_ratio_jk"] = "jk,pz_index,w_index"
        f.attrs["single_ratio"] = "[M(pz,w)/M(pz_ref,w)] * m/E(pz)"
        f.attrs["double_ratio"] = (
            "[M(pz,w)/M(pz,w_ref)] / [M(pz_ref,w)/M(pz_ref,w_ref)]")
        f.attrs["nu"] = "2 pi pz w / Ls"
        f.attrs["part"] = "real"
        f.attrs["flow"] = flow
        f.attrs["pz_ref"] = PZ_REF
        f.attrs["w_ref"] = W_REF

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for i, (mean, err, title) in enumerate(
            [(single_mean, single_err, r"single ratio $\times\, m/E$"),
             (double_mean, double_err, "double ratio")]):
        for ipz_plot, pz in enumerate(pz_list):
            axes[i].errorbar(nu[ipz_plot], mean[ipz_plot], yerr=err[ipz_plot],
                             marker="o", ls="-", lw=1, ms=4, capsize=3,
                             label=f"pz={pz}")
        axes[i].axhline(1.0, color="k", ls=":", lw=1, alpha=0.5)
        axes[i].set_xlabel(r"$\nu = 2\pi\, p_z w / L_s$")
        axes[i].set_title(title)
    axes[0].set_ylabel("Re")
    axes[1].legend(fontsize=8, ncol=2)
    axes[0].set_ylim(-0.2, 1.2)
    axes[1].set_ylim(-0.2, 1.2)
    fig.suptitle(f"{operator}   {frame}   flow={flow * 0.1:.1f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"flow {flow}: saved {stem}", flush=True)
print("finished", flush=True)
