"""
Shows what a kernel table looks like, in momentum space and in position space.

    python3 plot_kernel.py <file> [picture.png]

<file> is one of the files the optimization writes in its run folder:
    state.json           -> the last ACCEPTED table (the best so far)
    theta/theta_0007.json -> the table tried at step 7
The picture goes to <file without .json>_kernel.png unless a name is given.

The kernel of one quark is K(d) = sum_j a_j G_j(d) exp(+i 2pi/L k_j d_z) (opt_kernel.py, the boost +k of prop2). Its 3D FFT
f(n) is what multiplies the quark field in momentum space: a sum of Gaussians, one per row of the table, with height a_j,
center k_j = mom_frac_j * mom_max and width ~ 2/rho. The script builds K from the table and Fourier transforms it; the
two are the same kernel seen from the two sides.

Top row, momentum space (n in units of 2pi/L): f along z with every single Gaussian drawn below the sum, f across,
and the map f(n_z, n_x). Bottom row, position space (d in lattice units): |K|, Re K and Im K along z, |K| across,
and the map |K|(d_z, d_x). The starting kernel is drawn in every line plot for comparison. Every curve is divided
by its own largest value, so only shapes are compared.
Needs numpy and matplotlib; no GPU.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for tools in (os.path.join(HERE, "..", "..", "2pt_production", "tools"), os.path.join(HERE, "..", "..", "tools")):  # laptop layout, Delta layout
    if os.path.exists(os.path.join(tools, "coulomb_smearing.py")):
        sys.path.insert(0, tools)
from opt_kernel import kernel_from_table  # noqa: E402

# ======== settings: the same numbers as in the setup file of the run ========
L = 32                                      # spatial lattice size
MOM_MAX = 6                                 # k_j = mom_frac_j * MOM_MAX
START_TABLE = [[1.0, 4.5, 2.25, 0.8]]       # the kernel every curve is compared with: the production kernel, where the run starts

# colors: this kernel, the starting kernel and the third curve keep the same color in every panel
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983", "#0b0b0b"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])       # size: one hue, light to dark
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#e34948", "#f0efec", "#2a78d6"])                          # sign: red negative, gray zero, blue positive


def read_table(name):
    content = json.load(open(name))
    if "accepted" in content:  # state.json
        acc = content["accepted"]
        return np.array(acc["table"], float), f"accepted table of step {acc['step']} (E = {acc['E']:.5f}, noise ratio {acc['noise_ratio']:.3f})"
    return np.array(content["table"], float), f"table of step {content['step']}"


def centered(a, axis):
    """Index 0..L-1 -> -L/2..L/2-1, so that zero momentum (or zero displacement) is in the middle."""
    return np.roll(a, L // 2, axis=axis)


def both_spaces(table):
    K = kernel_from_table(np, [L, L, L], table, MOM_MAX, +1)  # [dz, dy, dx]
    f = np.fft.fftn(K).real  # [nz, ny, nx]; the kernel is Hermitian, so f is real
    return K, f


table, label = read_table(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].rsplit(".json", 1)[0] + "_kernel.png"
K, f = both_spaces(table)
K0, f0 = both_spaces(np.array(START_TABLE, float))
axis_values = np.arange(-L // 2, L // 2)
nz_peak = int(np.argmax(np.abs(f[:, 0, 0])))  # the momentum along z where the kernel is largest

print(label)
print("      a          rho_T     rho_z     mom_frac")
for j, (a, rho_T, rho_z, frac) in enumerate(table):
    print(f"  {j + 1:2d}  {a:+.5f}   {rho_T:.4f}    {rho_z:.4f}    {frac:+.4f}")
d = (np.arange(L) + L // 2) % L - L // 2
for name, kernel in (("this kernel", K), ("starting kernel", K0)):
    w = np.abs(kernel) ** 2
    rms_z = np.sqrt((d[:, None, None] ** 2 * w).sum() / w.sum())
    rms_x = np.sqrt((d[None, None, :] ** 2 * w).sum() / w.sum())
    f_k = np.fft.fftn(kernel).real[:, 0, 0]
    peak = int(np.argmax(np.abs(f_k)))
    print(f"{name}: rms size of |K|^2 along z {rms_z:.3f}, across {rms_x:.3f}; momentum peak at n_z = {peak if peak < L // 2 else peak - L}")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": INK, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})  # fmt: skip
fig, ax = plt.subplots(2, 3, figsize=(16, 10.5))
fig.suptitle(f"Smearing kernel: {label}", x=0.01, ha="left", fontsize=13, color=INK)

# ---- momentum space ----
a = ax[0, 0]
scale = np.abs(f).max()
for j, row in enumerate(table):  # every single Gaussian of the sum, with its weight
    if row[0] != 0:
        single = both_spaces(np.array([row]))[1] * 1.0
        a.plot(axis_values, centered(single[:, 0, 0], 0) / scale, color=GRAY, lw=0.8, label="single Gaussians (a_j x Gaussian j)" if j == 0 else None)
a.plot(axis_values, centered(f0[:, 0, 0], 0) / np.abs(f0).max(), color=ORANGE, lw=2, ls="--", label="starting kernel")
a.plot(axis_values, centered(f[:, 0, 0], 0) / scale, color=BLUE, lw=2, label="this kernel (the sum)")
a.set(title="Momentum space, along z  (n_x = n_y = 0)", xlabel="n_z  (quark momentum along z, units of 2pi/L)", ylabel="f / largest |f|")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)      # below the panel, so that it never covers a curve

a = ax[0, 1]
a.plot(axis_values, centered(f0[int(np.argmax(np.abs(f0[:, 0, 0]))), 0, :], 0) / np.abs(f0).max(), color=ORANGE, lw=2, ls="--", label="starting kernel")
a.plot(axis_values, centered(f[nz_peak, 0, :], 0) / scale, color=BLUE, lw=2, label="this kernel")
a.set(title="Momentum space, across  (n_y = 0, n_z at the peak)", xlabel="n_x  (units of 2pi/L)", ylabel="f / largest |f|")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)      # below the panel, so that it never covers a curve

a = ax[0, 2]
image = centered(centered(f[:, 0, :], 0), 1) / scale  # [nz, nx]
m = a.imshow(image.T, origin="lower", extent=[-L / 2 - 0.5, L / 2 - 0.5, -L / 2 - 0.5, L / 2 - 0.5], cmap=DIVERGING, vmin=-1, vmax=1, aspect="equal")
a.set(title="Momentum space map  f(n_z, n_x)  at n_y = 0", xlabel="n_z", ylabel="n_x")
a.grid(False)
fig.colorbar(m, ax=a, label="f / largest |f|   (blue positive, red negative)")

# ---- position space ----
a = ax[1, 0]
scale_K = np.abs(K).max()
a.plot(axis_values, centered(np.abs(K0[:, 0, 0]), 0) / np.abs(K0).max(), color=GRAY, lw=1.5, ls="--", label="|K| of the starting kernel")
a.plot(axis_values, centered(K[:, 0, 0].imag, 0) / scale_K, color=AQUA, lw=1.5, marker="o", ms=3.5, label="Im K")
a.plot(axis_values, centered(K[:, 0, 0].real, 0) / scale_K, color=ORANGE, lw=1.5, marker="o", ms=3.5, label="Re K")
a.plot(axis_values, centered(np.abs(K[:, 0, 0]), 0) / scale_K, color=BLUE, lw=2, marker="o", ms=4, label="|K|")
a.set(title="Position space, along z  (d_x = d_y = 0)", xlabel="d_z  (lattice units)", ylabel="K / largest |K|")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)      # below the panel, so that it never covers a curve

a = ax[1, 1]
a.plot(axis_values, centered(np.abs(K0[0, 0, :]), 0) / np.abs(K0).max(), color=GRAY, lw=1.5, ls="--", label="|K| of the starting kernel")
a.plot(axis_values, centered(np.abs(K[0, 0, :]), 0) / scale_K, color=BLUE, lw=2, marker="o", ms=4, label="|K|")
a.set(title="Position space, across  (d_y = d_z = 0)", xlabel="d_x  (lattice units)", ylabel="|K| / largest |K|")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)      # below the panel, so that it never covers a curve

a = ax[1, 2]
image = centered(centered(np.abs(K[:, 0, :]), 0), 1) / scale_K  # [dz, dx]
m = a.imshow(image.T, origin="lower", extent=[-L / 2 - 0.5, L / 2 - 0.5, -L / 2 - 0.5, L / 2 - 0.5], cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="equal")
a.set(title="Position space map  |K|(d_z, d_x)  at d_y = 0", xlabel="d_z", ylabel="d_x")
a.grid(False)
fig.colorbar(m, ax=a, label="|K| / largest |K|")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out, dpi=130)
print("picture:", out)
