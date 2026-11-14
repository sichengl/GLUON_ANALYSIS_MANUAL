"""
Checks the files written by a mock run (run_mock.sh) against an independent calculation in the production order:
whole-lattice arrays, the kernel placed at the source, the mock inverse, sink smearing by FFT, site-local contraction
with the momentum phase, for every gamma pair.

  A. correlator matrix (w = 0) of every node, source, gamma pair, pz, forward/backward tau: must agree to rounding.
  B. sink derivative (w >= 1): against a finite difference of the same calculation with only the SINK kernel changed.
  C. the identity the method rests on: derivative with the kernel changed at BOTH ends = D + D^dagger. The mock solver is
     the free Wilson operator, where the identity is exact for every source (in QCD it holds on average over cfgs).

Usage: python3 check_mock_output.py <scratch>/run [<scratch>/code]      (default code folder: next to run)
Checks results, not speed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
run_dir = sys.argv[1]
code_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(run_dir.rstrip("/")), "code")
sys.path.insert(0, os.path.join(HERE, "stubs"))
sys.path.insert(0, code_dir)

from mock_free import free_wilson_inverse, mock_mass, gamma  # noqa: E402
from opt_kernel import kernel_from_table, table_to_vector, vector_to_table, parameter_names  # noqa: E402
from opt_check import corr_name, theta_name, read_json  # noqa: E402
import glob  # noqa: E402
import importlib  # noqa: E402

setup = importlib.import_module(os.path.basename(glob.glob(os.path.join(code_dir, "opt_setup_*.py"))[0])[:-3])  # the mock setup of this run folder: importing it does not start MPI

p = setup.delta_params
L, T = p.GLs, p.GLt
G4, G5 = gamma(8), gamma(15)
gammas = [gamma(n) for n in setup.opt_gamma_ids]
A_list = [G5 @ g for g in gammas]
B_list = [(G4 @ g.conj().T @ G4) @ G5 for g in gammas]
pz_list, t_max = setup.opt_pz_list, setup.opt_t_max


def solutions(table, cfg, src_pos):
    """prop1 (source boosts -k) and prop2 (+k) on the whole lattice, unsmeared at the sink."""
    out = []
    for sign in (-1, +1):
        K = kernel_from_table(np, [L, L, L], table, p.mom_max, sign)
        source = np.zeros((T, L, L, L, 4, 4, 3, 3), complex)
        x0, y0, z0, t0 = src_pos
        eta = np.roll(K, (z0, y0, x0), axis=(0, 1, 2))
        source[t0] = eta[..., None, None, None, None] * np.eye(4)[:, :, None, None] * np.eye(3)[None, None]
        out.append(free_wilson_inverse(source, mock_mass(cfg)))
    return out


def correlator(S1, S2, sink_table, src_pos):
    """C[fb, a, b, pz, tau]: production order, sink smearing then the site sum with exp(-2 pi i P.(x - x_src)/L)."""
    smear = lambda S, K: np.fft.ifftn(np.fft.fftn(S, axes=(1, 2, 3)) * np.fft.fftn(K)[None, :, :, :, None, None, None, None], axes=(1, 2, 3))
    p1 = smear(S1, kernel_from_table(np, [L, L, L], sink_table, p.mom_max, -1))
    p2 = smear(S2, kernel_from_table(np, [L, L, L], sink_table, p.mom_max, +1))
    z = np.arange(L).reshape(L, 1, 1)
    t0 = src_pos[3]
    times = [(t0 + np.arange(t_max + 1)) % T, (t0 - np.arange(t_max + 1)) % T]
    C = np.zeros((2, len(gammas), len(gammas), len(pz_list), t_max + 1), complex)
    for a, A in enumerate(A_list):
        for b, B in enumerate(B_list):
            site = np.einsum("tzyxjiba,jk,tzyxklba,li->tzyx", p1.conj(), A, p2, B)
            for ipz, pz in enumerate(pz_list):
                full = (site * np.exp(-2j * np.pi * pz * (z - src_pos[2]) / L)).sum(axis=(1, 2, 3))
                for fb in range(2):
                    C[fb, a, b, ipz] = full[times[fb]]
    return C


steps = sorted(int(name.split("_")[1]) for name in os.listdir(f"{run_dir}/corr"))
print(f"run {run_dir}: steps {steps}, {len(setup.opt_cfg_index)} nodes")
worst_A = worst_B = worst_C = 0
for step in [steps[0], steps[-1]]:
    table = np.array(read_json(theta_name(run_dir, step))["table"])
    names = parameter_names(len(table))
    for worker in range(len(setup.opt_cfg_index)):
        if not os.path.exists(corr_name(run_dir, step, worker)):  # a lost node (many-node folders)
            continue
        data = np.load(corr_name(run_dir, step, worker))
        assert np.array_equal(data["table"], table)
        cfg, sources = int(data["cfg"]), data["source_list"]
        for i_src in [0, len(sources) - 1] if worker else range(len(sources)):  # all sources of node 0, two of the others
            src_pos = [int(v) for v in sources[i_src]]
            S1, S2 = solutions(table, cfg, src_pos)
            ref = correlator(S1, S2, table, src_pos)
            out = data["C"][i_src]  # [fb, a, b, pz, w, tau]
            worst_A = max(worst_A, abs(out[:, :, :, :, 0] - ref).max() / abs(ref).max())
            if worker == 0 and i_src == 0:
                vec = table_to_vector(table)
                for i in range(len(vec)):  # every parameter
                    h = 1e-5
                    up, dn = vec.copy(), vec.copy()
                    up[i] += h
                    dn[i] -= h
                    # B: only the sink kernel changes
                    fd = (correlator(S1, S2, vector_to_table(up), src_pos) - correlator(S1, S2, vector_to_table(dn), src_pos)) / (2 * h)
                    D = out[:, :, :, :, 1 + i]
                    scale = max(abs(fd).max(), 1e-12 * abs(ref).max())
                    worst_B = max(worst_B, abs(D - fd).max() / scale)
                    # C: the kernel changes at both ends (new inversions) vs D + D^dagger
                    if i % 3 == 0:
                        both = (correlator(*solutions(vector_to_table(up), cfg, src_pos), vector_to_table(up), src_pos)
                                - correlator(*solutions(vector_to_table(dn), cfg, src_pos), vector_to_table(dn), src_pos)) / (2 * h)  # fmt: skip
                        sym = D + D.conj().swapaxes(1, 2)
                        scale = max(abs(both).max(), 1e-12 * abs(ref).max())
                        err = abs(sym - both).max() / scale
                        worst_C = max(worst_C, err)
    print(f" step {step}: A (correlator matrix) {worst_A:.1e}   B (sink derivative) {worst_B:.1e}   C (both ends = D + D^dag) {worst_C:.1e}")
assert worst_A < 1e-10 and worst_B < 1e-5 and worst_C < 1e-5
print("all checks passed")
