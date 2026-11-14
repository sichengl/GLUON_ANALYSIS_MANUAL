"""
Laptop check of the decision logic of opt_check.py on a toy model (no lattice): fake nodes write correlator files from a
formula in which the effective energy has a known minimum, the noise grows when rho_z of Gaussian 1 shrinks, and the
noise can be partly reduced by moving mom_frac. The real check_step() then runs the whole loop.

Looked at: E falls, every accepted point is below the noise cut, a kernel step that raises E is rejected, the mom_frac
scan moves in the direction that lowers the noise, the run stops by itself, a repeated check (restart) changes nothing,
and (many-node settings) a node that stops delivering is dropped and the run goes on with the same cfgs in every comparison.
Usage: python3 test_check_toy.py <empty scratch directory>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "2pt_production", "tools"))

from opt_check import check_step, write_start, corr_name, theta_name, state_name, read_json, stop_name  # noqa: E402
from opt_kernel import table_to_vector, parameter_names  # noqa: E402

scratch = sys.argv[1]
start = np.array([[1.0, 4.5, 2.25, 0.8], [0.0, 3.0, 1.5, 0.5], [0.0, 6.0, 3.0, 1.1]])
names = parameter_names(len(start))
n_cfg, n_src, t_max = 6, 8, 6
pz_list = [0, 1]
rng = np.random.default_rng(3)
z = rng.normal(size=(n_cfg, n_src, 2, len(pz_list), t_max + 1))  # the same "gauge noise" at every step, as with fixed cfgs and sources

best = {"rhoT1": 5.0, "rhoz1": 1.0}  # where the toy energy is lowest


def toy(table):
    """C[cfg, src, fb, 1, 1, pz, w, tau] of the toy model."""
    vec = table_to_vector(table)
    value = dict(zip(names, vec))
    E = 0.9 + 0.02 * (value["rhoT1"] - best["rhoT1"]) ** 2 + 0.05 * (value["rhoz1"] - best["rhoz1"]) ** 2
    dE = np.zeros(len(vec))
    dE[names.index("rhoT1")] = 0.04 * (value["rhoT1"] - best["rhoT1"])
    dE[names.index("rhoz1")] = 0.10 * (value["rhoz1"] - best["rhoz1"])
    quiet_frac = 0.8 + 0.25 * (2.25 - value["rhoz1"])  # the mom_frac with the least noise moves when rho_z changes
    sigma = 0.01 * (1 + 0.6 * max(0.0, 2.25 - value["rhoz1"])) * (1 + 6 * (value["frac1"] - quiet_frac) ** 2)
    tau = np.arange(t_max + 1)
    signal = -np.exp(-E * tau) * np.array([1.0, 0.5])[:, None]  # [pz, tau], negative like the real G5G4 correlator
    C = np.zeros((n_cfg, n_src, 2, 1, 1, len(pz_list), 1 + len(vec), t_max + 1), complex)
    C[:, :, :, 0, 0, :, 0, :] = signal * (1 + sigma * z)
    for i in range(len(vec)):
        C[:, :, :, 0, 0, :, 1 + i, :] = 0.5 * (-tau * dE[i]) * signal * (1 + sigma * z)  # sink-only derivative = half of the full one
    return C


def run(noise_scan, tag, lose=None):
    """lose = (node, step): from that step on the node writes nothing, as if it had died (tests the lost-node rule)."""
    opt_dir = f"{scratch}/{tag}"
    s = {
        "n_workers": n_cfg, "start_table": start, "mom_max": 6, "L": 32, "gamma_names": ["G5G4"], "gamma_time_sign": [-1],
        "pz_list": pz_list, "pz_target": [1], "pz_weight": [1.0], "t_score": 2, "t_noise": 4,
        "noise_cut": 1.2, "noise_sample": "source", "noise_scan": noise_scan, "tilt_step": 0.1, "tilt_max": 0.6, "tilt_min_gain": 0.02,
        "scale": [0.05, 0.25, 0.25, 0.05], "lower": [-5, 0.5, 0.5, -0.5], "upper": [5, 8, 8, 2], "free_columns": [True, True, True, True],
        "step_start": 1.0, "step_grow": 1.5, "step_shrink": 0.5, "step_max": 4.0, "step_min": 0.05,
        "grad_nsigma": 1.5, "max_steps": 80, "wait_minutes": 1, "poll_seconds": 0.01,
    }  # fmt: skip
    if lose is not None:  # the settings of a many-node run: go on without a node that does not deliver
        s.update({"max_lost": 1, "straggler_fraction": 0.5, "straggler_minutes": 0.001})
    write_start(opt_dir, s)
    step = 0
    while True:
        C = toy(np.array(read_json(theta_name(opt_dir, step))["table"]))
        for worker in range(n_cfg):  # every fake node writes its file
            if lose is not None and worker == lose[0] and step >= lose[1]:
                continue
            name = corr_name(opt_dir, step, worker)
            os.makedirs(os.path.dirname(name), exist_ok=True)
            np.savez(name, C=C[worker])
        go_on = check_step(opt_dir, step, s)
        before = open(state_name(opt_dir)).read()
        assert check_step(opt_dir, step, s) == go_on and open(state_name(opt_dir)).read() == before  # a repeated check (restart) changes nothing
        if not go_on:
            break
        step += 1
    state = read_json(state_name(opt_dir))
    history = state["history"]
    accepted = [h for h in history if "ACCEPT" in h["decision"]]
    rejected = [h for h in history if "REJECT" in h["decision"]]
    acc = state["accepted"]
    final = dict(zip(names, table_to_vector(acc["table"])))
    print(f"\n#### {tag}: {len(history)} steps, {len(accepted)} accepted, {len(rejected)} rejected, stop: {state['stop_reason']}")
    print(f"#### E {history[0]['E']:.5f} -> {acc['E']:.5f}, noise ratio {acc['noise_ratio']:.3f}, rhoT1 {final['rhoT1']:.3f}, rhoz1 {final['rhoz1']:.3f}, frac1 {final['frac1']:.3f}")
    assert os.path.exists(stop_name(opt_dir)) and acc["E"] < history[0]["E"] and acc["noise_ratio"] <= s["noise_cut"]
    assert final["rhoz1"] < 2.25 and len(rejected) > 0
    if lose is not None:
        alive = [w for w in range(n_cfg) if w != lose[0]]
        assert state["alive"] == alive and os.path.exists(f"{opt_dir}/lost/worker_{lose[0]:02d}") and len(acc["E_jk"]) == len(alive)
        assert "LOST NODES" in open(f"{opt_dir}/status.log").read()
        print(f"#### node {lose[0]} lost at step {lose[1]}: the run went on with nodes {state['alive']}")
    return final


always = run("always", "always")
over_cut = run("over_cut", "over_cut")
assert always["frac1"] > 0.8  # the scan moved mom_frac toward the quiet side
run("always", "lost_during_scan", lose=(5, 5))   # a node dies while a mom_frac scan is running
run("always", "lost_at_start", lose=(2, 0))      # a node never delivers
print("\nall checks passed")
