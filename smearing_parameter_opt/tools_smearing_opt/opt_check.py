"""
Node-0 part of the smearing parameter optimization: read the correlators of all nodes, measure, decide, print the status.

Every node (one cfg each) writes for step k the file corr/step_k/worker_w.npz with

    C[src, fb, gamma_sink, gamma_source, pz, w, tau]
                                w = 0: pion correlator with the current kernel at source and sink
                                w = 1 + i: its derivative with respect to parameter i, sink end only
                                fb = 0 forward (t_src + tau), 1 backward (t_src - tau)

check_step() then does, for step k:
  1. score      E = sum over the target pz of weight * [ln|C(t)| - ln|C(t+1)|]     (the effective energy; lower = less contamination)
                With several gammas, C is the correlator of the best combination of them: v^dag C v, v from the GEVP of the
                gamma matrices C(t) and C(t+1). This optimizes the Dirac structure at every step, exactly and with no inversion.
  2. gradient   dE/d parameter, from dC/d parameter = D + D^dag, D = sink-only derivative (one gamma: 2 Re D)
  3. noise      relative error of C(t_noise), divided by the same number at step 0 (the starting kernel)
  4. decision   A kernel step (gradient descent) that raises E is REJECTED: half the step, back to the last accepted point.
                A kernel step that lowers E is followed by a MOM_FRAC SCAN (setting noise_scan = "always"): all mom_frac move
                together, which changes only the noise, until the noise stops falling. The quietest point (the unshifted one
                unless a shift lowers the noise by more than tilt_min_gain) is ACCEPTED if its noise is below the cut; otherwise
                the kernel step is REJECTED. With noise_scan = "over_cut" the scan is run only
                when the noise of the kernel step is above the cut, and stops at the first point below it.
  5. writes theta/theta_(k+1).json for the nodes, state.json for a restart, and the status (job output and status.log).

All numbers that steer this are in the settings dictionary built in the setup file. Pure numpy: runs on a laptop.
"""

import json
import os
import time

import numpy as np

from opt_kernel import table_to_vector, vector_to_table, parameter_names, per_parameter, tilt_table

# ---------------- files shared by the nodes ----------------


def theta_name(opt_dir, step):
    return f"{opt_dir}/theta/theta_{step:04d}.json"


def corr_name(opt_dir, step, worker):
    return f"{opt_dir}/corr/step_{step:04d}/worker_{worker:02d}.npz"


def state_name(opt_dir):
    return f"{opt_dir}/state.json"


def stop_name(opt_dir):
    return f"{opt_dir}/STOP"


def status_name(opt_dir):
    return f"{opt_dir}/status.log"


def lost_name(opt_dir, worker):
    """Marker of a node that is out of the run (it failed at the start, or it did not deliver a step in time)."""
    return f"{opt_dir}/lost/worker_{worker:02d}"


def mark_lost(opt_dir, worker, reason):
    os.makedirs(f"{opt_dir}/lost", exist_ok=True)
    with open(lost_name(opt_dir, worker), "w") as f:
        f.write(reason + "\n")


def write_json(name, content):
    """Written as .tmp and renamed when complete, so no node ever reads a half-written file."""
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with open(f"{name}.tmp", "w") as f:
        json.dump(content, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f"{name}.tmp", name)


def read_json(name):
    with open(name) as f:
        return json.load(f)


def wait_for_files(names, opt_dir, wait_minutes, poll_seconds, also_stop_at=None):
    """True when every file is there, False if the STOP file (or the file also_stop_at) appears first. RuntimeError after wait_minutes."""
    start = time.time()
    while True:
        if all(os.path.exists(name) for name in names):
            return True
        if os.path.exists(stop_name(opt_dir)) or (also_stop_at is not None and os.path.exists(also_stop_at)):
            return False
        if time.time() - start > 60 * wait_minutes:
            missing = [name for name in names if not os.path.exists(name)]
            raise RuntimeError(f"waited {wait_minutes} minutes, still missing {missing}")
        time.sleep(poll_seconds)


def wait_for_workers(opt_dir, step, alive, s):
    """
    Waits for the result files of step `step` from the nodes in `alive`. Returns the nodes that delivered, or None if the
    STOP file appears.
    Without the setting max_lost (or with 0) every node must deliver: RuntimeError after wait_minutes (small runs).
    With max_lost > 0 (many nodes): a node with a lost marker is not waited for, and once the fraction straggler_fraction
    of the nodes has delivered, the others get straggler_minutes more; after that the step goes on without them.
    """
    start, enough_since = time.time(), None
    while True:
        have = [w for w in alive if os.path.exists(corr_name(opt_dir, step, w))]
        pending = [w for w in alive if w not in have and not os.path.exists(lost_name(opt_dir, w))]
        if not pending:
            return have
        if os.path.exists(stop_name(opt_dir)):
            return None
        if s.get("max_lost", 0) > 0 and len(have) >= s["straggler_fraction"] * len(alive):
            enough_since = enough_since or time.time()
            if time.time() - enough_since > 60 * s["straggler_minutes"]:
                return have
        if time.time() - start > 60 * s["wait_minutes"]:
            raise RuntimeError(f"step {step}: waited {s['wait_minutes']} minutes, {len(have)} of {len(alive)} nodes delivered, still missing {pending[:20]}")
        time.sleep(s["poll_seconds"])


def load_step(opt_dir, step, workers):
    """C[cfg, src, fb, gamma_sink, gamma_source, pz, w, tau] of the given nodes."""
    return np.stack([np.load(corr_name(opt_dir, step, worker))["C"] for worker in workers])


def write_start(opt_dir, s):
    """Step 0 is the starting table of the setup file."""
    if not os.path.exists(theta_name(opt_dir, 0)):
        write_json(theta_name(opt_dir, 0), {"step": 0, "mode": "descent", "note": "starting table", "table": np.array(s["start_table"], float).tolist()})


def say(opt_dir, text):
    """The status goes to the job output of node 0 and is appended to status.log."""
    print(text, flush=True)
    with open(status_name(opt_dir), "a") as f:
        f.write(text + "\n")


# ---------------- measurement ----------------


def jackknife_error(samples):
    samples = np.asarray(samples)
    n = len(samples)
    return np.sqrt((n - 1) / n * ((samples - samples.mean(axis=0)) ** 2).sum(axis=0))


def lowest_state(A, B):
    """
    The Dirac structure: A = C(t) and B = C(t+1) are matrices in the gammas (sink gamma, source gamma). The operator
    sum_g v_g * (operator with gamma g) has the correlator v^dag C v, and the v with the slowest decay solves the GEVP
        B v = lambda A v,    largest lambda,    E = -ln(lambda) = ln|v^dag A v| - ln|v^dag B v|.
    With one gamma this is lambda = C(t+1) / C(t), the plain effective energy.
    Returns lambda and v (v^dag v = 1, largest component real and positive).
    """
    values, vectors = np.linalg.eig(np.linalg.solve(A, B))
    i = np.argmax(values.real)
    v = vectors[:, i] / np.linalg.norm(vectors[:, i])
    v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
    return values[i].real, v


def measure(C, s):
    """C[cfg, src, fb, gamma_sink, gamma_source, pz, w, tau] -> score, gradient and noise, with jackknife errors over the cfgs
    (over the sources if there is only one cfg: a one-node test)."""
    n_cfg, n_src = C.shape[:2]
    n_pz = C.shape[5]
    t, t_noise = s["t_score"], s["t_noise"]
    target = [s["pz_list"].index(pz) for pz in s["pz_target"]]
    weight = np.array(s["pz_weight"], float)

    # forward and backward averaged, with the time-reflection sign s_a s_b of every gamma pair: [cfg, src, a, b, pz, w, tau]
    sign = np.array(s["gamma_time_sign"], float)
    c = (C[:, :, 0] + (sign[:, None] * sign[None, :])[None, None, :, :, None, None, None] * C[:, :, 1]) / 2
    # the kernel at both ends: the matrix is Hermitian in the gammas (on average), and its derivative is D + D^dag,
    # D = sink-only derivative (one gamma: 2 Re D)
    c0 = c[:, :, :, :, :, 0, :]  # [cfg, src, a, b, pz, tau]
    c0 = (c0 + c0.conj().swapaxes(2, 3)) / 2
    d = c[:, :, :, :, :, 1:, :]  # [cfg, src, a, b, pz, par, tau]
    dc = d + d.conj().swapaxes(2, 3)
    c0_cfg, dc_cfg = c0.mean(axis=1), dc.mean(axis=1)  # sources averaged: one matrix per cfg
    if n_cfg == 1:  # one-node test: a jackknife over one cfg is impossible, so the errors come from the spread over its sources
        c0_cfg, dc_cfg = c0[0], dc[0]
    n_jk = len(c0_cfg)

    def score(c0_mean, dc_mean):  # [a, b, pz, tau] and [a, b, pz, par, tau], averaged over cfgs
        lnC_t, lnC_t1, E_pz, g_pz, vectors = np.zeros(n_pz), np.zeros(n_pz), np.zeros(n_pz), np.zeros((n_pz, dc_mean.shape[3])), []
        for ipz in range(n_pz):
            A, B = c0_mean[:, :, ipz, t], c0_mean[:, :, ipz, t + 1]
            lam, v = lowest_state(A, B)
            vAv, vBv = (v.conj() @ A @ v).real, (v.conj() @ B @ v).real
            dA = np.einsum("a,abp,b->p", v.conj(), dc_mean[:, :, ipz, :, t], v).real
            dB = np.einsum("a,abp,b->p", v.conj(), dc_mean[:, :, ipz, :, t + 1], v).real
            lnC_t[ipz], lnC_t1[ipz], E_pz[ipz] = np.log(abs(vAv)), np.log(abs(vBv)), -np.log(lam)
            g_pz[ipz] = -(dB - lam * dA) / (lam * vAv)  # d E_pz / d parameter, with v kept fixed (first order: v does not enter)
            vectors.append(v)
        return lnC_t, lnC_t1, E_pz, weight @ E_pz[target], weight @ g_pz[target], np.array(vectors)

    central = score(c0_cfg.mean(axis=0), dc_cfg.mean(axis=0))
    total0, totald = c0_cfg.sum(axis=0), dc_cfg.sum(axis=0)  # jackknife sample i = mean without sample i = (total - sample i) / (n - 1)
    jk = [score((total0 - c0_cfg[i]) / (n_jk - 1), (totald - dc_cfg[i]) / (n_jk - 1)) for i in range(n_jk)]
    errors = [jackknife_error([sample[k] for sample in jk]) for k in range(5)]

    # noise: relative error of the projected correlator v^dag C(t_noise) v, from the spread over cfgs and over all sources
    v = central[5]  # [pz, gamma]
    x = np.einsum("pa,csabp,pb->csp", v.conj(), c0[:, :, :, :, :, t_noise], v).real  # [cfg, src, pz]
    mean = np.abs(x.mean(axis=(0, 1)))
    noise_src = x.reshape(n_cfg * n_src, -1).std(axis=0, ddof=1) / np.sqrt(n_cfg * n_src) / mean
    noise_cfg = x.mean(axis=1).std(axis=0, ddof=1) / np.sqrt(n_cfg) / mean if n_cfg > 1 else noise_src

    return {
        "lnC_t": central[0], "lnC_t_err": errors[0], "lnC_t1": central[1], "lnC_t1_err": errors[1],
        "E_pz": central[2], "E_pz_err": errors[2], "E": float(central[3]), "E_err": float(errors[3]),
        "E_jk": np.array([sample[3] for sample in jk]), "grad": central[4], "grad_err": errors[4],
        "noise_cfg": noise_cfg, "noise_src": noise_src, "v": v, "errors_from": f"{n_cfg} cfgs" if n_cfg > 1 else f"the {n_src} sources of ONE cfg (test only)",
    }  # fmt: skip


# ---------------- the two kinds of move ----------------


def propose_descent(table, grad, grad_err, step_size, s):
    """
    One gradient-descent step from `table`. Every parameter is measured in units of its scale (settings "scale"), so that
    a weight and a width can be compared. Steps:
      - gradient components below grad_nsigma errors are set to zero (they are noise);
      - the direction that moves all mom_frac together (tilt_table) is removed: it changes only the noise, see noise scan;
      - the step has length step_size in scaled units, opposite to the gradient; the result is kept inside the bounds.
    Returns (new table, text) or (None, reason) if no gradient component is significant.
    """
    table = np.array(table, float)
    n_gauss = len(table)
    grad, grad_err = np.array(grad), np.array(grad_err)
    scale = per_parameter(s["scale"], n_gauss)
    free = per_parameter(s["free_columns"], n_gauss).astype(bool)
    names = parameter_names(n_gauss)

    use = free & (np.abs(grad) >= s["grad_nsigma"] * grad_err)
    if not use.any():
        return None, f"no gradient component above {s['grad_nsigma']} sigma"
    g = np.where(use, grad * scale, 0.0)

    h = 1e-4
    tilt = (table_to_vector(tilt_table(table, +h, s["mom_max"], s["L"])) - table_to_vector(tilt_table(table, -h, s["mom_max"], s["L"]))) / (2 * h) / scale
    no_effect = per_parameter([0, 1, 1, 1], n_gauss) * np.repeat(table[:, 0] == 0, 4)[1:]  # shape parameters of a Gaussian with weight 0 do nothing
    tilt = np.where(free & (no_effect == 0), tilt, 0.0)
    if tilt @ tilt > 0:
        g = g - (g @ tilt) / (tilt @ tilt) * tilt
    if not np.any(g):
        return None, "the gradient is along the noise-only direction"

    vector = table_to_vector(table)
    change = -step_size * g / np.sqrt(g @ g) * scale
    new = np.clip(vector + change, per_parameter(s["lower"], n_gauss), per_parameter(s["upper"], n_gauss))
    order = np.argsort(-np.abs((new - vector) / scale))[:6]
    text = ", ".join(f"{names[i]} {vector[i]:.4f} -> {new[i]:.4f}" for i in order if new[i] != vector[i])
    return vector_to_table(new), text


def next_shift(points, first_direction, tilt_step, tilt_max, min_gain):
    """
    mom_frac scan: which common mom_frac shift to try next, from the points tried so far (first point: shift 0).
    Try one step in first_direction (the direction that helped in the last scan); if the noise fell keep going that way,
    otherwise try one step the other way and keep going while the noise falls. "Fell" means by more than min_gain
    (a fraction): smaller changes are within the error of the noise estimate.
    None: the scan is over (the noise stopped falling, or the shift reached tilt_max).
    """
    fell = lambda new, old: new["noise_ratio"] < old["noise_ratio"] * (1 - min_gain)
    if len(points) == 1:
        return first_direction * tilt_step
    last, before = points[-1], points[-2]
    if len(points) == 2:
        return 2 * last["shift"] if fell(last, points[0]) else -last["shift"]
    if len(points) == 3 and last["shift"] * before["shift"] < 0:  # the first direction was no better, the other one was just tried
        return 2 * last["shift"] if fell(last, points[0]) else None
    if fell(last, before) and abs(last["shift"]) + tilt_step <= tilt_max + 1e-9:
        return last["shift"] + np.sign(last["shift"]) * tilt_step
    return None


# ---------------- status text ----------------


def with_error(value, error):
    """0.91113(62): the value with its error in the last digits."""
    if not np.isfinite(error) or error <= 0:
        return f"{value:.6g}"
    digits = max(0, int(np.ceil(-np.log10(error))) + 1)
    return f"{value:.{digits}f}({round(error * 10**digits):d})"


def table_text(table):
    lines = ["      a          rho_T     rho_z     mom_frac"]
    for j, (a, rho_T, rho_z, frac) in enumerate(table):
        lines.append(f"  {j + 1:2d}  {a:+.5f}   {rho_T:.4f}    {rho_z:.4f}    {frac:+.4f}")
    return "\n".join(lines)


def status_text(step, mode, note, m, noise_ratio, s, seconds):
    t, t_noise = s["t_score"], s["t_noise"]
    names = s["gamma_names"]
    lines = [f"\n================ step {step} | {mode} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {seconds:.0f} s since the last check ================"]
    if note:
        lines.append(f" this point: {note}")
    dirac = " | Dirac structure v (size, phase in degrees)" if len(names) > 1 else ""
    lines.append(f"  pz | ln|C(t={t})|        ln|C(t={t + 1})|        E = ln C({t}) - ln C({t + 1}) | noise ratio at t={t_noise}: from sources / from cfgs{dirac}")
    for ipz, pz in enumerate(s["pz_list"]):
        star = "*" if pz in s["pz_target"] else " "
        v_text = " | " + ", ".join(f"{name} {abs(x):.3f} {np.degrees(np.angle(x)):+.0f}" for name, x in zip(names, m["v"][ipz])) if len(names) > 1 else ""
        lines.append(f" {star}{pz:2d} | {with_error(m['lnC_t'][ipz], m['lnC_t_err'][ipz]):16s} {with_error(m['lnC_t1'][ipz], m['lnC_t1_err'][ipz]):16s} "
                     f"{with_error(m['E_pz'][ipz], m['E_pz_err'][ipz]):24s} | {noise_ratio['source'][ipz]:.3f} / {noise_ratio['cfg'][ipz]:.3f}{v_text}")  # fmt: skip
    lines.append(f" errors: jackknife over {m['errors_from']}")
    lines.append(f" (* = in the score, weights {s['pz_weight']}" + (f"; C = v^dag C v, v from the GEVP of C({t}) and C({t + 1}) in the gammas {names})" if len(names) > 1 else ")"))
    return "\n".join(lines)


# ---------------- one check ----------------


def check_step(opt_dir, step, s):
    """Checks step `step` and writes the next table. Returns True to go on, False when the optimization has stopped."""
    state = read_json(state_name(opt_dir)) if os.path.exists(state_name(opt_dir)) else None

    # restart: this step was already checked, only make sure its result is on disk
    if state is not None and state["last_checked"] >= step:
        if state["stop_reason"] is None and not os.path.exists(theta_name(opt_dir, step + 1)):
            write_json(theta_name(opt_dir, step + 1), state["next"])
        return state["stop_reason"] is None

    # the nodes still in the run, and those that delivered this step
    alive = state["alive"] if state is not None else list(range(s["n_workers"]))
    have = wait_for_workers(opt_dir, step, alive, s)
    if have is None:
        return False
    lost = [w for w in alive if w not in have]
    ratio = lambda now, ref: np.where(np.array(ref) > 0, now / np.where(np.array(ref) > 0, ref, 1), 1.0)
    target = [s["pz_list"].index(pz) for pz in s["pz_target"]]
    too_many_lost = None
    if lost:
        # these nodes are out of the run from now on. Every number compared between steps must come from the SAME cfgs, so the
        # reference noise (step 0), the accepted point and the points of a running mom_frac scan are measured again without them.
        for w in lost:
            if not os.path.exists(lost_name(opt_dir, w)):
                mark_lost(opt_dir, w, f"no result file for step {step} in time")
        say(opt_dir, f"\n LOST NODES at step {step}: {lost}. The run goes on with {len(have)} of {s['n_workers']} cfgs.")
        if s["n_workers"] - len(have) > s.get("max_lost", 0):
            too_many_lost = f"{s['n_workers'] - len(have)} nodes lost, more than max_lost = {s.get('max_lost', 0)}"
        elif state is not None:
            m0 = measure(load_step(opt_dir, 0, have), s)
            state["ref_noise"] = {"source": m0["noise_src"].tolist(), "cfg": m0["noise_cfg"].tolist()}
            for point in [state["accepted"]] + (state["scan"]["points"] if state["scan"] else []):
                mp = measure(load_step(opt_dir, point["step"], have), s)
                noise = mp["noise_src"] if s["noise_sample"] == "source" else mp["noise_cfg"]
                point.update({"E": mp["E"], "E_err": mp["E_err"], "E_jk": mp["E_jk"].tolist(), "grad": mp["grad"].tolist(), "grad_err": mp["grad_err"].tolist(),
                              "noise_ratio": float(ratio(noise, state["ref_noise"][s["noise_sample"]])[target].max())})  # fmt: skip
    if too_many_lost is not None:
        say(opt_dir, f"\n STOP: {too_many_lost}")
        if state is not None:
            state["last_checked"], state["stop_reason"], state["alive"] = step, too_many_lost, have
            write_json(state_name(opt_dir), state)
        with open(stop_name(opt_dir), "w") as f:
            f.write(too_many_lost + "\n")
        return False

    C = load_step(opt_dir, step, have)  # [cfg, src, fb, gamma_sink, gamma_source, pz, w, tau]
    here = read_json(theta_name(opt_dir, step))
    table = np.array(here["table"], float)
    m = measure(C, s)

    # noise relative to step 0 (the starting kernel)
    if step == 0:
        state = {"mode": "descent", "step_size": s["step_start"], "scan": None, "tilt_direction": 1, "history": [], "time": time.time(),
                 "ref_noise": {"source": m["noise_src"].tolist(), "cfg": m["noise_cfg"].tolist()}}  # fmt: skip
    state["alive"] = have
    noise_ratio = {"source": ratio(m["noise_src"], state["ref_noise"]["source"]), "cfg": ratio(m["noise_cfg"], state["ref_noise"]["cfg"])}
    N = float(noise_ratio[s["noise_sample"]][target].max())

    say(opt_dir, status_text(step, state["mode"], here.get("note", ""), m, noise_ratio, s, time.time() - state["time"]))
    this_point = {"step": step, "table": table.tolist(), "E": m["E"], "E_err": m["E_err"], "E_jk": m["E_jk"].tolist(),
                  "grad": m["grad"].tolist(), "grad_err": m["grad_err"].tolist(), "noise_ratio": N, "shift": here.get("shift", 0.0)}  # fmt: skip
    mode = state["mode"]
    accept = None  # the point that becomes the accepted one in this check, if any
    reject = None  # or the reason why the kernel step is given up

    if step == 0:
        say(opt_dir, f" score E = {with_error(m['E'], m['E_err'])}    noise ratio 1 by definition (this is the reference)")
        decision = "START: this is the reference point"
        state["accepted"] = this_point
    else:
        acc = state["accepted"]
        dE = m["E"] - acc["E"]
        dE_err = float(jackknife_error(m["E_jk"] - np.array(acc["E_jk"])))
        say(opt_dir, f" score E = {with_error(m['E'], m['E_err'])}    last accepted (step {acc['step']}) {acc['E']:.6f}    difference {with_error(dE, dE_err)}")
        say(opt_dir, f" noise ratio (largest target pz, from {s['noise_sample']}s) {N:.3f}    cut {s['noise_cut']}")

        if mode == "descent":  # this point is a kernel step
            if dE >= 0:
                reject = "E went up"
            elif s["noise_scan"] == "always" or N > s["noise_cut"]:
                decision = "E went down. MOM_FRAC SCAN: move all mom_frac together to lower the noise"
                state["mode"] = "noise_scan"
                state["scan"] = {"points": [this_point]}
            else:
                accept = this_point

        elif mode == "noise_scan":  # this point is a mom_frac trial of the scan
            state["scan"]["points"].append(this_point)
            decision = "mom_frac scan goes on"
            if s["noise_scan"] == "over_cut" and dE < 0 and N <= s["noise_cut"]:
                accept = this_point  # below the cut: enough

        if state["mode"] == "noise_scan" and accept is None:
            points = state["scan"]["points"]
            say(opt_dir, " mom_frac scan so far (common shift: noise ratio): " + ", ".join(f"{p['shift']:+.2f}: {p['noise_ratio']:.3f}" for p in points))
            shift = next_shift(points, state["tilt_direction"], s["tilt_step"], s["tilt_max"], s["tilt_min_gain"])
            if shift is not None:
                state["next"] = {"step": step + 1, "mode": "noise_scan", "shift": float(shift), "note": f"mom_frac scan, common shift {shift:+.2f} from the kernel of step {points[0]['step']}",
                                 "table": tilt_table(points[0]["table"], shift, s["mom_max"], s["L"]).tolist()}  # fmt: skip
                next_text = f"mom_frac scan: common shift {shift:+.2f} from the kernel of step {points[0]['step']}"
            else:  # the scan is over: among the points whose E is below the accepted E, the unshifted one unless a shifted one is really quieter
                best = None
                for p in points:  # in the order tried: shift 0 first
                    if p["E"] < acc["E"] and (best is None or p["noise_ratio"] < best["noise_ratio"] * (1 - s["tilt_min_gain"])):
                        best = p
                if best is not None and best["noise_ratio"] <= s["noise_cut"]:
                    accept = best
                else:
                    reject = "the mom_frac scan found no point below the noise cut"

        if accept is not None:
            new_size = min(state["step_size"] * s["step_grow"], s["step_max"])
            decision = f"ACCEPT the point of step {accept['step']} (common mom_frac shift {accept['shift']:+.2f}, noise ratio {accept['noise_ratio']:.3f}). step size {state['step_size']:.3f} -> {new_size:.3f}"
            if accept["shift"] != 0:
                state["tilt_direction"] = int(np.sign(accept["shift"]))
            state["accepted"], state["step_size"] = accept, new_size
        if reject is not None:
            decision = f"REJECT ({reject}). step size {state['step_size']:.3f} -> {state['step_size'] * s['step_shrink']:.3f}, back to step {acc['step']}"
            state["step_size"] *= s["step_shrink"]
        if accept is not None or reject is not None:
            state["mode"], state["scan"] = "descent", None

    say(opt_dir, f" decision: {decision}")

    # the next point is a kernel step (gradient descent) from the accepted point, unless a mom_frac scan is running
    stop_reason = None
    acc = state["accepted"]
    if state["mode"] == "descent":
        n_big = int((np.abs(acc["grad"]) >= s["grad_nsigma"] * np.array(acc["grad_err"])).sum())
        scale = per_parameter(s["scale"], len(acc["table"]))
        names = parameter_names(len(acc["table"]))
        order = np.argsort(-np.abs(np.array(acc["grad"]) * scale))[:6]
        say(opt_dir, f" gradient at the accepted point (step {acc['step']}): {n_big} of {len(names)} components above {s['grad_nsigma']} sigma. "
                     "Largest, per unit of scale: " + ", ".join(f"{names[i]} {with_error(acc['grad'][i] * scale[i], acc['grad_err'][i] * scale[i])}" for i in order))  # fmt: skip
        if state["step_size"] < s["step_min"]:
            stop_reason = f"the step size fell below {s['step_min']}: no better point nearby within the noise cut"
        else:
            new_table, text = propose_descent(acc["table"], acc["grad"], acc["grad_err"], state["step_size"], s)
            if new_table is None:
                stop_reason = text
            else:
                state["next"] = {"step": step + 1, "mode": "descent", "note": f"kernel step of size {state['step_size']:.3f} from step {acc['step']}", "table": new_table.tolist()}
                next_text = f"kernel step from step {acc['step']}, step size {state['step_size']:.3f}. Largest changes: {text}"
    if stop_reason is None and step + 1 >= s["max_steps"]:
        stop_reason = f"max_steps = {s['max_steps']} reached"

    state["history"].append({"step": step, "mode": mode, "E": m["E"], "E_err": m["E_err"], "noise_ratio": N, "decision": decision, "table": table.tolist(),
                             "v": [[[x.real, x.imag] for x in row] for row in m["v"]]})  # fmt: skip
    state["last_checked"], state["stop_reason"], state["time"] = step, stop_reason, time.time()
    write_json(state_name(opt_dir), state)

    if "ACCEPT" in decision:
        say(opt_dir, f" accepted table (step {acc['step']}):\n" + table_text(acc["table"]))
    if stop_reason is None:
        say(opt_dir, f" next (step {step + 1}): {next_text}")
        write_json(theta_name(opt_dir, step + 1), state["next"])
    else:
        say(opt_dir, f"\n STOP: {stop_reason}")
        say(opt_dir, f" best point: step {acc['step']}, E = {with_error(acc['E'], acc['E_err'])}, noise ratio {acc['noise_ratio']:.3f} "
                     f"(start: E = {state['history'][0]['E']:.6f})\n" + table_text(acc["table"]))  # fmt: skip
        with open(stop_name(opt_dir), "w") as f:
            f.write(stop_reason + "\n")
    return stop_reason is None
