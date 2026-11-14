# Smearing parameter optimization: design

Goal: find the pion interpolator (smearing kernel and Dirac structure) with the least excited-state contamination,
without fits, while the noise stays below a cut. Written 2026-11-11. Not yet run on a GPU.

## Files

| File | What it is |
|---|---|
| `delta_test/opt_setup_delta.py` | **Every choice**: cfgs, sources, starting table, gammas, score, noise cut, step sizes, paths |
| `delta_test/opt_worker_delta.py` | Runs on one node for one cfg, for the whole run. Node 0 also does the check |
| `delta_test/opt_worker_delta.sh` | Job array, 10 tasks = 10 nodes. Copy of the Delta production job script |
| `delta_one_gaussian/` | The simple first case: ONE Gaussian starting at the production kernel (rho_T, rho_z, mom_frac tuned), G5G4 only. Copies of the three `delta_test` files with their own names; only the starting table, the gammas, the run tag and the file names differ |
| `frontier_production/` | Frontier, 800 nodes (one stream-e cfg per node, 8 GCDs each), started through Flux. 13 Gaussians, G5G4, pz 6, noise from the cfgs, and the rule for lost nodes. The choices are listed in `FRONTIER_DECISIONS.pdf` |
| `frontier_test/` | The same on 10 nodes for 12 steps, to test the Frontier version before the 800-node job |
| `tools_smearing_opt/opt_kernel.py` | The kernel (sum of boosted Gaussians), its derivatives, the mom_frac move |
| `tools_smearing_opt/opt_contract.py` | The pion contraction with the sink done in momentum space |
| `tools_smearing_opt/opt_check.py` | Node 0: score, gradient, noise, decision, status text |
| `tools_smearing_opt/plot_kernel.py` | Picture of a kernel table in momentum space and in position space (`python3 plot_kernel.py state.json`) |
| `tools_smearing_opt/mock_check/` | Laptop checks (see the end) |

It also uses `2pt_production/tools/coulomb_smearing.py` and `pt2_comm_tools.py`, unchanged.

## The operator

**Kernel.** A sum of the production boosted Gaussians, one row of the table per Gaussian:

    table[j] = [a_j, rho_T_j, rho_z_j, mom_frac_j]        K(d) = sum_j a_j G_j(d) exp(+i 2pi/L k_j d_z),  k_j = mom_frac_j * 6

In momentum space each term is a Gaussian centered at k_j. `a_1 = 1` is fixed (an overall factor does nothing).
13 Gaussians give 51 parameters. One row `[1, 4.5, 2.25, 0.8]` alone is exactly the production kernel.

**Dirac structure.** The operator is `sum_g v_g * (operator with gamma g)`, g in G5, G5G4, G3G5. For a fixed kernel the best
`v` is the lowest state of the GEVP of the 3 x 3 gamma matrices C(t), C(t+1). It is solved exactly at every step from the
same propagators, so it needs no gradient and no inversion. `opt_gamma_ids = [7]` turns it off (G5G4 only).

## What is measured at every step

- **Score** `E = sum over target pz of weight * [ln|C(t)| - ln|C(t+1)|]`, with `C = v^dag C v`. The kernel is the same at
  source and sink, so E is an upper bound of the ground-state energy: lower E = less contamination.
- **Gradient** of E for all 51 parameters, from the same inversions (next section).
- **Noise** = relative error of C(t_noise), divided by its value at step 0 (the starting kernel).

## How one step is made short

1. **The node keeps its cfg.** Gauge fixing, HYP and the multigrid setup are done once, not once per step.
2. **The sink smearing is done as weights in momentum space.** Sink smearing multiplies the FFT of a propagator with the FFT of
   the kernel, so

       C(t, P) = 1/V sum_m W(m) M(t, m),   M from the FFT of the UNSMEARED solutions,   W(m) = kft2[m] conj(kft1[m - P])

   M does not know the kernel. One FFT and one contraction per source give C for the current kernel (W from the kernel) and its
   derivative for every parameter (W from the kernel derivative): the cost does not grow with the number of parameters.
   That derivative moves only the sink; with the kernel at both ends it is `D + D^dag` (one gamma: `2 Re D`), on average over cfgs.
3. **Only what is needed is contracted:** pion only, tsep 0..10 forward and backward, 7 momenta.
4. **Only small files are written:** one `.npz` per node and step.

So a step costs one inversion pair per source (16 per node) plus the contraction, on all nodes at once.
Not used, possible later: a looser solver tolerance for the tuning; more right-hand sides per solve.

## What node 0 decides (opt_check.py)

    kernel step (gradient descent from the last accepted point)
        E went up      -> REJECT: half the step size, try again from the accepted point
        E went down    -> MOM_FRAC SCAN: all mom_frac move together (this changes only the noise), one step in the direction
                          that helped last time; keep going while the noise falls, otherwise try the other direction
                          quietest point below the noise cut -> ACCEPT it, step size x 1.5, next kernel step from there
                          no point below the cut             -> REJECT the kernel step

- The gradient step: every parameter is in units of its scale; components below 1.5 errors are set to zero; the common
  mom_frac direction is taken out (it is the scan's job); the result stays inside the bounds.
- Every mom_frac trial is one more step of inversions, so with `opt_noise_scan = "always"` a kernel step costs at least 3 steps.
  `"over_cut"` scans only when the noise is above the cut.
- The run stops when the step size falls below `opt_step_min`, when no gradient component is significant, or at `opt_max_steps`.

## How the nodes talk

Only through small files in `opt_dir`:

    theta/theta_0007.json          the table of step 7 (written by node 0)
    corr/step_0007/worker_03.npz   C[src, fb, gamma_sink, gamma_source, pz, w, tau] of node 3 (w = 0 correlator, w >= 1 derivatives)
    state.json                     what node 0 needs for a restart, and the history of all steps
    status.log                     the status text of every step (also in the job output of task 0)
    STOP                           written when the run is over: all nodes end

One-node test: `opt_cfg_index = [0]`, a new `opt_run_tag`, `sbatch --array=0`. With one cfg the errors come from the spread
over its 16 sources (the status says so); good for testing the code and the timings, not for results.

Lost nodes (many-node runs only, settings `opt_max_lost`, `opt_straggler_fraction`, `opt_straggler_minutes`): a node that fails at the start,
or does not deliver a step in time, gets a file in `lost/` and is out for the rest of the run. Node 0 then measures the reference, the accepted
point and a running scan again without its cfg, so every comparison uses the same cfgs. Without these settings every node must deliver.

Restart: `sbatch` again. Every node continues at the first step it has not written. A new run needs a new `opt_run_tag`.

## Laptop checks (results only, no timing)

    cd tools_smearing_opt/mock_check
    python3 test_kernel_contract.py              # kernel, momentum-space sum = production order, derivative weights
    python3 test_check_toy.py <scratch dir>      # the decisions of node 0 on a toy model
    bash run_mock.sh <scratch dir> 4             # the REAL worker on 3 mock nodes x 4 ranks (numpy for cupy/PyQUDA, free Wilson solver)
    python3 check_mock_output.py <scratch dir>/run

Results 2026-11-11: momentum-space sum = sink smearing + contraction to 1e-15; worker output (4 ranks and 1 rank) = an
independent whole-lattice calculation to 1e-15; sink derivative = finite difference to 1e-7; derivative with the kernel at
both ends = `D + D^dag` to 1e-7 (exact in the free mock theory). A mock run killed during step 2 and started again continued at step 2 and ended with the same tables and bitwise the same correlators as an uninterrupted run.

## Not covered by the checks

- Nothing ran on a GPU: cupy, the real PyQUDA calls and the timings are untested. `latt_info.lexico/evenodd(..., backend="cupy")`
  is the same call as in the current `coulomb_smearing.py`, which has not run on a GPU either.
- Whether 10 cfgs x 16 sources resolve the gradient and the noise ratio well enough: the first real run shows it (the status
  prints every error).
- 51 parameters on 160 sources can fit the noise of the sample: the final table must be checked on other cfgs.
