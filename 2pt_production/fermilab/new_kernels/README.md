# New smearing kernels: Fermilab test

Tests the single-quark kernels of `coulomb_smearing_kernels.py` (the note `pion_smearing_kernels.pdf`, Secs. 1-7) in the pion two-point
function (the first test, pion only), now in the Model M scan for the pion and the proton.
The first goal is to check that they work. The second is to see whether any of them gives a better signal than the reference Gaussian
G_ref = [rho_T, rho_z, mom_frac] = [4.5, 2.25, 0.8].

| file | what it is |
|---|---|
| `coulomb_smearing_kernels.py` | the kernel package (drop-in for `coulomb_smearing.py`, plus the new kernels) |
| `new_kernels_2pt_setup_fermilab.py` | setup: lattice, cfgs, sources, and the operators: `operator_list` is the 13-operator Model M scan; the 14 of the first test are `test1_operator_list` |
| `new_kernels_tools.py` | `build_kernel`, `kernel_moments`, and the GPU checks run by the job before any inversion |
| `2pt_coulomb_new_kernels_32srcs_fermilab.py` | the job: the smearing-tuning job with kernels instead of `[rho_T, rho_z, mom_frac]`, pion and proton |
| `2pt_coulomb_new_kernels_32srcs_fermilab.sh` | sbatch: one cfg per array task by default, N_CFG cfgs per task with `--export=ALL,N_CFG=<n>` |
| `fermilab_package_new_kernels.py` | packs the per-cfg files into one pion and one proton file (run on Fermilab) |
| `quick_look_new_kernels.py` | local analysis: R, noise, pull, and imag against G_ref on single gamma pairs, plus the note's controlled pairs |
| `gevp_look_new_kernels.py` | local analysis: E_eff of the per-operator 3x3 gamma GEVP-reconstructed 2pt, R and noise against G_ref, the Model M scan curves |
| `kernel_radii.py` | prints the rms radii and momentum of every operator (CPU), after editing `operator_list` |
| `test_coulomb_smearing_kernels.py` | CPU unit tests of the package (numpy + scipy) |

## The operators of the first test (`test1_operator_list`)

| operator | what it tests |
|---|---|
| G_ref | the reference |
| G_rz1.5 | the width control of two_momentum (the tuning's [4.5, 1.5, 0.8]) |
| two_scale | a narrow core plus a broad tail |
| two_momentum | resolved sidebands at k0 ± 3.6 |
| M_plus, M_minus | the transverse-dependent mean momentum and its opposite-sign control |
| V | transverse-dependent splitting |
| MV | Models M and V together |
| W | transverse-dependent width |
| beta, gpd | conditional momentum amplitudes |
| stretched | exponential tails |
| power_tail | an r^-6 tail |
| polynomial | a nodeless polynomial envelope |

The operators that change the envelope (two_scale, W, stretched, power_tail, polynomial) have the same rms radii as G_ref, transverse 3.18
and along z 1.125. For them a change in signal is a change of shape, not of size. The reasons for every parameter are in the setup.

## Steps

1. On a laptop or the login node, run the unit tests and look at the operators:
   `python3 test_coulomb_smearing_kernels.py` (13/13 pass), then `python3 kernel_radii.py`.
2. Copy this folder to `2pt_production/fermilab/new_kernels` on Fermilab. The sbatch takes `pt2_comm_tools.py` from `../../tools`.
   Everything else comes from this folder.
3. Submit the test, 4 cfgs: `mkdir -p logs && sbatch 2pt_coulomb_new_kernels_32srcs_fermilab.sh`
4. Check the log of each task:
   - one line `KERNEL CHECK <name>: cpu-gpu ..., K(-d) - K(d)^* ...` per operator (14 in the first test, 13 in the Model M scan), all far below 1e-12;
   - for G_ref and G_rz1.5, `new source and sink vs coulomb_boosted_source / sink at +-k`, also far below 1e-12;
   - the run then proceeds as the smearing tuning does.
5. Pack on Fermilab: `python3 fermilab_package_new_kernels.py`. Copy the packed file home and run:
   `python3 quick_look_new_kernels.py <packed file> --per_tsrc` (and `--gamma G5`, `--gamma G3G5`).
   If the smearing tuning `coulomb_tune1` ran on the same cfgs, add `--tune1 <its packed pion file>`.
   G_ref and G_rz1.5 must reproduce its [4.5, 2.25, 0.8] and [4.5, 1.5, 0.8] (relative difference far below 1e-6).
6. Working properly means:
   - every kernel check passes;
   - the tuning check passes (if available);
   - the imaginary parts are compatible with zero;
   - no operator is flagged by the checks line of the quick look.

   The checks line uses a Bonferroni cut. A flagged pull is usually a different excited-state contamination at t_n. The kernel is wrong
   only if its E_eff stays away from G_ref at late t: look at its page in the pdf, or rerun with `--t_n 12` or `--t_n 14`.
7. More statistics: `sbatch --array=4-39 2pt_coulomb_new_kernels_32srcs_fermilab.sh`. The results go to the same folder, then pack and look
   again. To measure a different operator list, set a new `operator_tag` in the setup: the job refuses to mix lists in one folder.

The 4-cfg test checks the plumbing. The pion was nearly insensitive to the smearing in the first Delta analysis, so expect differences of a
few percent at most, and judge the signal on the larger run.

## Model M scan (current `operator_list`, `operator_tag = "model_M_scan1"`): pion and proton, GEVP analysis

The first test (`test1_operator_list`, kept in the setup) found two things about Model M, the transverse-dependent mean boost:
- M_plus (a larger boost near the profile center) has a better overlap than G_ref but more noise.
- M_minus is the reverse.

The scan maps that trade-off with 13 operators: G_ref, plus Model M at several strengths and three correlation lengths R_c.
- **Strength** is the spread of the local boost over the profile, in units of the test's M_plus. It is -1, -0.5, +0.5, +1, +1.5 and +2
  at R_c = 3.18, and -1, +1 and +2 at R_c = 2.25 and 4.5.
- At equal spread every R_c is equally far from G_ref, so R_c only changes where across the profile the boost moves.
- All operators keep G_ref's envelope (the same size) and its mean boost of 4.8.

Both the pion and the proton are saved, with all three gammas (G5, G5G4, G3G5) at sink and source. The proton uses the kernel K for all three
quarks, with the contraction of the smearing tuning.

Run all 40 cfgs in one job. That is about 36 min per cfg, so about 24 h; ask for 27 h for margin:

    sbatch --array=0 --time=27:00:00 --export=ALL,N_CFG=40 2pt_coulomb_new_kernels_32srcs_fermilab.sh

If the job runs out of time, resubmit the same command: finished cfgs and t_src are skipped. If lq2_gpu does not allow 27 h, run two
tasks of 20 cfgs instead (the same folder of results):

    sbatch --array=0,20 --time=14:00:00 --export=ALL,N_CFG=20 2pt_coulomb_new_kernels_32srcs_fermilab.sh
`fermilab_package_new_kernels.py` writes one pion and one proton file, with the proton projected to positive parity.

Analysis: `gevp_look_new_kernels.py <packed file>`, for the pion or the proton file. It follows
`local_analysis/build_local_GEVP_pion.ipynb`:
- for each operator, a 3x3 gamma GEVP on each half (forward, backward), with t0 2 and the ground-state vector at t_ref 3;
- the reconstructed 2pt, v0^dag C v0;
- its E_eff, with a jackknife over cfgs.

It writes:
- R and noise against G_ref on the reconstructed 2pt;
- a summary page of R and noise against the Model M strength, one line per R_c;
- the E_eff pages.

`quick_look_new_kernels.py` gives the same comparison on single diagonal gamma pairs.

## Reading the result

- R = [C(7)/C(3)] / [same for G_ref], the overlap measure of the smearing tuning. R > 1 means less excited-state contamination than G_ref.
- noise = err E_eff(t_n) / err of G_ref. Below 1 means a more precise signal.
- The controlled pairs (note, Sec. 8.2) are compared directly by the quick look:
  - M_plus / M_minus: the sign of the transverse-momentum correlation (the note's main hypothesis);
  - two_momentum / G_rz1.5: sidebands at the same z width;
  - V / two_momentum: transverse-dependent against constant splitting.
- Against G_ref: W (width), two_scale, stretched, power_tail and polynomial (envelope shape at a fixed size), MV.
- beta: its w(x | s) is narrow (0.8-1.0 units against the 2.26 of the envelope). It acts as Model M with delta_mu 0.9, a dose point between
  G_ref and M_plus, not as a new line shape. At this envelope no conditional kernel can carry much line-shape content.
- gpd is the literal GPD template. Its mean quark momentum is 1.7 units, against k0 = 4.8, so its result mostly reflects a low boost.
