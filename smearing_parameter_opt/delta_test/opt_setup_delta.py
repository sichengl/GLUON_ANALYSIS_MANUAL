#Smearing parameter optimization on Delta, stream e: every node holds ONE cfg for the whole run (10 nodes = 10 cfgs), with 2t x 2x x 2y x 2z = 16 sources
#per cfg. The kernel is a sum of boosted Gaussians (tools_smearing_opt/opt_kernel.py); its parameters are changed step by step to lower the pion
#effective energy E = ln C(t) - ln C(t+1), while the noise must stay below a cut. Lattice, solver, gauge fixing and HYP are those of the Delta
#production setup (2pt_production/delta/final_decision_for_production_delta/smear_v2/sepq_2pt_setup_delta_final.py), copied line by line.
#EVERY choice of the optimization is in this file.
import numpy as np
from pt2_comm_tools import TwoPtParams          # 2pt_production/tools/pt2_comm_tools.py: the job script copies it into the run folder


#======== parameters of the 2pt run, kept in the class; those not set here keep the defaults of TwoPtParams ========
#(temporal_shift 5, spatial_shift 3, t_boundary -1, anisotropy 1.0, xi_0 1.0, mrhs 12, gauge_fix_dir 3, hyp_dir_ignore -1)
delta_params = TwoPtParams()

#lattice and cfgs
delta_params.GLs = 32
delta_params.GLt = 96
delta_params.cfg_first = 204
delta_params.cfg_step = 6

#sources of the first cfg; they move by temporal_shift and spatial_shift from one cfg to the next (a subset of the production sources)
delta_params.t_base = np.arange(0, delta_params.GLt, 48)    # 2 t_src
delta_params.x_base = np.arange(0, delta_params.GLs, 16)    # 2 x
delta_params.y_base = np.arange(0, delta_params.GLs, 16)    # 2 y
delta_params.z_base = np.arange(0, delta_params.GLs, 16)    # 2 z                      # 2 x 2 x 2 x 2 = 16 sources

#QUDA: grid of 4 GPUs along t, clover Wilson Dirac operator, multigrid solver
delta_params.grid_size = [1, 1, 1, 4]
delta_params.mass = -0.05138
delta_params.tol = 1e-10
delta_params.maxiter = 1000
delta_params.clover_coeff_t = 1.04243
delta_params.clover_coeff_r = 1.04243
delta_params.multigrid = [[4, 4, 4, 4], [2, 2, 2, 2]]

#Coulomb gauge fixing by overrelaxation, then one HYP step
delta_params.gauge_fix_Nsteps = 25000
delta_params.gauge_fix_verbose_interval = 500
delta_params.gauge_fix_relax_boost = 1.7
delta_params.gauge_fix_tolerance = 1e-12
delta_params.gauge_fix_reunit_interval = 10
delta_params.gauge_fix_stopWtheta = 1
delta_params.hyp_n_steps = 1
delta_params.hyp_alpha1 = 0.75
delta_params.hyp_alpha2 = 0.6
delta_params.hyp_alpha3 = 0.3

#40 GB A100s: the momentum-space contraction runs on nt_chunk time slices at a time
delta_params.nt_chunk = 4

#quark boost of Gaussian j: k_j = mom_frac_j * mom_max along z, in units of 2pi/L
delta_params.mom_max = 6


#======== cfgs: one per node ========
cfg_list = np.arange(delta_params.cfg_first, delta_params.cfg_first + 800 * delta_params.cfg_step, delta_params.cfg_step)    # stream e 204, 210, ..., as in production
opt_cfg_index = list(range(0, 200, 20))         # node w holds cfg_list[opt_cfg_index[w]]: cfgs 204, 324, ..., 1284 (10 nodes; 120 trajectories apart)


#======== STARTING POINT: one row per Gaussian, [weight a, rho_T, rho_z, mom_frac] ========
#Row 1 is the production kernel [4.5, 2.25, 0.8] and its weight stays 1 (only relative weights matter). The other 12 Gaussians start
#with weight 0, so step 0 measures exactly the production kernel: it is the reference for the noise. Their widths and boosts below are where they
#start once their weight is switched on: rho_T 3, 4.5 and 6 (rho_z = rho_T / 2), each at four boosts around the production one.
#13 Gaussians x 4 - 1 fixed weight = 51 parameters. To add Gaussians, add rows.
opt_start_table = np.array([
    #  a     rho_T  rho_z  mom_frac
    [1.0,    4.5,   2.25,  0.8],        #  1: the production kernel
    [0.0,    4.5,   2.25,  0.2],        #  2: same widths, boost 1.2
    [0.0,    4.5,   2.25,  0.5],        #  3: same widths, boost 3.0
    [0.0,    4.5,   2.25,  1.1],        #  4: same widths, boost 6.6
    [0.0,    4.5,   2.25,  1.4],        #  5: same widths, boost 8.4
    [0.0,    3.0,   1.5,   0.2],        #  6: narrower in space
    [0.0,    3.0,   1.5,   0.5],        #  7
    [0.0,    3.0,   1.5,   1.1],        #  8
    [0.0,    3.0,   1.5,   1.4],        #  9
    [0.0,    6.0,   3.0,   0.2],        # 10: wider in space
    [0.0,    6.0,   3.0,   0.5],        # 11
    [0.0,    6.0,   3.0,   1.1],        # 12
    [0.0,    6.0,   3.0,   1.4],        # 13
])


#======== what is measured ========
#Dirac structure: the gammas of the interpolators (PyQUDA numbering, as gamma_ids in production). With more than one, every step computes the
#full gamma x gamma matrix from the same propagators and the score uses the best combination v (GEVP of C(t) and C(t+1), printed at every step).
#One entry, e.g. [7], turns this off: the score is then the plain effective energy of that gamma.
opt_gamma_ids = [15, 7, 11]         # Gamma(15) = G5, Gamma(7) = G5G4 = -G4G5, Gamma(11) = G3G5
opt_gamma_names = ["G5", "G5G4", "G3G5"]
opt_gamma_time_sign = [+1, -1, +1]  # C_ab(Lt - t) = s_a s_b C_ab(t): used when forward and backward are averaged (time_reflection_note of production)
opt_pz_list = [0, 1, 2, 3, 4, 5, 6] # momenta computed and printed at every step
opt_pz_target = [4, 5, 6]           # momenta in the score
opt_pz_weight = [1/3, 1/3, 1/3]     # their weights: score E = sum weight * [ln|C(t)| - ln|C(t+1)|], C = v^dag C v with the GEVP vector v
opt_t_score = 4                     # t of the score
opt_t_noise = 8                     # noise = relative error of C(t_noise) (the same combination v)
opt_t_max = 10                      # tsep kept: 0 .. opt_t_max, forward and backward of every source

#======== the noise condition ========
opt_noise_cut = 1.2                 # a point is allowed only if noise / noise(step 0) <= this, at every target pz
opt_noise_sample = "source"         # spread taken over all "source"s (160 samples) or over "cfg" averages (10 samples, rougher); both are printed
opt_noise_scan = "always"           # "always": after every kernel step that lowers E, move all mom_frac together until the noise stops falling
                                    #           (every mom_frac trial is one more step of inversions: a kernel step then costs at least 3 steps);
                                    # "over_cut": only when the noise of the kernel step is above the cut, and stop at the first point below it
opt_tilt_step = 0.1                 # mom_frac scan: all mom_frac move together, in steps of this (change of mom_frac of Gaussian 1)
opt_tilt_max = 0.6                  # largest shift tried in one scan
opt_tilt_min_gain = 0.02            # a mom_frac trial counts as quieter only if the noise falls by more than this fraction

#======== the gradient-descent step ========
#Every parameter is measured in units of its scale. A step of size 1 moves the parameters by about one scale in total.
#                   a      rho_T  rho_z  mom_frac
opt_scale        = [0.05,  0.25,  0.25,  0.05]
opt_lower        = [-5.0,  0.5,   0.5,  -0.5]       # a parameter never leaves [lower, upper]
opt_upper        = [+5.0,  8.0,   8.0,   2.0]
opt_free_columns = [True,  True,  True,  True]      # False freezes that column for all Gaussians
opt_step_start = 1.0                # first step size
opt_step_grow = 1.5                 # after an ACCEPT: step size x this (up to opt_step_max)
opt_step_shrink = 0.5               # after a REJECT: step size x this
opt_step_max = 4.0
opt_step_min = 0.05                 # stop when the step size falls below this
opt_grad_nsigma = 1.5               # gradient components smaller than this many (jackknife) errors are treated as zero
opt_kernel_h = 1e-5                 # step of the numerical derivative of the KERNEL formula (not of the correlator)
opt_max_steps = 100

#======== waiting for the other nodes ========
opt_wait_minutes = 60               # a node stops with an error if the file it waits for does not come within this time
opt_start_wait_minutes = 720        # the same for the first two waits of a node, when the other nodes may still be in the queue
opt_poll_seconds = 5                # how often a waiting node looks at the disk


#======== path definitions ========
opt_run_tag = "delta_test_run1"     # a new run needs a new tag: all files of a run live in opt_dir
opt_dir = f"/work/hdd/biqz/sliu54/smearing_parameter_opt/{opt_run_tag}"
quda_resource_path = "/u/sliu54/QUDA_resources/.cache/quda"
gauge_file_prefix = "/work/hdd/biqz/MILC/l3296f211b630m0074m037m440e/l3296f211b630m0074m037m440e"     # stream e, copied from Frontier with Globus


#======== the settings handed to tools_smearing_opt/opt_check.py (nothing new: the numbers above under the names it uses) ========
opt_settings = {
    "n_workers": len(opt_cfg_index), "start_table": opt_start_table, "mom_max": delta_params.mom_max, "L": delta_params.GLs,
    "gamma_names": opt_gamma_names, "gamma_time_sign": opt_gamma_time_sign,
    "pz_list": opt_pz_list, "pz_target": opt_pz_target, "pz_weight": opt_pz_weight, "t_score": opt_t_score, "t_noise": opt_t_noise,
    "noise_cut": opt_noise_cut, "noise_sample": opt_noise_sample, "noise_scan": opt_noise_scan, "tilt_step": opt_tilt_step, "tilt_max": opt_tilt_max, "tilt_min_gain": opt_tilt_min_gain,
    "scale": opt_scale, "lower": opt_lower, "upper": opt_upper, "free_columns": opt_free_columns,
    "step_start": opt_step_start, "step_grow": opt_step_grow, "step_shrink": opt_step_shrink, "step_max": opt_step_max, "step_min": opt_step_min,
    "grad_nsigma": opt_grad_nsigma, "max_steps": opt_max_steps, "wait_minutes": opt_wait_minutes, "poll_seconds": opt_poll_seconds,
}
