import numpy as np
from pt2_comm_tools import TwoPtParams          # tools/pt2_comm_tools.py: the job script copies it into the run folder

#Class built by claude
delta_params = TwoPtParams()

#lattice and cfgs
delta_params.GLs = 32
delta_params.GLt = 96
delta_params.cfg_first = 204
delta_params.cfg_step = 6

#sources of the first cfg; they move by temporal_shift and spatial_shift from one cfg to the next
delta_params.t_base = np.arange(0, delta_params.GLt, 24)    # 4 t_src
delta_params.x_base = np.arange(0, delta_params.GLs, 16)    # 2 x
delta_params.y_base = np.arange(0, delta_params.GLs, 16)    # 2 y
delta_params.z_base = np.arange(0, delta_params.GLs, 16)    # 2 z                      # 4 x 2 x 2 x 2 = 32 sources

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

#40 GB A100s: the site-local contractions run on nt_chunk local time slices at a time
delta_params.nt_chunk = 4

#quark boost k = mom_frac * mom_max along z, interpolators
delta_params.mom_min = 0
delta_params.mom_max = 6
delta_params.gamma_ids = [15, 7, 11]               # Gamma(15) = G5, Gamma(7) = G5G4 = -G4G5, Gamma(11) = G3G5

#sink momenta: the 99 p_f of the production's plain case (px, py in 0, -1, 1 and pz in -2 .. 8), which also hold the p_f and p_i of its q != 0 cases
delta_params.pz_list = list(range(delta_params.mom_min, delta_params.mom_max + 1))
delta_params.pf_plain_list = [[px, py, pz] for px in [0,-1,1] for py in [0,-1,1] for pz in range(delta_params.mom_min - 2, delta_params.mom_max + 3)]

#forward and backward time slices of the *_fb output (taken after the roll, so index 0 is t_src)
delta_params.tsep_max = 20
forward_index  = np.arange(delta_params.tsep_max)                           # 0, 1, 2, ..., 19:   C(t_src + tsep)
backward_index = (-np.arange(delta_params.tsep_max)) % delta_params.GLt     # 0, 95, 94, ..., 77: C(t_src - tsep)
delta_params.fb_index = {"pt2_forward": forward_index, "pt2_backward": backward_index}      # dataset name in the .h5 file: its time slices after the roll

#job: one cfg per job (array task)
delta_params.n_cfg = 1


#======== smearing operators: one row [rho_T, rho_z, mom_frac] each, the same smearing at source and sink ========
#Edit the rows freely. A new list needs a new smear_tag: the job refuses to write a list into a folder that holds another one.
#A band in (rho, mom_frac) instead of the full product: a large radius only works at large pz, so the large radii get the large boosts and the
#small radii the small ones; no operator is spent on a large radius with a small boost or a small radius with a large one.
#All anisotropic, rho_z = rho_T / 2. 9 radii rho_T 2.0 .. 6.0, closer around the production 3.25, each with a window of 3 or 4 boosts out of
#mom_frac 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 that slides up with rho_T (about P/3 per quark for the proton and P/2 for the pion at pz = mom_max
#are 0.33 and 0.5). The boost shapes the operator less as rho_z shrinks (its momentum profile along z is ~ exp(-rho_z^2 p^2 / 4)), so at the
#small radii (rho_z 1.0 and 1.25) the boosts give closer operators; the band puts them at the small boosts of low pz.
#x = in smear_list
#   rho_T   rho_z   mom_frac 0.3  0.4  0.5  0.6  0.7  0.8
#   2.0     1.0              x    x    x
#   2.5     1.25             x    x    x    x
#   3.0     1.5                   x    x    x
#   3.25    1.625                 x    x    x    x
#   3.5     1.75                       x    x    x
#   4.0     2.0                        x    x    x    x
#   4.5     2.25                            x    x    x
#   5.0     2.5                             x    x    x
#   6.0     3.0                             x    x    x
smear_tag = "coulomb_run3_pion_final_decision_20op_32src_200cfg"    
smear_list = [
    #rho_T  rho_z     mom_frac
    #======pion, pz = 0 (G5G4)======
    [1.0,   1.0,      0],
    [1.1,   1.1,      0],
    [1.25,  1.25,     0],
    [1.5,   1.5,      0],
    [1.75,  1.75,     0],
    
    #======pion, pz = 1-6 (G5G4)======
    [1.25,  1.25 / 2,  0.8],
    [1.5,   1.5 / 2,  0.8],
    [1.75,  1.75 / 2, 0.8],
    [4.5,   4.5 / 2,  0.8],
    [6.0,   6.0 / 2,  0.8],

    [1.25,  1.25 / 1.25,  0.8],
    [1.5,   1.5 / 1.25,  0.8],
    [1.75,  1.75 / 1.25, 0.8],
    [4.5,   4.5 / 1.25,  0.8],
    [6.0,   6.0 / 1.25,  0.8],

    
    
    #======proton, pz = 0======
    [5.5,   5.5,      0],
    [5.75,  5.75,     0],
    [6.0,   6.0,      0],
    #=====proton, pz = 1-6======
    [5.75,  5.75 / 2,  0.8],

    #=====reference============
    [3.25,   3.25 / 3,  0.8],    
]
assert len(smear_list) <= 30, "at most 30 smearings: the time limit of the sbatch script assumes it"
assert len({tuple(row) for row in smear_list}) == len(smear_list), "smear_list has a repeated row"
assert all(len(row) == 3 and row[0] > 0 and row[1] > 0 for row in smear_list), "every row is [rho_T > 0, rho_z > 0, mom_frac]"
smear_k_list = [np.array([0, 0, frac * delta_params.mom_max]) for rho_T, rho_z, frac in smear_list]     # boost in units of 2pi/L
smear_names = [f"rT{rho_T:.4g}_rz{rho_z:.4g}_f{frac:.4g}" for rho_T, rho_z, frac in smear_list]


#======== cfgs and sources (not in the class) ========
cfg_list = np.arange(delta_params.cfg_first, delta_params.cfg_first + 200 * 1 * delta_params.cfg_step, 1 * delta_params.cfg_step)    # stream e 204, 234, ..., 1374
cfg_measure_spacing = 1
n_spatial_src = len(delta_params.x_base) * len(delta_params.y_base) * len(delta_params.z_base)


#======== notes written into the .h5 attributes ========
smearing_note = ("boosted Gaussian in Coulomb gauge, no gauge links: K(d) = exp(-(dx^2+dy^2)/rho_T^2 - dz^2/rho_z^2) exp(+i 2pi/L k.d), d = x - y, "
                 "Gaussian part normalized to sum 1; the same K at source and sink (diagonal correlators only); k = (0, 0, mom_frac * mom_max), "
                 "smear_list rows [rho_T, rho_z, mom_frac]; rho_z = rho_T equals the width of Wuppertal smearing with the same rho")
gauge_fix_note = (f"Coulomb gauge fixing of the unsmeared links by overrelaxation, fixingOVR(gauge_dir={delta_params.gauge_fix_dir}, "
                  f"Nsteps={delta_params.gauge_fix_Nsteps}, verbose_interval={delta_params.gauge_fix_verbose_interval}, "
                  f"relax_boost={delta_params.gauge_fix_relax_boost}, tolerance={delta_params.gauge_fix_tolerance}, "
                  f"reunit_interval={delta_params.gauge_fix_reunit_interval}, stopWtheta={delta_params.gauge_fix_stopWtheta}), before HYP")
momentum_note = ("sink phase exp(-2 pi i p.(x - x_src)/L) (conjugate of PyQUDA MomentumPhase): the momentum_list dataset gives the physical sink "
                 "momentum p_f of every entry; the quarks are boosted toward physical +z, so +pz has the best overlap")
source_note = (f"spatial sources averaged: for every t_src, plain mean over the {n_spatial_src} positions x_src_list x y_src_list x z_src_list "
               "(q = 0, no phase weight)")
time_reflection_note = ("C_ab(Lt - t) = s_a s_b C_ab(t) with s = +1 for G5, -1 for G5G4, +1 for G3G5 (a = gamma_sink, b = gamma_source): "
                        "the elements pairing G5G4 with G5 or G3G5 are odd, include the sign when averaging forward and backward")
time_index_note = (f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {delta_params.tsep_max - 1}; "
                   "index 0 is the source slice in both. dim_time gives the time after the roll, (t - t_src) % Lt, of every tsep; the other time slices are not saved")


#======== path definitions ========
current_dir = "/projects/biqz/sliu54/2pt_production/coulomb_run3_pion_final_decision_20op_32src_200cfg"
quda_resource_path = "/u/sliu54/QUDA_resources/.cache/quda"
gauge_file_prefix = "/work/hdd/biqz/MILC/l3296f211b630m0074m037m440e/l3296f211b630m0074m037m440e"     # stream e, copied from Frontier with Globus
