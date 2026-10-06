#Fermilab smearing tuning: 30 Coulomb-gauge boosted smearings, each written directly as one row [rho_T, rho_z, mom_frac] of smear_list
#(not the product shape_list x mom_frac_list of sepq_2pt_setup_fermilab_classused.py), with the same smearing at source and sink: diagonal
#correlators only. 32 sources per cfg (8 t_src x 2 x x 2 y x 1 z) on 40 stream-d cfgs 204, 234, ..., 1374. These are the cfgs of the earlier
#64-source Coulomb and Wuppertal tests (local_analysis/test_2pts), and the 32 sources are half of their 64 (same shifts, z_base cut to [0]),
#so every operator here can be compared with those runs on the same gauge fields.
import numpy as np
from pt2_comm_tools import TwoPtParams          # tools/pt2_comm_tools.py: the job script copies it into the run folder


#======== parameters of the 2pt run, kept in the class; those not set here keep the defaults of TwoPtParams ========
#(temporal_shift 5, spatial_shift 3, t_boundary -1, anisotropy 1.0, xi_0 1.0, mrhs 12, gauge_fix_dir 3, hyp_dir_ignore -1, --icfg 0)
#The class fields shape_list, mom_frac_list, k_list and the q lists are not used here: the smearings are smear_list below, all at q = 0.
fermilab_params = TwoPtParams()

#lattice and cfgs
fermilab_params.GLs = 32
fermilab_params.GLt = 96
fermilab_params.cfg_first = 204
fermilab_params.cfg_step = 6

#sources of the first cfg; they move by temporal_shift and spatial_shift from one cfg to the next
fermilab_params.t_base = np.arange(0, fermilab_params.GLt, 12)    # 8 t_src
fermilab_params.x_base = np.arange(0, fermilab_params.GLs, 16)    # 2 x
fermilab_params.y_base = np.arange(0, fermilab_params.GLs, 16)    # 2 y
fermilab_params.z_base = np.array([0])                            # 1 z: 8 x 2 x 2 x 1 = 32 sources

#QUDA: grid of 4 GPUs along t, clover Wilson Dirac operator, multigrid solver
fermilab_params.grid_size = [1, 1, 1, 4]
fermilab_params.mass = -0.05138
fermilab_params.tol = 1e-10
fermilab_params.maxiter = 1000
fermilab_params.clover_coeff_t = 1.04243
fermilab_params.clover_coeff_r = 1.04243
fermilab_params.multigrid = [[4, 4, 4, 4], [2, 2, 2, 2]]

#Coulomb gauge fixing by overrelaxation, then one HYP step
fermilab_params.gauge_fix_Nsteps = 20000
fermilab_params.gauge_fix_verbose_interval = 500
fermilab_params.gauge_fix_relax_boost = 1.7
fermilab_params.gauge_fix_tolerance = 1e-12
fermilab_params.gauge_fix_reunit_interval = 10
fermilab_params.gauge_fix_stopWtheta = 1
fermilab_params.hyp_n_steps = 1
fermilab_params.hyp_alpha1 = 0.75
fermilab_params.hyp_alpha2 = 0.6
fermilab_params.hyp_alpha3 = 0.3

#quark boost k = mom_frac * mom_max along z, interpolators
fermilab_params.mom_min = 0
fermilab_params.mom_max = 6
fermilab_params.gamma_ids = [15, 7]                # Gamma(15) = G5, Gamma(7) = G5G4 = -G4G5

#sink momenta: the 99 p_f of the production's plain case (px, py in 0, -1, 1 and pz in -2 .. 8), which also hold the p_f and p_i of its q != 0 cases
fermilab_params.pz_list = list(range(fermilab_params.mom_min, fermilab_params.mom_max + 1))
fermilab_params.pf_plain_list = [[px, py, pz] for px in [0,-1,1] for py in [0,-1,1] for pz in range(fermilab_params.mom_min - 2, fermilab_params.mom_max + 3)]

#forward and backward time slices of the *_fb output (taken after the roll, so index 0 is t_src)
fermilab_params.tsep_max = 20
forward_index  = np.arange(fermilab_params.tsep_max)                           # 0, 1, 2, ..., 19:   C(t_src + tsep)
backward_index = (-np.arange(fermilab_params.tsep_max)) % fermilab_params.GLt     # 0, 95, 94, ..., 77: C(t_src - tsep)
fermilab_params.fb_index = {"pt2_forward": forward_index, "pt2_backward": backward_index}      # dataset name in the .h5 file: its time slices after the roll

#job: one cfg per job (array task)
fermilab_params.n_cfg = 1


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
smear_tag = "coulomb_tune1"
smear_list = [
    #rho_T  rho_z       mom_frac
    [2.0,   2.0 / 2,    0.3],
    [2.0,   2.0 / 2,    0.4],
    [2.0,   2.0 / 2,    0.5],
    [2.5,   2.5 / 2,    0.3],
    [2.5,   2.5 / 2,    0.4],
    [2.5,   2.5 / 2,    0.5],
    [2.5,   2.5 / 2,    0.6],
    [3.0,   3.0 / 2,    0.4],
    [3.0,   3.0 / 2,    0.5],
    [3.0,   3.0 / 2,    0.6],
    [3.25,  3.25 / 2,   0.4],
    [3.25,  3.25 / 2,   0.5],
    [3.25,  3.25 / 2,   0.6],      # aniso, frac 0.6 of the earlier tests and of the production
    [3.25,  3.25 / 2,   0.7],      # aniso, frac 0.7 of the frac0p65-0p7-0p75 test
    [3.5,   3.5 / 2,    0.5],
    [3.5,   3.5 / 2,    0.6],
    [3.5,   3.5 / 2,    0.7],
    [4.0,   4.0 / 2,    0.5],
    [4.0,   4.0 / 2,    0.6],
    [4.0,   4.0 / 2,    0.7],
    [4.0,   4.0 / 2,    0.8],
    [4.5,   4.5 / 2,    0.6],
    [4.5,   4.5 / 2,    0.7],
    [4.5,   4.5 / 2,    0.8],
    [5.0,   5.0 / 2,    0.6],
    [5.0,   5.0 / 2,    0.7],
    [5.0,   5.0 / 2,    0.8],
    [6.0,   6.0 / 2,    0.6],
    [6.0,   6.0 / 2,    0.7],
    [6.0,   6.0 / 2,    0.8],
]
assert len(smear_list) <= 30, "at most 30 smearings: the time limit of the sbatch script assumes it"
assert len({tuple(row) for row in smear_list}) == len(smear_list), "smear_list has a repeated row"
assert all(len(row) == 3 and row[0] > 0 and row[1] > 0 for row in smear_list), "every row is [rho_T > 0, rho_z > 0, mom_frac]"
smear_k_list = [np.array([0, 0, frac * fermilab_params.mom_max]) for rho_T, rho_z, frac in smear_list]     # boost in units of 2pi/L
smear_names = [f"rT{rho_T:.4g}_rz{rho_z:.4g}_f{frac:.4g}" for rho_T, rho_z, frac in smear_list]


#======== cfgs and sources (not in the class) ========
cfg_list = np.arange(fermilab_params.cfg_first, fermilab_params.cfg_first + 40 * 5 * fermilab_params.cfg_step, 5 * fermilab_params.cfg_step)    # stream d 204, 234, ..., 1374
cfg_measure_spacing = 1
n_spatial_src = len(fermilab_params.x_base) * len(fermilab_params.y_base) * len(fermilab_params.z_base)


#======== notes written into the .h5 attributes ========
smearing_note = ("boosted Gaussian in Coulomb gauge, no gauge links: K(d) = exp(-(dx^2+dy^2)/rho_T^2 - dz^2/rho_z^2) exp(+i 2pi/L k.d), d = x - y, "
                 "Gaussian part normalized to sum 1; the same K at source and sink (diagonal correlators only); k = (0, 0, mom_frac * mom_max), "
                 "smear_list rows [rho_T, rho_z, mom_frac]; rho_z = rho_T equals the width of Wuppertal smearing with the same rho")
gauge_fix_note = (f"Coulomb gauge fixing of the unsmeared links by overrelaxation, fixingOVR(gauge_dir={fermilab_params.gauge_fix_dir}, "
                  f"Nsteps={fermilab_params.gauge_fix_Nsteps}, verbose_interval={fermilab_params.gauge_fix_verbose_interval}, "
                  f"relax_boost={fermilab_params.gauge_fix_relax_boost}, tolerance={fermilab_params.gauge_fix_tolerance}, "
                  f"reunit_interval={fermilab_params.gauge_fix_reunit_interval}, stopWtheta={fermilab_params.gauge_fix_stopWtheta}), before HYP")
momentum_note = ("sink phase exp(-2 pi i p.(x - x_src)/L) (conjugate of PyQUDA MomentumPhase): the momentum_list dataset gives the physical sink "
                 "momentum p_f of every entry; the quarks are boosted toward physical +z, so +pz has the best overlap")
source_note = (f"spatial sources averaged: for every t_src, plain mean over the {n_spatial_src} positions x_src_list x y_src_list x z_src_list "
               "(q = 0, no phase weight)")
time_reflection_note = ("C_ab(Lt - t) = s_a s_b C_ab(t) with s = +1 for G5, -1 for G5G4 (a = gamma_sink, b = gamma_source): "
                        "the G5-G5G4 elements are odd, include the sign when averaging forward and backward")
time_index_note = (f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {fermilab_params.tsep_max - 1}; "
                   "index 0 is the source slice in both. dim_time gives the time after the roll, (t - t_src) % Lt, of every tsep; the other time slices are not saved")


#======== path definitions ========
current_dir = "/lustre2/gluonp0/sliu1/2pt_production/smear_tuning"
quda_resource_path = "/lustre2/gluonp0/sliu1/.cache/quda"
gauge_file_prefix = "/lustre2/gluonp0/MILC/l3296f211b630m0074m037m440d/l3296f211b630m0074m037m440d"
