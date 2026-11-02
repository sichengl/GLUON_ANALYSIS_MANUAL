#Fermilab boost test at narrow rho_z, rho_z 1.5 run (large-pz pion, DIAGONAL run) on stream e: the first 200 cfgs (204, 210, ..., 1398) with the
#8t x 4x x 4y x 8z = 1024 sources per cfg, the 9 smearings of smear_list, each at source and sink only (no smearing GEVP), three gammas (G5, G5G4, G3G5).
#Copy of ../smear_fermilab_v3_followup_k6/sepq_2pt_setup_fermilab_final.py: only smear_list, smear_names, smear_tag and smearing_note differ (same cfgs,
#sources, solver and current_dir; the output goes to its own smear_tag folder). The sources of a cfg are those of Delta's at every third t_src.
import numpy as np
from pt2_comm_tools import TwoPtParams          # tools/pt2_comm_tools.py: the job script copies it into the run folder


#======== parameters of the 2pt run, kept in the class; those not set here keep the defaults of TwoPtParams ========
#(temporal_shift 5, spatial_shift 3, t_boundary -1, anisotropy 1.0, xi_0 1.0, mrhs 12, gauge_fix_dir 3, hyp_dir_ignore -1, --icfg 0, --n 20)
delta_params = TwoPtParams()

#lattice and cfgs
delta_params.GLs = 32
delta_params.GLt = 96
delta_params.cfg_first = 204
delta_params.cfg_step = 6

#sources of the first cfg; they move by temporal_shift and spatial_shift from one cfg to the next
delta_params.t_base = np.arange(0, delta_params.GLt, 12)    # 8 t_src
delta_params.x_base = np.arange(0, delta_params.GLs, 8)     # 4 x
delta_params.y_base = np.arange(0, delta_params.GLs, 8)     # 4 y
delta_params.z_base = np.arange(0, delta_params.GLs, 4)     # 8 z                      # 8 x 4 x 4 x 8 = 1024 sources

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

#smearings [rho_T, rho_z, mom_frac], each with its own quark boost k = mom_frac * mom_max along z; interpolators
delta_params.smear_list = [[4.5, 1.5, 0.9],      # boost scan at rho_z 1.5
                           [4.5, 1.5, 0.95],
                           [4.5, 1.5, 1.0],      # also in v4 and the follow-up (same cfgs and sources): checks this diagonal code against the final script
                           [4.5, 1.5, 1.05],
                           [4.5, 1.5, 1.1],
                           [4.5, 1.5, 1.2],
                           [4.5, 1.5, 1.3],
                           [3.5, 1.5, 1.0],      # rho_T one below 4.5
                           [5.5, 1.5, 1.0]]      # rho_T one above 4.5
delta_params.mom_min = 0
delta_params.mom_max = 6
delta_params.k_list = [np.array([0, 0, frac * delta_params.mom_max]) for rho_T, rho_z, frac in delta_params.smear_list]
delta_params.gamma_ids = [15, 7, 11]                # Gamma(15) = G5, Gamma(7) = G5G4 = -G4G5, Gamma(11) = G3G5

#momentum transfers and sink momenta
delta_params.q_forward_list     = [[ 0, 0, 0]]
delta_params.q_sym_0xi_list     = [[ 2, 0, 0],[-2, 0, 0],[ 0, 2, 0],[ 0,-2, 0],[ 2, 2, 0],[ 2,-2, 0],[-2, 2, 0],[-2,-2, 0]]
delta_params.q_sym_non0xi_list  = [[ 0, 0, 2],[ 2, 0, 2],[-2, 0, 2],[ 0, 2, 2],[ 0,-2, 2],[ 2, 2, 2],[ 2,-2, 2],[-2, 2, 2],[-2,-2, 2],
                                   [ 0, 0,-2],[ 2, 0,-2],[-2, 0,-2],[ 0, 2,-2],[ 0,-2,-2],[ 2, 2,-2],[ 2,-2,-2],[-2, 2,-2],[-2,-2,-2]]
delta_params.q_asy_0xi_list     = [[ 1, 0, 0],[-1, 0, 0],[ 0, 1, 0],[ 0,-1, 0],[ 1, 1, 0],[ 1,-1, 0],[-1, 1, 0],[-1,-1, 0]]
delta_params.q_asy_non0xi_list  = [[ 0, 0, 1],[ 1, 0, 1],[-1, 0, 1],[ 0, 1, 1],[ 0,-1, 1],[ 1, 1, 1],[ 1,-1, 1],[-1, 1, 1],[-1,-1, 1],
                                   [ 0, 0,-1],[ 1, 0,-1],[-1, 0,-1],[ 0, 1,-1],[ 0,-1,-1],[ 1, 1,-1],[ 1,-1,-1],[-1, 1,-1],[-1,-1,-1],
                                   [ 0, 0, 2],[ 1, 0, 2],[-1, 0, 2],[ 0, 1, 2],[ 0,-1, 2],[ 1, 1, 2],[ 1,-1, 2],[-1, 1, 2],[-1,-1, 2],
                                   [ 0, 0,-2],[ 1, 0,-2],[-1, 0,-2],[ 0, 1,-2],[ 0,-1,-2],[ 1, 1,-2],[ 1,-1,-2],[-1, 1,-2],[-1,-1,-2]]
delta_params.q_plain_list       = [[ 0, 0, 0]]
delta_params.pz_list = list(range(delta_params.mom_min, delta_params.mom_max + 1))
delta_params.pf_plain_list = [[px, py, pz] for px in [0,-1,1] for py in [0,-1,1] for pz in range(delta_params.mom_min - 2, delta_params.mom_max + 3)]

#forward and backward time slices of the *_fb output (taken after the roll, so index 0 is t_src)
delta_params.tsep_max = 20
forward_index  = np.arange(delta_params.tsep_max)                           # 0, 1, 2, ..., 19:   C(t_src + tsep)
backward_index = (-np.arange(delta_params.tsep_max)) % delta_params.GLt     # 0, 95, 94, ..., 77: C(t_src - tsep)
delta_params.fb_index = {"pt2_forward": forward_index, "pt2_backward": backward_index}      # dataset name in the .h5 file: its time slices after the roll


#======== cfgs, smearing operators and momentum tables (not in the class) ========
cfg_list = np.arange(delta_params.cfg_first, delta_params.cfg_first + 800 * delta_params.cfg_step, delta_params.cfg_step)    # stream e 204, 210, ..., 1398
cfg_measure_spacing = 1
src_phase_sign = -1
n_spatial_src = len(delta_params.x_base) * len(delta_params.y_base) * len(delta_params.z_base)
smear_names = ["rho4p5z1p5k5p4", "rho4p5z1p5k5p7", "rho4p5z1p5k6", "rho4p5z1p5k6p3", "rho4p5z1p5k6p6", "rho4p5z1p5k7p2", "rho4p5z1p5k7p8", "rho3p5z1p5k6", "rho5p5z1p5k6"]     # one name per row of delta_params.smear_list, unique (the notebooks label operators by name)
smear_tag = "coulomb_momfrac_rhoz1p5_diag"      # short and fixed (a name built from 9 smear_names would pass the 255-character file name limit); _diag: one smear axis, the same smearing at source and sink
#a new smear_list needs a new smear_tag: the job refuses an output folder that holds another smear_list

P_list = np.array([[0, 0, pz] for pz in delta_params.pz_list])
pf_forward_table    = P_list + np.array(delta_params.q_forward_list)[:, None]
pf_sym_0xi_table    = P_list + np.array(delta_params.q_sym_0xi_list)[:, None] // 2
pf_sym_non0xi_table = P_list + np.array(delta_params.q_sym_non0xi_list)[:, None] // 2
pf_asy_0xi_table    = P_list + np.array(delta_params.q_asy_0xi_list)[:, None]
pf_asy_non0xi_table = P_list + np.array(delta_params.q_asy_non0xi_list)[:, None]

pf_forward_index_list    = [[delta_params.pf_plain_list.index(pf) for pf in row] for row in pf_forward_table.tolist()]
pf_sym_0xi_index_list    = [[delta_params.pf_plain_list.index(pf) for pf in row] for row in pf_sym_0xi_table.tolist()]
pf_sym_non0xi_index_list = [[delta_params.pf_plain_list.index(pf) for pf in row] for row in pf_sym_non0xi_table.tolist()]
pf_asy_0xi_index_list    = [[delta_params.pf_plain_list.index(pf) for pf in row] for row in pf_asy_0xi_table.tolist()]
pf_asy_non0xi_index_list = [[delta_params.pf_plain_list.index(pf) for pf in row] for row in pf_asy_non0xi_table.tolist()]


#======== notes written into the .h5 attributes ========
smearing_note = ("boosted Gaussian in Coulomb gauge, no gauge links: K(d) = exp(-(dx^2+dy^2)/rho_T^2 - dz^2/rho_z^2) exp(+i 2pi/L k.d), d = x - y, "
                 "Gaussian part normalized to sum 1; smearing i of smear_list, [rho_T, rho_z, mom_frac], is used at source AND sink (diagonal run, no smearing "
                 "cross terms), quark boost k = mom_frac * mom_max along z: [4.5, 1.5, 0.9], [4.5, 1.5, 0.95], [4.5, 1.5, 1], [4.5, 1.5, 1.05], [4.5, 1.5, 1.1], [4.5, 1.5, 1.2], [4.5, 1.5, 1.3], [3.5, 1.5, 1], [5.5, 1.5, 1]")
gauge_fix_note = (f"Coulomb gauge fixing of the unsmeared links by overrelaxation, fixingOVR(gauge_dir={delta_params.gauge_fix_dir}, "
                  f"Nsteps={delta_params.gauge_fix_Nsteps}, verbose_interval={delta_params.gauge_fix_verbose_interval}, "
                  f"relax_boost={delta_params.gauge_fix_relax_boost}, tolerance={delta_params.gauge_fix_tolerance}, "
                  f"reunit_interval={delta_params.gauge_fix_reunit_interval}, stopWtheta={delta_params.gauge_fix_stopWtheta}), before HYP")
momentum_note = ("sink phase exp(-2 pi i p.(x - x_src)/L) (conjugate of PyQUDA MomentumPhase): the momentum_list dataset gives the physical sink "
                 "momentum p_f of every entry; the quarks are boosted toward physical +z, so +pz has the best overlap")
source_note = (f"spatial sources contracted: for every t_src, mean over the {n_spatial_src} positions x_src_list x y_src_list x z_src_list, weighted by "
               "exp(-i 2pi/Ls q.x_src) with q from the q_list dataset. This phase is ALREADY APPLIED: do not apply it again in the 3pt build")
time_reflection_note = ("C_ab(Lt - t) = s_a s_b C_ab(t) with s = +1 for G5, -1 for G5G4, +1 for G3G5 (a = gamma_sink, b = gamma_source): "
                        "the elements pairing G5G4 with G5 or G3G5 are odd, include the sign when averaging forward and backward")
time_index_note = (f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {delta_params.tsep_max - 1}; "
                   "index 0 is the source slice in both. dim_time gives the time after the roll, (t - t_src) % Lt, of every tsep; the other time slices are not saved")
transfer_note = ("P_f = P_i + q (physical momenta, equal to the labels): source weight exp(-i 2pi/Ls q.x_src), gluon operator "
                    "Fourier transformed with exp(+i 2pi/Ls q.z) (FF q_phase_sign = +1). The initial-state 2pt of a 3pt at sink "
                    "label pf and transfer q is at label pf - q.")


#======== path definitions ========
current_dir = "/lustre2/gluonp0/sliu1/2pt_production/smear_test_1024src"
quda_resource_path = "/lustre2/gluonp0/sliu1/.cache/quda"                 # the tuning cache of the earlier Fermilab 2pt runs (same lattice and grid)
gauge_file_prefix = "/lustre2/gluonp0/sliu1/MILC/l3296f211b630m0074m037m440e/l3296f211b630m0074m037m440e"     # stream e, copied from Frontier with Globus: only the cfgs copied so far can run
