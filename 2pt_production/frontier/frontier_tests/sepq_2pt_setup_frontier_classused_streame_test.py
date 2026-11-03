#Frontier test on stream e: 10 cfgs with the 8t x 4x x 4y x 8z = 1024 sources of the old stream-e 2pt data (Frontier, 799 cfgs),
#to compare the physics (energies vs pz) with it. Same as sepq_2pt_setup_frontier_classused.py except the source grid, the gauge
#stream, the 10 cfgs and the output folder (the stream-d production also writes 1024src files for the same cfg numbers).
import numpy as np
from pt2_comm_tools import TwoPtParams          # tools/pt2_comm_tools.py: the job script copies it into the run folder


#======== parameters of the 2pt run, kept in the class; those not set here keep the defaults of TwoPtParams ========
#(temporal_shift 5, spatial_shift 3, t_boundary -1, anisotropy 1.0, xi_0 1.0, mrhs 12, gauge_fix_dir 3, hyp_dir_ignore -1, --icfg 0, --n 20)
frontier_params = TwoPtParams()

#lattice and cfgs
frontier_params.GLs = 32
frontier_params.GLt = 96
frontier_params.cfg_first = 204
frontier_params.cfg_step = 6

#sources of the first cfg; they move by temporal_shift and spatial_shift from one cfg to the next
frontier_params.t_base = np.arange(0, frontier_params.GLt, 12)    # 8 t_src
frontier_params.x_base = np.arange(0, frontier_params.GLs, 8)     # 4 x
frontier_params.y_base = np.arange(0, frontier_params.GLs, 8)     # 4 y
frontier_params.z_base = np.arange(0, frontier_params.GLs, 4)     # 8 z

#QUDA: grid of 4 GPUs along t, clover Wilson Dirac operator, multigrid solver
frontier_params.grid_size = [1, 1, 1, 4]
frontier_params.mass = -0.05138
frontier_params.tol = 1e-10
frontier_params.maxiter = 1000
frontier_params.clover_coeff_t = 1.04243
frontier_params.clover_coeff_r = 1.04243
frontier_params.multigrid = [[4, 4, 4, 4], [2, 2, 2, 2]]

#Coulomb gauge fixing by overrelaxation, then one HYP step
frontier_params.gauge_fix_Nsteps = 20000
frontier_params.gauge_fix_verbose_interval = 500
frontier_params.gauge_fix_relax_boost = 1.7
frontier_params.gauge_fix_tolerance = 1e-12
frontier_params.gauge_fix_reunit_interval = 10
frontier_params.gauge_fix_stopWtheta = 1
frontier_params.hyp_n_steps = 1
frontier_params.hyp_alpha1 = 0.75
frontier_params.hyp_alpha2 = 0.6
frontier_params.hyp_alpha3 = 0.3

#smearing shapes [rho_T, rho_z], quark boosts, interpolators
rho_T = 3.25
frontier_params.shape_list = [[rho_T, rho_T], [rho_T, rho_T / 2], [rho_T, rho_T / 3]]
frontier_params.mom_frac_list = [0.6, 0.3]
frontier_params.mom_min = 0
frontier_params.mom_max = 6
frontier_params.k_list = [np.array([0, 0, frac * frontier_params.mom_max]) for frac in frontier_params.mom_frac_list]
frontier_params.gamma_ids = [15, 7]                # Gamma(15) = G5, Gamma(7) = G5G4 = -G4G5

#momentum transfers and sink momenta
frontier_params.q_forward_list     = [[ 0, 0, 0]]
frontier_params.q_sym_0xi_list     = [[ 2, 0, 0],[-2, 0, 0],[ 0, 2, 0],[ 0,-2, 0],[ 2, 2, 0],[ 2,-2, 0],[-2, 2, 0],[-2,-2, 0]]
frontier_params.q_sym_non0xi_list  = [[ 0, 0, 2],[ 2, 0, 2],[-2, 0, 2],[ 0, 2, 2],[ 0,-2, 2],[ 2, 2, 2],[ 2,-2, 2],[-2, 2, 2],[-2,-2, 2],
                                      [ 0, 0,-2],[ 2, 0,-2],[-2, 0,-2],[ 0, 2,-2],[ 0,-2,-2],[ 2, 2,-2],[ 2,-2,-2],[-2, 2,-2],[-2,-2,-2]]
frontier_params.q_asy_0xi_list     = [[ 1, 0, 0],[-1, 0, 0],[ 0, 1, 0],[ 0,-1, 0],[ 1, 1, 0],[ 1,-1, 0],[-1, 1, 0],[-1,-1, 0]]
frontier_params.q_asy_non0xi_list  = [[ 0, 0, 1],[ 1, 0, 1],[-1, 0, 1],[ 0, 1, 1],[ 0,-1, 1],[ 1, 1, 1],[ 1,-1, 1],[-1, 1, 1],[-1,-1, 1],
                                      [ 0, 0,-1],[ 1, 0,-1],[-1, 0,-1],[ 0, 1,-1],[ 0,-1,-1],[ 1, 1,-1],[ 1,-1,-1],[-1, 1,-1],[-1,-1,-1],
                                      [ 0, 0, 2],[ 1, 0, 2],[-1, 0, 2],[ 0, 1, 2],[ 0,-1, 2],[ 1, 1, 2],[ 1,-1, 2],[-1, 1, 2],[-1,-1, 2],
                                      [ 0, 0,-2],[ 1, 0,-2],[-1, 0,-2],[ 0, 1,-2],[ 0,-1,-2],[ 1, 1,-2],[ 1,-1,-2],[-1, 1,-2],[-1,-1,-2]]
frontier_params.q_plain_list       = [[ 0, 0, 0]]
frontier_params.pz_list = list(range(frontier_params.mom_min, frontier_params.mom_max + 1))
frontier_params.pf_plain_list = [[px, py, pz] for px in [0,-1,1] for py in [0,-1,1] for pz in range(frontier_params.mom_min - 2, frontier_params.mom_max + 3)]

#forward and backward time slices of the *_fb output (taken after the roll, so index 0 is t_src)
frontier_params.tsep_max = 20
forward_index  = np.arange(frontier_params.tsep_max)                           # 0, 1, 2, ..., 19:   C(t_src + tsep)
backward_index = (-np.arange(frontier_params.tsep_max)) % frontier_params.GLt     # 0, 95, 94, ..., 77: C(t_src - tsep)
frontier_params.fb_index = {"pt2_forward": forward_index, "pt2_backward": backward_index}      # dataset name in the .h5 file: its time slices after the roll


#======== cfgs, smearing operators and momentum tables (not in the class) ========
cfg_list = np.arange(frontier_params.cfg_first, frontier_params.cfg_first + 10 * frontier_params.cfg_step, frontier_params.cfg_step)    # stream e 204, 210, ..., 258
cfg_measure_spacing = 1
src_phase_sign = -1
n_spatial_src = len(frontier_params.x_base) * len(frontier_params.y_base) * len(frontier_params.z_base)
shape_names = ["iso", "aniso2", "aniso3"]      # rho_z = rho_T, rho_T/2, rho_T/3
smear_list = [[(i_shape,shape),(i_frac,frac)] for i_shape, shape in enumerate(frontier_params.shape_list) for i_frac, frac in enumerate(frontier_params.mom_frac_list)]
frac_str = [("%.3f" % frac).rstrip("0").rstrip(".").replace(".", "p") for frac in frontier_params.mom_frac_list]
smear_tag = f"coulomb_rhoT{rho_T}_{'-'.join(shape_names)}_frac{'-'.join(frac_str)}"

P_list = np.array([[0, 0, pz] for pz in frontier_params.pz_list])
pf_forward_table    = P_list + np.array(frontier_params.q_forward_list)[:, None]
pf_sym_0xi_table    = P_list + np.array(frontier_params.q_sym_0xi_list)[:, None] // 2
pf_sym_non0xi_table = P_list + np.array(frontier_params.q_sym_non0xi_list)[:, None] // 2
pf_asy_0xi_table    = P_list + np.array(frontier_params.q_asy_0xi_list)[:, None]
pf_asy_non0xi_table = P_list + np.array(frontier_params.q_asy_non0xi_list)[:, None]

pf_forward_index_list    = [[frontier_params.pf_plain_list.index(pf) for pf in row] for row in pf_forward_table.tolist()]
pf_sym_0xi_index_list    = [[frontier_params.pf_plain_list.index(pf) for pf in row] for row in pf_sym_0xi_table.tolist()]
pf_sym_non0xi_index_list = [[frontier_params.pf_plain_list.index(pf) for pf in row] for row in pf_sym_non0xi_table.tolist()]
pf_asy_0xi_index_list    = [[frontier_params.pf_plain_list.index(pf) for pf in row] for row in pf_asy_0xi_table.tolist()]
pf_asy_non0xi_index_list = [[frontier_params.pf_plain_list.index(pf) for pf in row] for row in pf_asy_non0xi_table.tolist()]


#======== notes written into the .h5 attributes ========
smearing_note = ("boosted Gaussian in Coulomb gauge, no gauge links: K(d) = exp(-(dx^2+dy^2)/rho_T^2 - dz^2/rho_z^2) exp(+i 2pi/L k.d), d = x - y, "
                 "Gaussian part normalized to sum 1; the same K at source and sink; rho_z = rho_T equals the width of N40 rho3.25 Wuppertal smearing")
gauge_fix_note = (f"Coulomb gauge fixing of the unsmeared links by overrelaxation, fixingOVR(gauge_dir={frontier_params.gauge_fix_dir}, "
                  f"Nsteps={frontier_params.gauge_fix_Nsteps}, verbose_interval={frontier_params.gauge_fix_verbose_interval}, "
                  f"relax_boost={frontier_params.gauge_fix_relax_boost}, tolerance={frontier_params.gauge_fix_tolerance}, "
                  f"reunit_interval={frontier_params.gauge_fix_reunit_interval}, stopWtheta={frontier_params.gauge_fix_stopWtheta}), before HYP")
momentum_note = ("sink phase exp(-2 pi i p.(x - x_src)/L) (conjugate of PyQUDA MomentumPhase): the momentum_list dataset gives the physical sink "
                 "momentum p_f of every entry; the quarks are boosted toward physical +z, so +pz has the best overlap")
source_note = (f"spatial sources contracted: for every t_src, mean over the {n_spatial_src} positions x_src_list x y_src_list x z_src_list, weighted by "
               "exp(-i 2pi/Ls q.x_src) with q from the q_list dataset. This phase is ALREADY APPLIED: do not apply it again in the 3pt build")
time_reflection_note = ("C_ab(Lt - t) = s_a s_b C_ab(t) with s = +1 for G5, -1 for G5G4 (a = gamma_sink, b = gamma_source): "
                        "the G5-G5G4 elements are odd, include the sign when averaging forward and backward")
time_index_note = (f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {frontier_params.tsep_max - 1}; "
                   "index 0 is the source slice in both. dim_time gives the time after the roll, (t - t_src) % Lt, of every tsep; the other time slices are not saved")
transfer_note = ("P_f = P_i + q (physical momenta, equal to the labels): source weight exp(-i 2pi/Ls q.x_src), gluon operator "
                    "Fourier transformed with exp(+i 2pi/Ls q.z) (FF q_phase_sign = +1). The initial-state 2pt of a 3pt at sink "
                    "label pf and transfer q is at label pf - q.")


#======== path definitions ========
current_dir = "/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/2pt_production/streame_test"
quda_resource_path = "/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/2pt_production/.cache/quda"
gauge_file_prefix = "/lustre/orion/lgt132/world-shared/DATA/MILC/a09m310/gauge/l3296f211b630m0074m037m440e"     # stream e, same folder as stream d
