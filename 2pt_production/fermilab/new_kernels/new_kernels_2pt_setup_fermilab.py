#Fermilab test of the new single-quark smearing kernels of coulomb_smearing_kernels.py (the note pion_smearing_kernels.pdf, Secs. 1-7):
#14 operators, each one a kernel at source and sink (diagonal correlators only), pion only, G5, G5G4 and G3G5 at sink and source (3 x 3),
#q = 0 at the 99 sink momenta of pf_plain_list. 32 sources per cfg (4 t_src x 2 x x 2 y x 2 z) on the 40 stream-d cfgs 204, 234, ..., 1374:
#the same cfgs and sources as the Fermilab smearing tuning (smear_tuning/smear_tune_2pt_setup_fermilab.py, smear_tag coulomb_tune1), whose
#operator [4.5, 2.25, 0.8] is the reference G_ref here, so G_ref can be checked against that run cfg by cfg (quick_look_new_kernels.py --tune1).
import numpy as np
from pt2_comm_tools import TwoPtParams          # tools/pt2_comm_tools.py: the job script copies it into the run folder


#======== parameters of the 2pt run, kept in the class; those not set here keep the defaults of TwoPtParams ========
#(temporal_shift 5, spatial_shift 3, t_boundary -1, anisotropy 1.0, xi_0 1.0, mrhs 12, gauge_fix_dir 3, hyp_dir_ignore -1, --icfg 0)
fermilab_params = TwoPtParams()

#lattice and cfgs
fermilab_params.GLs = 32
fermilab_params.GLt = 96
fermilab_params.cfg_first = 204
fermilab_params.cfg_step = 6

#sources of the first cfg; they move by temporal_shift and spatial_shift from one cfg to the next (as in the smearing tuning)
fermilab_params.t_base = np.arange(0, fermilab_params.GLt, 24)    # 4 t_src
fermilab_params.x_base = np.arange(0, fermilab_params.GLs, 16)    # 2 x
fermilab_params.y_base = np.arange(0, fermilab_params.GLs, 16)    # 2 y
fermilab_params.z_base = np.arange(0, fermilab_params.GLs, 16)    # 2 z                      # 4 x 2 x 2 x 2 = 32 sources

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

#momenta in units of 2pi/L along z, interpolators
fermilab_params.mom_min = 0
fermilab_params.mom_max = 6
fermilab_params.gamma_ids = [15, 7, 11]               # Gamma(15) = G5, Gamma(7) = G5G4 = -G4G5, Gamma(11) = G3G5

#sink momenta: the 99 p_f of the production's plain case (px, py in 0, -1, 1 and pz in -2 .. 8)
fermilab_params.pz_list = list(range(fermilab_params.mom_min, fermilab_params.mom_max + 1))
fermilab_params.pf_plain_list = [[px, py, pz] for px in [0,-1,1] for py in [0,-1,1] for pz in range(fermilab_params.mom_min - 2, fermilab_params.mom_max + 3)]

#forward and backward time slices of the *_fb output (taken after the roll, so index 0 is t_src)
fermilab_params.tsep_max = 20
forward_index  = np.arange(fermilab_params.tsep_max)                              # 0, 1, 2, ..., 19:   C(t_src + tsep)
backward_index = (-np.arange(fermilab_params.tsep_max)) % fermilab_params.GLt     # 0, 95, 94, ..., 77: C(t_src - tsep)
fermilab_params.fb_index = {"pt2_forward": forward_index, "pt2_backward": backward_index}      # dataset name in the .h5 file: its time slices after the roll

#job: one cfg per job (array task)
fermilab_params.n_cfg = 1


#======== operators: the new kernels, each one deforming the reference Gaussian G_ref ========
#Reference: the best large-pz pion Gaussian of the Delta tuning, [rho_T, rho_z, mom_frac] = [4.5, 4.5 / 2, 0.8], quark boost k0 = 0.8 * 6 = 4.8.
#Units (coulomb_smearing_kernels.py): lengths in lattice units, amplitude convention exp(-r^2 / rho^2); momenta in units of 2pi/L along z.
#Every operator is the same kernel at source and sink; the job uses the kernel for the quark (prop2) and its conjugate (boost -k) for the
#antiquark (prop1), as coulomb_boosted_source(+k) / (-k) in the smearing tuning.
#Deformation strengths are tied to the reference: correlation length R_c = rho_T / sqrt(2) (then f_bar = 2/3, Eq. (18)), momentum shift
#delta_mu = 0.75 k0, splitting delta = 0.75 k0 = 3.6. Two coherent packets are resolved only for delta > sqrt(2)/rho_z = 3.2 units: at
#delta = 2.4 two_momentum would be 99.95% the Gaussian [4.5, 1.73, 0.8] (Eq. (14)), a narrower width and not sidebands.
#For scale: one step of the tuning grid ([4.5, 2.25] -> [5, 2.5] or [4, 2]) is 1 - overlap^2 = 0.017 - 0.021 away from G_ref; W, polynomial and
#two_scale are 1 - 2 such steps away, M_plus 2, two_momentum 10 (of which 1.5 steps are not a width change).
#The operators that change the envelope (two_scale, W, stretched, power_tail, polynomial) have their rms radii, transverse and longitudinal,
#matched to G_ref (3.18 and 1.125): the Delta tuning found the overlap set by the widths, so a change in signal here is a change of shape,
#not of size. M_plus and M_minus change only the phase (|K| = |K_G|). two_momentum, V, MV and beta keep G_env of G_ref, but their momentum
#factor narrows |K| along z (rms_z 0.749, 0.897, 0.897, 1.083 against 1.125): their comparison with G_ref is not at a fixed size, so
#two_momentum has the width control G_rz1.5 (rms_z 0.750). kernel_radii.py prints these numbers.
rho_T_ref, rho_z_ref, frac_ref = 4.5, 4.5 / 2, 0.8
k0 = frac_ref * fermilab_params.mom_max           # 4.8, the same number as k of [4.5, 2.25, 0.8] in the smearing tuning
R_c = 3.18                                        # rho_T_ref / sqrt(2)
P_z = fermilab_params.mom_max                     # pion momentum of the x-parametrized kernels (Eq. (28)): k = x P_z

operator_tag = "new_kernels_test1"
operator_list = [
    #name             function of coulomb_smearing_kernels  keyword arguments (the job adds xp = cupy and spatial_size)
    ["G_ref",         "gaussian_kernel",                    {"k0": k0, "rho_T": rho_T_ref, "rho_z": rho_z_ref}],                          # Eq. (6), the reference
    ["G_rz1.5",       "gaussian_kernel",                    {"k0": k0, "rho_T": rho_T_ref, "rho_z": 4.5 / 3}],                            # control of two_momentum: its rms_z; also
                                                                                                                                          #   [4.5, 1.5, 0.8] of the smearing tuning
    ["two_scale",     "two_scale_kernel",                   {"k0": k0, "A": 1.0, "rho_T1": 2.7, "rho_z1": 1.35,                          # Eq. (9): core 0.6 G_ref, tail 1.5 G_ref
                                                             "B": 0.2303, "rho_T2": 6.75, "rho_z2": 3.375}],                              #          B: rms matched
    ["two_momentum",  "two_momentum_kernel",                {"k0": k0, "rho_T": rho_T_ref, "rho_z": rho_z_ref, "delta": 3.6}],             # Eq. (11): resolved packets at 1.2 and 8.4
    ["M_plus",        "model_M_kernel",                     {"k0": k0, "rho_T": rho_T_ref, "rho_z": rho_z_ref, "delta_mu": 3.6, "R_c": R_c}],   # Eq. (16): larger k at small d_perp
    ["M_minus",       "model_M_kernel",                     {"k0": k0, "rho_T": rho_T_ref, "rho_z": rho_z_ref, "delta_mu": -3.6, "R_c": R_c}],  # control: the opposite correlation
    ["V",             "model_V_kernel",                     {"k0": k0, "rho_T": rho_T_ref, "rho_z": rho_z_ref, "delta0": 3.6, "R_c": R_c}],     # Eq. (20): vs two_momentum (same splitting)
    ["MV",            "model_MV_kernel",                    {"k0": k0, "rho_T": rho_T_ref, "rho_z": rho_z_ref, "delta_mu": 3.6, "R_c": R_c,
                                                             "delta0": 3.6, "R_g": R_c}],                                                 # Eq. (26): M_plus and V together
    ["W",             "model_W_kernel",                     {"k0": k0, "rho_T": 4.978, "rho_z0": 1.104, "alpha": 1.5, "R_c": R_c}],       # Eq. (23): 2.5 times longer along z at the center
                                                                                                                                          #   than outside, rms matched
    ["beta",          "beta_kernel",                        {"P_z": P_z, "rho_T": rho_T_ref, "rho_z": rho_z_ref, "a0": 4.2, "a1": 0.9,    # Eq. (30): <x> 0.85 at the center, 0.70 outside,
                                                             "b0": 1.8, "b1": -0.9, "R_c": R_c, "n_x": 200}],                             #   mean momentum 4.82 = k0. w(x | s) is narrow (std
                                                                                                                                          #   0.8 - 1.0 units against 2.26 of G_ref), so this acts
                                                                                                                                          #   as Model M with delta_mu 0.9 (a dose point between
                                                                                                                                          #   G_ref and M_plus), not as a new line shape
    ["gpd",           "gpd_kernel",                         {"P_z": P_z, "rho_T": rho_T_ref, "rho_z": rho_z_ref, "alpha_b": 11.5,         # Eq. (34), literal template: alpha' = 1 GeV^-2 at
                                                             "a_q": -0.5, "b_q": 2.0, "n_x": 200}],                                       #   a = 0.0582 fm; mean momentum only 1.7 (valence x),
                                                                                                                                          #   so it mostly tests a low boost
    ["stretched",     "stretched_kernel",                   {"k0": k0, "rho_T": 2.62, "rho_z": 1.686, "nu_T": 1.0, "nu_z": 1.0}],         # Eq. (35): exponential tails, rms matched
    ["power_tail",    "power_tail_kernel",                  {"k0": k0, "R_T": 5.958, "R_z": 2.979, "p": 3.0}],                            # Eq. (37): r^-6 tail, rms matched
    ["polynomial",    "polynomial_kernel",                  {"k0": k0, "rho_T": 3.112, "rho_z": 1.556, "c_T": 1.0, "c_z": 1.0}],          # Eq. (38): no node (factor >= 0.25), fatter tail,
                                                                                                                                          #   rms matched
]
assert len(operator_list) <= 20, "at most 20 operators: the time limit of the sbatch script assumes it"
assert len({row[0] for row in operator_list}) == len(operator_list), "operator_list has a repeated name"
assert operator_list[0][0] == "G_ref", "the first operator is the reference: quick_look_new_kernels.py compares the others with it"
operator_names = [row[0] for row in operator_list]


#======== cfgs and sources (not in the class) ========
cfg_list = np.arange(fermilab_params.cfg_first, fermilab_params.cfg_first + 40 * 5 * fermilab_params.cfg_step, 5 * fermilab_params.cfg_step)    # stream d 204, 234, ..., 1374
cfg_measure_spacing = 1
n_spatial_src = len(fermilab_params.x_base) * len(fermilab_params.y_base) * len(fermilab_params.z_base)


#======== notes written into the .h5 attributes ========
smearing_note = ("single-quark kernels of coulomb_smearing_kernels.py in Coulomb gauge, no gauge links: source eta(x) = K(x - x_src), sink "
                 "sum_y K(x - y) prop(y), d = x - y; K = N G_env(s, dz) exp(i mu(s) dz) chi(dz | s) (Eq. (40) of pion_smearing_kernels.pdf), "
                 "real envelope normalized to sum 1; lengths in lattice units, momenta in units of 2pi/L along z; prop2 (quark) smeared with K, "
                 "prop1 (antiquark, used conjugated) with K^* (all momenta reversed); the same K at source and sink (diagonal correlators only); "
                 "operator_list_json gives the function and parameters of every operator")
gauge_fix_note = (f"Coulomb gauge fixing of the unsmeared links by overrelaxation, fixingOVR(gauge_dir={fermilab_params.gauge_fix_dir}, "
                  f"Nsteps={fermilab_params.gauge_fix_Nsteps}, verbose_interval={fermilab_params.gauge_fix_verbose_interval}, "
                  f"relax_boost={fermilab_params.gauge_fix_relax_boost}, tolerance={fermilab_params.gauge_fix_tolerance}, "
                  f"reunit_interval={fermilab_params.gauge_fix_reunit_interval}, stopWtheta={fermilab_params.gauge_fix_stopWtheta}), before HYP")
momentum_note = ("sink phase exp(-2 pi i p.(x - x_src)/L) (conjugate of PyQUDA MomentumPhase): the momentum_list dataset gives the physical sink "
                 "momentum p_f of every entry; the quarks are boosted toward physical +z, so +pz has the best overlap")
source_note = (f"spatial sources averaged: for every t_src, plain mean over the {n_spatial_src} positions x_src_list x y_src_list x z_src_list "
               "(q = 0, no phase weight)")
time_reflection_note = ("C_ab(Lt - t) = s_a s_b C_ab(t) with s = +1 for G5, -1 for G5G4, +1 for G3G5 (a = gamma_sink, b = gamma_source): "
                        "the elements pairing G5G4 with G5 or G3G5 are odd, include the sign when averaging forward and backward")
time_index_note = (f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {fermilab_params.tsep_max - 1}; "
                   "index 0 is the source slice in both. dim_time gives the time after the roll, (t - t_src) % Lt, of every tsep; the other time slices are not saved")


#======== path definitions ========
current_dir = "/lustre2/gluonp0/sliu1/2pt_production/new_kernels"
quda_resource_path = "/lustre2/gluonp0/sliu1/.cache/quda"
gauge_file_prefix = "/lustre2/gluonp0/MILC/l3296f211b630m0074m037m440d/l3296f211b630m0074m037m440d"
