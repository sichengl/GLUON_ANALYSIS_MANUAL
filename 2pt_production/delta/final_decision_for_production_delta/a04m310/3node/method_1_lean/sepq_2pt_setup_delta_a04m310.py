#NODE TEST setup for a04m310 (l64192f211b700m00316m0158m188, 64^3 x 192) on 3 Delta gpuA100x8 node(s), 24 GPUs along t (8 time slices each), folder a04m310/3node/method_1_lean.
#Copy of a09m310/delta_2pt_production_a09m310_stream_e_direction_x/sepq_2pt_setup_delta_final_a09m310_streame.py (md5 fd51fd80): the same 2pt
#(pion + proton, sepq kinematics, two smearings, three gammas, 99 sink momenta); every changed value is marked NODE TEST. Choices still to be
#confirmed for the production: the smearing widths (a09's in fm, scale 2 = a09/a04 (0.0871/~0.042)), tsep_max (a09's 20 in fm), the source grid
#(this test: 24 sources per cfg; production: 3072 on a09's spacing in fm) and, for a04, the mass and c_SW.
import os
import numpy as np
from pt2_comm_tools import TwoPtParams          # tools/pt2_comm_tools.py: the job script copies it into the run folder


#======== parameters of the 2pt run, kept in the class; those not set here keep the defaults of TwoPtParams ========
#(temporal_shift 5, spatial_shift 3, t_boundary -1, anisotropy 1.0, xi_0 1.0, mrhs 12, gauge_fix_dir 3, hyp_dir_ignore -1, --icfg 0, --n 20)
delta_params = TwoPtParams()

#lattice and cfgs
delta_params.GLs = 64                                     # NODE TEST
delta_params.GLt = 192                                    # NODE TEST
delta_params.cfg_first = 1002                              # NODE TEST: the first cfg of the Delta copy
delta_params.cfg_step = 6

#sources of the first cfg; they move by temporal_shift and spatial_shift from one cfg to the next
delta_params.t_base = np.arange(0, delta_params.GLt, 64)     # 3 t_src (NODE TEST: test statistics; production 24)
delta_params.x_base = np.arange(0, delta_params.GLs, 32)     # 2 x
delta_params.y_base = np.arange(0, delta_params.GLs, 32)     # 2 y
delta_params.z_base = np.arange(0, delta_params.GLs, 32)     # 2 z                      # 3 x 2 x 2 x 2 = 24 sources (NODE TEST: test statistics; production 3072)

#QUDA: grid of 24 GPUs along t (NODE TEST), clover Wilson Dirac operator, multigrid solver
delta_params.grid_size = [1, 1, 1, 24]                # NODE TEST: 3 node(s) x 8 GPUs
delta_params.mass = -0.022                          # NODE TEST: TEST VALUES until the a04 tuning (Sergey's middle trial mass; c_SW guess until plaquette_csw_delta_a04m310)
delta_params.tol = 1e-8
delta_params.maxiter = 1000
delta_params.clover_coeff_t = 1.029
delta_params.clover_coeff_r = 1.029
delta_params.multigrid = [[4, 4, 4, 2], [2, 2, 2, 2]]        # NODE TEST: local t 8 -> 4 -> 2, x y z 64 -> 16 -> 8: every coarse extent even
delta_params.mrhs = 1                                    # NODE TEST: columns solved together (default 12); each one costs QUDA and PyQUDA workspace

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
delta_params.nt_chunk = 1                                # NODE TEST

#smearings [rho_T, rho_z, mom_frac], each with its own quark boost k = mom_frac * mom_max along z; interpolators
delta_params.smear_list = [[9.0, 4.5, 0.8],       # boosted, for pz >= 2   (NODE TEST: a09's [4.5, 2.25, 0.8] in fm)
                           [3.0, 3.0, 0.0]]        # unboosted, for pz 0-1 (NODE TEST: a09's [1.5, 1.5, 0] in fm)
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
delta_params.tsep_max = 42                                # NODE TEST: a09's 20 in fm
forward_index  = np.arange(delta_params.tsep_max)                           # 0, 1, 2, ..., 41:   C(t_src + tsep)
backward_index = (-np.arange(delta_params.tsep_max)) % delta_params.GLt     # 0, 191, 190, ..., 151: C(t_src - tsep)
delta_params.fb_index = {"pt2_forward": forward_index, "pt2_backward": backward_index}      # dataset name in the .h5 file: its time slices after the roll


#======== cfgs, smearing operators and momentum tables (not in the class) ========
#NODE TEST: the cfgs are the complete gauge files <ensemble>.<N> in the Delta gauge folder with N >= cfg_first (a file still being copied stops here)
ensemble = "l64192f211b700m00316m0158m188"
gauge_dir = "/work/hdd/biqz/MILC/l64192f211b700m00316m0158m188"                      # copied from Frontier with Globus
gauge_file_bytes = delta_params.GLs**3 * delta_params.GLt * 4 * 18 * 4 + 96     # MILC single precision links + 96-byte header
assert os.path.isdir(gauge_dir), f"gauge folder {gauge_dir} does not exist"
found = {}
for name in os.listdir(gauge_dir):
    head, dot, tail = name.rpartition(".")
    if head == ensemble and tail.isdigit():
        found[int(tail)] = os.path.getsize(os.path.join(gauge_dir, name))
not_complete = sorted((cfg, size) for cfg, size in found.items() if size != gauge_file_bytes)
assert not not_complete, f"{len(not_complete)} gauge files are not {gauge_file_bytes} bytes (still being copied?), e.g. (cfg, bytes) {not_complete[:5]}"
cfg_list = np.array([cfg for cfg in sorted(found) if cfg >= delta_params.cfg_first])
assert len(cfg_list) > 0, f"no gauge file {ensemble}.<cfg> with cfg >= {delta_params.cfg_first} in {gauge_dir}"
cfg_measure_spacing = 1
src_phase_sign = -1
n_spatial_src = len(delta_params.x_base) * len(delta_params.y_base) * len(delta_params.z_base)
smear_names = ["rho9z4p5", "rho3"]           # one name per row of delta_params.smear_list
frac_str = [("%.3f" % frac).rstrip("0").rstrip(".").replace(".", "p") for rho_T, rho_z, frac in delta_params.smear_list]
smear_tag = f"a04m310_nodetest_3node_method_1_lean_coulomb_{'-'.join(smear_names)}_frac{'-'.join(frac_str)}_smear"      # a04m310_nodetest_3node_method_1_lean_: the ensemble, the node count and the method (NODE TEST), in every folder and file name (the job script adds _direction_x); _smear: one smear axis per end (the _paired files of the earlier runs have separate shape and frac axes)

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
                 "Gaussian part normalized to sum 1; source and sink smearings each run over the rows of smear_list, [rho_T, rho_z, mom_frac] with "
                 "quark boost k = mom_frac * mom_max along z: [9.0, 4.5, 0.8] (k = 4.8) and [3.0, 3.0, 0] (k = 0)")
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
current_dir = "/work/hdd/biqz/sliu54/2pt_production/a04m310_node_test/3node_method_1_lean"      # NODE TEST: one folder per node count and method
quda_resource_path = "/u/sliu54/QUDA_resources/.cache/quda_a04m310_24gpu_3node_method_1_lean"      # NODE TEST: its own tuning cache (lattice, grid and method)
gauge_file_prefix = f"{gauge_dir}/{ensemble}"     # gauge file of cfg N: {gauge_file_prefix}.{N}
