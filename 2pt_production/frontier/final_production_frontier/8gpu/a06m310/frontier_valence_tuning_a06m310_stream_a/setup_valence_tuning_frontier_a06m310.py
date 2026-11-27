#Valence tuning on a06m310 (MILC l48144f211b672m0048m024m286a, stream a, 48^3 x 144) on Frontier, 8 GCDs per run, for two scripts:
#  plaquette_csw_frontier_a06m310.py  the plaquette of the HYP-smeared links and the tadpole clover coefficient c_SW = P^(-3/4)
#  pion_point_frontier_a06m310.py     the clover-on-HISQ pion 2pt with point sources, to measure the pion mass
#The clover action is the one of the a09m310 production (HYP-smeared links, one step 0.75/0.6/0.3) with the a06m310 values of PNDME
#(arXiv:1606.07049, Table 2) and of Sergey's table: c_SW = 1.03493, bare light mass -0.0398. PNDME's pion for these values: 319.6 MeV at a = 0.0582 fm,
#a M_pi = 0.0943.
import os
import numpy as np
from pt2_comm_tools import TwoPtParams          # tools/pt2_comm_tools.py: the job script copies it into the run folder


#======== parameters, kept in the class; those not set here keep the defaults of TwoPtParams ========
tune_params = TwoPtParams()

#lattice
tune_params.GLs = 48
tune_params.GLt = 144

#cfgs: cfg_list (below) is read from the gauge folder, so the sources move with the POSITION of a cfg in cfg_list, not with its number:
#cfg_first = 0 and cfg_step = 1 make make_sources take the position (the scripts pass positions 0, 1, 2, ...)
tune_params.cfg_first = 0
tune_params.cfg_step = 1

#point sources of the first cfg: 16 t_src x 2 x x 2 y x 4 z = 256 per cfg; they move by temporal_shift and spatial_shift from one cfg to the next
tune_params.t_base = np.arange(0, tune_params.GLt, 9)      # 16 t_src, 9 apart (about 0.52 fm)
tune_params.x_base = np.arange(0, tune_params.GLs, 24)     # 2 x
tune_params.y_base = np.arange(0, tune_params.GLs, 24)     # 2 y
tune_params.z_base = np.arange(0, tune_params.GLs, 12)     # 4 z
tune_params.temporal_shift = 5                              # coprime to 144
tune_params.spatial_shift = 5                               # coprime to 48 (the default 3 is not: make_sources checks it)

#QUDA: 8 GCDs along t (18 time slices each), clover Wilson Dirac operator, multigrid solver
tune_params.grid_size = [1, 1, 1, 8]
tune_params.mass = -0.0398
tune_params.tol = 1e-8
tune_params.maxiter = 1000
tune_params.clover_coeff_t = 1.03493
tune_params.clover_coeff_r = 1.03493
tune_params.multigrid = [[4, 4, 4, 3], [2, 2, 2, 3]]        # local 48^3 x 18 -> coarse 12^3 x 6 -> 6^3 x 2: every coarse extent even (the a09 blocks would give 18/4)
tune_params.mrhs = 12

#HYP smearing of the links in the Dirac operator: the a09m310 production values
tune_params.hyp_n_steps = 1
tune_params.hyp_alpha1 = 0.75
tune_params.hyp_alpha2 = 0.6
tune_params.hyp_alpha3 = 0.3
tune_params.hyp_dir_ignore = -1                             # all four directions smeared
tune_params.hyp_compute_plaquette = True
tune_params.hyp_compute_qcharge = False

n_spatial_src = len(tune_params.x_base) * len(tune_params.y_base) * len(tune_params.z_base)


#======== paths ========
ensemble = "l48144f211b672m0048m024m286a"
ens_tag = "a06m310_streama"                                 # in every folder and file name of the output
gauge_dir = "/lustre/orion/lgt132/world-shared/DATA/MILC/a06m310/l48144f211b672m0048m024m286a"
gauge_file_prefix = f"{gauge_dir}/{ensemble}"               # gauge file of cfg N: {gauge_file_prefix}.{N}
out_dir = f"/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/2pt_production/{ens_tag}_valence_tuning"
pion_tag = f"{ens_tag}_point_m{tune_params.mass}_csw{tune_params.clover_coeff_t}"     # a06m310_streama_point_m-0.0398_csw1.03493
pion_dir = f"{out_dir}/pion_{pion_tag}"                     # one folder per mass and c_SW: cfg<N>/pion_<pion_tag>_cfg<N>_t<t_idx>.npz
quda_resource_path = "/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/2pt_production/.cache/quda_a06m310_8gcd"     # its own tuning cache: other lattice


#======== cfgs: the gauge files in gauge_dir, by cfg number; the first n_cfg_use of them are measured ========
#A MILC file holds GLs^3 x GLt sites x 4 links x 18 reals x 4 bytes (single precision) and a 96-byte header. Every file of the ensemble in gauge_dir must
#have this size: a file still being copied stops the run here, so every rank of every run sees the same list.
n_cfg_use = 50
gauge_file_bytes = tune_params.GLs**3 * tune_params.GLt * 4 * 18 * 4 + 96      # 4,586,471,520
assert os.path.isdir(gauge_dir), f"gauge folder {gauge_dir} does not exist (not copied yet?)"
found = {}
for name in os.listdir(gauge_dir):
    head, dot, tail = name.rpartition(".")
    if head == ensemble and tail.isdigit():
        found[int(tail)] = os.path.getsize(os.path.join(gauge_dir, name))
assert found, f"no gauge file {ensemble}.<cfg> in {gauge_dir}"
not_complete = sorted((cfg, size) for cfg, size in found.items() if size != gauge_file_bytes)
assert not not_complete, f"{len(not_complete)} gauge files are not {gauge_file_bytes} bytes (still being copied?), e.g. (cfg, bytes) {not_complete[:5]}"
assert len(found) >= n_cfg_use, f"{len(found)} gauge files in {gauge_dir}, {n_cfg_use} needed"
cfg_list = np.array(sorted(found)[:n_cfg_use])
