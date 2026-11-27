#Valence tuning on a04m310 (MILC l64192f211b700m00316m0158m188, 64^3 x 192) on Delta, one gpuA100x8 node (8 A100-40GB, NVLink) per run, step 1: the plaquette of the
#HYP-smeared links and the tadpole clover coefficient c_SW = P^(-3/4) (plaquette_csw_delta_a04m310.py).
#Delta copy of frontier/final_production_frontier/8gpu/a04m310/frontier_valence_tuning_a04m310/setup_valence_tuning_frontier_a04m310.py:
#only the GPU grid (and its comment) and the paths differ. The bare light mass comes after c_SW is fixed (Sergey's first guesses
#-0.020, -0.022, -0.024, then a fit of M_pi^2 = A (m - B)); no published values exist for this ensemble.
import os
import numpy as np
from pt2_comm_tools import TwoPtParams          # tools/pt2_comm_tools.py: the job script copies it into the run folder


#======== parameters, kept in the class; those not set here keep the defaults of TwoPtParams ========
tune_params = TwoPtParams()

#lattice
tune_params.GLs = 64
tune_params.GLt = 192

#QUDA: 8 GPUs along t (24 time slices each); no Dirac operator in step 1
tune_params.grid_size = [1, 1, 1, 8]
tune_params.mass = None                                     # to be tuned after c_SW
tune_params.clover_coeff_t = None                           # the result of step 1
tune_params.clover_coeff_r = None

#HYP smearing of the links in the Dirac operator: the a09m310 production values (c_SW must come from the same smearing)
tune_params.hyp_n_steps = 1
tune_params.hyp_alpha1 = 0.75
tune_params.hyp_alpha2 = 0.6
tune_params.hyp_alpha3 = 0.3
tune_params.hyp_dir_ignore = -1                             # all four directions smeared
tune_params.hyp_compute_plaquette = True
tune_params.hyp_compute_qcharge = False


#======== paths ========
ensemble = "l64192f211b700m00316m0158m188"
ens_tag = "a04m310"                                         # in every folder and file name of the output
gauge_dir = "/work/hdd/biqz/MILC/l64192f211b700m00316m0158m188"                      # copied from Frontier with Globus (10 cfgs, 1002-1056)
gauge_file_prefix = f"{gauge_dir}/{ensemble}"               # gauge file of cfg N: {gauge_file_prefix}.{N}
out_dir = f"/work/hdd/biqz/sliu54/2pt_production/{ens_tag}_valence_tuning"
quda_resource_path = "/u/sliu54/QUDA_resources/.cache/quda_a04m310_8gpu"     # its own tuning cache: other lattice


#======== cfgs: the gauge files in gauge_dir, by cfg number; the first n_cfg_use of them are measured ========
#A MILC file holds GLs^3 x GLt sites x 4 links x 18 reals x 4 bytes (single precision) and a 96-byte header. Every file of the ensemble in gauge_dir must
#have this size: a file still being copied stops the run here, so every rank sees the same list.
n_cfg_use = 10
gauge_file_bytes = tune_params.GLs**3 * tune_params.GLt * 4 * 18 * 4 + 96      # 14,495,514,720
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
