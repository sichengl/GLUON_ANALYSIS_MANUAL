#Clover-on-HISQ pion 2pt on a06m310 with point sources and point sinks, to measure the pion mass for the clover action of the setup
#(setup_valence_tuning_frontier_a06m310.py: c_SW, bare mass, HYP links as in the a09m310 production, 256 sources per cfg).
#Per point source at x0 = (x, y, z, t0), with S = the propagator from x0 (all 12 spin-color columns):
#    C(t) = sum_x sum_{spins, colors} |S(x, t; x0)|^2   (the gamma5 - gamma5 pion through gamma5-hermiticity, zero momentum)
#For each t_src the 16 spatial sources are averaged and the time axis is rolled so that index 0 is t_src: C[tau] = C(t_src + tau), tau = 0 .. 143.
#No gauge fixing (point sources need none). One file per cfg and t_src, written when that t_src is done, so a job killed at its time limit
#loses at most one t_src and the next job skips what is on disk: {pion_dir}/cfg<N>/pion_<pion_tag>_cfg<N>_t<t_idx>.npz
#Run: python3 -m mpi4py pion_point_frontier_a06m310.py --icfg <position in cfg_list> --n <number of cfgs>   (8 ranks, grid [1,1,1,8])
import os
import argparse
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, source
from time import perf_counter
from cupy.cuda.runtime import deviceSynchronize
from pt2_comm_tools import release_cupy_cache, log_gpu_memory    # tools/pt2_comm_tools.py
from setup_valence_tuning_frontier_a06m310 import tune_params, cfg_list, gauge_file_prefix, pion_dir, pion_tag, quda_resource_path, n_spatial_src

import sys, traceback
from mpi4py import MPI

#STOP ON ERROR: an uncaught error on any rank prints its traceback and kills all ranks of this run at once (as in the production scripts)
def abort_on_error(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = abort_on_error


parser = argparse.ArgumentParser()
parser.add_argument("--icfg", type=int, default=0)      # position in cfg_list of the first cfg of this run
parser.add_argument("--n", type=int, default=1)         # number of cfgs of this run
args = parser.parse_args()


#======== cfgs and sources of this run: the sources move with the position in cfg_list (setup: cfg_first 0, cfg_step 1) ========
positions = np.arange(args.icfg, min(args.icfg + args.n, len(cfg_list)))
measurement_list = cfg_list[positions]
t_src, x_src, y_src, z_src = tune_params.make_sources(positions)


#======== QUDA initialization ========
tune_params.init_quda(quda_resource_path)
latt_info = tune_params.make_latt_info()
if latt_info.mpi_rank == 0:
    print({"cfgs": measurement_list.tolist(), "mass": tune_params.mass, "c_SW": tune_params.clover_coeff_t, "tol": tune_params.tol,
           "multigrid": tune_params.multigrid, "t_src": t_src.tolist(), "x_src": x_src.tolist(), "y_src": y_src.tolist(), "z_src": z_src.tolist()})
dirac = tune_params.make_dirac(latt_info)


def part_name(cfg, t_idx):
    return f"{pion_dir}/cfg{cfg}/pion_{pion_tag}_cfg{cfg}_t{t_idx}.npz"

def all_on_disk(names):
    #rank 0 checks and tells the other ranks, so that every rank skips the same work (gatherLattice is collective)
    found = all(os.path.exists(name) for name in names) if latt_info.mpi_rank == 0 else None
    return latt_info.mpi_comm.bcast(found, root=0)


gauge = None
for i_cfg, cfg in enumerate(measurement_list):

    if all_on_disk([part_name(cfg, t_idx) for t_idx in range(len(t_src[i_cfg]))]):
        core.getLogger().info(f"SKIP CFG #{cfg}: all its t_src are saved")
        continue
    if latt_info.mpi_rank == 0:
        os.makedirs(f"{pion_dir}/cfg{cfg}", exist_ok=True)
    gauge = None
    release_cupy_cache()

    #READ GAUGE
    deviceSynchronize()
    s = perf_counter()
    gauge = io.readMILCGauge(f"{gauge_file_prefix}.{cfg}", checksum=True, reunitarize_sigma=1e-6)
    deviceSynchronize()
    core.getLogger().info(f"READ GAUGE #{cfg}: {perf_counter() - s} secs")

    #HYP, as in the production (no gauge fixing here)
    s = perf_counter()
    core.getLogger().info(f"plaq_hyp_before = {gauge.plaquette()}")
    gauge.hypSmear(tune_params.hyp_n_steps, tune_params.hyp_alpha1, tune_params.hyp_alpha2, tune_params.hyp_alpha3, tune_params.hyp_dir_ignore,
                   tune_params.hyp_compute_plaquette, tune_params.hyp_compute_qcharge)
    deviceSynchronize()
    plaq_hyp = [float(v) for v in gauge.plaquette()]
    core.getLogger().info(f"plaq_hyp_after = {plaq_hyp}")
    core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE (also builds the multigrid setup)
    release_cupy_cache()
    dirac.loadGauge(gauge)

    for t_idx, t0 in enumerate(t_src[i_cfg]):
        if all_on_disk([part_name(cfg, t_idx)]):
            core.getLogger().info(f"SKIP T_SRC {t0}: already saved")
            continue
        corr = cp.zeros(latt_info.Lt, "<c16")              # (Lt_local,): sum over the spatial sources of this t_src
        for x0 in x_src[i_cfg]:
            for y0 in y_src[i_cfg]:
                for z0 in z_src[i_cfg]:
                    deviceSynchronize()
                    s = perf_counter()
                    point_source = source.propagator(latt_info, "point", [int(x0), int(y0), int(z0), int(t0)])     # 12 columns, 1 at x0
                    release_cupy_cache()
                    prop = core.invertPropagator(dirac, point_source, mrhs=tune_params.mrhs)
                    corr += contract("wtzyxjiba,wtzyxjiba->t", prop.data.conj(), prop.data)     # sum over sites, spins and colors of |S|^2
                    point_source = prop = None
                    deviceSynchronize()
                    core.getLogger().info(f"SOURCE {[int(x0), int(y0), int(z0), int(t0)]}: {perf_counter() - s} secs")
        log_gpu_memory("AFTER T_SRC")

        #SAVE THIS T_SRC: all ranks gather (collective), rank 0 rolls, averages over the spatial sources and writes the file
        corr_all = core.gatherLattice(corr.get(), [0, -1, -1, -1])         # rank 0: (GLt,) in absolute time
        if latt_info.mpi_rank == 0:
            corr_all = np.roll(corr_all, -int(t0)) / n_spatial_src        # C[tau] = C(t0 + tau)
            name = part_name(cfg, t_idx)
            with open(f"{name}.tmp", "wb") as f:
                np.savez(f, corr=corr_all, cfg=cfg, t_src=int(t0), x_src=x_src[i_cfg], y_src=y_src[i_cfg], z_src=z_src[i_cfg],
                         mass=tune_params.mass, csw=tune_params.clover_coeff_t, tol=tune_params.tol, plaquette_hyp=np.array(plaq_hyp),
                         note=("corr[tau] = C(t_src + tau), C(t) = mean over the spatial sources of sum_x |S(x,t; x_src,t_src)|^2 summed over spins and "
                               "colors (gamma5-gamma5 pion, zero momentum, point source and sink, HYP-smeared clover action); "
                               "plaquette_hyp = (total, spatial, temporal) after HYP"))
            os.replace(f"{name}.tmp", name)
        core.getLogger().info(f"SAVE T_SRC {t0} (index {t_idx}) of cfg {cfg}")

    core.getLogger().info(f"END CFG #{cfg}")
