import os
import argparse
import numpy as np
import cupy as cp
from pyquda_utils import core, io, gamma
from pyquda.field import LatticePropagator
from time import perf_counter
from cupy.cuda.runtime import deviceSynchronize
from coulomb_smearing import coulomb_gauge_theta                                                  # 2pt_production/tools/coulomb_smearing.py
from pt2_comm_tools import release_cupy_cache, log_gpu_memory                                     # 2pt_production/tools/pt2_comm_tools.py
from opt_kernel import kernel_from_table, kernel_ft_with_derivatives                              # tools_smearing_opt/opt_kernel.py
from opt_contract import time_slices, source_lexico_from_kernel, pair_weights, resolved_pion      # tools_smearing_opt/opt_contract.py
from opt_check import theta_name, corr_name, read_json, wait_for_files, write_start, check_step, lost_name, mark_lost   # tools_smearing_opt/opt_check.py
from opt_setup_frontier_test import *  # delta_params, cfg list, starting table and every opt_ setting, paths (the only line that differs from ../frontier_production/opt_worker_frontier.py)

import sys, traceback
from mpi4py import MPI

#STOP ON ERROR: an uncaught error on any rank prints its traceback and kills all ranks of this run at once, so the job ends as FAILED instead of
#hanging until walltime (the same lines as in the production script)
def abort_on_error(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)
sys.excepthook = abort_on_error

#FRONTIER copy of ../delta_test/opt_worker_delta.py (8 GCDs per node, grid [1,1,1,8], many nodes). It differs in three places, each marked #LOST NODE
#or by the setup import: a node whose gauge fixing fails marks itself as lost and ends (instead of an error that would stop nothing but itself),
#and a waiting node ends when node 0 has marked it as lost. Everything else is the same.
#Smearing parameter optimization, the part that runs on ONE node (the GPUs of the grid in the setup) for ONE cfg and stays alive for the whole run:
#  once:        read the gauge field, Coulomb gauge fixing, HYP, multigrid setup (the lines of the production script)
#  every step:  wait for the kernel table of the step, invert the 16 sources smeared with it (one prop1/prop2 pair per source),
#               contract in momentum space (the gamma x gamma correlator matrix and its derivative for every kernel parameter from the same contraction),
#               write the small result file, wait for the next table.
#Node 0 (--worker 0) also does the check: when all nodes have written their file it measures, decides and writes the next table
#(tools_smearing_opt/opt_check.py), and prints the status of every step in its job output and in opt_dir/status.log.
#The nodes talk only through small files in opt_dir (theta/, corr/, STOP).

parser = argparse.ArgumentParser()
parser.add_argument("--worker", type=int, default=0)      # which node this is: it holds cfg_list[opt_cfg_index[worker]]
args = parser.parse_args()


#======== configuration and sources of this node ========
worker = args.worker
cfg = cfg_list[opt_cfg_index[worker]]
t_src, x_src, y_src, z_src = delta_params.make_sources([cfg])
source_list = [[x0, y0, z0, t0] for t0 in t_src[0] for x0 in x_src[0] for y0 in y_src[0] for z0 in z_src[0]]      # the same sources at every step


#LOST NODE: a node that was marked as lost in an earlier job of this run stays out (every rank sees the same file, so all ranks end here)
if os.path.exists(lost_name(opt_dir, worker)):
    print(f"WORKER {worker} (cfg {cfg}) ENDS: marked as lost in {lost_name(opt_dir, worker)}", flush=True)
    sys.exit(0)


#======== QUDA initialization ========
delta_params.init_quda(quda_resource_path)
latt_info = delta_params.make_latt_info()
dirac = delta_params.make_dirac(latt_info)
comm, rank = latt_info.mpi_comm, latt_info.mpi_rank
if list(latt_info.grid_size[:3]) != [1, 1, 1]:
    raise RuntimeError(f"the 3D FFTs need an unsplit spatial grid, got grid {latt_info.grid_size}")
if rank == 0:
    print({"worker": worker, "cfg": int(cfg), "source_list": np.array(source_list).tolist(), "opt_dir": opt_dir}, flush=True)


#======== gamma matrices (cupy arrays: after core.init), as in the production contraction ========
G4, G5 = gamma.gamma(8), gamma.gamma(15)
gamma_list = [gamma.gamma(n) for n in opt_gamma_ids]                                      # the interpolators Gamma(n), n in opt_gamma_ids
G5G_sink_list = [G5 @ gamma_sink for gamma_sink in gamma_list]                            # G5 Gamma_sink
Gbar_G5_source_list = [(G4 @ gamma_source.conj().T @ G4) @ G5 for gamma_source in gamma_list]     # gamma_source_bar G5
n_gamma = len(gamma_list)


#======== ONCE: gauge field, Coulomb gauge fixing, HYP, multigrid setup (the lines of the production script) ========
#READ GAUGE
deviceSynchronize()
s = perf_counter()
gauge_file = f"{gauge_file_prefix}.{cfg}"
gauge = io.readMILCGauge(gauge_file,checksum=True, reunitarize_sigma=1e-6)
deviceSynchronize()
core.getLogger().info(f"READ GAUGE #{cfg}: {perf_counter() - s} secs")

#COULOMB GAUGE FIXING of the unsmeared links
deviceSynchronize()
s = perf_counter()
gauge.fixingOVR(gauge_dir=delta_params.gauge_fix_dir, Nsteps=delta_params.gauge_fix_Nsteps, verbose_interval=delta_params.gauge_fix_verbose_interval,
                relax_boost=delta_params.gauge_fix_relax_boost, tolerance=delta_params.gauge_fix_tolerance,
                reunit_interval=delta_params.gauge_fix_reunit_interval, stopWtheta=delta_params.gauge_fix_stopWtheta)     # overrelaxation, since QUDA's FFT method runs on one GPU only
theta = coulomb_gauge_theta(latt_info, gauge)
deviceSynchronize()
core.getLogger().info(f"COULOMB GAUGE FIXING: {perf_counter() - s} secs, theta {theta:.3e}")
if not theta <= 2e-6:                           # QUDA stops silently after Nsteps
    #LOST NODE: this cfg cannot be used. The node marks itself as lost, so that node 0 does not wait for it, and ends (theta is the same on every rank)
    if rank == 0:
        mark_lost(opt_dir, worker, f"cfg {cfg}: Coulomb gauge fixing stopped at theta {theta:.3e} after {delta_params.gauge_fix_Nsteps} steps")
    core.getLogger().info(f"WORKER {worker} (cfg {cfg}) ENDS: Coulomb gauge fixing stopped at theta {theta:.3e}, node marked as lost")
    sys.exit(0)

#HYP on the gauge-fixed links: HYP is gauge covariant, so the propagators live in the same Coulomb gauge as the smearing
deviceSynchronize()
s = perf_counter()
gauge_hyp = gauge.copy()
del gauge
gauge_hyp.hypSmear(delta_params.hyp_n_steps, delta_params.hyp_alpha1, delta_params.hyp_alpha2, delta_params.hyp_alpha3, delta_params.hyp_dir_ignore,
                   delta_params.hyp_compute_plaquette, delta_params.hyp_compute_qcharge)
deviceSynchronize()
core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

#LOAD GAUGE (also builds the multigrid setup): kept for all steps
release_cupy_cache()
dirac.loadGauge(gauge_hyp)


#======== first step of this node: 0 for a new run; after a restart, the first step whose result file this node has not written ========
step = 0
if rank == 0:
    os.makedirs(f"{opt_dir}/theta", exist_ok=True)
    if worker == 0:
        write_start(opt_dir, opt_settings)                        # node 0 writes the starting table as step 0
    while os.path.exists(theta_name(opt_dir, step)) and os.path.exists(corr_name(opt_dir, step, worker)):
        step += 1
step = comm.bcast(step, root=0)

spatial_size = latt_info.size[:3]                                 # [Lx, Ly, Lz]
n_w = 4 * len(opt_start_table)                                    # the correlator itself + one derivative per parameter (4 per Gaussian - 1 fixed weight)
n_done = 0                                                        # steps this node has done since it started

while True:

    #WAIT for the kernel table of this step. Rank 0 looks at the disk and tells the other ranks. On node 0 the table comes from its own
    #check of the previous step, which first waits for the result files of all nodes.
    #The first two waits of a node can be long (the other nodes may still be in the queue): opt_start_wait_minutes; later ones opt_wait_minutes.
    table = None
    if rank == 0:
        wait_minutes = opt_start_wait_minutes if n_done < 2 else opt_wait_minutes
        go_on = True
        if worker == 0 and not os.path.exists(theta_name(opt_dir, step)):
            go_on = check_step(opt_dir, step - 1, {**opt_settings, "wait_minutes": wait_minutes})
        if go_on and wait_for_files([theta_name(opt_dir, step)], opt_dir, wait_minutes, opt_poll_seconds, also_stop_at=lost_name(opt_dir, worker)):      #LOST NODE: also ends when node 0 has marked this node as lost
            table = np.array(read_json(theta_name(opt_dir, step))["table"], float)
    table = comm.bcast(table, root=0)
    if table is None:                                             # the optimization has stopped (STOP file), or this node was marked as lost
        break

    deviceSynchronize()
    step_started = perf_counter()
    core.getLogger().info(f"STEP {step} STARTS")

    #KERNELS of this step: sum of boosted Gaussians, boosts -k_j for prop1 and +k_j for prop2 as in production, and the momentum-space
    #weights of the sink: W[pz, 0] for the correlator, W[pz, 1 + i] for its derivative with respect to parameter i
    kernel1 = kernel_from_table(cp, spatial_size, table, delta_params.mom_max, -1)
    kernel2 = kernel_from_table(cp, spatial_size, table, delta_params.mom_max, +1)
    kft1, dkft1 = kernel_ft_with_derivatives(cp, spatial_size, table, delta_params.mom_max, -1, opt_kernel_h)
    kft2, dkft2 = kernel_ft_with_derivatives(cp, spatial_size, table, delta_params.mom_max, +1, opt_kernel_h)
    W = pair_weights(cp, kft1, kft2, dkft1, dkft2, opt_pz_list)
    del kft1, dkft1, kft2, dkft2

    corr = np.zeros((len(source_list), 2, n_gamma, n_gamma, len(opt_pz_list), n_w, opt_t_max + 1), "<c16")      # [src, fb, gamma_sink, gamma_source, pz, w, tau]: this rank fills the time slices it holds

    for i_src, src_pos in enumerate(source_list):

        deviceSynchronize()
        inner_loop = perf_counter()
        core.getLogger().info(f"SOURCE POSITION = {src_pos}")

        #SRC: the kernel placed at the source, -k for prop1 and +k for prop2
        #prop1 is conjugated, and will be used for anti-quark
        #prop2 is not conjugated, and will be used for quark
        prop1_inv = prop2_inv = S1 = S2 = None      # drop the previous source's propagators before allocating new ones
        prop1_inv = LatticePropagator(latt_info, latt_info.evenodd(source_lexico_from_kernel(cp, latt_info.size, latt_info.gt * latt_info.Lt, kernel1, src_pos), False, backend="cupy"))
        prop2_inv = LatticePropagator(latt_info, latt_info.evenodd(source_lexico_from_kernel(cp, latt_info.size, latt_info.gt * latt_info.Lt, kernel2, src_pos), False, backend="cupy"))

        #INVERT
        deviceSynchronize()
        release_cupy_cache()
        s = perf_counter()
        prop1_inv = core.invertPropagator(dirac, prop1_inv, mrhs=delta_params.mrhs)
        prop2_inv = core.invertPropagator(dirac, prop2_inv, mrhs=delta_params.mrhs)
        deviceSynchronize()
        core.getLogger().info(f"INVERT 2 propagagors: {perf_counter() - s} secs")

        #CONTRACT in momentum space, only on the time slices t_src + tau and t_src - tau (tau = 0 .. opt_t_max) that this rank holds
        s = perf_counter()
        t_local = time_slices(src_pos[3], delta_params.GLt, opt_t_max) - latt_info.gt * latt_info.Lt        # (2, opt_t_max + 1): [fb, tau]
        fb_mine, tau_mine = np.nonzero((t_local >= 0) & (t_local < latt_info.Lt))
        if len(fb_mine) > 0:
            S1 = latt_info.lexico(prop1_inv.data, False, backend="cupy")      # [t, z, y, x, spin, spin, color, color], unsmeared
            S2 = latt_info.lexico(prop2_inv.data, False, backend="cupy")
            for c in range(0, len(fb_mine), delta_params.nt_chunk):
                fb_chunk, tau_chunk = fb_mine[c:c + delta_params.nt_chunk], tau_mine[c:c + delta_params.nt_chunk]
                t_chunk = cp.asarray(t_local[fb_chunk, tau_chunk])
                out = cp.asnumpy(resolved_pion(cp, S1[t_chunk], S2[t_chunk], G5G_sink_list, Gbar_G5_source_list, W, opt_pz_list))       # [gamma_sink, gamma_source, pz, w, time slice of the chunk]
                for n in range(len(fb_chunk)):
                    corr[i_src, fb_chunk[n], :, :, :, :, tau_chunk[n]] = out[:, :, :, :, n]
        #the momentum-space sum is relative to the origin: this phase makes it relative to the source, exp(-2 pi i P.(x - x_src)/L) as in production
        corr[i_src] *= np.exp(2j * np.pi * np.array(opt_pz_list) * src_pos[2] / delta_params.GLs)[None, None, None, :, None, None]
        deviceSynchronize()
        core.getLogger().info(f"CONTRACT: {perf_counter() - s} secs")
        core.getLogger().info(f"UNTIL CONTRACTION: {perf_counter() - inner_loop} secs")

    prop1_inv = prop2_inv = S1 = S2 = W = None
    log_gpu_memory(f"END STEP {step}")

    #SAVE: the ranks add up their time slices; rank 0 writes the result file of this node (as .tmp, renamed when complete)
    corr = comm.allreduce(corr)
    if rank == 0:
        name = corr_name(opt_dir, step, worker)
        os.makedirs(os.path.dirname(name), exist_ok=True)
        with open(f"{name}.tmp", "wb") as f:
            np.savez(f, C=corr, table=table, step=step, cfg=cfg, source_list=np.array(source_list), pz_list=np.array(opt_pz_list), gamma_ids=np.array(opt_gamma_ids))
            f.flush(); os.fsync(f.fileno())
        os.replace(f"{name}.tmp", name)
    core.getLogger().info(f"STEP {step} DONE: {perf_counter() - step_started} secs for {len(source_list)} sources")
    step += 1
    n_done += 1

dirac.freeGauge()
core.getLogger().info(f"WORKER {worker} (cfg {cfg}) ENDS at step {step}")
