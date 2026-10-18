import os
import shutil
import argparse
import h5py
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, phase, gamma
from time import perf_counter
from cupy.cuda.runtime import deviceSynchronize
from tqdm import tqdm
from coulomb_smearing import sink_smear_evenodd, coulomb_gauge_theta
from pt2_comm_tools import make_eps, release_cupy_cache, log_gpu_memory # tools/pt2_comm_tools.py
from frontier_2pt_tools import use_run_communicator, invert_coulomb_boosted_source
from sepq_2pt_setup_frontier_final import *  # frontier_params (the TwoPtParams of this run), cfg list, smearing operators, momentum tables, notes, paths

#Final run on Frontier: the Delta final script (delta/final_decision_for_production_delta/2pt_coulomb_boosted_sepq_3072srcs_delta_final.py) with the
#setup sepq_2pt_setup_frontier_final.py, the same physics and output files. The changes fit the job on 1 GCD or the 2 GCDs of one MI250X:
#  1. INVERT: the source columns are built from the smearing profile, mrhs at a time, and the solutions wait in host memory while QUDA solves
#     (invert_coulomb_boosted_source replaces coulomb_boosted_source + core.invertPropagator); mrhs is set per layout (--mrhs).
#  2. SINK SMEARING: done on nt_chunk time slices at a time inside the contraction, so the smeared propagators are never stored whole. The chunk
#     loop now runs outside the sink-gamma loop, so that each chunk is smeared once.
#  3. A cfg is cut into --nparts blocks of t_src, one per run (as in the stream-e test), so that it fits in Frontier's 12 h jobs; the run that
#     saves the last t_src of a cfg merges it. One srun can hold many runs (use_run_communicator).


parser = argparse.ArgumentParser()
parser.add_argument("--icfg", type=int, default=frontier_params.icfg)      # starting position in cfg_list, e.g., icfg=100 means we start with the 100th cfg in the list, namely, 204+6*100=804
parser.add_argument("--n", type=int, default=frontier_params.n_cfg)        # number of configs to measure in this one job (per run)
parser.add_argument("--gcds", type=int, default=1)                         # GCDs per run: grid [1, 1, 1, gcds]
parser.add_argument("--mrhs", type=int, default=frontier_params.mrhs)      # right-hand sides QUDA solves together (its memory grows with mrhs)
parser.add_argument("--nparts", type=int, default=1)                       # the t_src of every cfg are cut into nparts consecutive blocks, one per run
parser.add_argument("--part", type=int, default=0)                         # block of the first run of this srun: run r of the srun takes block part + r
args = parser.parse_args()


#======== configurations and sources of this run ========
#run r of this srun takes block b = part + r: block b % nparts of the t_src of cfg_list[icfg + b // nparts], then of every (runs / nparts)-th cfg after it
run_id, n_runs = use_run_communicator(args.gcds)
block = args.part + run_id
icfg0 = args.icfg + block // args.nparts
n = args.n
measurement_list = cfg_list[icfg0::max(1, n_runs // args.nparts) * cfg_measure_spacing][:n]
t_src, x_src, y_src, z_src = frontier_params.make_sources(measurement_list)
t_idx_mine = np.array_split(np.arange(len(frontier_params.t_base)), args.nparts)[block % args.nparts]


#======== QUDA initialization ========
frontier_params.grid_size = [1, 1, 1, args.gcds]
frontier_params.mrhs = args.mrhs
frontier_params.init_quda(quda_resource_path)
latt_info = frontier_params.make_latt_info()
if latt_info.mpi_rank == 0:
    print({**frontier_params.make_run_parameters(n, t_src, x_src, y_src, z_src), "tsep_max": frontier_params.tsep_max,
           "run": run_id, "runs": n_runs, "t_idx": t_idx_mine.tolist(), "grid_size": frontier_params.grid_size, "mrhs": frontier_params.mrhs})
dirac = frontier_params.make_dirac(latt_info)


#======== gamma matrices, epsilon tensor, containers (cupy arrays: after core.init) ========
gamma_list, gamma_names = frontier_params.make_gamma_list()                   # the interpolators Gamma(n), n in frontier_params.gamma_ids, and their names
G4, G5, charge = gamma.gamma(8), gamma.gamma(15), gamma.gamma(10)     # g4, g5 and C = g2 g4, in PyQUDA's DeGrand-Rossi convention
parity_p = (gamma.gamma(0) + gamma.gamma(8)) / 2                      # (1 + g4) / 2
parity_m = (gamma.gamma(0) - gamma.gamma(8)) / 2                      # (1 - g4) / 2
eps_color = make_eps()
corr_dic = frontier_params.make_containers_one_tsrc(latt_info)
fb_dic = frontier_params.make_containers_one_tsrc_fb() if latt_info.mpi_rank == 0 else None
pion_forward,    proton_forward_dirac    = corr_dic["pion_forward"],    corr_dic["proton_forward_dirac"]
pion_sym_0xi,    proton_sym_0xi_dirac    = corr_dic["pion_sym_0xi"],    corr_dic["proton_sym_0xi_dirac"]
pion_sym_non0xi, proton_sym_non0xi_dirac = corr_dic["pion_sym_non0xi"], corr_dic["proton_sym_non0xi_dirac"]
pion_asy_0xi,    proton_asy_0xi_dirac    = corr_dic["pion_asy_0xi"],    corr_dic["proton_asy_0xi_dirac"]
pion_asy_non0xi, proton_asy_non0xi_dirac = corr_dic["pion_asy_non0xi"], corr_dic["proton_asy_non0xi_dirac"]
pion_plain,      proton_plain_dirac      = corr_dic["pion_plain"],      corr_dic["proton_plain_dirac"]
#column index tables on this GPU: they pick each case's p_f out of the sums over the 99 momenta of pf_plain_list
pf_forward_index    = cp.asarray(pf_forward_index_list)       # (1, 7)
pf_sym_0xi_index    = cp.asarray(pf_sym_0xi_index_list)       # (8, 7)
pf_sym_non0xi_index = cp.asarray(pf_sym_non0xi_index_list)    # (18, 7)
pf_asy_0xi_index    = cp.asarray(pf_asy_0xi_index_list)       # (8, 7)
pf_asy_non0xi_index = cp.asarray(pf_asy_non0xi_index_list)    # (36, 7)
mom_phase = phase.MomentumPhase(latt_info)
#host memory for the solutions of one source smearing (prop1 and prop2), column by column [spin * 3 + color, parity, t, z, y, x//2, spin, color]
host_shape = (12, 2, latt_info.Lt, latt_info.Lz, latt_info.Ly, latt_info.Lx // 2, 4, 3)
host_count = int(np.prod(host_shape))      # complex numbers; cupy's pinned memory pool rounds the allocation up, so take only these
prop1_host, prop2_host = [np.frombuffer(cp.cuda.alloc_pinned_memory(host_count * 16), "<c16", host_count).reshape(host_shape) for _ in range(2)]


#======== output: one pion and one proton file per case and cfg ========
gevp_dir = f"{current_dir}/{smear_tag}_GEVP_sepq_{len(frontier_params.t_base) * n_spatial_src}src_phyp_fb_tsep{frontier_params.tsep_max}"      # new directory: never mixed with the full-time files of *_tsrcsave.py
save_list = [   # (case, q list, p_f of every entry, frame)
    ("forward",    frontier_params.q_forward_list,    pf_forward_table,    "forward: p_f = p_i = (0,0,pz), pz = pz_list[iz]"),
    ("sym_0xi",    frontier_params.q_sym_0xi_list,    pf_sym_0xi_table,    "symmetric frame: P = (p_f + p_i)/2 = (0,0,pz), p_f = P + q/2, p_i = P - q/2, pz = pz_list[iz]"),
    ("sym_non0xi", frontier_params.q_sym_non0xi_list, pf_sym_non0xi_table, "symmetric frame: P = (p_f + p_i)/2 = (0,0,pz), p_f = P + q/2, p_i = P - q/2, pz = pz_list[iz]"),
    ("asy_0xi",    frontier_params.q_asy_0xi_list,    pf_asy_0xi_table,    "asymmetric frame: p_i = (0,0,pz), p_f = p_i + q, pz = pz_list[iz]"),
    ("asy_non0xi", frontier_params.q_asy_non0xi_list, pf_asy_non0xi_table, "asymmetric frame: p_i = (0,0,pz), p_f = p_i + q, pz = pz_list[iz]"),
    ("plain",      frontier_params.q_plain_list,      frontier_params.pf_plain_list,       "plain 2pt, q = 0, at every p_f and p_i = p_f - q of the other cases"),
]
#the notes written into the .h5 attributes (smearing_note, gauge_fix_note, momentum_note, source_note, time_reflection_note, time_index_note) are in the setup file

#======== saving per t_src ========
file_list = [(case, particle) for case, q_case, pf_case, frame_note in save_list for particle in ["pion", "proton"]]

def h5_name(cfg_dir, case, particle, cfg):
    return f"{cfg_dir}/{particle}_{smear_tag}_GEVP_sepq_{case}_phyp_fb_tsep{frontier_params.tsep_max}_cfg{cfg}.h5"

def part_name(cfg_dir, case, particle, cfg, t_idx, pt2):
    return f"{cfg_dir}/parts/{particle}_{smear_tag}_GEVP_sepq_{case}_phyp_fb_tsep{frontier_params.tsep_max}_cfg{cfg}_t{t_idx}_{pt2}.npy"

def all_on_disk(names):
    #rank 0 checks and tells the other ranks, so that every rank skips the same work (gatherLattice is collective)
    found = all(os.path.exists(name) for name in names) if latt_info.mpi_rank == 0 else None
    return latt_info.mpi_comm.bcast(found, root=0)

def take_merge(cfg_dir):
    #several runs can find all parts on disk at the same moment: only the run that creates the lock file first (O_EXCL) merges.
    #The name holds the Slurm job id and restart count, so a lock left in parts/ by an earlier, killed job does not block a resubmission.
    took = None
    if latt_info.mpi_rank == 0:
        lock = f"{cfg_dir}/parts/merge_job{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_RESTART_COUNT', '0')}"
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            took = True
        except (FileExistsError, FileNotFoundError):        # another run holds the lock, or has already merged and removed parts/
            took = False
    return latt_info.mpi_comm.bcast(took, root=0)

for i_cfg, cfg in tqdm(enumerate(measurement_list),desc=f"Processing cfgs"):

    cfg_dir = f"{gevp_dir}/cfg{cfg}"
    if all_on_disk([h5_name(cfg_dir, case, particle, cfg) for case, particle in file_list]):
        core.getLogger().info(f"SKIP CFG #{cfg}: its .h5 files are already saved")
        continue
    if latt_info.mpi_rank == 0:
        os.makedirs(f"{cfg_dir}/parts", exist_ok=True)      # also creates gevp_dir and cfg_dir

    for key in corr_dic:
        corr_dic[key][...] = 0
    prop1 = prop2 = prop1_inv = prop2_inv = pf_phases_plain = None      # drop the previous cfg's arrays before QUDA works on the new gauge field
    release_cupy_cache()

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
    gauge.fixingOVR(gauge_dir=frontier_params.gauge_fix_dir, Nsteps=frontier_params.gauge_fix_Nsteps, verbose_interval=frontier_params.gauge_fix_verbose_interval,
                    relax_boost=frontier_params.gauge_fix_relax_boost, tolerance=frontier_params.gauge_fix_tolerance,
                    reunit_interval=frontier_params.gauge_fix_reunit_interval, stopWtheta=frontier_params.gauge_fix_stopWtheta)     # overrelaxation, since QUDA's FFT method runs on one GPU only
    theta = coulomb_gauge_theta(latt_info, gauge)
    deviceSynchronize()
    core.getLogger().info(f"COULOMB GAUGE FIXING: {perf_counter() - s} secs, theta {theta:.3e}")
    if theta > 2e-6:                                # QUDA stops silently after Nsteps; cfg 204 ends at 7.8e-9, cfg 306 at 1.5e-6 after 20000 steps
        core.getLogger().info(f"SKIP CFG #{cfg}: Coulomb gauge fixing stopped at theta {theta:.3e} after {frontier_params.gauge_fix_Nsteps} steps")
        continue                                    # theta is an allreduce, so every rank skips the same cfg

    #HYP on the gauge-fixed links: HYP is gauge covariant, so the propagators live in the same Coulomb gauge as the smearing
    deviceSynchronize()
    s = perf_counter()
    gauge_hyp = gauge.copy()
    del gauge
    core.getLogger().info(f"DOING HYP SMEARING")
    core.getLogger().info(f"plaq_hyp_before = {gauge_hyp.plaquette()}")
    gauge_hyp.hypSmear(frontier_params.hyp_n_steps, frontier_params.hyp_alpha1, frontier_params.hyp_alpha2, frontier_params.hyp_alpha3, frontier_params.hyp_dir_ignore,
                       frontier_params.hyp_compute_plaquette, frontier_params.hyp_compute_qcharge)
    deviceSynchronize()
    core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
    core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE (also builds the multigrid setup)
    release_cupy_cache()
    dirac.loadGauge(gauge_hyp)

    for t_idx, t0 in enumerate(t_src[i_cfg]):
        if t_idx not in t_idx_mine:
            continue                                            # another run's block
        if all_on_disk([part_name(cfg_dir, case, particle, cfg, t_idx, pt2) for case, particle in file_list for pt2 in frontier_params.fb_index]):
            core.getLogger().info(f"SKIP T_SRC {t0}: already saved")
            continue
        for key in corr_dic:
            corr_dic[key][...] = 0                              # the containers hold one t_src: start it from zero
        for x_idx, x0 in enumerate(x_src[i_cfg]):
            for y_idx, y0 in enumerate(y_src[i_cfg]):
                for z_idx, z0 in enumerate(z_src[i_cfg]):

                    deviceSynchronize()
                    inner_loop = perf_counter()
                    core.getLogger().info(f"INNER LOOP STARTS")

                    src_pos = [x0, y0, z0, t0]
                    core.getLogger().info(f"SOURCE POSITION = {src_pos}")
                    #sink phases once, on the 99 momenta of pf_plain_list; conjugated so that the label is the physical momentum
                    pf_phases_plain = None                                                     # drop the previous source position's phases first
                    pf_phases_plain = mom_phase.getPhases(frontier_params.pf_plain_list, src_pos).conj()      # (99, 2, Lt_local, Lz, Ly, Lx//2)
                    #source weights exp(-i 2pi/Ls q.x_src) / n_spatial_src, one per q of each case
                    x_vec = np.array([x0, y0, z0])
                    src_phase_forward    = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / frontier_params.GLs * (frontier_params.q_forward_list    @ x_vec)) / n_spatial_src)
                    src_phase_sym_0xi    = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / frontier_params.GLs * (frontier_params.q_sym_0xi_list    @ x_vec)) / n_spatial_src)
                    src_phase_sym_non0xi = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / frontier_params.GLs * (frontier_params.q_sym_non0xi_list @ x_vec)) / n_spatial_src)
                    src_phase_asy_0xi    = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / frontier_params.GLs * (frontier_params.q_asy_0xi_list    @ x_vec)) / n_spatial_src)
                    src_phase_asy_non0xi = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / frontier_params.GLs * (frontier_params.q_asy_non0xi_list @ x_vec)) / n_spatial_src)
                    src_phase_plain      = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / frontier_params.GLs * (frontier_params.q_plain_list      @ x_vec)) / n_spatial_src)

                    for smear_src in smear_list:
                        i_src_shape,src_shape = smear_src[0]
                        i_src_frac,src_frac   = smear_src[1]
                        rho_T_src, rho_z_src = frontier_params.shape_list[i_src_shape]
                        k_src = frontier_params.k_list[i_src_frac]

                        #SRC: boosted Gaussian in Coulomb gauge, -k for prop1 and +k for prop2
                        #prop1 is conjugated, and will be used for anti-quark
                        #prop2 is not conjugated, and will be used for quark
                        prop1 = prop2 = prop1_inv = prop2_inv = None      # drop the previous source smearing's propagators before allocating new ones
                        core.getLogger().info(f"SRC SMEARING: {shape_names[i_src_shape]} rho_T {rho_T_src} rho_z {rho_z_src}, frac {frontier_params.mom_frac_list[i_src_frac]}")

                        #INVERT: the source columns are built from the smearing profile, mrhs at a time, and the solutions go to host memory, so that
                        #QUDA has the GPU to itself while it solves (the same right-hand sides and solves as core.invertPropagator of coulomb_boosted_source)
                        deviceSynchronize()
                        release_cupy_cache()
                        s = perf_counter()
                        core.getLogger().info(f"SOLVING DIRAC EQ")
                        invert_coulomb_boosted_source(dirac, latt_info, -k_src, src_pos, rho_T_src, rho_z_src, frontier_params.mrhs, prop1_host)
                        if np.any(k_src):
                            invert_coulomb_boosted_source(dirac, latt_info, +k_src, src_pos, rho_T_src, rho_z_src, frontier_params.mrhs, prop2_host)
                        core.getLogger().info(f"INVERT 2 propagagors: {perf_counter() - s} secs")

                        #BACK TO THE GPU for the sink smearing and the contractions: QUDA's solver memory is free again
                        deviceSynchronize()
                        s = perf_counter()
                        prop1_inv = core.LatticePropagator(latt_info)
                        for col in range(12):
                            prop1_inv.setFermion(core.LatticeFermion(latt_info, cp.asarray(prop1_host[col])), col // 3, col % 3)
                        if np.any(k_src):
                            prop2_inv = core.LatticePropagator(latt_info)
                            for col in range(12):
                                prop2_inv.setFermion(core.LatticeFermion(latt_info, cp.asarray(prop2_host[col])), col // 3, col % 3)
                        else:
                            prop2_inv = prop1_inv
                        deviceSynchronize()
                        core.getLogger().info(f"TO GPU: {perf_counter() - s} secs")
                        log_gpu_memory("AFTER INVERT")

                        for smear_snk in smear_list:

                            i_snk_shape,snk_shape = smear_snk[0]
                            i_snk_frac,snk_frac   = smear_snk[1]
                            rho_T_sink, rho_z_sink = frontier_params.shape_list[i_snk_shape]
                            k_sink = frontier_params.k_list[i_snk_frac]

                            #SINK AND CONTRACT, nt_chunk local time slices at a time: each chunk of prop1_inv and prop2_inv is sink smeared with
                            #sink_smear_evenodd (what coulomb_boosted_sink does, slice by slice; the chunks start at even t, as its even-odd ordering needs)
                            #and contracted with every sink gamma, so the smeared propagators are never stored whole
                            deviceSynchronize()
                            s = perf_counter()
                            core.getLogger().info(f"SINK SMEARING: {shape_names[i_snk_shape]} rho_T {rho_T_sink} rho_z {rho_z_sink}, frac {frontier_params.mom_frac_list[i_snk_frac]}")
                            pion_open   = [cp.empty((4, 4, len(frontier_params.pf_plain_list), latt_info.Lt), "<c16") for gamma_sink in gamma_list]          # per sink gamma, (4,4,99,Lt_local)
                            proton_open = [cp.empty((4, 4, 4, 4, len(frontier_params.pf_plain_list), latt_info.Lt), "<c16") for gamma_sink in gamma_list]    # per sink gamma, (4,4,4,4,99,Lt_local)
                            for tc in range(0, latt_info.Lt, frontier_params.nt_chunk):
                                ts = slice(tc, tc + frontier_params.nt_chunk)
                                p1 = sink_smear_evenodd(prop1_inv.data[:, ts], latt_info.size[:3], -k_sink, rho_T_sink, rho_z_sink)
                                p2 = sink_smear_evenodd(prop2_inv.data[:, ts], latt_info.size[:3], +k_sink, rho_T_sink, rho_z_sink)
                                ph = pf_phases_plain[:, :, ts]

                                for i_gamma_sink, gamma_sink in enumerate(gamma_list):

                                    CG_sink, G5G_sink = charge @ gamma_sink, G5 @ gamma_sink
                                    diquark = contract("def,gh,wtzyxhjeb,wtzyxgida->wtzyxijabf", eps_color, CG_sink, p2, p2)

                                    # pion: colors and sink gamma summed at each site, then sites summed with the phases of the 99 momenta
                                    #       source spins l,i left open for the source gamma
                                    pion_open[i_gamma_sink][..., ts] = contract("pwtzyx,wtzyxjiba,jk,wtzyxklba->lipt", ph, p1.conj(), G5G_sink, p2)

                                    # proton: colors and epsilon summed at each site, both Wick terms (step 1), then sites summed with the phases (step 2)
                                    #         source spins i,j and sink spins l,k left open
                                    proton_site  = contract("abc,wtzyxijabf,wtzyxlkfc->wtzyxijlk", eps_color, diquark, p2)    # step 1, first term, 256 per site
                                    proton_site -= contract("abc,wtzyxkjcbf,wtzyxlifa->wtzyxijlk", eps_color, diquark, p2)   # step 1, second term
                                    proton_open[i_gamma_sink][..., ts] = contract("pwtzyx,wtzyxijlk->ijlkpt", ph, proton_site)       # step 2
                                    del diquark, proton_site
                                del p1, p2, ph

                            for i_gamma_sink, gamma_sink in enumerate(gamma_list):

                                pion_open_plain, proton_open_plain = pion_open[i_gamma_sink], proton_open[i_gamma_sink]

                                # each case: its p_f columns, laid out as (q, pz)
                                pion_open_forward      = pion_open_plain[:, :, pf_forward_index]                   # (4,4,1,7,Lt_local)
                                pion_open_sym_0xi      = pion_open_plain[:, :, pf_sym_0xi_index]                   # (4,4,8,7,Lt_local)
                                pion_open_sym_non0xi   = pion_open_plain[:, :, pf_sym_non0xi_index]                # (4,4,18,7,Lt_local)
                                pion_open_asy_0xi      = pion_open_plain[:, :, pf_asy_0xi_index]                   # (4,4,8,7,Lt_local)
                                pion_open_asy_non0xi   = pion_open_plain[:, :, pf_asy_non0xi_index]                # (4,4,36,7,Lt_local)
                                proton_open_forward    = proton_open_plain[:, :, :, :, pf_forward_index]           # (4,4,4,4,1,7,Lt_local)
                                proton_open_sym_0xi    = proton_open_plain[:, :, :, :, pf_sym_0xi_index]           # (4,4,4,4,8,7,Lt_local)
                                proton_open_sym_non0xi = proton_open_plain[:, :, :, :, pf_sym_non0xi_index]        # (4,4,4,4,18,7,Lt_local)
                                proton_open_asy_0xi    = proton_open_plain[:, :, :, :, pf_asy_0xi_index]           # (4,4,4,4,8,7,Lt_local)
                                proton_open_asy_non0xi = proton_open_plain[:, :, :, :, pf_asy_non0xi_index]        # (4,4,4,4,36,7,Lt_local)

                                for i_gamma_source, gamma_source in enumerate(gamma_list):

                                    gamma_source_bar = G4 @ gamma_source.conj().T @ G4

                                    pion_forward[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,li,liqzt->qzt", src_phase_forward, gamma_source_bar @ G5, pion_open_forward)                            # (1, 7, Lt_local)
                                    proton_forward_dirac[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,ij,ijlkqzt->lkqzt", src_phase_forward, charge @ gamma_source, proton_open_forward)                     # (4, 4, 1, 7, Lt_local)

                                    pion_sym_0xi[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,li,liqzt->qzt", src_phase_sym_0xi, gamma_source_bar @ G5, pion_open_sym_0xi)                            # (8, 7, Lt_local)
                                    proton_sym_0xi_dirac[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,ij,ijlkqzt->lkqzt", src_phase_sym_0xi, charge @ gamma_source, proton_open_sym_0xi)                     # (4, 4, 8, 7, Lt_local)

                                    pion_sym_non0xi[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,li,liqzt->qzt", src_phase_sym_non0xi, gamma_source_bar @ G5, pion_open_sym_non0xi)                      # (18, 7, Lt_local)
                                    proton_sym_non0xi_dirac[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,ij,ijlkqzt->lkqzt", src_phase_sym_non0xi, charge @ gamma_source, proton_open_sym_non0xi)               # (4, 4, 18, 7, Lt_local)

                                    pion_asy_0xi[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,li,liqzt->qzt", src_phase_asy_0xi, gamma_source_bar @ G5, pion_open_asy_0xi)                            # (8, 7, Lt_local)
                                    proton_asy_0xi_dirac[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,ij,ijlkqzt->lkqzt", src_phase_asy_0xi, charge @ gamma_source, proton_open_asy_0xi)                     # (4, 4, 8, 7, Lt_local)

                                    pion_asy_non0xi[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,li,liqzt->qzt", src_phase_asy_non0xi, gamma_source_bar @ G5, pion_open_asy_non0xi)                      # (36, 7, Lt_local)
                                    proton_asy_non0xi_dirac[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,ij,ijlkqzt->lkqzt", src_phase_asy_non0xi, charge @ gamma_source, proton_open_asy_non0xi)               # (4, 4, 36, 7, Lt_local)

                                    pion_plain[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,li,lipt->qpt", src_phase_plain, gamma_source_bar @ G5, pion_open_plain)                                 # (1, 99, Lt_local)
                                    proton_plain_dirac[i_snk_shape, i_src_shape, i_snk_frac, i_src_frac, i_gamma_sink, i_gamma_source] += contract(
                                    "q,ij,ijlkpt->lkqpt", src_phase_plain, charge @ gamma_source, proton_open_plain)                          # (4, 4, 1, 99, Lt_local)

                            deviceSynchronize()
                            core.getLogger().info(f"SINK SMEAR AND CONTRACT ALL: {perf_counter() - s} secs")

                    free, total = cp.cuda.runtime.memGetInfo()
                    pool = cp.get_default_memory_pool()
                    core.getLogger().info(f"GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")
                    core.getLogger().info(f"UNTIL CONTRACTION: {perf_counter() - inner_loop} secs")

        #SAVE THIS T_SRC: every rank gathers each slab in the same order (gatherLattice is collective), rank 0 rolls it, keeps its forward and backward
        #time slices and writes them to parts/. Each part is written as .tmp and renamed when complete, so a job killed while saving leaves no half-written part.
        deviceSynchronize()
        s = perf_counter()
        for case, particle in file_list:
            key = f"pion_{case}" if particle == "pion" else f"proton_{case}_dirac"
            corr = corr_dic[key]
            corr_slab = core.gatherLattice(corr.get(), [corr.ndim - 1, -1, -1, -1])                          # ALL ranks; time is the last axis
            if latt_info.mpi_rank == 0:
                #ROLL so that time index 0 is t_src:  rolled[..., tau] = C[..., (t_src + tau) % 96]
                corr_slab = np.roll(corr_slab, -t0, axis=-1)
                if particle == "proton":
                    corr_slab[..., frontier_params.GLt - t0:] *= -1                                  # antiperiodic sign for the times that wrapped around
                #FORWARD AND BACKWARD: pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. tsep_max-1
                for pt2, t_index in frontier_params.fb_index.items():
                    fb_dic[pt2][key][...] = corr_slab[..., t_index]
                    part = part_name(cfg_dir, case, particle, cfg, t_idx, pt2)
                    with open(f"{part}.tmp", "wb") as f:
                        np.save(f, fb_dic[pt2][key])
                    os.replace(f"{part}.tmp", part)
            del corr_slab
        core.getLogger().info(f"SAVE T_SRC {t0}: {perf_counter() - s} secs")

    #MERGE only once the parts of every t_src are on disk (the other runs' blocks too), and by one run only
    if not all_on_disk([part_name(cfg_dir, case, particle, cfg, t_idx, pt2) for t_idx in range(len(t_src[i_cfg])) for case, particle in file_list for pt2 in frontier_params.fb_index]):
        core.getLogger().info(f"END CFG #{cfg}: t_src of another run not saved yet, the run that saves the last one merges")
        continue
    if not take_merge(cfg_dir):
        core.getLogger().info(f"END CFG #{cfg}: another run merges")
        continue

    #MERGE: rank 0 builds the .h5 files from the parts, then deletes the parts. Each file has two datasets, pt2_forward and pt2_backward, with the attributes
    #of the full-time files of *_tsrcsave.py, except dim_spec ("tsep" for "time"), dim_time, the new time_index and the proton's how_to_project.
    #Files are written as .tmp and renamed when complete, so a job killed while merging leaves no half-filled file under the final name.
    deviceSynchronize()
    saving_started = perf_counter()
    for case, q_case, pf_case, frame_note in save_list:
        for particle in ["pion", "proton"]:
            key = f"pion_{case}" if particle == "pion" else f"proton_{case}_dirac"
            filename = h5_name(cfg_dir, case, particle, cfg)
            dim_head = ["shape_sink","shape_source","frac_sink","frac_source","gamma_sink","gamma_source"] + (["dirac_sink","dirac_source"] if particle == "proton" else [])
            dim_tail = ["t_src_list","q_list","momentum_list","tsep"] if case == "plain" else ["t_src_list","q_list","pz","tsep"]

            if latt_info.mpi_rank == 0:
                with h5py.File(f"{filename}.tmp", "w") as f:
                    f.create_dataset("q_list", data=np.array(q_case))
                    f.create_dataset("momentum_list", data=np.array(pf_case))                                   # p_f of every entry: [iq, iz] (cases) or [ip] (plain)
                    f.create_dataset("pz_list", data=np.array(frontier_params.pz_list))
                    f.create_dataset("shape_list", data=np.array(frontier_params.shape_list))
                    f.create_dataset("mom_frac_list", data=np.array(frontier_params.mom_frac_list))
                    f.create_dataset("k_list", data=np.array(frontier_params.k_list))
                    if particle == "proton":
                        f.create_dataset("parity_p", data=cp.asnumpy(parity_p))      # (1+g4)/2, for pt2_forward
                        f.create_dataset("parity_m", data=cp.asnumpy(parity_m))      # (1-g4)/2, for pt2_backward, enters with a minus sign
                    for pt2 in frontier_params.fb_index:                                             # pt2_forward and pt2_backward
                        fb = fb_dic[pt2][key]
                        dset = f.create_dataset(pt2, shape=fb.shape[:-3] + (len(t_src[i_cfg]),) + fb.shape[-3:], dtype="<c16")      # the t_src index is put back here; filled per t_src below
                        dset.attrs["kinematic_case"] = case
                        dset.attrs["frame"] = frame_note
                        dset.attrs["dim_spec"] = np.array(dim_head + dim_tail, dtype=h5py.string_dtype())
                        dset.attrs["gamma_list"] = np.array(gamma_names, dtype=h5py.string_dtype())
                        dset.attrs["shape_names"] = np.array(shape_names, dtype=h5py.string_dtype())
                        dset.attrs["shape_list"] = "rows of shape_list dataset: [rho_T, rho_z]; index order (sink, source)"
                        dset.attrs["mom_frac_list"] = "quark boost k = frac * mom_max in units of 2pi/L along z; index order (sink, source)"
                        dset.attrs["spatial_sources"] = source_note
                        dset.attrs["smearing"] = smearing_note
                        dset.attrs["gauge_fixing"] = gauge_fix_note
                        dset.attrs["momentum_convention"] = momentum_note
                        dset.attrs["momentum_transfer"] = transfer_note
                        dset.attrs["src_phase_sign"] = src_phase_sign                 # -1: weight exp(src_phase_sign * i 2pi/Ls q.x_src)
                        dset.attrs["gauge_file"] = gauge_file
                        dset.attrs["measurements"] = [cfg]
                        dset.attrs["x_src_list"] = x_src[i_cfg]
                        dset.attrs["y_src_list"] = y_src[i_cfg]
                        dset.attrs["z_src_list"] = z_src[i_cfg]
                        dset.attrs["t_src_list"] = t_src[i_cfg]
                        dset.attrs["time_index"] = time_index_note
                        dset.attrs["dim_time"] = frontier_params.fb_index[pt2]                       # time after the roll, (t - t_src) % Lt, of every tsep
                        if particle == "pion":
                            dset.attrs["time_reflection"] = "pt2_backward[..., tsep] = s_a s_b pt2_forward[..., tsep] for tsep >= 1: " + time_reflection_note
                            dset.attrs["propagator"] = "prop1 smeared with boost -k and prop2 with +k, at source and sink (quarks boosted toward physical +z)"
                        else:
                            dset.attrs["diquark"] = "C Gamma_sink at the sink, C Gamma_source at the source (equal to -Gamma_bar_source C for G5 and G5G4, an overall sign)"
                            dset.attrs["how_to_project"] = ("NOT projected. Positive parity: C_forward(tsep) = sum_kl parity_p[k,l] M[l,k] with M = pt2_forward, "
                                                            "C_backward(tsep) = -sum_kl parity_m[k,l] M[l,k] with M = pt2_backward, l = dirac_sink, k = dirac_source. "
                                                            "Antiperiodic sign already applied. pt2_backward[..., 0] is the source slice projected with parity_m: do not use it. "
                                                            "The time reflection holds for the projected correlators, not for M: C_backward(tsep) = s_a s_b C_forward(tsep) "
                                                            "for tsep >= 1: " + time_reflection_note)
                            dset.attrs["propagator"] = "prop2 (boost +k) for all three quark lines; source smearing = (shape_source, frac_source), sink smearing = (shape_sink, frac_sink)"

            if latt_info.mpi_rank == 0:
                with h5py.File(f"{filename}.tmp", "a") as f:
                    for pt2 in frontier_params.fb_index:
                        for t_idx in range(len(t_src[i_cfg])):                   # the slabs saved during the run, already rolled, signed and cut to tsep_max
                            f[pt2][..., t_idx, :, :, :] = np.load(part_name(cfg_dir, case, particle, cfg, t_idx, pt2))
                os.replace(f"{filename}.tmp", filename)

    #After all tsrc's files are gathered, delete the "parts" folder
    if latt_info.mpi_rank == 0:
        shutil.rmtree(f"{cfg_dir}/parts")                   
    core.getLogger().info(f"SAVING SECTION:{perf_counter()-saving_started} secs")

    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    core.getLogger().info(f"END CFG #{cfg}: GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")

dirac.freeGauge()