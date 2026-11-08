import os
import shutil
import socket
import argparse
import h5py
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, phase, gamma
from time import perf_counter
from cupy.cuda.runtime import deviceSynchronize
from tqdm import tqdm
from coulomb_smearing_notched import coulomb_boosted_source, coulomb_boosted_sink, coulomb_gauge_theta      # the notched copy in this folder
from pt2_comm_tools import make_eps, release_cupy_cache, log_gpu_memory # tools/pt2_comm_tools.py
from sepq_2pt_setup_fermilab_final import *  # delta_params (the TwoPtParams of this run), cfg list, smearing operators, momentum tables, notes, paths

import sys, traceback
from mpi4py import MPI

#STOP ON ERROR: an uncaught error on any rank prints its traceback and kills all ranks of this run at once, so the job ends as FAILED instead of
#hanging until walltime (PyQUDA's endQuda at exit waits for the other ranks, which are waiting in a collective for this one)
def abort_on_error(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)
sys.excepthook = abort_on_error

#Fermilab NOTCHED-kernel test (large-pz pion, DIAGONAL run): 2pt (pion + proton) with the 3 smearings of smear_list in the setup, rows
#[rho_T, rho_z, mom_frac, notch_R], each used at source AND sink only: no source x sink smearing matrix, so no smearing GEVP. Every smearing keeps its
#full 3 x 3 gamma matrix (G5, G5G4, G3G5), and only the plain q = 0 case is computed (the 99 sink momenta of pf_plain_list, (0,0,pz) among them).
#A source costs one pair of inversions and one sink smearing + contraction per smearing.
#Copy of ../smear_fermilab_4p5_1p4-1p6_highfrac/2pt_coulomb_boosted_sepq_1024srcs_fermilab_final.py: only these comments, the 4-entry unpacking of the
#smear_list rows, the notch_R passed to coulomb_boosted_source / coulomb_boosted_sink (coulomb_smearing_notched.py of this folder), the two log lines
#and the smear_list notes of the .h5 attributes differ. notch_R = 0 is the old kernel bit for bit, so smear 0 here equals [4.5, 1.0, 1.0] of
#../smear_fermilab_v3_followup_k6 on the same cfg.


parser = argparse.ArgumentParser()
parser.add_argument("--icfg", type=int, default=delta_params.icfg)      # starting position in cfg_list, e.g., icfg=100 means we start with the 100th cfg in the list, namely, 204+6*100=804
parser.add_argument("--n", type=int, default=delta_params.n_cfg)        # number of configs to measure in this one job
args = parser.parse_args()


#======== configurations and sources of this job ========
icfg0 = args.icfg
n = args.n
measurement_list = cfg_list[icfg0::cfg_measure_spacing][:n]
t_src, x_src, y_src, z_src = delta_params.make_sources(measurement_list)


#======== QUDA initialization ========
delta_params.init_quda(quda_resource_path)
latt_info = delta_params.make_latt_info()
if latt_info.mpi_rank == 0:
    print({**delta_params.make_run_parameters(n, t_src, x_src, y_src, z_src), "tsep_max": delta_params.tsep_max})
dirac = delta_params.make_dirac(latt_info)


#======== gamma matrices, epsilon tensor, containers (cupy arrays: after core.init) ========
gamma_list, gamma_names = delta_params.make_gamma_list()                   # the interpolators Gamma(n), n in delta_params.gamma_ids, and their names
G4, G5, charge = gamma.gamma(8), gamma.gamma(15), gamma.gamma(10)     # g4, g5 and C = g2 g4, in PyQUDA's DeGrand-Rossi convention
parity_p = (gamma.gamma(0) + gamma.gamma(8)) / 2                      # (1 + g4) / 2
parity_m = (gamma.gamma(0) - gamma.gamma(8)) / 2                      # (1 - g4) / 2
eps_color = make_eps()
#containers of one t_src with ONE smear axis (smearing i at source and sink) and the plain case only, as in the smear tuning:
#pion (smear, gamma_sink, gamma_source, q, p_f, local time), proton (smear, gamma_sink, gamma_source, dirac_sink, dirac_source, q, p_f, local time)
n_smear, n_gamma, n_q, n_pf = len(delta_params.smear_list), len(gamma_list), len(delta_params.q_plain_list), len(delta_params.pf_plain_list)
corr_dic = {"pion_plain":         cp.zeros((n_smear, n_gamma, n_gamma,       n_q, n_pf, latt_info.Lt), "<c16"),
            "proton_plain_dirac": cp.zeros((n_smear, n_gamma, n_gamma, 4, 4, n_q, n_pf, latt_info.Lt), "<c16")}
fb_dic = {pt2: {key: np.zeros(corr.shape[:-1] + (delta_params.tsep_max,), "<c16") for key, corr in corr_dic.items()}
          for pt2 in delta_params.fb_index} if latt_info.mpi_rank == 0 else None      # forward and backward time slices of one t_src, on the host of rank 0
pion_plain,      proton_plain_dirac      = corr_dic["pion_plain"],      corr_dic["proton_plain_dirac"]
mom_phase = phase.MomentumPhase(latt_info)


#======== output: one pion and one proton file per cfg, plain case only ========
gevp_dir = f"{current_dir}/{smear_tag}_{len(delta_params.t_base) * n_spatial_src}src_phyp_fb_tsep{delta_params.tsep_max}"      # own folder: smear_tag ends in _diag
save_list = [   # (case, q list, p_f of every entry, frame)
    ("plain",      delta_params.q_plain_list,      delta_params.pf_plain_list,       "plain 2pt, q = 0, at the 99 p_f of pf_plain_list"),
]
#the notes written into the .h5 attributes (smearing_note, gauge_fix_note, momentum_note, source_note, time_reflection_note, time_index_note) are in the setup file

#======== saving per t_src ========
file_list = [(case, particle) for case, q_case, pf_case, frame_note in save_list for particle in ["pion", "proton"]]

def h5_name(cfg_dir, case, particle, cfg):
    return f"{cfg_dir}/{particle}_{smear_tag}_{case}_phyp_fb_tsep{delta_params.tsep_max}_cfg{cfg}.h5"

def part_name(cfg_dir, case, particle, cfg, t_idx, pt2):
    return f"{cfg_dir}/parts/{particle}_{smear_tag}_{case}_phyp_fb_tsep{delta_params.tsep_max}_cfg{cfg}_t{t_idx}_{pt2}.npy"

def all_on_disk(names):
    #rank 0 checks and tells the other ranks, so that every rank skips the same work (gatherLattice is collective)
    found = all(os.path.exists(name) for name in names) if latt_info.mpi_rank == 0 else None
    return latt_info.mpi_comm.bcast(found, root=0)

#the folder keeps the smear_list it was made with: a job with another list would skip the parts and files of these operators as if they were its own
if latt_info.mpi_rank == 0:
    os.makedirs(gevp_dir, exist_ok=True)
    smear_file = f"{gevp_dir}/smear_list.npy"
    if os.path.exists(smear_file):
        if not np.array_equal(np.load(smear_file), np.array(delta_params.smear_list)):
            raise RuntimeError(f"{smear_file} holds another smear_list: set a new smear_tag in the setup for a new set of operators")
    else:
        tmp = f"{smear_file}.{socket.gethostname()}.{os.getpid()}.tmp"     # array tasks may start together: each writes its own file, then renames it
        with open(tmp, "wb") as f:
            np.save(f, np.array(delta_params.smear_list))
        os.replace(tmp, smear_file)

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
    gauge.fixingOVR(gauge_dir=delta_params.gauge_fix_dir, Nsteps=delta_params.gauge_fix_Nsteps, verbose_interval=delta_params.gauge_fix_verbose_interval,
                    relax_boost=delta_params.gauge_fix_relax_boost, tolerance=delta_params.gauge_fix_tolerance,
                    reunit_interval=delta_params.gauge_fix_reunit_interval, stopWtheta=delta_params.gauge_fix_stopWtheta)     # overrelaxation, since QUDA's FFT method runs on one GPU only
    theta = coulomb_gauge_theta(latt_info, gauge)
    deviceSynchronize()
    core.getLogger().info(f"COULOMB GAUGE FIXING: {perf_counter() - s} secs, theta {theta:.3e}")
    if theta > 2e-6:                                # QUDA stops silently after Nsteps; cfg 204 ends at 7.8e-9, cfg 306 at 1.5e-6 after 20000 steps
        core.getLogger().info(f"SKIP CFG #{cfg}: Coulomb gauge fixing stopped at theta {theta:.3e} after {delta_params.gauge_fix_Nsteps} steps")
        continue                                    # theta is an allreduce, so every rank skips the same cfg

    #HYP on the gauge-fixed links: HYP is gauge covariant, so the propagators live in the same Coulomb gauge as the smearing
    deviceSynchronize()
    s = perf_counter()
    gauge_hyp = gauge.copy()
    del gauge
    core.getLogger().info(f"DOING HYP SMEARING")
    core.getLogger().info(f"plaq_hyp_before = {gauge_hyp.plaquette()}")
    gauge_hyp.hypSmear(delta_params.hyp_n_steps, delta_params.hyp_alpha1, delta_params.hyp_alpha2, delta_params.hyp_alpha3, delta_params.hyp_dir_ignore,
                       delta_params.hyp_compute_plaquette, delta_params.hyp_compute_qcharge)
    deviceSynchronize()
    core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
    core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE (also builds the multigrid setup)
    release_cupy_cache()
    dirac.loadGauge(gauge_hyp)

    for t_idx, t0 in enumerate(t_src[i_cfg]):
        if all_on_disk([part_name(cfg_dir, case, particle, cfg, t_idx, pt2) for case, particle in file_list for pt2 in delta_params.fb_index]):
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
                    pf_phases_plain = mom_phase.getPhases(delta_params.pf_plain_list, src_pos).conj()      # (99, 2, Lt_local, Lz, Ly, Lx//2)
                    #source weight exp(-i 2pi/Ls q.x_src) / n_spatial_src of the plain case (q = 0: 1 / n_spatial_src)
                    x_vec = np.array([x0, y0, z0])
                    src_phase_plain      = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / delta_params.GLs * (delta_params.q_plain_list      @ x_vec)) / n_spatial_src)

                    for i_src, (rho_T_src, rho_z_src, frac_src, notch_src) in enumerate(delta_params.smear_list):
                        k_src = delta_params.k_list[i_src]

                        #SRC: boosted Gaussian in Coulomb gauge, -k for prop1 and +k for prop2
                        #prop1 is conjugated, and will be used for anti-quark
                        #prop2 is not conjugated, and will be used for quark
                        prop1 = prop2 = prop1_inv = prop2_inv = None      # drop the previous source smearing's propagators before allocating new ones
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"SRC SMEARING: {smear_names[i_src]} rho_T {rho_T_src} rho_z {rho_z_src}, frac {frac_src}, notch_R {notch_src}")
                        prop1_inv = coulomb_boosted_source(latt_info, -k_src, src_pos, rho_T_src, rho_z_src, notch_R=notch_src)
                        if np.any(k_src):
                            prop2_inv = coulomb_boosted_source(latt_info, +k_src, src_pos, rho_T_src, rho_z_src, notch_R=notch_src)
                        else:
                            prop2_inv = prop1_inv
                        
                        deviceSynchronize()
                        core.getLogger().info(f"SOURCE SMEAR: {perf_counter() - s} secs")

                        #INVERT
                        deviceSynchronize()
                        release_cupy_cache()
                        s = perf_counter()
                        core.getLogger().info(f"SOLVING DIRAC EQ")
                        prop1_inv = core.invertPropagator(dirac, prop1_inv, mrhs=delta_params.mrhs)
                        if np.any(k_src):
                            prop2_inv = core.invertPropagator(dirac, prop2_inv, mrhs=delta_params.mrhs)
                        else:
                            prop2_inv = prop1_inv
                        core.getLogger().info(f"INVERT 2 propagagors: {perf_counter() - s} secs")
                        log_gpu_memory("AFTER INVERT")

                        for i_snk, (rho_T_sink, rho_z_sink, frac_sink, notch_sink) in [(i_src, delta_params.smear_list[i_src])]:      # diagonal run: the sink smearing is the source's own

                            k_sink = delta_params.k_list[i_snk]

                            #SINK: the same smearing with the sink parameters, from the unsmeared solution each time
                            prop1 = prop2 = None      # drop the previous sink smearing's propagators before allocating new ones
                            deviceSynchronize()
                            s = perf_counter()
                            core.getLogger().info(f"SINK SMEARING: {smear_names[i_snk]} rho_T {rho_T_sink} rho_z {rho_z_sink}, frac {frac_sink}, notch_R {notch_sink}")
                            prop1 = coulomb_boosted_sink(latt_info, prop1_inv, -k_sink, rho_T_sink, rho_z_sink, notch_R=notch_sink)
                            prop2 = coulomb_boosted_sink(latt_info, prop2_inv, +k_sink, rho_T_sink, rho_z_sink, notch_R=notch_sink)
                            deviceSynchronize()
                            core.getLogger().info(f"SINK SMEAR: {perf_counter() - s} secs")

                            #CONTRACT
                            deviceSynchronize()
                            s = perf_counter()

                            for i_gamma_sink, gamma_sink in enumerate(gamma_list):

                                CG_sink, G5G_sink = charge @ gamma_sink, G5 @ gamma_sink
                                pion_open_plain   = cp.empty((4, 4, len(delta_params.pf_plain_list), latt_info.Lt), "<c16")          # (4,4,99,Lt_local)
                                proton_open_plain = cp.empty((4, 4, 4, 4, len(delta_params.pf_plain_list), latt_info.Lt), "<c16")    # (4,4,4,4,99,Lt_local)
                                for tc in range(0, latt_info.Lt, delta_params.nt_chunk):
                                    ts = slice(tc, tc + delta_params.nt_chunk)
                                    p1, p2, ph = prop1.data[:, ts], prop2.data[:, ts], pf_phases_plain[:, :, ts]

                                    diquark = contract("def,gh,wtzyxhjeb,wtzyxgida->wtzyxijabf", eps_color, CG_sink, p2, p2)

                                    # pion: colors and sink gamma summed at each site, then sites summed with the phases of the 99 momenta
                                    #       source spins l,i left open for the source gamma
                                    pion_open_plain[..., ts] = contract("pwtzyx,wtzyxjiba,jk,wtzyxklba->lipt", ph, p1.conj(), G5G_sink, p2)

                                    # proton: colors and epsilon summed at each site, both Wick terms (step 1), then sites summed with the phases (step 2)
                                    #         source spins i,j and sink spins l,k left open
                                    proton_site  = contract("abc,wtzyxijabf,wtzyxlkfc->wtzyxijlk", eps_color, diquark, p2)    # step 1, first term, 256 per site
                                    proton_site -= contract("abc,wtzyxkjcbf,wtzyxlifa->wtzyxijlk", eps_color, diquark, p2)   # step 1, second term
                                    proton_open_plain[..., ts] = contract("pwtzyx,wtzyxijlk->ijlkpt", ph, proton_site)       # step 2
                                    del diquark, proton_site, p1, p2, ph

                                for i_gamma_source, gamma_source in enumerate(gamma_list):

                                    gamma_source_bar = G4 @ gamma_source.conj().T @ G4

                                    pion_plain[i_src, i_gamma_sink, i_gamma_source] += contract(
                                    "q,li,lipt->qpt", src_phase_plain, gamma_source_bar @ G5, pion_open_plain)                                 # (1, 99, Lt_local)
                                    proton_plain_dirac[i_src, i_gamma_sink, i_gamma_source] += contract(
                                    "q,ij,ijlkpt->lkqpt", src_phase_plain, charge @ gamma_source, proton_open_plain)                          # (4, 4, 1, 99, Lt_local)

                            deviceSynchronize()
                            core.getLogger().info(f"CONTRACT ALL: {perf_counter() - s} secs")

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
                    corr_slab[..., delta_params.GLt - t0:] *= -1                                  # antiperiodic sign for the times that wrapped around
                #FORWARD AND BACKWARD: pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. tsep_max-1
                for pt2, t_index in delta_params.fb_index.items():
                    fb_dic[pt2][key][...] = corr_slab[..., t_index]
                    part = part_name(cfg_dir, case, particle, cfg, t_idx, pt2)
                    with open(f"{part}.tmp", "wb") as f:
                        np.save(f, fb_dic[pt2][key])
                        f.flush(); os.fsync(f.fileno())                 # wait until the part is really on disk: if the storage fails, an OSError here, never a damaged part under its final name
                    os.replace(f"{part}.tmp", part)
            del corr_slab
        core.getLogger().info(f"SAVE T_SRC {t0}: {perf_counter() - s} secs")

    #MERGE: rank 0 builds the .h5 files from the parts, then deletes the parts. Each file has two datasets, pt2_forward and pt2_backward, with the attributes
    #of the full-time files of *_tsrcsave.py, except dim_spec ("tsep" for "time"), dim_time, the new time_index and the proton's how_to_project.
    #Files are written as .tmp and renamed when complete, so a job killed while merging leaves no half-filled file under the final name.
    deviceSynchronize()
    saving_started = perf_counter()
    for case, q_case, pf_case, frame_note in save_list:
        for particle in ["pion", "proton"]:
            key = f"pion_{case}" if particle == "pion" else f"proton_{case}_dirac"
            filename = h5_name(cfg_dir, case, particle, cfg)
            dim_head = ["smear","gamma_sink","gamma_source"] + (["dirac_sink","dirac_source"] if particle == "proton" else [])
            dim_tail = ["t_src_list","q_list","momentum_list","tsep"] if case == "plain" else ["t_src_list","q_list","pz","tsep"]

            if latt_info.mpi_rank == 0:
                with h5py.File(f"{filename}.tmp", "w") as f:
                    f.create_dataset("q_list", data=np.array(q_case))
                    f.create_dataset("momentum_list", data=np.array(pf_case))                                   # p_f of every entry: [iq, iz] (cases) or [ip] (plain)
                    f.create_dataset("pz_list", data=np.array(delta_params.pz_list))
                    f.create_dataset("smear_list", data=np.array(delta_params.smear_list))                      # (smear, 4): rows [rho_T, rho_z, mom_frac, notch_R]
                    f.create_dataset("k_list", data=np.array(delta_params.k_list))
                    if particle == "proton":
                        f.create_dataset("parity_p", data=cp.asnumpy(parity_p))      # (1+g4)/2, for pt2_forward
                        f.create_dataset("parity_m", data=cp.asnumpy(parity_m))      # (1-g4)/2, for pt2_backward, enters with a minus sign
                    for pt2 in delta_params.fb_index:                                             # pt2_forward and pt2_backward
                        fb = fb_dic[pt2][key]
                        dset = f.create_dataset(pt2, shape=fb.shape[:-3] + (len(t_src[i_cfg]),) + fb.shape[-3:], dtype="<c16")      # the t_src index is put back here; filled per t_src below
                        dset.attrs["kinematic_case"] = case
                        dset.attrs["frame"] = frame_note
                        dset.attrs["dim_spec"] = np.array(dim_head + dim_tail, dtype=h5py.string_dtype())
                        dset.attrs["gamma_list"] = np.array(gamma_names, dtype=h5py.string_dtype())
                        dset.attrs["smear_names"] = np.array(smear_names, dtype=h5py.string_dtype())
                        dset.attrs["smear_list"] = np.array(delta_params.smear_list)                  # row i = [rho_T, rho_z, mom_frac, notch_R] of smear index i, at source and sink
                        dset.attrs["smear_list_note"] = "rows [rho_T, rho_z, mom_frac, notch_R]; smear index i is smearing i at source AND sink (diagonal run, no smearing cross terms); quark boost k = mom_frac * mom_max in units of 2pi/L along z (k_list dataset); notch_R: the z Fourier transform of the kernel times 1 - exp(-notch_R^2 q_z^2 / 4), 0 = no notch, inf = only q_z = 0 removed"
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
                        dset.attrs["dim_time"] = delta_params.fb_index[pt2]                       # time after the roll, (t - t_src) % Lt, of every tsep
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
                            dset.attrs["propagator"] = "prop2 (boost +k) for all three quark lines; the same smearing smear_list[smear] at source and sink"

            if latt_info.mpi_rank == 0:
                with h5py.File(f"{filename}.tmp", "a") as f:
                    for pt2 in delta_params.fb_index:
                        for t_idx in range(len(t_src[i_cfg])):                   # the slabs saved during the run, already rolled, signed and cut to tsep_max
                            f[pt2][..., t_idx, :, :, :] = np.load(part_name(cfg_dir, case, particle, cfg, t_idx, pt2))
                fd = os.open(f"{filename}.tmp", os.O_RDONLY); os.fsync(fd); os.close(fd)   # wait until the .h5 is really on disk, as for the parts
                os.replace(f"{filename}.tmp", filename)

    #After all tsrc's files are gathered, delete the "parts" folder
    if latt_info.mpi_rank == 0:
        shutil.rmtree(f"{cfg_dir}/parts")                   
    core.getLogger().info(f"SAVING SECTION:{perf_counter()-saving_started} secs")

    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    core.getLogger().info(f"END CFG #{cfg}: GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")

dirac.freeGauge()