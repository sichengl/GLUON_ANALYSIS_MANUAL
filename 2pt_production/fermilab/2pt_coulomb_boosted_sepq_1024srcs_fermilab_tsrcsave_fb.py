import os
import shutil
import argparse
import h5py
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, phase
from time import perf_counter
from cupy.cuda.runtime import deviceSynchronize
from tqdm import tqdm
from coulomb_smearing import coulomb_boosted_source, coulomb_boosted_sink, coulomb_gauge_theta
from sepq_2pt_setup_fermilab import *        # parameters, momentum lists, sources, paths, containers; make_gamma_eps() and make_containers_one_tsrc() need core.init first


parser = argparse.ArgumentParser()
parser.add_argument("--icfg", type=int, default=0)      # starting position in cfg_list, e.g., icfg=100 means we start with the 100th cfg in the list, namely, 204+6*100=804
parser.add_argument("--n", type=int, default=20)        # number of configs to measure in this one job
args = parser.parse_args()


#======== configurations and sources of this job ========
icfg0 = args.icfg
n = args.n
measurement_list = cfg_list[icfg0::cfg_measure_spacing][:n]
ncfg     = (measurement_list - 204) // 6
t_src, x_src, y_src, z_src = make_sources(ncfg)


#======== QUDA initialization ========
core.init([1, 1, 1, 4], resource_path=quda_resource_path)
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
if latt_info.mpi_rank == 0:
    print({**make_run_parameters(n, t_src, x_src, y_src, z_src), "tsep_max": tsep_max})
dirac = core.getDirac(latt_info, -0.05138, 1e-10, 1000, 1.0, 1.04243, 1.04243, [[4, 4, 4, 4],[2,2,2,2]])


#======== gamma matrices, epsilon tensor, containers (cupy arrays: after core.init) ========
gamma_list, G4, G5, charge, parity_p, parity_m, eps_color = make_gamma_eps()
#GPU containers without the t_src index (make_containers_one_tsrc in the setup): they hold only the t_src being computed, with all local time slices,
#are zeroed when it starts and are saved to parts/ when it ends, so the GPU never holds more than one t_src of results
corr_dic = make_containers_one_tsrc(latt_info)
#host containers of the forward and backward 2pt of one t_src, tsep_max time slices each (make_containers_one_tsrc_fb in the setup):
#rank 0 fills them from the gathered and rolled correlator when the t_src is saved, so only rank 0 needs them
fb_dic = make_containers_one_tsrc_fb() if latt_info.mpi_rank == 0 else None
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


#======== output: one pion and one proton file per case and cfg ========
gevp_dir = f"{current_dir}/{smear_tag}_GEVP_sepq_{len(t_base) * n_spatial_src}src_phyp_fb_tsep{tsep_max}"      # new directory: never mixed with the full-time files of *_tsrcsave.py
save_list = [   # (case, q list, p_f of every entry, frame)
    ("forward",    q_forward_list,    pf_forward_table,    "forward: p_f = p_i = (0,0,pz), pz = pz_list[iz]"),
    ("sym_0xi",    q_sym_0xi_list,    pf_sym_0xi_table,    "symmetric frame: P = (p_f + p_i)/2 = (0,0,pz), p_f = P + q/2, p_i = P - q/2, pz = pz_list[iz]"),
    ("sym_non0xi", q_sym_non0xi_list, pf_sym_non0xi_table, "symmetric frame: P = (p_f + p_i)/2 = (0,0,pz), p_f = P + q/2, p_i = P - q/2, pz = pz_list[iz]"),
    ("asy_0xi",    q_asy_0xi_list,    pf_asy_0xi_table,    "asymmetric frame: p_i = (0,0,pz), p_f = p_i + q, pz = pz_list[iz]"),
    ("asy_non0xi", q_asy_non0xi_list, pf_asy_non0xi_table, "asymmetric frame: p_i = (0,0,pz), p_f = p_i + q, pz = pz_list[iz]"),
    ("plain",      q_plain_list,      pf_plain_list,       "plain 2pt, q = 0, at every p_f and p_i = p_f - q of the other cases"),
]
smearing_note = ("boosted Gaussian in Coulomb gauge, no gauge links: K(d) = exp(-(dx^2+dy^2)/rho_T^2 - dz^2/rho_z^2) exp(+i 2pi/L k.d), d = x - y, "
                 "Gaussian part normalized to sum 1; the same K at source and sink; rho_z = rho_T equals the width of N40 rho3.25 Wuppertal smearing")
gauge_fix_note = "Coulomb gauge fixing of the unsmeared links by overrelaxation, fixingOVR(gauge_dir=3, Nsteps=20000, verbose_interval=500, relax_boost=1.7, tolerance=1e-12, reunit_interval=10, stopWtheta=1), before HYP"
momentum_note = ("sink phase exp(-2 pi i p.(x - x_src)/L) (conjugate of PyQUDA MomentumPhase): the momentum_list dataset gives the physical sink "
                 "momentum p_f of every entry; the quarks are boosted toward physical +z, so +pz has the best overlap")
source_note = (f"spatial sources contracted: for every t_src, mean over the {n_spatial_src} positions x_src_list x y_src_list x z_src_list, weighted by "
               "exp(-i 2pi/Ls q.x_src) with q from the q_list dataset. This phase is ALREADY APPLIED: do not apply it again in the 3pt build")
time_reflection_note = ("C_ab(Lt - t) = s_a s_b C_ab(t) with s = +1 for G5, -1 for G45 (a = gamma_sink, b = gamma_source): "
                        "the G5-G45 elements are odd, include the sign when averaging forward and backward")
time_index_note = (f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {tsep_max - 1}; index 0 is the source slice "
                   "in both. dim_time gives the time after the roll, (t - t_src) % Lt, of every tsep; the other time slices are not saved")

#======== saving per t_src ========
#The containers hold one t_src (no t_src index). As soon as all spatial sources of a t_src are done, rank 0 gathers and rolls every container, keeps its
#forward and backward tsep_max time slices (fb_index in the setup) and writes them to {cfg_dir}/parts/ (one .npy per file, t_src and direction).
#When all t_src of a cfg are done, the parts are merged into .h5 files with two datasets, pt2_forward and pt2_backward, and deleted.
#A rerun of the same cfg skips every t_src whose parts are on disk, and skips the cfg if its .h5 files are on disk.
#If you change the sources or the smearing, delete the old parts/ folder by hand first.
file_list = [(case, particle) for case, q_case, pf_case, frame_note in save_list for particle in ["pion", "proton"]]

def h5_name(cfg_dir, case, particle, cfg):
    return f"{cfg_dir}/{particle}_{smear_tag}_GEVP_sepq_{case}_phyp_fb_tsep{tsep_max}_cfg{cfg}.h5"

def part_name(cfg_dir, case, particle, cfg, t_idx, pt2):
    return f"{cfg_dir}/parts/{particle}_{smear_tag}_GEVP_sepq_{case}_phyp_fb_tsep{tsep_max}_cfg{cfg}_t{t_idx}_{pt2}.npy"

def all_on_disk(names):
    #rank 0 checks and tells the other ranks, so that every rank skips the same work (gatherLattice is collective)
    found = all(os.path.exists(name) for name in names) if latt_info.mpi_rank == 0 else None
    return latt_info.mpi_comm.bcast(found, root=0)

for i_cfg, cfg in tqdm(enumerate(measurement_list),desc=f"Processing cfgs"):

    cfg_dir = f"{gevp_dir}/cfg{cfg}"
    if all_on_disk([h5_name(cfg_dir, case, particle, cfg) for case, particle in file_list]):
        core.getLogger().info(f"SKIP CFG #{cfg}: its .h5 files are already saved")
        continue
    if latt_info.mpi_rank == 0:
        os.makedirs(f"{cfg_dir}/parts", exist_ok=True)      # also creates gevp_dir and cfg_dir

    for key in corr_dic:
        corr_dic[key][...] = 0

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
    gauge.fixingOVR(gauge_dir=3, Nsteps=20000, verbose_interval=500, relax_boost=1.7, tolerance=1e-12, reunit_interval=10, stopWtheta=1)   # 3 = Coulomb; overrelaxation, since QUDA's FFT method runs on one GPU only
    theta = coulomb_gauge_theta(latt_info, gauge)
    deviceSynchronize()
    core.getLogger().info(f"COULOMB GAUGE FIXING: {perf_counter() - s} secs, theta {theta:.3e}")
    if theta > 1e-6:                                # QUDA stops silently after Nsteps; cfg 204 ends at 7.8e-9 on both machines, 1e-12 is never reached
        raise RuntimeError(f"Coulomb gauge fixing failed: theta {theta:.3e} after 20000 steps")

    #HYP on the gauge-fixed links: HYP is gauge covariant, so the propagators live in the same Coulomb gauge as the smearing
    deviceSynchronize()
    s = perf_counter()
    gauge_hyp = gauge.copy()
    del gauge
    core.getLogger().info(f"DOING HYP SMEARING")
    core.getLogger().info(f"plaq_hyp_before = {gauge_hyp.plaquette()}")
    gauge_hyp.hypSmear(1, 0.75, 0.6, 0.3, -1,True,True)
    deviceSynchronize()
    core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
    core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE
    dirac.loadGauge(gauge_hyp)

    for t_idx, t0 in enumerate(t_src[i_cfg]):
        if all_on_disk([part_name(cfg_dir, case, particle, cfg, t_idx, pt2) for case, particle in file_list for pt2 in fb_index]):
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
                    pf_phases_plain = mom_phase.getPhases(pf_plain_list, src_pos).conj()      # (99, 2, Lt_local, Lz, Ly, Lx//2)
                    #source weights exp(-i 2pi/Ls q.x_src) / n_spatial_src, one per q of each case
                    x_vec = np.array([x0, y0, z0])
                    src_phase_forward    = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / Ls * (q_forward_list    @ x_vec)) / n_spatial_src)
                    src_phase_sym_0xi    = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / Ls * (q_sym_0xi_list    @ x_vec)) / n_spatial_src)
                    src_phase_sym_non0xi = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / Ls * (q_sym_non0xi_list @ x_vec)) / n_spatial_src)
                    src_phase_asy_0xi    = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / Ls * (q_asy_0xi_list    @ x_vec)) / n_spatial_src)
                    src_phase_asy_non0xi = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / Ls * (q_asy_non0xi_list @ x_vec)) / n_spatial_src)
                    src_phase_plain      = cp.asarray(np.exp(src_phase_sign * 1j * 2 * np.pi / Ls * (q_plain_list      @ x_vec)) / n_spatial_src)

                    for smear_src in smear_list:
                        i_src_shape,src_shape = smear_src[0]
                        i_src_frac,src_frac   = smear_src[1]
                        rho_T_src, rho_z_src = shape_list[i_src_shape]
                        k_src = k_list[i_src_frac]

                        #SRC: boosted Gaussian in Coulomb gauge, -k for prop1 and +k for prop2
                        #prop1 is conjugated, and will be used for anti-quark
                        #prop2 is not conjugated, and will be used for quark
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"SRC SMEARING: {shape_names[i_src_shape]} rho_T {rho_T_src} rho_z {rho_z_src}, frac {mom_frac_list[i_src_frac]}")
                        prop1_inv = coulomb_boosted_source(latt_info, -k_src, src_pos, rho_T_src, rho_z_src)
                        prop2_inv = coulomb_boosted_source(latt_info, +k_src, src_pos, rho_T_src, rho_z_src)
                        deviceSynchronize()
                        core.getLogger().info(f"SOURCE SMEAR: {perf_counter() - s} secs")

                        #INVERT
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"SOLVING DIRAC EQ")
                        prop1_inv = core.invertPropagator(dirac, prop1_inv, mrhs=12)
                        prop2_inv = core.invertPropagator(dirac, prop2_inv, mrhs=12)
                        core.getLogger().info(f"INVERT 2 propagagors: {perf_counter() - s} secs")

                        for smear_snk in smear_list:

                            i_snk_shape,snk_shape = smear_snk[0]
                            i_snk_frac,snk_frac   = smear_snk[1]
                            rho_T_sink, rho_z_sink = shape_list[i_snk_shape]
                            k_sink = k_list[i_snk_frac]

                            #SINK: the same smearing with the sink parameters, from the unsmeared solution each time
                            deviceSynchronize()
                            s = perf_counter()
                            core.getLogger().info(f"SINK SMEARING: {shape_names[i_snk_shape]} rho_T {rho_T_sink} rho_z {rho_z_sink}, frac {mom_frac_list[i_snk_frac]}")
                            prop1 = coulomb_boosted_sink(latt_info, prop1_inv, -k_sink, rho_T_sink, rho_z_sink)
                            prop2 = coulomb_boosted_sink(latt_info, prop2_inv, +k_sink, rho_T_sink, rho_z_sink)
                            deviceSynchronize()
                            core.getLogger().info(f"SINK SMEAR: {perf_counter() - s} secs")

                            #CONTRACT
                            deviceSynchronize()
                            s = perf_counter()

                            for i_gamma_sink, gamma_sink in enumerate(gamma_list):

                                diquark = contract("def,gh,wtzyxhjeb,wtzyxgida->wtzyxijabf",eps_color, charge @ gamma_sink, prop2.data, prop2.data)

                                # pion: colors and sink gamma summed at each site, then sites summed with the phases of the 99 momenta
                                #       source spins l,i left open for the source gamma
                                pion_open_plain = contract("pwtzyx,wtzyxjiba,jk,wtzyxklba->lipt",
                                                           pf_phases_plain, prop1.data.conj(), G5 @ gamma_sink, prop2.data)          # (4,4,99,Lt_local)

                                # proton: colors and epsilon summed at each site, both Wick terms (step 1), then sites summed with the phases (step 2)
                                #         source spins i,j and sink spins l,k left open
                                proton_site  = contract("abc,wtzyxijabf,wtzyxlkfc->wtzyxijlk", eps_color, diquark, prop2.data)    # step 1, first term, 256 per site
                                proton_site -= contract("abc,wtzyxkjcbf,wtzyxlifa->wtzyxijlk", eps_color, diquark, prop2.data)   # step 1, second term
                                proton_open_plain = contract("pwtzyx,wtzyxijlk->ijlkpt", pf_phases_plain, proton_site)           # step 2, (4,4,4,4,99,Lt_local)

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
                    corr_slab[..., Lt - t0:] *= -1                                  # antiperiodic sign for the times that wrapped around
                #FORWARD AND BACKWARD: pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. tsep_max-1
                for pt2, t_index in fb_index.items():
                    fb_dic[pt2][key][...] = corr_slab[..., t_index]
                    part = part_name(cfg_dir, case, particle, cfg, t_idx, pt2)
                    with open(f"{part}.tmp", "wb") as f:
                        np.save(f, fb_dic[pt2][key])
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
            dim_head = ["shape_sink","shape_source","frac_sink","frac_source","gamma_sink","gamma_source"] + (["dirac_sink","dirac_source"] if particle == "proton" else [])
            dim_tail = ["t_src_list","q_list","momentum_list","tsep"] if case == "plain" else ["t_src_list","q_list","pz","tsep"]

            if latt_info.mpi_rank == 0:
                with h5py.File(f"{filename}.tmp", "w") as f:
                    f.create_dataset("q_list", data=np.array(q_case))
                    f.create_dataset("momentum_list", data=np.array(pf_case))                                   # p_f of every entry: [iq, iz] (cases) or [ip] (plain)
                    f.create_dataset("pz_list", data=np.array(pz_list))
                    f.create_dataset("shape_list", data=np.array(shape_list))
                    f.create_dataset("mom_frac_list", data=np.array(mom_frac_list))
                    f.create_dataset("k_list", data=np.array(k_list))
                    if particle == "proton":
                        f.create_dataset("parity_p", data=cp.asnumpy(parity_p))      # (1+g4)/2, for pt2_forward
                        f.create_dataset("parity_m", data=cp.asnumpy(parity_m))      # (1-g4)/2, for pt2_backward, enters with a minus sign
                    for pt2 in fb_index:                                             # pt2_forward and pt2_backward
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
                        dset.attrs["dim_time"] = fb_index[pt2]                       # time after the roll, (t - t_src) % Lt, of every tsep
                        if particle == "pion":
                            dset.attrs["time_reflection"] = "pt2_backward[..., tsep] = s_a s_b pt2_forward[..., tsep] for tsep >= 1: " + time_reflection_note
                            dset.attrs["propagator"] = "prop1 smeared with boost -k and prop2 with +k, at source and sink (quarks boosted toward physical +z)"
                        else:
                            dset.attrs["diquark"] = "C Gamma_sink at the sink, C Gamma_source at the source (equal to -Gamma_bar_source C for G5 and G45, an overall sign)"
                            dset.attrs["how_to_project"] = ("NOT projected. Positive parity: C_forward(tsep) = sum_kl parity_p[k,l] M[l,k] with M = pt2_forward, "
                                                            "C_backward(tsep) = -sum_kl parity_m[k,l] M[l,k] with M = pt2_backward, l = dirac_sink, k = dirac_source. "
                                                            "Antiperiodic sign already applied. pt2_backward[..., 0] is the source slice projected with parity_m: do not use it. "
                                                            "The time reflection holds for the projected correlators, not for M: C_backward(tsep) = s_a s_b C_forward(tsep) "
                                                            "for tsep >= 1: " + time_reflection_note)
                            dset.attrs["propagator"] = "prop2 (boost +k) for all three quark lines; source smearing = (shape_source, frac_source), sink smearing = (shape_sink, frac_sink)"

            if latt_info.mpi_rank == 0:
                with h5py.File(f"{filename}.tmp", "a") as f:
                    for pt2 in fb_index:
                        for t_idx in range(len(t_src[i_cfg])):                   # the slabs saved during the run, already rolled, signed and cut to tsep_max
                            f[pt2][..., t_idx, :, :, :] = np.load(part_name(cfg_dir, case, particle, cfg, t_idx, pt2))
                os.replace(f"{filename}.tmp", filename)

    if latt_info.mpi_rank == 0:
        shutil.rmtree(f"{cfg_dir}/parts")                   # only after every .h5 file of this cfg is complete
    core.getLogger().info(f"SAVING SECTION:{perf_counter()-saving_started} secs")
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    core.getLogger().info(f"END CFG #{cfg}: GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")
dirac.freeGauge()