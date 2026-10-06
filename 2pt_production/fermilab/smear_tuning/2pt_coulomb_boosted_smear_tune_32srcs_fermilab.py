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
from coulomb_smearing import coulomb_boosted_source, coulomb_boosted_sink, coulomb_gauge_theta
from pt2_comm_tools import make_eps           # tools/pt2_comm_tools.py
from smear_tune_2pt_setup_fermilab import * # fermilab_params (the TwoPtParams of this run), cfg list, smear_list, momenta, notes, paths

#Smearing tuning: every smearing of smear_list ([rho_T, rho_z, mom_frac]) is used at source and sink, and only these diagonal correlators are
#computed, so a source costs one pair of inversions and one contraction per smearing. Pion and proton, G5 and G5G4 at sink and source (2 x 2),
#q = 0 at the 99 sink momenta of pf_plain_list, the spatial sources of each t_src averaged. The contraction is the plain case of
#2pt_coulomb_boosted_sepq_1024srcs_fermilab_tsrcsave_fb_classused.py, with the smearing pair (i, i) only.


parser = argparse.ArgumentParser()
parser.add_argument("--icfg", type=int, default=fermilab_params.icfg)      # position in cfg_list of the first cfg of this job: --icfg 3 measures cfg_list[3] = 294
parser.add_argument("--n", type=int, default=fermilab_params.n_cfg)        # number of configs to measure in this one job
args = parser.parse_args()


#======== configurations and sources of this job ========
icfg0 = args.icfg
n = args.n
measurement_list = cfg_list[icfg0::cfg_measure_spacing][:n]
t_src, x_src, y_src, z_src = fermilab_params.make_sources(measurement_list)


#======== QUDA initialization ========
fermilab_params.init_quda(quda_resource_path)
latt_info = fermilab_params.make_latt_info()
if latt_info.mpi_rank == 0:
    print({"GLs": fermilab_params.GLs, "GLt": fermilab_params.GLt, "cfgs_to_meas": measurement_list, "mom_max": fermilab_params.mom_max,
           "x_src_list_shifted": x_src, "y_src_list_shifted": y_src, "z_src_list_shifted": z_src, "t_src_list_shifted": t_src,
           "pf_plain_list": fermilab_params.pf_plain_list, "tsep_max": fermilab_params.tsep_max, "smear_tag": smear_tag})
    for i_smear, (rho_T, rho_z, frac) in enumerate(smear_list):
        print(f"smearing {i_smear:2d}: {smear_names[i_smear]:24s} rho_T {rho_T:.4g} rho_z {rho_z:.4g} mom_frac {frac:.4g} k {smear_k_list[i_smear]}")
dirac = fermilab_params.make_dirac(latt_info)


#======== gamma matrices, epsilon tensor, containers (cupy arrays: after core.init) ========
gamma_list, gamma_names = fermilab_params.make_gamma_list()                   # the interpolators Gamma(n), n in fermilab_params.gamma_ids, and their names
G4, G5, charge = gamma.gamma(8), gamma.gamma(15), gamma.gamma(10)     # g4, g5 and C = g2 g4, in PyQUDA's DeGrand-Rossi convention
parity_p = (gamma.gamma(0) + gamma.gamma(8)) / 2                      # (1 + g4) / 2
parity_m = (gamma.gamma(0) - gamma.gamma(8)) / 2                      # (1 - g4) / 2
eps_color = make_eps()
n_smear, n_gamma, n_pf = len(smear_list), len(gamma_list), len(fermilab_params.pf_plain_list)
#one t_src at a time: (smear, gamma_sink, gamma_source, [dirac_sink, dirac_source,] p_f, local time)
corr_dic = {
    "pion":   cp.zeros((n_smear, n_gamma, n_gamma,       n_pf, latt_info.Lt), "<c16"),
    "proton": cp.zeros((n_smear, n_gamma, n_gamma, 4, 4, n_pf, latt_info.Lt), "<c16"),
}
pion_plain, proton_plain_dirac = corr_dic["pion"], corr_dic["proton"]
#forward and backward 2pt of one t_src on the host of rank 0, tsep_max time slices instead of GLt
fb_dic = {pt2: {particle: np.zeros(corr.shape[:-1] + (fermilab_params.tsep_max,), "<c16") for particle, corr in corr_dic.items()}
          for pt2 in fermilab_params.fb_index} if latt_info.mpi_rank == 0 else None
mom_phase = phase.MomentumPhase(latt_info)


#======== output: one pion and one proton file per cfg ========
gevp_dir = f"{current_dir}/{smear_tag}_{len(fermilab_params.t_base) * n_spatial_src}src_phyp_fb_tsep{fermilab_params.tsep_max}"

def h5_name(cfg_dir, particle, cfg):
    return f"{cfg_dir}/{particle}_{smear_tag}_phyp_fb_tsep{fermilab_params.tsep_max}_cfg{cfg}.h5"

def part_name(cfg_dir, particle, cfg, t_idx, pt2):
    return f"{cfg_dir}/parts/{particle}_{smear_tag}_phyp_fb_tsep{fermilab_params.tsep_max}_cfg{cfg}_t{t_idx}_{pt2}.npy"

def all_on_disk(names):
    #rank 0 checks and tells the other ranks, so that every rank skips the same work (gatherLattice is collective)
    found = all(os.path.exists(name) for name in names) if latt_info.mpi_rank == 0 else None
    return latt_info.mpi_comm.bcast(found, root=0)

#the folder keeps the smear_list it was made with: a job with another list would skip the parts and files of these operators as if they were its own
if latt_info.mpi_rank == 0:
    os.makedirs(gevp_dir, exist_ok=True)
    smear_file = f"{gevp_dir}/smear_list.npy"
    if os.path.exists(smear_file):
        if not np.array_equal(np.load(smear_file), np.array(smear_list)):
            raise RuntimeError(f"{smear_file} holds another smear_list: set a new smear_tag in the setup for a new set of operators")
    else:
        tmp = f"{smear_file}.{socket.gethostname()}.{os.getpid()}.tmp"     # array tasks may start together: each writes its own file, then renames it
        with open(tmp, "wb") as f:
            np.save(f, np.array(smear_list))
        os.replace(tmp, smear_file)

for i_cfg, cfg in tqdm(enumerate(measurement_list),desc=f"Processing cfgs"):

    cfg_dir = f"{gevp_dir}/cfg{cfg}"
    if all_on_disk([h5_name(cfg_dir, particle, cfg) for particle in corr_dic]):
        core.getLogger().info(f"SKIP CFG #{cfg}: its .h5 files are already saved")
        continue
    if latt_info.mpi_rank == 0:
        os.makedirs(f"{cfg_dir}/parts", exist_ok=True)      # also creates cfg_dir

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
    gauge.fixingOVR(gauge_dir=fermilab_params.gauge_fix_dir, Nsteps=fermilab_params.gauge_fix_Nsteps, verbose_interval=fermilab_params.gauge_fix_verbose_interval,
                    relax_boost=fermilab_params.gauge_fix_relax_boost, tolerance=fermilab_params.gauge_fix_tolerance,
                    reunit_interval=fermilab_params.gauge_fix_reunit_interval, stopWtheta=fermilab_params.gauge_fix_stopWtheta)     # overrelaxation, since QUDA's FFT method runs on one GPU only
    theta = coulomb_gauge_theta(latt_info, gauge)
    deviceSynchronize()
    core.getLogger().info(f"COULOMB GAUGE FIXING: {perf_counter() - s} secs, theta {theta:.3e}")
    if theta > 1e-6:                                # QUDA stops silently after Nsteps; cfg 204 ends at 7.8e-9 on both machines, 1e-12 is never reached
        raise RuntimeError(f"Coulomb gauge fixing failed: theta {theta:.3e} after {fermilab_params.gauge_fix_Nsteps} steps")

    #HYP on the gauge-fixed links: HYP is gauge covariant, so the propagators live in the same Coulomb gauge as the smearing
    deviceSynchronize()
    s = perf_counter()
    gauge_hyp = gauge.copy()
    del gauge
    core.getLogger().info(f"DOING HYP SMEARING")
    core.getLogger().info(f"plaq_hyp_before = {gauge_hyp.plaquette()}")
    gauge_hyp.hypSmear(fermilab_params.hyp_n_steps, fermilab_params.hyp_alpha1, fermilab_params.hyp_alpha2, fermilab_params.hyp_alpha3, fermilab_params.hyp_dir_ignore,
                       fermilab_params.hyp_compute_plaquette, fermilab_params.hyp_compute_qcharge)
    deviceSynchronize()
    core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
    core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE
    dirac.loadGauge(gauge_hyp)

    for t_idx, t0 in enumerate(t_src[i_cfg]):
        if all_on_disk([part_name(cfg_dir, particle, cfg, t_idx, pt2) for particle in corr_dic for pt2 in fermilab_params.fb_index]):
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
                    pf_phases_plain = mom_phase.getPhases(fermilab_params.pf_plain_list, src_pos).conj()      # (99, 2, Lt_local, Lz, Ly, Lx//2)

                    for i_smear, (rho_T, rho_z, frac) in enumerate(smear_list):
                        k_smear = smear_k_list[i_smear]

                        #SRC: boosted Gaussian in Coulomb gauge, -k for prop1 and +k for prop2
                        #prop1 is conjugated, and will be used for anti-quark
                        #prop2 is not conjugated, and will be used for quark
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"SRC SMEARING {i_smear}: {smear_names[i_smear]}, rho_T {rho_T} rho_z {rho_z}, frac {frac}")
                        prop1_inv = coulomb_boosted_source(latt_info, -k_smear, src_pos, rho_T, rho_z)
                        prop2_inv = coulomb_boosted_source(latt_info, +k_smear, src_pos, rho_T, rho_z)
                        deviceSynchronize()
                        core.getLogger().info(f"SOURCE SMEAR: {perf_counter() - s} secs")

                        #INVERT
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"SOLVING DIRAC EQ")
                        prop1_inv = core.invertPropagator(dirac, prop1_inv, mrhs=fermilab_params.mrhs)
                        prop2_inv = core.invertPropagator(dirac, prop2_inv, mrhs=fermilab_params.mrhs)
                        core.getLogger().info(f"INVERT 2 propagagors: {perf_counter() - s} secs")

                        #SINK: the same smearing as the source
                        deviceSynchronize()
                        s = perf_counter()
                        prop1 = coulomb_boosted_sink(latt_info, prop1_inv, -k_smear, rho_T, rho_z)
                        prop2 = coulomb_boosted_sink(latt_info, prop2_inv, +k_smear, rho_T, rho_z)
                        del prop1_inv, prop2_inv                        # one sink per source smearing: the unsmeared solutions are not needed again
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
                            #         source spins i,j (for C Gamma_source below) and k, sink spin l left open
                            proton_site  = contract("abc,wtzyxijabf,wtzyxlkfc->wtzyxijlk", eps_color, diquark, prop2.data)    # step 1, first term, 256 per site
                            proton_site -= contract("abc,wtzyxkjcbf,wtzyxlifa->wtzyxijlk", eps_color, diquark, prop2.data)   # step 1, second term
                            proton_open_plain = contract("pwtzyx,wtzyxijlk->ijlkpt", pf_phases_plain, proton_site)           # step 2, (4,4,4,4,99,Lt_local)

                            for i_gamma_source, gamma_source in enumerate(gamma_list):

                                gamma_source_bar = G4 @ gamma_source.conj().T @ G4

                                #plain mean over the spatial sources of this t_src
                                pion_plain[i_smear, i_gamma_sink, i_gamma_source] += contract(
                                "li,lipt->pt", gamma_source_bar @ G5, pion_open_plain) / n_spatial_src                       # (99, Lt_local)
                                proton_plain_dirac[i_smear, i_gamma_sink, i_gamma_source] += contract(
                                "ij,ijlkpt->lkpt", charge @ gamma_source, proton_open_plain) / n_spatial_src                 # (4, 4, 99, Lt_local)

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
        for particle, corr in corr_dic.items():
            corr_slab = core.gatherLattice(corr.get(), [corr.ndim - 1, -1, -1, -1])                          # ALL ranks; time is the last axis
            if latt_info.mpi_rank == 0:
                #ROLL so that time index 0 is t_src:  rolled[..., tau] = C[..., (t_src + tau) % 96]
                corr_slab = np.roll(corr_slab, -t0, axis=-1)
                if particle == "proton":
                    corr_slab[..., fermilab_params.GLt - t0:] *= -1                                  # antiperiodic sign for the times that wrapped around
                #FORWARD AND BACKWARD: pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. tsep_max-1
                for pt2, t_index in fermilab_params.fb_index.items():
                    fb_dic[pt2][particle][...] = corr_slab[..., t_index]
                    part = part_name(cfg_dir, particle, cfg, t_idx, pt2)
                    with open(f"{part}.tmp", "wb") as f:
                        np.save(f, fb_dic[pt2][particle])
                    os.replace(f"{part}.tmp", part)
            del corr_slab
        core.getLogger().info(f"SAVE T_SRC {t0}: {perf_counter() - s} secs")

    #MERGE: rank 0 builds the .h5 files from the parts, then deletes the parts. Each file has two datasets, pt2_forward and pt2_backward.
    #Files are written as .tmp and renamed when complete, so a job killed while merging leaves no half-filled file under the final name.
    deviceSynchronize()
    saving_started = perf_counter()
    for particle in corr_dic:
        filename = h5_name(cfg_dir, particle, cfg)
        dim_head = ["smear","gamma_sink","gamma_source"] + (["dirac_sink","dirac_source"] if particle == "proton" else [])
        dim_tail = ["t_src_list","momentum_list","tsep"]

        if latt_info.mpi_rank == 0:
            with h5py.File(f"{filename}.tmp", "w") as f:
                f.create_dataset("momentum_list", data=np.array(fermilab_params.pf_plain_list))           # p_f of every entry
                f.create_dataset("smear_list", data=np.array(smear_list))                                 # rows [rho_T, rho_z, mom_frac]
                f.create_dataset("k_list", data=np.array(smear_k_list))                                   # boost of every smearing, units of 2pi/L
                if particle == "proton":
                    f.create_dataset("parity_p", data=cp.asnumpy(parity_p))      # (1+g4)/2, for pt2_forward
                    f.create_dataset("parity_m", data=cp.asnumpy(parity_m))      # (1-g4)/2, for pt2_backward, enters with a minus sign
                for pt2 in fermilab_params.fb_index:                                             # pt2_forward and pt2_backward
                    fb = fb_dic[pt2][particle]
                    dset = f.create_dataset(pt2, shape=fb.shape[:-2] + (len(t_src[i_cfg]),) + fb.shape[-2:], dtype="<c16")      # the t_src index is put back here; filled per t_src below
                    dset.attrs["kinematic_case"] = "plain"
                    dset.attrs["frame"] = "plain 2pt, q = 0, at every p_f of momentum_list"
                    dset.attrs["dim_spec"] = np.array(dim_head + dim_tail, dtype=h5py.string_dtype())
                    dset.attrs["gamma_list"] = np.array(gamma_names, dtype=h5py.string_dtype())
                    dset.attrs["smear_names"] = np.array(smear_names, dtype=h5py.string_dtype())
                    dset.attrs["smear_list"] = "rows of smear_list dataset: [rho_T, rho_z, mom_frac]; the same smearing at sink and source, smear index i means the pair (i, i)"
                    dset.attrs["mom_frac"] = f"quark boost k = mom_frac * mom_max = mom_frac * {fermilab_params.mom_max} in units of 2pi/L along z"
                    dset.attrs["spatial_sources"] = source_note
                    dset.attrs["smearing"] = smearing_note
                    dset.attrs["gauge_fixing"] = gauge_fix_note
                    dset.attrs["momentum_convention"] = momentum_note
                    dset.attrs["gauge_file"] = gauge_file
                    dset.attrs["measurements"] = [cfg]
                    dset.attrs["x_src_list"] = x_src[i_cfg]
                    dset.attrs["y_src_list"] = y_src[i_cfg]
                    dset.attrs["z_src_list"] = z_src[i_cfg]
                    dset.attrs["t_src_list"] = t_src[i_cfg]
                    dset.attrs["time_index"] = time_index_note
                    dset.attrs["dim_time"] = fermilab_params.fb_index[pt2]                       # time after the roll, (t - t_src) % Lt, of every tsep
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
                        dset.attrs["propagator"] = "prop2 (boost +k) for all three quark lines; the same smearing at source and sink"

            with h5py.File(f"{filename}.tmp", "a") as f:
                for pt2 in fermilab_params.fb_index:
                    for t_idx in range(len(t_src[i_cfg])):                   # the slabs saved during the run, already rolled, signed and cut to tsep_max
                        f[pt2][..., t_idx, :, :] = np.load(part_name(cfg_dir, particle, cfg, t_idx, pt2))
            os.replace(f"{filename}.tmp", filename)

    if latt_info.mpi_rank == 0:
        shutil.rmtree(f"{cfg_dir}/parts")                   # only after every .h5 file of this cfg is complete
    core.getLogger().info(f"SAVING SECTION:{perf_counter()-saving_started} secs")
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    core.getLogger().info(f"END CFG #{cfg}: GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")
dirac.freeGauge()
