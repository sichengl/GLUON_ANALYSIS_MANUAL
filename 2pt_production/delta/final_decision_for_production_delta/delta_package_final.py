#Packs the merged per-cfg files of the final run (2pt_coulomb_boosted_sepq_3072srcs_delta_final.py) into one pion and one proton file with a cfg axis,
#for the local analysis. Run it on Delta, CPU only, with the setup next to it: python3 delta_package_final.py
#Choose the momentum transfers in the block below. Only finished cfgs are packed: those with all 12 .h5 files on disk (the merge writes each file
#under its final name only when it is complete) that open with the same layout as the first one; the others are printed.
#It can run while the production runs. Running it again later packs every cfg finished by then into a new file (the cfg count is in the name).
import os
import sys
import numpy as np
import h5py

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..", "..", "tools")]        # the setup imports pt2_comm_tools from 2pt_production/tools
from sepq_2pt_setup_delta_final import delta_params, cfg_list, smear_tag, current_dir, n_spatial_src, smear_names


#======== what to pack ========
q_pack_list = [[0, 0, 0]]          # momentum transfers q, in units of 2pi/L; add more, e.g. [[0, 0, 0], [1, 0, 0], [0, 0, 1]]
frame = "asy"                      # frame of the q != 0 entries: "sym" (P = (p_f + p_i)/2 = (0,0,pz)) or "asy" (p_i = (0,0,pz)); q = 0 is the forward case
project_proton = False             # False: keep the proton's open Dirac indices; True: positive-parity projection, as in delta_package_smear_tune.py
save_dir = "/work/hdd/biqz/sliu54/2pt_production/packed/final_production"      # /work/hdd: /projects is too small


tsep_max = delta_params.tsep_max
n_src = len(delta_params.t_base) * n_spatial_src
gevp_dir = f"{current_dir}/{smear_tag}_GEVP_sepq_{n_src}src_phyp_fb_tsep{tsep_max}"     # gevp_dir of the job script
all_cases = ["forward", "sym_0xi", "sym_non0xi", "asy_0xi", "asy_non0xi", "plain"]
pt2_names = ["pt2_forward", "pt2_backward"]
os.makedirs(save_dir, exist_ok=True)

def h5_name(case, particle, cfg):
    return f"{gevp_dir}/cfg{cfg}/{particle}_{smear_tag}_GEVP_sepq_{case}_phyp_fb_tsep{tsep_max}_cfg{cfg}.h5"

#finished cfgs: the merge has written all 12 files
cfg_done = [int(cfg) for cfg in cfg_list if all(os.path.exists(h5_name(case, particle, cfg)) for case in all_cases for particle in ["pion", "proton"])]
print(f"{len(cfg_done)} of {len(cfg_list)} cfgs finished in {gevp_dir}")
print(f"not finished: {[int(cfg) for cfg in cfg_list if int(cfg) not in cfg_done]}")
assert cfg_done, "no cfg to pack"

#the case that holds each q: q = 0 from the forward case, the others from the cases of the chosen frame
q_pack_list = [[int(v) for v in q] for q in q_pack_list]
assert len({tuple(q) for q in q_pack_list}) == len(q_pack_list), f"a q appears twice in q_pack_list: {q_pack_list}"
frame_cases = {"sym": ["sym_0xi", "sym_non0xi"], "asy": ["asy_0xi", "asy_non0xi"]}[frame]
q_in_case = {}
with h5py.File(h5_name("forward", "pion", cfg_done[0]), "r") as f:
    q_in_case["forward"] = f["q_list"][:].tolist()
for case in frame_cases:
    with h5py.File(h5_name(case, "pion", cfg_done[0]), "r") as f:
        q_in_case[case] = f["q_list"][:].tolist()
q_case = []
for q in q_pack_list:
    found = [case for case in ["forward"] + frame_cases if q in q_in_case[case]]
    assert found, f"q = {q} is not in the {frame} frame; it has q = 0 and {[qq for case in frame_cases for qq in q_in_case[case]]}"
    q_case.append(found[0])

#per case: the q indices to read (increasing, as h5py needs) and where each one goes on the q axis of the packed arrays
read_plan = []
for case in dict.fromkeys(q_case):                                    # the cases in the order of q_pack_list, each once
    positions = [i for i, c in enumerate(q_case) if c == case]
    iq = [q_in_case[case].index(q_pack_list[i]) for i in positions]
    iq_sorted = sorted(iq)
    read_plan.append((case, iq_sorted, [iq_sorted.index(i) for i in iq], positions))

#layout, momenta and notes from the first cfg; every other cfg must have the same layout and q lists
ref_shape, ref_q = {}, {}
for case, _, _, _ in read_plan:
    for particle in ["pion", "proton"]:
        with h5py.File(h5_name(case, particle, cfg_done[0]), "r") as f:
            ref_shape[case, particle] = f["pt2_forward"].shape
            ref_q[case] = f["q_list"][:]
pf_table = np.zeros((len(q_pack_list), len(delta_params.pz_list), 3), int)    # p_f of every packed entry [iq, iz]
for case, iq_sorted, reorder, positions in read_plan:
    with h5py.File(h5_name(case, "pion", cfg_done[0]), "r") as f:
        pf_table[positions] = f["momentum_list"][:][iq_sorted][reorder]
        pion_attrs = dict(f["pt2_forward"].attrs)
with h5py.File(h5_name("forward", "proton", cfg_done[0]), "r") as f:
    parity_p = f["parity_p"][:]                                       # (1 + gamma4) / 2
    parity_m = f["parity_m"][:]                                       # (1 - gamma4) / 2
    proton_attrs = dict(f["pt2_forward"].attrs)
n_tsrc = ref_shape[read_plan[0][0], "pion"][4]                     # (smear, smear, g, g, t_src, q, pz, tsep)

#keep only the cfgs whose files open with that layout (a damaged or half-copied file is listed and left out)
cfg_pack = []
for cfg in cfg_done:
    try:
        for case, _, _, _ in read_plan:
            for particle in ["pion", "proton"]:
                with h5py.File(h5_name(case, particle, cfg), "r") as f:
                    assert all(f[pt2].shape == ref_shape[case, particle] for pt2 in pt2_names), "dataset shape differs"
                    assert np.array_equal(f["q_list"][:], ref_q[case]), "q_list differs"
        cfg_pack.append(cfg)
    except Exception as err:
        print(f"cfg {cfg} left out: {type(err).__name__}: {err}")
n_cfg = len(cfg_pack)
print(f"packing {n_cfg} cfgs, q = {q_pack_list} from {q_case}")

#packed arrays: (cfg, smear_sink, smear_source, g_snk, g_src, [dirac_snk, dirac_src,] t_src, q, pz, tsep); smear i = row i of smear_list
n_smear, n_q, n_pz, n_gamma = len(delta_params.smear_list), len(q_pack_list), len(delta_params.pz_list), len(delta_params.gamma_ids)
pion_shape = (n_smear, n_smear, n_gamma, n_gamma, n_tsrc, n_q, n_pz, tsep_max)
proton_shape = (n_smear, n_smear, n_gamma, n_gamma) + (() if project_proton else (4, 4)) + (n_tsrc, n_q, n_pz, tsep_max)
q_tag = "-".join("".join(f"m{-v}" if v < 0 else f"{v}" for v in q) for q in q_pack_list) if n_q <= 4 else f"{n_q}q"
name_tag = f"{smear_tag}_GEVP_q{q_tag}_{frame}_fb_tsep{tsep_max}_{n_src}src_ncfg{n_cfg}"
filenames = {particle: f"{save_dir}/2pt_{name_tag}_{particle}{'_projected' if particle == 'proton' and project_proton else ''}.h5" for particle in ["pion", "proton"]}
out = {particle: h5py.File(f"{filenames[particle]}.tmp", "w") for particle in ["pion", "proton"]}     # renamed when complete
dsets = {(particle, pt2): out[particle].create_dataset(pt2, shape=(n_cfg,) + shape, dtype="<c16")
         for particle, shape in [("pion", pion_shape), ("proton", proton_shape)] for pt2 in pt2_names}
src_lists = {name: np.zeros((n_cfg, n), int) for name, n in [("t_src_list", n_tsrc), ("x_src_list", len(delta_params.x_base)),
                                                              ("y_src_list", len(delta_params.y_base)), ("z_src_list", len(delta_params.z_base))]}

for icfg, cfg in enumerate(cfg_pack):
    print("reading cfg", cfg, flush=True)
    for particle, shape in [("pion", pion_shape), ("proton", (n_smear, n_smear, n_gamma, n_gamma, 4, 4, n_tsrc, n_q, n_pz, tsep_max))]:
        corr = {pt2: np.zeros(shape, "<c16") for pt2 in pt2_names}
        for case, iq_sorted, reorder, positions in read_plan:
            with h5py.File(h5_name(case, particle, cfg), "r") as f:
                if particle == "pion" and case == read_plan[0][0]:
                    for name in src_lists:
                        src_lists[name][icfg] = f["pt2_forward"].attrs[name]
                if particle == "proton":
                    assert np.array_equal(f["pt2_forward"].attrs["t_src_list"], src_lists["t_src_list"][icfg]), f"cfg {cfg}: pion and proton t_src differ"
                for pt2 in pt2_names:
                    data = f[pt2][..., iq_sorted, :, :][..., reorder, :, :]      # (smear_snk, smear_src, g_snk, g_src, [dirac,] t_src, q, pz, tsep)
                    corr[pt2][..., positions, :, :] = data
        if particle == "proton" and project_proton:
            corr["pt2_forward"] = np.einsum("kl,abcdlktqpT->abcdtqpT", parity_p, corr["pt2_forward"])         # sum_kl parity_p[k,l] M[l,k]
            corr["pt2_backward"] = -np.einsum("kl,abcdlktqpT->abcdtqpT", parity_m, corr["pt2_backward"])      # -sum_kl parity_m[k,l] M[l,k]
        for pt2 in pt2_names:
            dsets[particle, pt2][icfg] = corr[pt2]

dirac = [] if project_proton else ["dirac_sink", "dirac_source"]
common_attrs = {
    "gamma_list": pion_attrs["gamma_list"],
    "smearing": pion_attrs["smearing"] + ". Packed smearing axes: smear_sink, smear_source = 0 or 1, rows of the smear_list dataset [rho_T, rho_z, mom_frac]",
    "gauge_fixing": pion_attrs["gauge_fixing"],
    "momentum_convention": pion_attrs["momentum_convention"],
    "momentum_transfer": pion_attrs["momentum_transfer"],
    "spatial_sources": pion_attrs["spatial_sources"],
    "time_index": pion_attrs["time_index"],
    "smear_list": pion_attrs["smear_list"],                           # rows [rho_T, rho_z, mom_frac] of smear_sink and smear_source
    "kinematics": (f"q_list[iq] packed from kinematic_case_list[iq]; frame {frame}: pf_list[iq, iz] = p_f, pi_list[iq, iz] = p_i = p_f - q, "
                   "pz = pz_list[iz] (forward: p_f = p_i = (0,0,pz); sym: P = (p_f + p_i)/2 = (0,0,pz); asy: p_i = (0,0,pz))"),
    "source_files": gevp_dir,
}
for particle in ["pion", "proton"]:
    f = out[particle]
    f.create_dataset("cfg_list", data=np.array(cfg_pack))
    for name in src_lists:
        f.create_dataset(name, data=src_lists[name])
    f.create_dataset("q_list", data=np.array(q_pack_list))
    f.create_dataset("kinematic_case_list", data=np.array(q_case, dtype=h5py.string_dtype()))
    f.create_dataset("pf_list", data=pf_table)
    f.create_dataset("pi_list", data=pf_table - np.array(q_pack_list)[:, None])
    f.create_dataset("pz_list", data=np.array(delta_params.pz_list))
    f.create_dataset("smear_list", data=pion_attrs["smear_list"])     # rows [rho_T, rho_z, mom_frac], as the production saved them
    f.create_dataset("smear_names", data=np.array(smear_names, dtype=h5py.string_dtype()))
    for key in common_attrs:
        f.attrs[key] = common_attrs[key]
    if particle == "pion":
        f.attrs["dim_pt2"] = "cfg, smear_sink, smear_source, gamma_sink, gamma_source, t_src, q, pz, tsep"
        f.attrs["time_reflection"] = pion_attrs["time_reflection"]
        f.attrs["propagator"] = pion_attrs["propagator"]
    else:
        f.create_dataset("parity_p", data=parity_p)
        f.create_dataset("parity_m", data=parity_m)
        f.attrs["dim_pt2"] = ", ".join(["cfg", "smear_sink", "smear_source", "gamma_sink", "gamma_source"] + dirac + ["t_src", "q", "pz", "tsep"])
        f.attrs["diquark"] = proton_attrs["diquark"]
        f.attrs["propagator"] = proton_attrs["propagator"]
        f.attrs["projection"] = ("positive parity: forward sum_kl parity_p[k,l] M[l,k], backward -sum_kl parity_m[k,l] M[l,k] (l = dirac_sink, k = dirac_source); "
                                 "backward tsep = 0 is the source slice projected with parity_m, do not use it") if project_proton else proton_attrs["how_to_project"]
    f.close()
    os.replace(f"{filenames[particle]}.tmp", filenames[particle])
    print("saved", filenames[particle])
