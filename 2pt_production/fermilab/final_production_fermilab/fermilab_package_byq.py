#Packs the merged per-cfg files of the final run (2pt_coulomb_boosted_sepq_3072srcs_fermilab_final.py) per frame and momentum transfer q, for the local
#analysis. For every (frame, q) of pack_list it writes three files per particle of particle_list, each with a cfg axis:
#  C3input: the 2pt of the case that holds q (source weight exp(-i 2pi/Ls q.x_src)); the input of the 3pt with the FF of the same q.
#           Its ensemble mean is zero for q != 0, so it is not a 2pt for the ratio.
#  C2pf:    the plain q = 0 2pt at the sink momenta p_f(pz) of this (frame, q): denominators of the ratio and the GEVP at the sink
#  C2pi:    the plain q = 0 2pt at the source momenta p_i(pz) = p_f(pz) - q: denominators of the ratio and the GEVP at the source
#Frame "forward" (q = 0) writes the C3input file only: at q = 0 that file is also the C2pf and the C2pi file.
#Every file holds one q and has no q axis: to get a q axis, stack the C3input files of several (frame, q).
#Run it on Fermilab, CPU only, with the setup next to it: python3 fermilab_package_byq.py (the Fermilab copy of delta_package_byq.py)
#Only finished cfgs are packed: those with all 12 .h5 files on disk (the merge writes each file under its final name only when it is complete)
#that open with the same layout as the first one; the others are printed. Running it again later packs every cfg finished by then into new files.
import os
import sys
import numpy as np
import h5py

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..", "..", "tools")]        # the setup imports pt2_comm_tools from 2pt_production/tools
from sepq_2pt_setup_fermilab_final import delta_params, cfg_list, smear_tag, current_dir, n_spatial_src, smear_names


#======== what to pack ========
pack_list = [["sym", [2, 0, 0]], ["sym", [-2, 0, 0]], ["sym", [0, 2, 0]], ["sym", [0, -2, 0]]]   # (frame, q), q in units of 2pi/L, one set of files per entry; e.g. [["sym", [2, 0, 0]], ["asy", [1, 0, 0]], ["forward", [0, 0, 0]]]
                                   # frame "sym": P = (p_f + p_i)/2 = (0,0,pz); "asy": p_i = (0,0,pz); "forward": q = 0, p_f = p_i = (0,0,pz)
particle_list = ["pion"]           # ["pion"] or ["pion", "proton"]: the particles to pack (the pion files also give the source positions)
project_proton = False             # False: keep the proton's open Dirac indices; True: positive-parity projection (described in the projection attribute)
save_dir = "/lustre2/gluonp0/sliu1/2pt_production/packed/final_production_byq"      # lustre2, next to the per-cfg files


tsep_max = delta_params.tsep_max
n_src = len(delta_params.t_base) * n_spatial_src
gevp_dir = f"{current_dir}/{smear_tag}_GEVP_sepq_{n_src}src_phyp_fb_tsep{tsep_max}"     # gevp_dir of the job script
all_cases = ["forward", "sym_0xi", "sym_non0xi", "asy_0xi", "asy_non0xi", "plain"]
frame_cases = {"forward": ["forward"], "sym": ["sym_0xi", "sym_non0xi"], "asy": ["asy_0xi", "asy_non0xi"]}
pt2_names = ["pt2_forward", "pt2_backward"]
assert particle_list in (["pion"], ["pion", "proton"]), f"particle_list {particle_list}: use ['pion'] or ['pion', 'proton']"
os.makedirs(save_dir, exist_ok=True)

def h5_name(case, particle, cfg):
    return f"{gevp_dir}/cfg{cfg}/{particle}_{smear_tag}_GEVP_sepq_{case}_phyp_fb_tsep{tsep_max}_cfg{cfg}.h5"

#finished cfgs: the merge has written all 12 files
cfg_done = [int(cfg) for cfg in cfg_list if all(os.path.exists(h5_name(case, particle, cfg)) for case in all_cases for particle in ["pion", "proton"])]
print(f"{len(cfg_done)} of {len(cfg_list)} cfgs finished in {gevp_dir}")
print(f"not finished: {[int(cfg) for cfg in cfg_list if int(cfg) not in cfg_done]}")
assert cfg_done, "no cfg to pack"

#q lists of every case and the momenta of the plain case, from the first cfg
q_in_case = {}
for case in all_cases:
    with h5py.File(h5_name(case, "pion", cfg_done[0]), "r") as f:
        q_in_case[case] = f["q_list"][:].tolist()
        if case == "plain":
            p_plain = f["momentum_list"][:].tolist()                  # (99, 3): momenta of the plain 2pt, q = 0

#read plan, one entry per output file (per particle): the case to read, its q index, and for the plain case the momentum indices of p_f(pz) or p_i(pz)
pack_list = [[frame, [int(v) for v in q]] for frame, q in pack_list]
assert len({(frame, tuple(q)) for frame, q in pack_list}) == len(pack_list), f"an entry appears twice in pack_list: {pack_list}"
read_plan = []          # (frame, q, role, case, iq, ip, pf, pi): ip = None keeps the pz axis of the case
for frame, q in pack_list:
    assert frame in frame_cases, f"frame {frame}: use 'sym', 'asy' or 'forward'"
    assert (frame == "forward") == (q == [0, 0, 0]), f"({frame}, {q}): q = [0, 0, 0] is the forward frame, and the forward frame has q = [0, 0, 0] only"
    found = [case for case in frame_cases[frame] if q in q_in_case[case]]
    assert found, f"q = {q} is not in the {frame} frame; it has {[qq for case in frame_cases[frame] for qq in q_in_case[case]]}"
    case = found[0]
    iq = q_in_case[case].index(q)
    with h5py.File(h5_name(case, "pion", cfg_done[0]), "r") as f:
        pf = f["momentum_list"][iq]                                   # (pz, 3): p_f of every pz of this q
    pi = pf - np.array(q)                                             # p_i = p_f - q (physical labels of the final 2pt)
    read_plan.append((frame, q, "C3input", case, iq, None, pf, pi))
    if frame != "forward":
        for role, momenta in [("C2pf", pf), ("C2pi", pi)]:
            missing = [p for p in momenta.tolist() if p not in p_plain]
            assert not missing, f"({frame}, {q}) {role}: momenta {missing} are not in the plain case"
            read_plan.append((frame, q, role, "plain", 0, [p_plain.index(p) for p in momenta.tolist()], pf, pi))
cases_used = list(dict.fromkeys(case for _, _, _, case, _, _, _, _ in read_plan))

#layout, momenta and notes from the first cfg; every other cfg must have the same layout, q lists and momenta
ref_shape, ref_q, ref_p = {}, {}, {}
for case in cases_used:
    for particle in ["pion", "proton"]:
        with h5py.File(h5_name(case, particle, cfg_done[0]), "r") as f:
            ref_shape[case, particle] = f["pt2_forward"].shape
            ref_q[case] = f["q_list"][:]
            ref_p[case] = f["momentum_list"][:]
with h5py.File(h5_name("forward", "pion", cfg_done[0]), "r") as f:
    pion_attrs = dict(f["pt2_forward"].attrs)
with h5py.File(h5_name("forward", "proton", cfg_done[0]), "r") as f:
    parity_p = f["parity_p"][:]                                       # (1 + gamma4) / 2
    parity_m = f["parity_m"][:]                                       # (1 - gamma4) / 2
    proton_attrs = dict(f["pt2_forward"].attrs)
n_tsrc = ref_shape[cases_used[0], "pion"][4]                    # (smear, smear, g, g, t_src, q, pz or p, tsep)

#keep only the cfgs whose files open with that layout (a damaged or half-copied file is listed and left out)
cfg_pack = []
for cfg in cfg_done:
    try:
        for case in cases_used:
            for particle in ["pion", "proton"]:
                with h5py.File(h5_name(case, particle, cfg), "r") as f:
                    assert all(f[pt2].shape == ref_shape[case, particle] for pt2 in pt2_names), "dataset shape differs"
                    assert np.array_equal(f["q_list"][:], ref_q[case]), "q_list differs"
                    assert np.array_equal(f["momentum_list"][:], ref_p[case]), "momentum_list differs"
        cfg_pack.append(cfg)
    except Exception as err:
        print(f"cfg {cfg} left out: {type(err).__name__}: {err}")
n_cfg = len(cfg_pack)
print(f"packing {n_cfg} cfgs, (frame, q) = {pack_list}")

#packed arrays: (cfg, smear_sink, smear_source, g_snk, g_src, [dirac_snk, dirac_src,] t_src, pz, tsep); smear i = row i of smear_list
n_smear, n_pz, n_gamma = len(delta_params.smear_list), len(delta_params.pz_list), len(delta_params.gamma_ids)
pion_shape = (n_smear, n_smear, n_gamma, n_gamma, n_tsrc, n_pz, tsep_max)
proton_shape = (n_smear, n_smear, n_gamma, n_gamma) + (() if project_proton else (4, 4)) + (n_tsrc, n_pz, tsep_max)

def file_name(frame, q, role, particle):
    name_tag = f"{smear_tag}_GEVP_{frame}_q{q[0]}_{q[1]}_{q[2]}_{role}_fb_tsep{tsep_max}_{n_src}src_ncfg{n_cfg}"
    return f"{save_dir}/2pt_{name_tag}_{particle}{'_projected' if particle == 'proton' and project_proton else ''}.h5"

filenames = {(i, particle): file_name(frame, q, role, particle) for i, (frame, q, role, _, _, _, _, _) in enumerate(read_plan) for particle in particle_list}
out = {key: h5py.File(f"{filenames[key]}.tmp", "w") for key in filenames}      # renamed when complete
dsets = {(key, pt2): out[key].create_dataset(pt2, shape=(n_cfg,) + (pion_shape if key[1] == "pion" else proton_shape), dtype="<c16")
         for key in out for pt2 in pt2_names}
src_lists = {name: np.zeros((n_cfg, n), int) for name, n in [("t_src_list", n_tsrc), ("x_src_list", len(delta_params.x_base)),
                                                              ("y_src_list", len(delta_params.y_base)), ("z_src_list", len(delta_params.z_base))]}

for icfg, cfg in enumerate(cfg_pack):
    print("reading cfg", cfg, flush=True)
    for particle, shape in [("pion", pion_shape), ("proton", (n_smear, n_smear, n_gamma, n_gamma, 4, 4, n_tsrc, n_pz, tsep_max))]:
        if particle not in particle_list:
            continue
        for i, (frame, q, role, case, iq, ip, pf, pi) in enumerate(read_plan):
            corr = {pt2: np.zeros(shape, "<c16") for pt2 in pt2_names}
            with h5py.File(h5_name(case, particle, cfg), "r") as f:
                if particle == "pion" and i == 0:
                    for name in src_lists:
                        src_lists[name][icfg] = f["pt2_forward"].attrs[name]
                if particle == "proton":
                    assert np.array_equal(f["pt2_forward"].attrs["t_src_list"], src_lists["t_src_list"][icfg]), f"cfg {cfg}: pion and proton t_src differ"
                for pt2 in pt2_names:
                    if ip is None:
                        data = f[pt2][..., iq, :, :]                  # (smear_snk, smear_src, g_snk, g_src, [dirac,] t_src, pz, tsep)
                    else:
                        ip_sorted = sorted(ip)                        # increasing, as h5py needs; then back to the order of pz
                        data = f[pt2][..., iq, ip_sorted, :][..., [ip_sorted.index(j) for j in ip], :]
                    corr[pt2][...] = data
            if particle == "proton" and project_proton:
                corr["pt2_forward"] = np.einsum("kl,abcdlktpT->abcdtpT", parity_p, corr["pt2_forward"])          # sum_kl parity_p[k,l] M[l,k]
                corr["pt2_backward"] = -np.einsum("kl,abcdlktpT->abcdtpT", parity_m, corr["pt2_backward"])       # -sum_kl parity_m[k,l] M[l,k]
            for pt2 in pt2_names:
                dsets[(i, particle), pt2][icfg] = corr[pt2]

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
    "source_files": gevp_dir,
}
role_notes = {
    "C3input": ("2pt of case {case} at q = {q}, weighted at the source with exp(-i 2pi/Ls q.x_src): the 3pt input, to be contracted with the FF of the same q. "
                "Sink momentum p_f = pf_list[iz] = momentum_list[iz]; in the 3pt the source momentum is p_i = pi_list[iz] = p_f - q. "
                "Its ensemble mean is zero for q != 0: the 2pts of the ratio are in the C2pf and C2pi files of this (frame, q)."),
    "C2pf": "plain q = 0 2pt (case plain) at the sink momenta p_f = pf_list[iz] = momentum_list[iz] of ({frame}, q = {q}): ratio denominators and GEVP at the sink",
    "C2pi": "plain q = 0 2pt (case plain) at the source momenta p_i = pi_list[iz] = momentum_list[iz] of ({frame}, q = {q}): ratio denominators and GEVP at the source",
}
frame_note = {"forward": "forward: q = 0, p_f = p_i = (0,0,pz); this file is also the C2pf and the C2pi file",
              "sym": "symmetric: P = (p_f + p_i)/2 = (0,0,pz), p_f = P + q/2, p_i = P - q/2",
              "asy": "asymmetric: p_i = (0,0,pz), p_f = p_i + q"}
for i, (frame, q, role, case, iq, ip, pf, pi) in enumerate(read_plan):
    for particle in particle_list:
        f = out[i, particle]
        f.create_dataset("cfg_list", data=np.array(cfg_pack))
        for name in src_lists:
            f.create_dataset(name, data=src_lists[name])
        f.create_dataset("q", data=np.array(q))
        f.create_dataset("pf_list", data=pf)                          # (pz, 3): p_f of this (frame, q)
        f.create_dataset("pi_list", data=pi)                          # (pz, 3): p_i = p_f - q
        f.create_dataset("momentum_list", data=pf if role in ["C3input", "C2pf"] else pi)     # (pz, 3): sink momentum of the 2pt in this file
        f.create_dataset("pz_list", data=np.array(delta_params.pz_list))
        f.create_dataset("smear_list", data=pion_attrs["smear_list"])     # rows [rho_T, rho_z, mom_frac], as the production saved them
        f.create_dataset("smear_names", data=np.array(smear_names, dtype=h5py.string_dtype()))
        for key in common_attrs:
            f.attrs[key] = common_attrs[key]
        f.attrs["frame"] = frame
        f.attrs["role"] = role
        f.attrs["kinematic_case"] = case
        f.attrs["kinematics"] = frame_note[frame] + ", pz = pz_list[iz]. " + role_notes[role].format(case=case, q=q, frame=frame)
        if particle == "pion":
            f.attrs["dim_pt2"] = "cfg, smear_sink, smear_source, gamma_sink, gamma_source, t_src, pz, tsep"
            f.attrs["time_reflection"] = pion_attrs["time_reflection"]
            f.attrs["propagator"] = pion_attrs["propagator"]
        else:
            f.create_dataset("parity_p", data=parity_p)
            f.create_dataset("parity_m", data=parity_m)
            f.attrs["dim_pt2"] = ", ".join(["cfg", "smear_sink", "smear_source", "gamma_sink", "gamma_source"] + dirac + ["t_src", "pz", "tsep"])
            f.attrs["diquark"] = proton_attrs["diquark"]
            f.attrs["propagator"] = proton_attrs["propagator"]
            f.attrs["projection"] = ("positive parity: forward sum_kl parity_p[k,l] M[l,k], backward -sum_kl parity_m[k,l] M[l,k] (l = dirac_sink, k = dirac_source); "
                                     "backward tsep = 0 is the source slice projected with parity_m, do not use it") if project_proton else proton_attrs["how_to_project"]
        f.close()
        os.replace(f"{filenames[i, particle]}.tmp", filenames[i, particle])
        print("saved", filenames[i, particle])
