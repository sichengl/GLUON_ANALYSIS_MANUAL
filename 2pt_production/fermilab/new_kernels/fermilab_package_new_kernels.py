#Packs the per-cfg files of the new-kernel runs (2pt_coulomb_new_kernels_32srcs_fermilab.py) into one pion and one proton file with a cfg axis,
#for the local analysis (gevp_look_new_kernels.py, quick_look_new_kernels.py). Run it on Fermilab once the job is done, CPU only, from any folder:
#python3 fermilab_package_new_kernels.py
#It reads the output folder, cfgs and operator_tag from the setup next to it, packs every cfg whose two files are on disk (the missing ones are
#printed), keeps the sink momenta of momentum_output_list and projects the proton to positive parity (the per-cfg files keep the open Dirac indices),
#as fermilab_package_smear_tune.py does for the smearing tuning.
import os
import sys
import json
import numpy as np
import h5py

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..", "..", "tools")]        # the setup imports pt2_comm_tools from 2pt_production/tools
from new_kernels_2pt_setup_fermilab import fermilab_params, cfg_list, operator_tag, operator_list, current_dir, n_spatial_src

momentum_output_list = [[0, 0, pz] for pz in fermilab_params.pz_list]      # (0,0,pz), pz = 0 .. 6, as in the packed smearing-tuning files
tsep_max = fermilab_params.tsep_max
n_src = len(fermilab_params.t_base) * n_spatial_src
pt2_dir = f"{current_dir}/{operator_tag}_{n_src}src_phyp_fb_tsep{tsep_max}"   # gevp_dir of the job script
save_dir = "/project/gluonp0/sicheng/gluon_production/2pt_production/packed"
os.makedirs(save_dir, exist_ok=True)
particles = ["pion", "proton"]

def h5_name(particle, cfg):
    return f"{pt2_dir}/cfg{cfg}/{particle}_{operator_tag}_phyp_fb_tsep{tsep_max}_cfg{cfg}.h5"

cfg_done = [int(cfg) for cfg in cfg_list if all(os.path.exists(h5_name(particle, cfg)) for particle in particles)]
print(f"{len(cfg_done)} of {len(cfg_list)} cfgs on disk in {pt2_dir}; missing: {[int(cfg) for cfg in cfg_list if int(cfg) not in cfg_done]}")
assert cfg_done, "no cfg to pack"
n_cfg = len(cfg_done)

#lists, projectors and notes from the first cfg; every other cfg must have the same lists
with h5py.File(h5_name("pion", cfg_done[0]), "r") as f:
    momentum_list = f["momentum_list"][:]
    operator_json = f.attrs["operator_list_json"]
    operator_names = f["operator_names"][:]
    kernel_moments = f["kernel_moments"][:]
    pion_attrs = dict(f["pt2_forward"].attrs)
    pion_shape = f["pt2_forward"].shape                 # (operator, g_snk, g_src, t_src, p_f, tsep)
with h5py.File(h5_name("proton", cfg_done[0]), "r") as f:
    parity_p = f["parity_p"][:]                         # (1 + gamma4) / 2
    parity_m = f["parity_m"][:]                         # (1 - gamma4) / 2
    proton_attrs = dict(f["pt2_forward"].attrs)
assert json.loads(operator_json) == json.loads(json.dumps(operator_list)), "the files hold another operator_list than the setup"

# where the wanted momenta sit in the files
mom_index = []
for mom in momentum_output_list:
    matches = np.where((momentum_list == mom).all(axis=1))[0]
    assert len(matches) == 1, f"momentum {mom} is not in the file"
    mom_index.append(int(matches[0]))
mom_index_sorted = sorted(mom_index)                                   # h5py needs increasing indices
mom_reorder = [mom_index_sorted.index(i) for i in mom_index]           # back to the order of momentum_output_list

n_op, n_gamma, _, n_tsrc = pion_shape[:4]
n_mom = len(momentum_output_list)
pt2_names = ["pt2_forward", "pt2_backward"]

# (cfg, operator, g_snk, g_src, t_src, pf, tsep); the proton projected
pion   = {pt2: np.zeros((n_cfg, n_op, n_gamma, n_gamma, n_tsrc, n_mom, tsep_max), "<c16") for pt2 in pt2_names}
proton = {pt2: np.zeros((n_cfg, n_op, n_gamma, n_gamma, n_tsrc, n_mom, tsep_max), "<c16") for pt2 in pt2_names}
t_src_all = np.zeros((n_cfg, n_tsrc), "<i8")

for icfg, cfg in enumerate(cfg_done):
    print("reading cfg", cfg, flush=True)
    with h5py.File(h5_name("pion", cfg), "r") as f:
        assert np.array_equal(f["momentum_list"][:], momentum_list) and f.attrs["operator_list_json"] == operator_json, f"cfg {cfg}: lists differ"
        t_src_all[icfg] = f["pt2_forward"].attrs["t_src_list"]
        for pt2 in pt2_names:
            pion[pt2][icfg] = f[pt2][:, :, :, :, mom_index_sorted, :][..., mom_reorder, :]
    with h5py.File(h5_name("proton", cfg), "r") as f:
        assert np.array_equal(f["momentum_list"][:], momentum_list) and f.attrs["operator_list_json"] == operator_json, f"cfg {cfg}: lists differ"
        assert np.array_equal(f["pt2_forward"].attrs["t_src_list"], t_src_all[icfg]), f"cfg {cfg}: pion and proton t_src differ"
        for pt2, parity, sign in [("pt2_forward", parity_p, +1), ("pt2_backward", parity_m, -1)]:
            data = f[pt2][:, :, :, :, :, :, mom_index_sorted, :][..., mom_reorder, :]     # (operator, g_snk, g_src, dirac_snk, dirac_src, t_src, pf, tsep)
            proton[pt2][icfg] = sign * np.einsum("kl,sablktpT->sabtpT", parity, data)        # forward sum_kl parity_p[k,l] M[l,k], backward -sum_kl parity_m[k,l] M[l,k]

print("pion   forward/backward", pion["pt2_forward"].shape)
print("proton forward/backward (projected)", proton["pt2_forward"].shape)

common_attrs = {
    "dim_pt2": "cfg, operator, gamma_sink, gamma_source, t_src, pf, tsep",
    "operator_list_json": operator_json,
    "time_index": f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {tsep_max - 1}; index 0 is the source slice in both",
    "source_files": pt2_dir,
}
for key in ["gamma_list", "operators", "reference", "smearing", "momentum_convention", "spatial_sources", "time_reflection"]:
    common_attrs[key] = pion_attrs[key]

for particle, corr in [("pion", pion), ("proton", proton)]:
    filename = f"{save_dir}/2pt_{operator_tag}_{n_op}ops_{particle}_tsrc_fb_tsep{tsep_max}_{n_src}src_ncfg{n_cfg}.h5"
    with h5py.File(filename, "w") as f:
        for pt2 in pt2_names:
            f.create_dataset(pt2, data=corr[pt2])           # (cfg, operator, g_snk, g_src, t_src, pf, tsep)
        f.create_dataset("pf_list", data=np.array(momentum_output_list))
        f.create_dataset("cfg_list", data=np.array(cfg_done))
        f.create_dataset("t_src_list", data=t_src_all)
        f.create_dataset("operator_names", data=operator_names)
        f.create_dataset("kernel_moments", data=kernel_moments)    # rows [rms_T, rms_z, r_source, mean_k, width_k]
        for key in common_attrs:
            f.attrs[key] = common_attrs[key]
        if particle == "proton":
            f.create_dataset("parity_p", data=parity_p)
            f.create_dataset("parity_m", data=parity_m)
            f.attrs["projection"] = ("positive parity: forward sum_kl parity_p[k,l] M[l,k], backward -sum_kl parity_m[k,l] M[l,k] (l = dirac_sink, k = dirac_source); "
                                     "backward tsep = 0 is the source slice projected with parity_m, do not use it; the time reflection "
                                     "C_backward(tsep) = s_a s_b C_forward(tsep), tsep >= 1, holds for these projected correlators")
            f.attrs["diquark"] = proton_attrs["diquark"]
    print("saved", filename)
