#Packs the per-cfg files of the new-kernel test (2pt_coulomb_new_kernels_32srcs_fermilab.py) into one pion file with a cfg axis, for the local
#analysis (quick_look_new_kernels.py). Run it on Fermilab once the array job is done, CPU only, from any folder: python3 fermilab_package_new_kernels.py
#It reads the output folder, cfgs and operator_tag from the setup next to it, packs every cfg whose file is on disk (the missing ones are printed)
#and keeps the sink momenta of momentum_output_list.
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

def h5_name(particle, cfg):
    return f"{pt2_dir}/cfg{cfg}/{particle}_{operator_tag}_phyp_fb_tsep{tsep_max}_cfg{cfg}.h5"

cfg_done = [int(cfg) for cfg in cfg_list if os.path.exists(h5_name("pion", cfg))]
print(f"{len(cfg_done)} of {len(cfg_list)} cfgs on disk in {pt2_dir}; missing: {[int(cfg) for cfg in cfg_list if int(cfg) not in cfg_done]}")
assert cfg_done, "no cfg to pack"
n_cfg = len(cfg_done)

#lists and notes from the first cfg; every other cfg must have the same lists
with h5py.File(h5_name("pion", cfg_done[0]), "r") as f:
    momentum_list = f["momentum_list"][:]
    operator_json = f.attrs["operator_list_json"]
    operator_names = f["operator_names"][:]
    kernel_moments = f["kernel_moments"][:]
    pion_attrs = dict(f["pt2_forward"].attrs)
    pion_shape = f["pt2_forward"].shape                 # (operator, g_snk, g_src, t_src, p_f, tsep)
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

# (cfg, operator, g_snk, g_src, t_src, pf, tsep)
pion = {pt2: np.zeros((n_cfg, n_op, n_gamma, n_gamma, n_tsrc, n_mom, tsep_max), "<c16") for pt2 in pt2_names}
t_src_all = np.zeros((n_cfg, n_tsrc), "<i8")

for icfg, cfg in enumerate(cfg_done):
    print("reading cfg", cfg, flush=True)
    with h5py.File(h5_name("pion", cfg), "r") as f:
        assert np.array_equal(f["momentum_list"][:], momentum_list) and f.attrs["operator_list_json"] == operator_json, f"cfg {cfg}: lists differ"
        t_src_all[icfg] = f["pt2_forward"].attrs["t_src_list"]
        for pt2 in pt2_names:
            pion[pt2][icfg] = f[pt2][:, :, :, :, mom_index_sorted, :][..., mom_reorder, :]

print("pion forward/backward", pion["pt2_forward"].shape)

filename = f"{save_dir}/2pt_{operator_tag}_{n_op}ops_pion_tsrc_fb_tsep{tsep_max}_{n_src}src_ncfg{n_cfg}.h5"
with h5py.File(filename, "w") as f:
    for pt2 in pt2_names:
        f.create_dataset(pt2, data=pion[pt2])           # (cfg, operator, g_snk, g_src, t_src, pf, tsep)
    f.create_dataset("pf_list", data=np.array(momentum_output_list))
    f.create_dataset("cfg_list", data=np.array(cfg_done))
    f.create_dataset("t_src_list", data=t_src_all)
    f.create_dataset("operator_names", data=operator_names)
    f.create_dataset("kernel_moments", data=kernel_moments)    # rows [rms_T, rms_z, r_source, mean_k, width_k]
    f.attrs["operator_list_json"] = operator_json
    f.attrs["dim_pt2"] = "cfg, operator, gamma_sink, gamma_source, t_src, pf, tsep"
    for key in ["gamma_list", "operators", "reference", "smearing", "momentum_convention", "spatial_sources", "time_reflection"]:
        f.attrs[key] = pion_attrs[key]
    f.attrs["time_index"] = f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {tsep_max - 1}; index 0 is the source slice in both"
    f.attrs["source_files"] = pt2_dir
print("saved", filename)
