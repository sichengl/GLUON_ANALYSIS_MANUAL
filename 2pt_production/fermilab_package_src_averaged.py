import os
import numpy as np
import h5py

q_output_list = [[0, 0, 0]]
momentum_output_list = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6]]
t_sink_max = 20
smear_tag = "coulomb_rhoT3.25_iso-aniso_frac0p6-0p3"
pt2_dir = f"/lustre2/gluonp0/sliu1/2pt_production/{smear_tag}_GEVP_qsrc_physp_ez"
cfg_list = list(range(204, 204 + 30 * 40, 30))          # 40 configs: 204, 234, ..., 1374
n_cfg = len(cfg_list)
Lt = 96
save_dir = "/project/gluonp0/sicheng/gluon_production/2pt_production/packed"
os.makedirs(save_dir, exist_ok=True)

forward_index = np.arange(t_sink_max)                   # 0, 1, ..., 19
backward_index = (-np.arange(t_sink_max)) % Lt          # 0, 95, 94, ..., 77

#read projectors from h5
with h5py.File(f"{pt2_dir}/proton_{smear_tag}_GEVP_DIRAC_qsrc_physp_cfg{cfg_list[0]}.h5", "r") as f:
    proton_shape = f["proton_gamma"].shape              # (2, 2, 2, 2, 2, 2, 4, 4, t_src, q, mom, Lt)
    parity_p = f["parity_p"][:]                         # (1 + gamma4) / 2
    parity_m = f["parity_m"][:]                         # (1 - gamma4) / 2

# where the wanted q's and momenta sit in the files (taken from the first config)
with h5py.File(f"{pt2_dir}/pion_{smear_tag}_GEVP_qsrc_physp_cfg{cfg_list[0]}.h5", "r") as f:
    q_list = f["q_list"][:]
    momentum_list = f["momentum_list"][:]
    pion_shape = f["pion_gamma"].shape                  # (2, 2, 2, 2, 2, 2, t_src, q, mom, Lt)
with h5py.File(f"{pt2_dir}/proton_{smear_tag}_GEVP_DIRAC_qsrc_physp_cfg{cfg_list[0]}.h5", "r") as f:
    proton_shape = f["proton_gamma"].shape              # (2, 2, 2, 2, 2, 2, 4, 4, t_src, q, mom, Lt)

q_index = []
for q in q_output_list:
    matches = np.where((q_list == q).all(axis=1))[0]
    assert len(matches) == 1, f"q {q} is not in the file"
    q_index.append(int(matches[0]))

mom_index = []
for mom in momentum_output_list:
    matches = np.where((momentum_list == mom).all(axis=1))[0]
    assert len(matches) == 1, f"momentum {mom} is not in the file"
    mom_index.append(int(matches[0]))
mom_index_sorted = sorted(mom_index)                                   # h5py needs increasing indices
mom_reorder = [mom_index_sorted.index(i) for i in mom_index]           # back to the order of momentum_output_list

n_q = len(q_output_list)
n_mom = len(momentum_output_list)
n_tsrc = pion_shape[6]
n_shape = pion_shape[0]      # 2: iso, aniso
n_frac  = pion_shape[2]      # 2: boost 0.6, 0.3
n_gamma = pion_shape[4]      # 2: G5, G45
n_tsrc  = pion_shape[6]      # 8
n_dirac = proton_shape[6]    # 4
n_q     = len(q_output_list)
n_mom   = len(momentum_output_list)

# (cfg, shape_snk, shape_src, frac_snk, frac_src, g_snk, g_src, t_src, q, mom, tsep)
pion_forward  = np.zeros((n_cfg, n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, n_tsrc, n_q, n_mom, t_sink_max), "<c16")
pion_backward = np.zeros((n_cfg, n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, n_tsrc, n_q, n_mom, t_sink_max), "<c16")

# (cfg, shape_snk, shape_src, frac_snk, frac_src, g_snk, g_src, dirac_snk, dirac_src, t_src, q, mom, tsep)
proton_forward  = np.zeros((n_cfg, n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, n_dirac, n_dirac, n_tsrc, n_q, n_mom, t_sink_max), "<c16")
proton_backward = np.zeros((n_cfg, n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, n_dirac, n_dirac, n_tsrc, n_q, n_mom, t_sink_max), "<c16")

t_src_all = np.zeros((n_cfg, n_tsrc), "<i8")

for icfg, cfg in enumerate(cfg_list):
    print("reading cfg", cfg, flush=True)
    with h5py.File(f"{pt2_dir}/pion_{smear_tag}_GEVP_qsrc_physp_cfg{cfg}.h5", "r") as f:
        assert np.array_equal(f["q_list"][:], q_list) and np.array_equal(f["momentum_list"][:], momentum_list), f"cfg {cfg}: lists differ"
        t_src_all[icfg] = f["pion_gamma"].attrs["t_src_list"]
        for i_q in range(n_q):
            data = f["pion_gamma"][:, :, :, :, :, :, :, q_index[i_q], mom_index_sorted, :]      # (2,2,2,2,2,2, t_src, mom, Lt)
            data = data[..., mom_reorder, :]
            pion_forward[icfg, ..., i_q, :, :] = data[..., forward_index]
            pion_backward[icfg, ..., i_q, :, :] = data[..., backward_index]
    with h5py.File(f"{pt2_dir}/proton_{smear_tag}_GEVP_DIRAC_qsrc_physp_cfg{cfg}.h5", "r") as f:
        assert np.array_equal(f["proton_gamma"].attrs["t_src_list"], t_src_all[icfg]), f"cfg {cfg}: pion and proton t_src differ"
        for i_q in range(n_q):
            data = f["proton_gamma"][:, :, :, :, :, :, :, :, :, q_index[i_q], mom_index_sorted, :]   # (2,2,2,2,2,2, 4,4, t_src, mom, Lt)
            data = data[..., mom_reorder, :]
            proton_forward[icfg, ..., i_q, :, :] = data[..., forward_index]
            proton_backward[icfg, ..., i_q, :, :] = data[..., backward_index]

print("pion   forward/backward", pion_forward.shape)
print("proton forward/backward", proton_forward.shape)

proton_forward_projected  =  np.einsum("kl,nabcdeflktqpT->nabcdeftqpT", parity_p, proton_forward)
proton_backward_projected = -np.einsum("kl,nabcdeflktqpT->nabcdeftqpT", parity_m, proton_backward)
print("proton projected", proton_forward_projected.shape)

# metadata copied from the first config's files
with h5py.File(f"{pt2_dir}/pion_{smear_tag}_GEVP_qsrc_physp_cfg{cfg_list[0]}.h5", "r") as f:
    shape_list = f["shape_list"][:]                     # rows [rho_T, rho_z]
    mom_frac_list = f["mom_frac_list"][:]
    k_list = f["k_list"][:]
    pion_attrs = dict(f["pion_gamma"].attrs)
with h5py.File(f"{pt2_dir}/proton_{smear_tag}_GEVP_DIRAC_qsrc_physp_cfg{cfg_list[0]}.h5", "r") as f:
    diquark_note = f["proton_gamma"].attrs["diquark"]

common_attrs = {
    "dim_pt2": "cfg, shape_sink, shape_source, frac_sink, frac_source, gamma_sink, gamma_source, t_src, q, pf, tsep",
    "gamma_list": pion_attrs["gamma_list"],
    "shape_names": pion_attrs["shape_names"],
    "smearing": pion_attrs["smearing"],
    "momentum_convention": pion_attrs["momentum_convention"],
    "spatial_sources": pion_attrs["spatial_sources"],
    "time_reflection": pion_attrs["time_reflection"],
    "time_index": f"pt2_forward[..., tsep] = C(t_src + tsep), pt2_backward[..., tsep] = C(t_src - tsep), tsep = 0 .. {t_sink_max - 1}; index 0 is the source slice in both",
    "source_files": pt2_dir,
}

pion_file = f"{save_dir}/2pt_{smear_tag}_GEVP_qsrc_physp_pion_tsrc_q{n_q}_fb_tsep{t_sink_max}_64src_ncfg{n_cfg}.h5"
with h5py.File(pion_file, "w") as f:
    f.create_dataset("pt2_forward", data=pion_forward)          # (cfg, shape_snk, shape_src, frac_snk, frac_src, g_snk, g_src, t_src, q, pf, tsep)
    f.create_dataset("pt2_backward", data=pion_backward)
    f.create_dataset("pf_list", data=np.array(momentum_output_list))
    f.create_dataset("q_list", data=np.array(q_output_list))
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.create_dataset("t_src_list", data=t_src_all)
    f.create_dataset("shape_list", data=shape_list)
    f.create_dataset("mom_frac_list", data=mom_frac_list)
    f.create_dataset("k_list", data=k_list)
    for key in common_attrs:
        f.attrs[key] = common_attrs[key]

proton_file = f"{save_dir}/2pt_{smear_tag}_GEVP_qsrc_physp_proton_tsrc_q{n_q}_fb_tsep{t_sink_max}_64src_ncfg{n_cfg}.h5"
with h5py.File(proton_file, "w") as f:
    f.create_dataset("pt2_forward", data=proton_forward_projected)     # (cfg, shape_snk, shape_src, frac_snk, frac_src, g_snk, g_src, t_src, q, pf, tsep)
    f.create_dataset("pt2_backward", data=proton_backward_projected)
    f.create_dataset("pf_list", data=np.array(momentum_output_list))
    f.create_dataset("q_list", data=np.array(q_output_list))
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.create_dataset("t_src_list", data=t_src_all)
    f.create_dataset("shape_list", data=shape_list)
    f.create_dataset("mom_frac_list", data=mom_frac_list)
    f.create_dataset("k_list", data=k_list)
    f.create_dataset("parity_p", data=parity_p)
    f.create_dataset("parity_m", data=parity_m)
    for key in common_attrs:
        f.attrs[key] = common_attrs[key]
    f.attrs["projection"] = "positive parity: forward sum_kl parity_p[k,l] M[l,k], backward -sum_kl parity_m[k,l] M[l,k] (l = dirac_sink, k = dirac_source); backward tsep = 0 is the source slice projected with parity_m, do not use it"
    f.attrs["diquark"] = diquark_note

print("saved", pion_file)
print("saved", proton_file)