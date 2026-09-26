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

forward_index = np.arange(t_sink_max)                   # 0, 1, ..., 19
backward_index = (-np.arange(t_sink_max)) % Lt          # 0, 95, 94, ..., 77

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

# containers: (cfg, shape_snk, shape_src, frac_snk, frac_src, g_snk, g_src, [dirac_snk, dirac_src,] t_src, q, mom, tsep)
pion_forward = np.zeros((n_cfg,) + pion_shape[:7] + (n_q, n_mom, t_sink_max), "<c16")
pion_backward = np.zeros_like(pion_forward)
proton_forward = np.zeros((n_cfg,) + proton_shape[:9] + (n_q, n_mom, t_sink_max), "<c16")
proton_backward = np.zeros_like(proton_forward)
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