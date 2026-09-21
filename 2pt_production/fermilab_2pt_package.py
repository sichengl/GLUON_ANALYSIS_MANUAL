import os
import h5py
import numpy as np

# settings
pt2_dir  = "/project/gluonp0/sicheng/gluon_production/2pt_production/N40_rho3.25_GEVP_ez_momfrac0p6"
save_dir = "/project/gluonp0/sicheng/gluon_production/2pt_production/packed"
os.makedirs(save_dir, exist_ok=True)

cfg_list = list(range(204, 204 + 30*10, 30))     
ncfg = len(cfg_list)
Ls = 32
T  = 96
rho      = "3.25"
mom_frac = "0p6"
tsep_max = 18                                    # keep t = 0 .. 17 after the source (forward) and before it (backward)
pf_list = [[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5], [0,0,6]]      # sink momenta to keep
q_list  = [[0,0,0], [-2,0,0], [2,0,0], [0,2,0], [0,-2,0]]                      # momentum transfers for the source phase
src_phase_sign = -1                              # phase exp(-i 2pi/Ls q.x_src), same convention as the local 3pt build

#masterpiece of claude
forward_index  = np.arange(tsep_max)             # 0, 1, 2, ..., 17
backward_index = (-np.arange(tsep_max)) % T      # 0, 95, 94, ..., 79

# fixed information from the first config
first_file = f"{pt2_dir}/proton_N40_rho{rho}_frac{mom_frac}_GEVP_DIRAC_cfg{cfg_list[0]}.h5"
with h5py.File(first_file, "r") as f:
    momentum_list = f["momentum_list"][:]                  # [81, 3]
    proton_shape  = f["proton_gamma"].shape                # (3, 3, 4, 4, nt, nx, ny, nz, 81, 96)
    parity_p      = f["parity_p"][:]                       # (1 + g4) / 2
    parity_m      = f["parity_m"][:]                       # (1 - g4) / 2
    gamma_list    = f["proton_gamma"].attrs["gamma_list"]
nt = proton_shape[4]
nx = proton_shape[5]
ny = proton_shape[6]
nz = proton_shape[7]
n_spatial = nx * ny * nz
npf = len(pf_list)
nq  = len(q_list)

# position of each wanted momentum inside the 81-momentum list
pf_index = []
for pf in pf_list:
    for i in range(len(momentum_list)):
        if momentum_list[i][0] == pf[0] and momentum_list[i][1] == pf[1] and momentum_list[i][2] == pf[2]:
            pf_index.append(i)
            break
assert len(pf_index) == npf, "some momentum in pf_list is not in the file"

# empty containers, per source, time already cut to the forward / backward windows
pion_fwd_raw   = np.zeros((ncfg, 3, 3, nt, nx, ny, nz, npf, tsep_max), "<c16")          # [cfg, gs, gsrc, tsrc, xsrc, ysrc, zsrc, pf, tsep]
pion_bwd_raw   = np.zeros((ncfg, 3, 3, nt, nx, ny, nz, npf, tsep_max), "<c16")
proton_fwd_raw = np.zeros((ncfg, 3, 3, 4, 4, nt, nx, ny, nz, npf, tsep_max), "<c16")    # [cfg, gs, gsrc, l, k, tsrc, xsrc, ysrc, zsrc, pf, tsep], open Dirac
proton_bwd_raw = np.zeros((ncfg, 3, 3, 4, 4, nt, nx, ny, nz, npf, tsep_max), "<c16")
t_src_all = np.zeros((ncfg, nt), "<i8")
x_src_all = np.zeros((ncfg, nx), "<i8")
y_src_all = np.zeros((ncfg, ny), "<i8")
z_src_all = np.zeros((ncfg, nz), "<i8")

# loop over configs: fill the containers
for icfg in range(ncfg):
    cfg = cfg_list[icfg]
    print("reading cfg", cfg, flush=True)
    pion_file   = f"{pt2_dir}/pion_N40_rho{rho}_frac{mom_frac}_GEVP_cfg{cfg}.h5"
    proton_file = f"{pt2_dir}/proton_N40_rho{rho}_frac{mom_frac}_GEVP_DIRAC_cfg{cfg}.h5"
    with h5py.File(pion_file, "r") as f:
        pion = f["pion_gamma"][:][:, :, :, :, :, :, pf_index, :]                      # [3, 3, nt, nx, ny, nz, npf, 96]
    with h5py.File(proton_file, "r") as f:
        M = f["proton_gamma"][:][:, :, :, :, :, :, :, :, pf_index, :]                 # [3, 3, 4, 4, nt, nx, ny, nz, npf, 96]
        t_src_all[icfg] = f["proton_gamma"].attrs["t_src_list"]
        x_src_all[icfg] = f["proton_gamma"].attrs["x_src_list"]
        y_src_all[icfg] = f["proton_gamma"].attrs["y_src_list"]
        z_src_all[icfg] = f["proton_gamma"].attrs["z_src_list"]
    pion_fwd_raw[icfg]   = pion[..., forward_index]
    pion_bwd_raw[icfg]   = pion[..., backward_index]
    proton_fwd_raw[icfg] = M[..., forward_index]
    proton_bwd_raw[icfg] = M[..., backward_index]

# phase tensor: phase[cfg, q, x, y, z] = exp(sign * i 2pi/Ls q.x_src) / N_spatial   (the 1/N makes the sum a mean)
q_arr = np.array(q_list)                                                                     # [q, 3]
q_dot_x = (q_arr[None, :, 0, None, None, None] * x_src_all[:, None, :, None, None]
         + q_arr[None, :, 1, None, None, None] * y_src_all[:, None, None, :, None]
         + q_arr[None, :, 2, None, None, None] * z_src_all[:, None, None, None, :])       # [cfg, q, x, y, z]
phase = np.exp(src_phase_sign * 1j * 2 * np.pi / Ls * q_dot_x) / n_spatial                  # [cfg, q, x, y, z]

# contract the spatial sources with the phase tensor (q index 0 is the plain average)
pion_fwd = np.einsum("cqxyz,cijtxyzpT->cijtqpT", phase, pion_fwd_raw)                  # [cfg, gs, gsrc, tsrc, q, pf, tsep]
pion_bwd = np.einsum("cqxyz,cijtxyzpT->cijtqpT", phase, pion_bwd_raw)
proton_open_fwd = np.einsum("cqxyz,cijlktxyzpT->cijlktqpT", phase, proton_fwd_raw)     # [cfg, gs, gsrc, l, k, tsrc, q, pf, tsep], open Dirac
proton_open_bwd = np.einsum("cqxyz,cijlktxyzpT->cijlktqpT", phase, proton_bwd_raw)

# positive-parity projection: forward Tr[P+ M], backward -Tr[P- M]
proton_fwd =  np.einsum("kl,cijlktqpT->cijtqpT", parity_p, proton_open_fwd)            # [cfg, gs, gsrc, tsrc, q, pf, tsep]
proton_bwd = -np.einsum("kl,cijlktqpT->cijtqpT", parity_m, proton_open_bwd)

# sanity check: at the source slice forward and backward are the same lattice time
assert np.allclose(pion_fwd[:, :, :, :, 0, :, 0], pion_bwd[:, :, :, :, 0, :, 0]), "pion source slice differs"

# save: pion, projected proton, open-Dirac proton
tag = f"2pt_N40_rho{rho}_GEVP_ez_momfrac{mom_frac}"
common_attrs = {
    "gamma_list": gamma_list,
    "spatial_sources": f"mean over {nx} x {ny} x {nz} positions per tsrc, weighted by exp({src_phase_sign:+d} i 2pi/Ls q.x_src); q index 0 is the plain mean",
    "src_phase": f"exp({src_phase_sign:+d} i 2pi/Ls q.x_src) ALREADY APPLIED; do not apply it again in the 3pt build",
    "time_index": "index t = t0 + t (forward) or t0 - t (backward); index 0 is the source slice in both; antiperiodic sign already applied",
}

with h5py.File(f"{save_dir}/{tag}_pion_tsrc_q{nq}_fb_tsep{tsep_max}_ncfg{ncfg}.h5", "w") as f:
    f.create_dataset("pt2_forward",  data=pion_fwd)          # [cfg, gamma_sink, gamma_source, tsrc, q, pf, tsep]   C2(t0 + tsep)
    f.create_dataset("pt2_backward", data=pion_bwd)          # [cfg, gamma_sink, gamma_source, tsrc, q, pf, tsep]   C2(t0 - tsep)
    f.create_dataset("pf_list", data=np.array(pf_list))
    f.create_dataset("q_list", data=q_arr)
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.create_dataset("t_src_list", data=t_src_all)
    f.attrs["dim_pt2"] = "cfg, gamma_sink, gamma_source, tsrc, q, pf, tsep"
    f.attrs["plain_2pt"] = "pt2_forward[:, :, :, :, 0].mean(axis=3) -> [cfg, gamma_sink, gamma_source, pf, tsep]"
    for key in common_attrs:
        f.attrs[key] = common_attrs[key]

with h5py.File(f"{save_dir}/{tag}_proton_tsrc_q{nq}_fb_tsep{tsep_max}_ncfg{ncfg}.h5", "w") as f:
    f.create_dataset("pt2_forward",  data=proton_fwd)        # [cfg, gamma_sink, gamma_source, tsrc, q, pf, tsep]   positive parity, C2(t0 + tsep)
    f.create_dataset("pt2_backward", data=proton_bwd)        # [cfg, gamma_sink, gamma_source, tsrc, q, pf, tsep]   positive parity, C2(t0 - tsep)
    f.create_dataset("pf_list", data=np.array(pf_list))
    f.create_dataset("q_list", data=q_arr)
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.create_dataset("t_src_list", data=t_src_all)
    f.create_dataset("parity_p", data=parity_p)
    f.create_dataset("parity_m", data=parity_m)
    f.attrs["dim_pt2"] = "cfg, gamma_sink, gamma_source, tsrc, q, pf, tsep"
    f.attrs["projection"] = "forward: sum_kl parity_p[k,l] M[l,k]; backward: -sum_kl parity_m[k,l] M[l,k]; positive-parity proton both ways"
    f.attrs["plain_2pt"] = "pt2_forward[:, :, :, :, 0].mean(axis=3) -> [cfg, gamma_sink, gamma_source, pf, tsep]"
    for key in common_attrs:
        f.attrs[key] = common_attrs[key]

with h5py.File(f"{save_dir}/{tag}_proton_open_tsrc_q{nq}_fb_tsep{tsep_max}_ncfg{ncfg}.h5", "w") as f:
    f.create_dataset("pt2_forward",  data=proton_open_fwd)   # [cfg, gamma_sink, gamma_source, dirac_sink, dirac_source, tsrc, q, pf, tsep]   M(t0 + tsep), NOT projected
    f.create_dataset("pt2_backward", data=proton_open_bwd)   # [cfg, gamma_sink, gamma_source, dirac_sink, dirac_source, tsrc, q, pf, tsep]   M(t0 - tsep), NOT projected
    f.create_dataset("pf_list", data=np.array(pf_list))
    f.create_dataset("q_list", data=q_arr)
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.create_dataset("t_src_list", data=t_src_all)
    f.create_dataset("parity_p", data=parity_p)
    f.create_dataset("parity_m", data=parity_m)
    f.attrs["dim_pt2"] = "cfg, gamma_sink, gamma_source, dirac_sink, dirac_source, tsrc, q, pf, tsep"
    f.attrs["how_to_project"] = "NOT projected. positive parity: forward sum_kl parity_p[k,l] M[l,k], backward -sum_kl parity_m[k,l] M[l,k]"
    for key in common_attrs:
        f.attrs[key] = common_attrs[key]

print("saved to", save_dir)