from pathlib import Path
import h5py
import numpy as np

#build 3pt for local analysis
#vacuum subtracted
#save both forward and backward propagation in one file (two datasets)

FF_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GPD_calc/FF_data")
PT2_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GPD_calc_v3/2pt_production")

cfg_list = [c for c in range(204, 204 + 6*800, 6) if c != 4314]
Ls, T = 32, 96
pt2_rho, pt2_mom_frac = "3.25", "0p6"
x_src_list = list(range(0, 32, 8))
tsrc_list = list(range(0, 96, 12))
tins_max = 18
tsep_max = tins_max
gamma = 5
pf_list = [[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,0,5],[0,0,6]]
pt2_tmp_name = PT2_DIR / f"N40_rho{pt2_rho}_G{gamma}_ez_momfrac{pt2_mom_frac}" / f"pion_ix0_x0_N40_rho{pt2_rho}_frac{pt2_mom_frac}_G{gamma}_cfg204.h5"
pt2_mom_list = h5py.File(pt2_tmp_name,"r")["momentum_list"][:]
mom_to_idx = {tuple(p): i for i, p in enumerate(pt2_mom_list.tolist())}
pf_index_list = [mom_to_idx[tuple(pf)] for pf in pf_list]
tgf_index_list = [20,25,30,35,40]
q_list = [[0,0,0]]

pt2_raw = np.zeros((len(cfg_list),8,4,4,8,len(pf_index_list),96),"<c16")
pt2_cfg_forward = np.zeros((len(cfg_list),8,4,4,8,len(pf_index_list),18),"<c16")
pt2_cfg_backward = np.zeros((len(cfg_list),8,4,4,8,len(pf_index_list),18),"<c16")
FF_cfg_forward = np.zeros((len(cfg_list),8,6,6,len(tgf_index_list),10,len(q_list),tins_max), "<c16")
FF_cfg_backward = np.zeros((len(cfg_list),8,6,6,len(tgf_index_list),10,len(q_list),tins_max), "<c16")
forward_index = np.arange(tsep_max)[0:18]
backward_index = (-np.arange(tsep_max)) % T
backward_index = backward_index[0:18]
for icfg , cfg in enumerate(cfg_list):
    print(f"reading cfg = {cfg}")
    ncfg = (cfg - 204) // 6
    shift_x = (ncfg * 3) % Ls
    shift_t = (ncfg * 5) % T

    for ix, xsrc in enumerate(x_src_list):
        shifted_x = (xsrc + shift_x) % Ls
        pt2_data_name = PT2_DIR / f"N40_rho{pt2_rho}_G{gamma}_ez_momfrac{pt2_mom_frac}" / f"pion_ix{ix}_x{shifted_x}_N40_rho{pt2_rho}_frac{pt2_mom_frac}_G{gamma}_cfg{cfg}.h5"
        with h5py.File(pt2_data_name, "r") as f:
            pt2_raw[icfg,:,ix,...] = f[f"pion_{gamma}"][0, :, 0, :, :, :, :][..., pf_index_list, :]
            #print(f["pion_45"][0, :, 0, :, :, :, :].shape)
            momentum_list = f["momentum_list"][:]

    #roll pt2 
    for itsrc, tsrc in enumerate(tsrc_list):
        t_roll = ( tsrc + shift_t ) % 96
        rolled_2pt = np.roll(pt2_raw[icfg,itsrc],-t_roll,axis=-1)
        pt2_cfg_forward[icfg,itsrc,...] = rolled_2pt[...,forward_index]
        pt2_cfg_backward[icfg,itsrc,...] = rolled_2pt[...,backward_index]


    with h5py.File(FF_DIR / f"FF_opp_symmetric_asymmetric_flow0-40_cfg{cfg}.h5", "r") as f:
        q_data_list = f["symmetric_qlist"][:]
        w_data_list = f["wilson_line_list"][:]
        q_to_idx = {tuple(q): i for i, q in enumerate(q_data_list.tolist())}
        q_index_list = [q_to_idx[tuple(q)] for q in q_list]
        FF = f["symmetric_corr"][0, :, :, 20:41:5, 0:10, :, :][..., q_index_list, :]
    for itsrc, tsrc in enumerate(tsrc_list):
        t0 = (tsrc + shift_t) % 96
        rolled_FF = np.roll(FF, -t0, axis=-1)
        FF_cfg_forward[icfg,itsrc] = rolled_FF[..., forward_index]   # [tsrc, munu, rhosig, tgf, w, q, tau]
        FF_cfg_backward[icfg,itsrc] = rolled_FF[..., backward_index]

    assert np.array_equal(pt2_cfg_forward[icfg, ..., 0], pt2_cfg_backward[icfg, ..., 0]), f"cfg {cfg}: 2pt source slice differs"
    assert np.array_equal(FF_cfg_forward[icfg, ..., 0], FF_cfg_backward[icfg, ..., 0]), f"cfg {cfg}: FF source slice differs"

pt2_tag = f"2pt_N40_rho{pt2_rho}_G{gamma}_ez_momfrac{pt2_mom_frac}"
q_tag = "q000"
tgf_tag = "-".join(str(i) for i in tgf_index_list)
SAVE_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/3pt_production_simple")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

with h5py.File(SAVE_DIR / f"{pt2_tag}_fb_tsep{tsep_max}_ncfg{len(cfg_list)}.h5", "w") as f:
    f.create_dataset("pt2_forward", data=pt2_cfg_forward)     # [cfg, tsrc, xsrc, ysrc, zsrc, pf, tsep]   C2(t0 + tsep)
    f.create_dataset("pt2_backward", data=pt2_cfg_backward)   # [cfg, tsrc, xsrc, ysrc, zsrc, pf, tsep]   C2(t0 - tsep)
    f.create_dataset("pf_list", data=np.array(pf_list))
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.attrs["dim_pt2"] = "cfg, tsrc, xsrc, ysrc, zsrc, pf, tsep"
    f.attrs["gamma"] = f"G{gamma}"
    f.attrs["time_index"] = "index t = t0 + t (forward) or t0 - t (backward); index 0 is the source slice in both"

with h5py.File(SAVE_DIR / f"FF_{q_tag}_tgf{tgf_tag}_fb_tins{tins_max}_ncfg{len(cfg_list)}.h5", "w") as f:
    f.create_dataset("FF_forward", data=FF_cfg_forward)       # [cfg, tsrc, munu, rhosig, tgf, w, q, tau]   O(t0 + tau)
    f.create_dataset("FF_backward", data=FF_cfg_backward)     # [cfg, tsrc, munu, rhosig, tgf, w, q, tau]   O(t0 - tau)
    f.create_dataset("q_list", data=np.array(q_list))
    f.create_dataset("tgf_list", data=np.array(tgf_index_list))
    f.create_dataset("w_list", data=w_data_list[0:10])
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.attrs["dim_FF"] = "cfg, tsrc, munu, rhosig, tgf, w, q, tau"
    f.attrs["time_index"] = "index tau = t0 + tau (forward) or t0 - tau (backward); index 0 is the source slice in both"
print("saved", SAVE_DIR)

