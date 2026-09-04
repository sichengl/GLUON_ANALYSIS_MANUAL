import h5py
import numpy as np
from opt_einsum import contract

def jackknife(x):
    n_cfg = x.shape[0]
    return (x.sum(axis=0) - x) / (n_cfg - 1)  

gamma = "45"
FF_path = "/Users/mcp3270/2pt_plot_meff/local_analysis/FF_q000_tgf20-25-30-35-40_tins18_ncfg799.h5"
pt2_path = f"/Users/mcp3270/2pt_plot_meff/local_analysis/forward_2pt_N40_rho3.25_G{gamma}_ez_momfrac0p6_tsep18_ncfg799.h5"
Ls = 32
SRC_PHASE_SIGN = -1    # in my convention pi = pf + q, phase for FF is iq(x-xsrc) so -1 
WL_PHASE_SIGN = +1     # Fix the centering of Wilson line
FF_file = h5py.File(FF_path, "r")
FF_cfg = FF_file["FF"]                                   # [cfg, tsrc, munu, rhosig, tgf, w, q, tau] = (799, 8, 6, 6, 5, 10, 1, 18)
q_list = FF_file["q_list"][:]                            # [q, 3] = (1, 3)
w_list = FF_file["w_list"][:]                            # [w] = (10,)
print(FF_cfg.shape)
pt2_file = h5py.File(pt2_path, "r")
pt2_cfg = pt2_file["pt2"][:]                           # [cfg, tsrc, xsrc, ysrc, zsrc, pf, tsep] = (799, 8, 4, 4, 8, 7, 18)
cfg_list = pt2_file["cfg_list"][:]                       # [cfg] = (799,)
print(pt2_cfg.shape)

#In my FF production code, Wilson line is not from -w/2 to w/2 so we need an extra phase factor to restore the convention (exactly)
wl_phase = np.exp(WL_PHASE_SIGN * 1j * np.pi / Ls * contract(w_list,["w"], q_list[:, 2],  ["q"], ["w", "q"]))                                  # [w, q] = (10, 1)
TXTX = contract(FF_cfg[:, :, 3, 3,:,:,:,:], ["cfg", "tsrc", "tgf", "w", "q", "tau"],wl_phase,["w", "q"], ["cfg", "tsrc", "tgf", "w", "q", "tau"])     # [cfg, tsrc, tgf, w, q, tau] = (799, 8, 5, 10, 1, 18)
TYTY = contract(FF_cfg[:, :, 4, 4,:,:,:,:], ["cfg", "tsrc", "tgf", "w", "q", "tau"],wl_phase,["w", "q"],["cfg", "tsrc", "tgf", "w", "q", "tau"])
XYXY = contract(FF_cfg[:, :, 0, 0,:,:,:,:], ["cfg", "tsrc", "tgf", "w", "q", "tau"],wl_phase,["w", "q"],["cfg", "tsrc", "tgf", "w", "q", "tau"])
TXTXpTYTY = TXTX + TYTY
TXTXpTYTYm2XYXY = TXTXpTYTY - 2 * XYXY
TXTXpTYTYp2XYXY = TXTXpTYTY + 2 * XYXY
operators = [TXTX, TYTY, XYXY, TXTXpTYTY, TXTXpTYTYm2XYXY,TXTXpTYTYp2XYXY]

shift = ((cfg_list - 204) // 6 * 3) % Ls                               # [cfg]   spatial source shift of each configuration
x_pos = (np.arange(0, 32, 8)[None, :] + shift[:, None]) % Ls           # [cfg, xsrc]
y_pos = (np.arange(0, 32, 8)[None, :] + shift[:, None]) % Ls           # [cfg, ysrc]
z_pos = (np.arange(0, 32, 4)[None, :] + shift[:, None]) % Ls           # [cfg, zsrc]
# [cfg, xsrc, ysrc, zsrc, q]
q_dot_x = (x_pos[:, :, None, None, None] * q_list[:, 0] + y_pos[:, None, :, None, None] * q_list[:, 1] + z_pos[:, None, None, :, None] * q_list[:, 2])  
src_phase = np.exp(SRC_PHASE_SIGN * 1j * 2 * np.pi / Ls * q_dot_x)     # [cfg, xsrc, ysrc, zsrc, q]

#FF phase can be contracted with pt2 first
pt2_src = contract(src_phase, ["cfg", "xsrc", "ysrc", "zsrc", "q"],
                   pt2_cfg,   ["cfg", "tsrc", "xsrc", "ysrc", "zsrc", "pf", "tsep"],
                              ["cfg", "tsrc", "pf", "tsep", "q"]) / (4 * 4 * 8)   # [cfg, tsrc, pf, tsep, q]


#Calculate the connected piece
pt2FF = []
for iop, op in enumerate(operators):
    print(iop)
    connected_term = contract(pt2_src, ["cfg", "tsrc", "pf", "tsep", "q"],
                              op,      ["cfg", "tsrc", "tgf", "w", "q", "tau"],
                                       ["cfg", "tgf", "w", "q", "tsep", "tau", "pf"])
    pt2FF.append(connected_term / 8)                     # [cfg, tgf, w, q, tsep, tau, pf]          


#Calculate the disconnected piece using my old method
#First <2pt>
pt2_avg = contract(pt2_src, ["cfg", "tsrc", "pf", "tsep", "q"],["cfg", "pf", "tsep", "q"]) / 8  
#Then FF operator
op_avg = []
for iop, op in enumerate(operators):
    op_avg.append(contract(op, ["cfg", "tsrc", "tgf", "w", "q", "tau"],["cfg", "tgf", "w", "q", "tau"]) / 8)

pt2_jk = jackknife(pt2_avg)                       
op_jk = [jackknife(a) for a in op_avg]            
pt2FF_jk = [jackknife(a) for a in pt2FF] 

pt3_vac_subtracted = []
for iop in range(len(operators)):
    vac_jk = contract(pt2_jk,     ["jk", "pf", "tsep", "q"],
                      op_jk[iop], ["jk", "tgf", "w", "q", "tau"],
                                  ["jk", "tgf", "w", "q", "tsep", "tau", "pf"])   # [jk, tgf, w, q, tsep, tau, pf] = (799, 5, 10, 1, 18, 18, 7)
    pt3_vac_subtracted.append(pt2FF_jk[iop] - vac_jk)                              # [jk, tgf, w, q, tsep, tau, pf]

op_names = ["TXTX", "TYTY", "XYXY", "TXTXpTYTY", "TXTXpTYTYm2XYXY","TXTXpTYTYp2XYXY"]
pf_list = pt2_file["pf_list"][:]                       # [pf, 3] = (7, 3)
tgf_list = FF_file["tgf_list"][:]                      # [tgf] = (5,)
out_path = (f"/Users/mcp3270/2pt_plot_meff/local_analysis/"
            f"pt3_jk_method1_G{gamma}_q000_tgf20-25-30-35-40_src{SRC_PHASE_SIGN:+d}_wl{WL_PHASE_SIGN:+d}.h5")

with h5py.File(out_path, "w") as f:
    for iop, name in enumerate(op_names):
        f.create_dataset(f"pt3_jk/{name}", data=pt3_vac_subtracted[iop])   # [jk, tgf, w, q, tsep, tau, pf] = (799, 5, 10, 1, 18, 18, 7)
    f.create_dataset("pt2_jk", data=pt2_jk)                                # [jk, pf, tsep, q] = (799, 7, 18, 1)
    f.create_dataset("operator_names", data=np.array(op_names, dtype="S"))
    f.create_dataset("pf_list", data=pf_list)
    f.create_dataset("tgf_list", data=tgf_list)
    f.create_dataset("w_list", data=w_list)
    f.create_dataset("q_list", data=q_list)
    f.create_dataset("cfg_list", data=cfg_list)
    f.attrs["dim_pt3_jk"] = "jk, tgf, w, q, tsep, tau, pf"
    f.attrs["dim_pt2_jk"] = "jk, pf, tsep, q"
    f.attrs["vacuum_subtraction"] = "method 1: average over sources, jackknife over cfgs, multiply <C2><O>, subtract per replicate"
    f.attrs["src_phase"] = f"exp({SRC_PHASE_SIGN:+d} i q.x_src), x_src = shifted source positions"
    f.attrs["wl_phase"] = f"exp({WL_PHASE_SIGN:+d} i pi q_z w / Ls), applied to the operator"
    f.attrs["jackknife"] = "delete-one, replicate k = mean over all cfgs except cfg_list[k]"
print("saved", out_path)

#multiply jackknifed 2pt and FF per-tsrc then average tsrc
pt2_src_jk = jackknife(pt2_src)                    # [jk, tsrc, pf, tsep, q] = (799, 8, 7, 18, 1)
op_src_jk = [jackknife(op) for op in operators]    # each [jk, tsrc, tgf, w, q, tau] = (799, 8, 5, 10, 1, 18)
pt3_vac_subtracted_m2 = []
for iop in range(len(operators)):
    vac_jk = contract(pt2_src_jk,    ["jk", "tsrc", "pf", "tsep", "q"],
                      op_src_jk[iop], ["jk", "tsrc", "tgf", "w", "q", "tau"],
                                      ["jk", "tgf", "w", "q", "tsep", "tau", "pf"]) / 8   # [jk, tgf, w, q, tsep, tau, pf]
    pt3_vac_subtracted_m2.append(pt2FF_jk[iop] - vac_jk)
out_path = (f"/Users/mcp3270/2pt_plot_meff/local_analysis/"
            f"pt3_jk_method2_G{gamma}_q000_tgf20-25-30-35-40_src{SRC_PHASE_SIGN:+d}_wl{WL_PHASE_SIGN:+d}.h5")
with h5py.File(out_path, "w") as f:
    for iop, name in enumerate(op_names):
        f.create_dataset(f"pt3_jk/{name}", data=pt3_vac_subtracted_m2[iop])   # [jk, tgf, w, q, tsep, tau, pf]
    f.create_dataset("pt2_jk", data=pt2_jk)
    f.create_dataset("operator_names", data=np.array(op_names, dtype="S"))
    f.create_dataset("pf_list", data=pf_list)
    f.create_dataset("tgf_list", data=tgf_list)
    f.create_dataset("w_list", data=w_list)
    f.create_dataset("q_list", data=q_list)
    f.create_dataset("cfg_list", data=cfg_list)
    f.attrs["dim_pt3_jk"] = "jk, tgf, w, q, tsep, tau, pf"
    f.attrs["dim_pt2_jk"] = "jk, pf, tsep, q"
    f.attrs["vacuum_subtraction"] = "method 2: jackknife over cfgs per source, multiply per source, average over sources"
    f.attrs["src_phase"] = f"exp({SRC_PHASE_SIGN:+d} i q.x_src), x_src = shifted source positions"
    f.attrs["wl_phase"] = f"exp({WL_PHASE_SIGN:+d} i pi q_z w / Ls), applied to the operator"
    f.attrs["jackknife"] = "delete-one, replicate k = mean over all cfgs except cfg_list[k]"
print("saved", out_path)

#Method 3: multiply the per-cfg source averages on each cfg, then jackknife (biased by ~1/n_tsrc of the signal; comparison only)
pt3_vac_subtracted_m3 = []
for iop in range(len(operators)):
    vac_cfg = contract(pt2_avg,     ["cfg", "pf", "tsep", "q"],
                       op_avg[iop], ["cfg", "tgf", "w", "q", "tau"],
                                    ["cfg", "tgf", "w", "q", "tsep", "tau", "pf"])   # [cfg, tgf, w, q, tsep, tau, pf]   product on each cfg
    pt3_vac_subtracted_m3.append(pt2FF_jk[iop] - jackknife(vac_cfg))                 # [jk, tgf, w, q, tsep, tau, pf]
out_path = (f"/Users/mcp3270/2pt_plot_meff/local_analysis/"
            f"pt3_jk_method3_G{gamma}_q000_tgf20-25-30-35-40_src{SRC_PHASE_SIGN:+d}_wl{WL_PHASE_SIGN:+d}.h5")
with h5py.File(out_path, "w") as f:
    for iop, name in enumerate(op_names):
        f.create_dataset(f"pt3_jk/{name}", data=pt3_vac_subtracted_m3[iop])   # [jk, tgf, w, q, tsep, tau, pf]
    f.create_dataset("pt2_jk", data=pt2_jk)
    f.create_dataset("operator_names", data=np.array(op_names, dtype="S"))
    f.create_dataset("pf_list", data=pf_list)
    f.create_dataset("tgf_list", data=tgf_list)
    f.create_dataset("w_list", data=w_list)
    f.create_dataset("q_list", data=q_list)
    f.create_dataset("cfg_list", data=cfg_list)
    f.attrs["dim_pt3_jk"] = "jk, tgf, w, q, tsep, tau, pf"
    f.attrs["dim_pt2_jk"] = "jk, pf, tsep, q"
    f.attrs["vacuum_subtraction"] = "method 3: multiply per-cfg source averages on each cfg, then jackknife (biased by ~1/n_tsrc of the signal; comparison only)"
    f.attrs["src_phase"] = f"exp({SRC_PHASE_SIGN:+d} i q.x_src), x_src = shifted source positions"
    f.attrs["wl_phase"] = f"exp({WL_PHASE_SIGN:+d} i pi q_z w / Ls), applied to the operator"
    f.attrs["jackknife"] = "delete-one, replicate k = mean over all cfgs except cfg_list[k]"
print("saved", out_path)
