from pathlib import Path
import h5py
import numpy as np

RAW_DIR = Path(
    "/lustre/orion/lgt132/scratch/sicheng/GPD_calc_v3/2pt_production"
    "/N40_rho3.25_G45_ez_momfrac0p6"
)
NAME_TEMPLATE = "pion_ix{ix}_x{x}_N40_rho3.25_frac0p6_G45_cfg{cfg}.h5"

CFG_FIRST, CFG_STEP, CFG_COUNT = 204, 6, 800
Gs = 32
Gt = 96
X_SRC_LIST = list(range(0, Gs, 8))
T_SRC_LIST = list(range(0, Gt, 12))
X_SRC_SHIFT = 3
T_SRC_SHIFT = 5
CFG_SKIP = [4314]
CFG_LIST_ALL = range(CFG_FIRST, CFG_FIRST + CFG_STEP * CFG_COUNT, CFG_STEP)
CFG_LIST = [c for c in CFG_LIST_ALL if c not in CFG_SKIP]
N_TSRC, N_XSRC, N_YSRC, N_ZSRC, N_MOM = 8, 4, 4, 8, 81

OUT_PATH = Path("./twopt_source_averaged.h5")
PROGRESS_EVERY = 25

pt2_cfg = np.zeros((N_TSRC, N_XSRC, N_YSRC, N_ZSRC, N_MOM, Gt),
                  dtype=np.complex128)
pt2_all = np.zeros((len(CFG_LIST), N_MOM, Gt), dtype=np.complex128)
momentum_list = None

for icfg, cfg in enumerate(CFG_LIST):

    ncfg = (cfg - CFG_FIRST) // CFG_STEP
    if icfg % PROGRESS_EVERY == 0:
        print(f"reading cfg = {cfg}", flush=True)

    spatial_shift = X_SRC_SHIFT * ncfg
    time_shift = T_SRC_SHIFT * ncfg

    for ix, x_base in enumerate(X_SRC_LIST):
        x_src = (x_base + spatial_shift) % Gs
        path = RAW_DIR / NAME_TEMPLATE.format(ix=ix, x=x_src, cfg=cfg)

        with h5py.File(path, "r") as f:
            pt2_cfg[:, ix, :, :, :, :] = f["pion_45"][0, :, 0, :, :, :, :]
            t_src_file = [int(v) for v in f["pion_45"].attrs["t_src_list"]]
            if momentum_list is None:
                momentum_list = np.asarray(f["momentum_list"][:],
                                           dtype=np.int64)

        t_src_built = [(t + time_shift) % Gt for t in T_SRC_LIST]
        if t_src_built != t_src_file:
            print(f"For cfg={cfg},ix={ix},mismatch in tsrc_list:"
                  f"t_src_file={t_src_file} while t_src_built={t_src_built}",
                  flush=True)

    spatial_mean = np.mean(pt2_cfg, axis=(1, 2, 3))

    for it, tsrc in enumerate(t_src_file):
        spatial_mean[it] = np.roll(spatial_mean[it], -tsrc, axis=-1)

    pt2_all[icfg, :, :] = np.mean(spatial_mean, axis=0)

    if PROGRESS_EVERY and (icfg + 1) % PROGRESS_EVERY == 0:
        print(f"  {icfg + 1}/{len(CFG_LIST)} configurations", flush=True)

with h5py.File(OUT_PATH, "w") as f:
    dataset = f.create_dataset("correlator_cfg", data=pt2_all)
    dataset.attrs["geometry"] = "n_cfg,n_mom,Gt=799,81,96"
    dataset.attrs["cfg_skipped"] = "4314"
    f.create_dataset("momentum_list", data=momentum_list)
    f.create_dataset("cfg_list", data=np.asarray(CFG_LIST, dtype=np.int64))
    
print(f"written: {OUT_PATH}  {pt2_all.shape}", flush=True)
