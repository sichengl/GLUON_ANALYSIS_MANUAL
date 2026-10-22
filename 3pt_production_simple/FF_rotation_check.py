#Time correlators of the gluon FF, per cfg, for the x <-> y rotation check of the gluon operator: for every q of q_list,
#G_ab(q, dt) = (1/Lt) sum_t O_a(q, t) conj(O_b(q, t + dt)), a, b in the planes XY, XT, YT (the three of O3 = TXTX + TYTY - 2 XYXY),
#O_a = the diagonal FF component (a, a), at the HYP steps of hyp_list and w = 0 .. n_w-1. Read from the per-cfg HYP files, written to one small file.
#Under the 90 degree rotation about z (x -> y, y -> -x): q = (qx, qy, qz) -> (-qy, qx, qz), XT -> YT, YT -> -XT, XY -> XY.
#Run it on Frontier, CPU only: python3 FF_rotation_check.py
from pathlib import Path
import h5py
import numpy as np

FF_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/FF_production/FF_data")
SAVE_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/3pt_production_simple")

#======== what to compute ========
q_list = [[2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0],
          [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
          [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]
cfg_list = list(range(204, 204 + 6 * 800, 6))     # stream e 204, 210, ..., 4998; cfgs without a file are listed and left out
hyp_list = [8]                                     # HYP steps
n_w = 10                                           # Wilson-line lengths w = 0 .. n_w-1
dt_max = 10                                        # dt = 0 .. dt_max
planes = {"XY": 0, "XT": 3, "YT": 4}               # plane index in the FF files: XY, YZ, XZ, XT, YT, ZT
Lt = 96


def FF_name(cfg):
    return FF_DIR / f"FF_opp_symmetric_asymmetric_hyp0-15_w0-15_cfg{cfg}.h5"

missing = [cfg for cfg in cfg_list if not FF_name(cfg).exists()]
cfg_used = [cfg for cfg in cfg_list if cfg not in missing]
print(f"{len(cfg_used)} of {len(cfg_list)} cfgs have an FF file; left out: {missing}")
assert cfg_used, "no FF file found"

#where each q is in the FF files: the symmetric or the asymmetric list
with h5py.File(FF_name(cfg_used[0]), "r") as f:
    q_sym = f["symmetric_qlist"][:].tolist()
    q_asy = f["asymmetric_qlist"][:].tolist()
    hyp_file = [int(v) for v in f["symmetric_corr"].attrs["number_of_smears"]]
    assert f["symmetric_corr"].attrs["dim_spec"] == "cfg, munu, rhosig, hyp, wilson_list, mom, t", "unexpected layout"
read_plan = []
for q in q_list:
    if q in q_sym:
        read_plan.append(("symmetric_corr", q_sym.index(q)))
    elif q in q_asy:
        read_plan.append(("asymmetric_corr", q_asy.index(q)))
    else:
        raise AssertionError(f"q = {q} is in no FF list")
ihyp = [hyp_file.index(h) for h in hyp_list]

#G (cfg, q, hyp, a, b, w, dt)
plane_index = list(planes.values())
G = np.zeros((len(cfg_used), len(q_list), len(hyp_list), len(planes), len(planes), n_w, dt_max + 1), "<c16")
for icfg, cfg in enumerate(cfg_used):
    print("reading cfg", cfg, flush=True)
    with h5py.File(FF_name(cfg), "r") as f:
        assert [int(v) for v in f["symmetric_corr"].attrs["config_list"]] == [cfg], f"cfg {cfg}: the file holds another cfg"
        for iq, (name, k) in enumerate(read_plan):
            for jh, h in enumerate(ihyp):
                O = np.stack([f[name][0, a, a, h, :n_w, k, :] for a in plane_index])          # (plane, w, t)
                for dt in range(dt_max + 1):
                    G[icfg, iq, jh, :, :, :, dt] = np.einsum("awt,bwt->abw", O, np.conj(np.roll(O, -dt, axis=-1))) / Lt

out = SAVE_DIR / f"FF_rotation_check_hyp{'-'.join(str(h) for h in hyp_list)}_w0-{n_w - 1}_dt0-{dt_max}_ncfg{len(cfg_used)}.h5"
with h5py.File(f"{out}.tmp", "w") as f:
    f.create_dataset("G", data=G)
    f.create_dataset("cfg_list", data=np.array(cfg_used))
    f.create_dataset("q_list", data=np.array(q_list))
    f.create_dataset("hyp_list", data=np.array(hyp_list))
    f.create_dataset("w_list", data=np.arange(n_w))
    f.create_dataset("planes", data=np.array(list(planes), dtype=h5py.string_dtype()))
    f.attrs["dim_G"] = "cfg, q, hyp, plane_a, plane_b, w, dt"
    f.attrs["definition"] = "G_ab(q, dt) = (1/Lt) sum_t O_a(q, t) conj(O_b(q, t + dt)), O_a = FF component (a, a) of the per-cfg HYP file, t periodic"
    f.attrs["source_files"] = str(FF_name("{cfg}"))
Path(f"{out}.tmp").replace(out)
print("saved", out)
