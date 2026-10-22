#Packs the gluon FF of the per-cfg HYP files (FF_production/FF_covshift_oddeven_trimag_negative_direction_hyp.py) per momentum transfer q, for the
#local analysis with the per-q 2pt of delta_package_byq.py: one file per q with a cfg axis, the operator on all Lt time slices (no roll to any source).
#The values are stored as they are on disk: volume sum with exp(+i 2pi/Ls q.x) at the start point x of the Wilson line, which runs from x to x - w z.
#No centering phase is applied. Pair FF(q) with the 2pt of the same q. The frame is not in the name: FF(q) is the same in every frame.
#Run it on Frontier, CPU only: python3 packFF_byq.py
from pathlib import Path
import h5py
import numpy as np

FF_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/FF_production/FF_data")
SAVE_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/3pt_production_simple/FF_byq")

#======== what to pack ========
q_pack_list = [[2, 0, 0], [-2, 0, 0]]              # one file per q, in units of 2pi/L; -q of the first q is for the sign check
cfg_list = list(range(204, 204 + 6 * 200, 6))     # the 200 cfgs of the Delta final run: stream e 204, 210, ..., 1398
n_hyp = 9                                          # HYP steps 0 .. n_hyp-1
n_w = 10                                           # Wilson-line lengths w = 0 .. n_w-1
Lt = 96


def FF_name(cfg):
    return FF_DIR / f"FF_opp_symmetric_asymmetric_hyp0-15_w0-15_cfg{cfg}.h5"

missing = [cfg for cfg in cfg_list if not FF_name(cfg).exists()]
assert not missing, f"no FF file for cfgs {missing}"

#where each q is in the FF files: the symmetric or the asymmetric list (q = 0 is in both; the symmetric one is used)
with h5py.File(FF_name(cfg_list[0]), "r") as f:
    q_sym = f["symmetric_qlist"][:].tolist()
    q_asy = f["asymmetric_qlist"][:].tolist()
    hyp_file = [int(v) for v in f["symmetric_corr"].attrs["number_of_smears"]]
    w_file = [int(v) for v in f["symmetric_corr"].attrs["wilson_line_list"]]
    file_attrs = dict(f.attrs)
    for kind in ["symmetric", "asymmetric"]:
        assert f[f"{kind}_corr"].attrs["dim_spec"] == "cfg, munu, rhosig, hyp, wilson_list, mom, t", f"{kind}_corr: unexpected layout"
        assert f[f"{kind}_corr"].shape[-1] == Lt, f"{kind}_corr: time axis is not {Lt} long"
assert hyp_file[:n_hyp] == list(range(n_hyp)), f"HYP steps of the FF files {hyp_file} do not start with 0 .. {n_hyp - 1}"
assert w_file[:n_w] == list(range(n_w)), f"Wilson lines of the FF files {w_file} do not start with 0 .. {n_w - 1}"
q_pack_list = [[int(v) for v in q] for q in q_pack_list]
assert len({tuple(q) for q in q_pack_list}) == len(q_pack_list), f"a q appears twice in q_pack_list: {q_pack_list}"
read_plan = []                                     # (dataset, q index) of every q
for q in q_pack_list:
    if q in q_sym:
        read_plan.append(("symmetric_corr", q_sym.index(q)))
    elif q in q_asy:
        read_plan.append(("asymmetric_corr", q_asy.index(q)))
    else:
        raise AssertionError(f"q = {q} is in no FF list; symmetric {q_sym}, asymmetric {q_asy}")
print(f"packing {len(cfg_list)} cfgs, q = {q_pack_list} from {[name for name, _ in read_plan]}")

#one file per q: FF (cfg, munu, rhosig, hyp, w, t), written as .tmp and renamed when complete
SAVE_DIR.mkdir(parents=True, exist_ok=True)
n_cfg = len(cfg_list)
filenames = [SAVE_DIR / f"FF_hyp0-{n_hyp - 1}_w0-{n_w - 1}_q{q[0]}_{q[1]}_{q[2]}_ncfg{n_cfg}.h5" for q in q_pack_list]
out = [h5py.File(f"{name}.tmp", "w") for name in filenames]
dsets = [f.create_dataset("FF", shape=(n_cfg, 6, 6, n_hyp, n_w, Lt), dtype="<c16") for f in out]

for icfg, cfg in enumerate(cfg_list):
    print("reading cfg", cfg, flush=True)
    with h5py.File(FF_name(cfg), "r") as f:
        assert [int(v) for v in f["symmetric_corr"].attrs["config_list"]] == [cfg], f"cfg {cfg}: the file holds cfg {f['symmetric_corr'].attrs['config_list']}"
        for (name, iq), dset in zip(read_plan, dsets):
            dset[icfg] = f[name][0, :, :, :n_hyp, :n_w, iq, :]         # (munu, rhosig, hyp, w, t)

for q, (name, iq), f, filename in zip(q_pack_list, read_plan, out, filenames):
    f.create_dataset("q", data=np.array(q))
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.create_dataset("hyp_list", data=np.arange(n_hyp))
    f.create_dataset("w_list", data=np.arange(n_w))
    f.attrs["dim_FF"] = "cfg, munu, rhosig, hyp, w, t"
    f.attrs["munu_order"] = "XY, YZ, XZ, XT, YT, ZT (the same order for rhosig)"
    f.attrs["time_index"] = f"t = absolute time slice 0 .. {Lt - 1}, as in the per-cfg files (not rolled to any source)"
    f.attrs["smearing"] = "HYP 4D, alpha = (0.75, 0.6, 0.3), level n = n steps, 0 = unsmeared"
    f.attrs["q_phase_sign"] = file_attrs["q_phase_sign"]
    f.attrs["momentum_phase"] = str(file_attrs["momentum_phase"]) + ", x = start point of the Wilson line"
    f.attrs["bilocal_direction"] = file_attrs["bilocal_direction"]
    f.attrs["centering"] = ("not applied: for the operator centered at x - w z/2 multiply by exp(-i pi q_z w / Ls) (= 1 for q_z = 0)")
    f.attrs["pairing"] = ("contract with the 2pt of the same q from delta_package_byq.py (source weight exp(-i 2pi/Ls q.x_src)): the 3pt has P_f = P_i + q. "
                          "The momentum_relation and physical_transfer_convention attributes of the per-cfg FF files are for the old unconjugated 2pt, not this one")
    f.attrs["source_dataset"] = f"{name}, q index {iq}"
    f.attrs["source_files"] = str(FF_name("{cfg}"))
    f.close()
    Path(f"{filename}.tmp").replace(filename)
    print("saved", filename)
