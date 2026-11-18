#Packs the gluon FF of the per-cfg files of FF_covshift_boost_direction_frontier.py per momentum transfer q, for the local analysis with the per-q 2pt of
#delta_package_byq_opt.py or its Frontier and Fermilab copies: one file per q with a cfg axis, the operator on all Lt time slices (no roll to any source).
#ONE packer for HYP and flow and for the three directions; both are chosen on the command line:
#    python3 packFF_byq_frontier.py --smear hyp --boost_direction x
#The Frontier copy of FF_production/delta/packFF_byq_delta.py: only this header and FF_DIR, SAVE_DIR changed. Every place that depends on the direction is marked DIRECTION, on the smearing SMEAR.
#The values are stored as they are on disk: volume sum with exp(+i 2pi/Ls q.x) at the start point x of the Wilson line, which runs from x to x - w
#along the boost direction. No centering phase is applied. Pair FF(q) with the 2pt of the same q and the same boost direction.
#q_pack_list is the same for every direction: the q of the files are labels as seen from the boost direction, (a, b, c) with c along the boost.
#All cfgs whose per-cfg file exists are packed; the number is in the file names (ncfg).
#Run it on Frontier, CPU only.
import argparse
from pathlib import Path
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--smear", type=str, required=True, choices=["hyp", "flow"])              # HYP smearing or Wilson flow
parser.add_argument("--boost_direction", type=str, required=True, choices=["x", "y", "z"])    # lattice axis of the Wilson line
args = parser.parse_args()
smear = args.smear
boost_direction = args.boost_direction

FF_DIR = Path(f"/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/FF_production/FF_data/direction_{boost_direction}")  # out_dir of FF_covshift_boost_direction_frontier.py
SAVE_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/FF_production/FF_byq")    # the new names carry _boost<x|y|z>

#======== what to pack ========
q_pack_list = [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0]]      # one file per q, in units of 2pi/L
cfg_all = list(range(204, 204 + 6 * 800, 6))       # stream e 204, 210, ..., 4998: those with a per-cfg file are packed
n_w = 10                                           # Wilson-line lengths w = 0 .. n_w-1
Lt = 96
#SMEAR: the tag in the names of the per-cfg files, and the number of smearing levels to pack
if smear == "hyp":
    file_tag = "hyp0-15_w0-15"
    n_level = 9                                    # HYP steps 0 .. 8
    smearing_note = "HYP 4D, alpha = (0.75, 0.6, 0.3), level n = n steps, 0 = unsmeared"
if smear == "flow":
    file_tag = "flow0-40_w0-9"
    n_level = 41                                   # all flow levels 0 .. 40
    smearing_note = "Wilson flow, level n = n x 10 steps of 0.01 (flow time 0.1 n), 0 = unsmeared"


#DIRECTION: the direction is in the names of the per-cfg files
def FF_name(cfg):
    return FF_DIR / f"FF_opp_symmetric_asymmetric_{file_tag}_boost{boost_direction}_cfg{cfg}.h5"

cfg_list = [cfg for cfg in cfg_all if FF_name(cfg).exists()]
print(f"{len(cfg_list)} of {len(cfg_all)} cfgs have a file {FF_name('<cfg>').name} in {FF_DIR}")
assert cfg_list, "no FF file to pack"

#where each q is in the FF files: the symmetric or the asymmetric list (q = 0 is in both; the symmetric one is used)
with h5py.File(FF_name(cfg_list[0]), "r") as f:
    q_sym = f["symmetric_qlist"][:].tolist()
    q_asy = f["asymmetric_qlist"][:].tolist()
    level_file = [int(v) for v in f["symmetric_corr"].attrs["number_of_smears"]]
    w_file = [int(v) for v in f["symmetric_corr"].attrs["wilson_line_list"]]
    file_attrs = dict(f.attrs)
    for kind in ["symmetric", "asymmetric"]:
        assert f[f"{kind}_corr"].attrs["dim_spec"] == f"cfg, munu, rhosig, {smear}, wilson_list, mom, t", f"{kind}_corr: unexpected layout"
        assert f[f"{kind}_corr"].shape[-1] == Lt, f"{kind}_corr: time axis is not {Lt} long"
assert level_file[:n_level] == list(range(n_level)), f"smearing levels of the FF files {level_file} do not start with 0 .. {n_level - 1}"
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

#one file per q: FF (cfg, munu, rhosig, smearing level, w, t), written as .tmp and renamed when complete
SAVE_DIR.mkdir(parents=True, exist_ok=True)
n_cfg = len(cfg_list)
filenames = [SAVE_DIR / f"FF_{smear}0-{n_level - 1}_w0-{n_w - 1}_boost{boost_direction}_q{q[0]}_{q[1]}_{q[2]}_ncfg{n_cfg}.h5" for q in q_pack_list]
out = [h5py.File(f"{name}.tmp", "w") for name in filenames]
dsets = [f.create_dataset("FF", shape=(n_cfg, 6, 6, n_level, n_w, Lt), dtype="<c16") for f in out]

for icfg, cfg in enumerate(cfg_list):
    print("reading cfg", cfg, flush=True)
    with h5py.File(FF_name(cfg), "r") as f:
        assert [int(v) for v in f["symmetric_corr"].attrs["config_list"]] == [cfg], f"cfg {cfg}: the file holds cfg {f['symmetric_corr'].attrs['config_list']}"
        assert f.attrs["boost_direction"] == boost_direction and f.attrs["smear"] == smear, f"cfg {cfg}: the file is {f.attrs['smear']}, direction {f.attrs['boost_direction']}"
        for (name, iq), dset in zip(read_plan, dsets):
            dset[icfg] = f[name][0, :, :, :n_level, :n_w, iq, :]       # (munu, rhosig, smearing level, w, t)

for q, (name, iq), f, filename in zip(q_pack_list, read_plan, out, filenames):
    f.create_dataset("q", data=np.array(q))
    f.create_dataset("cfg_list", data=np.array(cfg_list))
    f.create_dataset(f"{smear}_list", data=np.arange(n_level))
    f.create_dataset("w_list", data=np.arange(n_w))
    f.attrs["dim_FF"] = f"cfg, munu, rhosig, {smear}, w, t"
    f.attrs["munu_order"] = "XY, YZ, XZ, XT, YT, ZT (the same order for rhosig), as seen from the boost direction: see boost_direction_note"
    f.attrs["time_index"] = f"t = absolute time slice 0 .. {Lt - 1}, as in the per-cfg files (not rolled to any source)"
    f.attrs["smear"] = smear
    f.attrs["smearing"] = smearing_note
    f.attrs["boost_direction"] = boost_direction
    f.attrs["boost_direction_note"] = file_attrs["boost_direction_note"]
    f.attrs["q_phase_sign"] = file_attrs["q_phase_sign"]
    f.attrs["momentum_phase"] = str(file_attrs["momentum_phase"]) + ", x = start point of the Wilson line"
    f.attrs["bilocal_direction"] = file_attrs["bilocal_direction"]
    f.attrs["centering"] = ("not applied: for the operator centered at x - w/2 along the boost direction multiply by exp(-i pi q_z w / Ls), "
                            "q_z = the third component of the q label (= 1 for q_z = 0)")
    f.attrs["pairing"] = ("contract with the 2pt of the same q and the same boost direction from delta_package_byq_opt.py (source weight "
                          "exp(-i 2pi/Ls q.x_src)): the 3pt has P_f = P_i + q. The momentum_relation and physical_transfer_convention attributes of "
                          "the per-cfg FF files are for the old unconjugated 2pt, not this one")
    f.attrs["source_dataset"] = f"{name}, q index {iq}"
    f.attrs["source_files"] = str(FF_name("{cfg}"))
    f.close()
    Path(f"{filename}.tmp").replace(filename)
    print("saved", filename)
