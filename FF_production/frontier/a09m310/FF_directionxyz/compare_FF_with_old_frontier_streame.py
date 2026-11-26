#Compares an FF file of the new script (FF_covshift_boost_direction_frontier_streame.py, --boost_direction z) with the old FF of the same cfg.
#The Frontier copy of FF_production/delta/compare_FF_with_old_streame.py: only this header changed (the usage lines below).
#Run on a Frontier login node after the FF test job has finished. On Frontier the whole old per-cfg file is there (all 16 HYP levels, 16 lines, 5 + 27 q):
#    . /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh
#    cd /lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/FF_production/FF_data
#    python3 <this folder>/compare_FF_with_old_frontier_streame.py streame_direction_z/FF_opp_symmetric_asymmetric_hyp0-15_w0-15_boostz_cfg204.h5 FF_opp_symmetric_asymmetric_hyp0-15_w0-15_cfg204.h5
#Flow: the old flow files are /lustre/orion/lgt132/scratch/sicheng/GPD_calc/FF_data/FF_opp_symmetric_asymmetric_flow0-40_cfg<cfg>.h5 (read by
#3pt_production_simple/build_3pt.py); a dataset the old file does not have is reported as "not compared".
#The old file can be
#  - a whole old per-cfg file of the old Frontier production (FF_covshift_oddeven_trimag_negative_direction_hyp.py), as above, or a flow file with the same layout;
#  - or FF_production/delta/FF_old_reference_cfg204_hyp0-8_w0-9.h5 (cfg 204 of that production, 5 symmetric q, HYP levels 0-8, Wilson lines 0-9).
#Every dataset that both files have (symmetric_corr, asymmetric_corr) is compared on the smearing levels and Wilson lines that both have.
#Printed per smearing level, two measures:
#    (a) the largest difference divided by the largest old number of that level;
#    (b) the largest relative difference of a single number, |new - old| / |old|.
#Verdict: SAME if (a) is below --threshold (1e-8) and (b) below --threshold_single (1e-6) at every level. The old numbers come from the same machine,
#possibly with another software build, so the difference may be exactly zero or at the level of rounding (about 1e-12).
import argparse
import sys

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("new_file", type=str)
parser.add_argument("old_file", type=str)
parser.add_argument("--threshold", type=float, default=1e-8)
parser.add_argument("--threshold_single", type=float, default=1e-6)
args = parser.parse_args()

worst = 0.0
worst_single = 0.0
n_total = 0
with h5py.File(args.new_file, "r") as f_new, h5py.File(args.old_file, "r") as f_old:
    print(f"new: {args.new_file}")
    print(f"old: {args.old_file}")
    if "boost_direction" in f_new.attrs:
        assert f_new.attrs["boost_direction"] == "z", f"the old FF has the Wilson line along z; the new file has boost_direction {f_new.attrs['boost_direction']}"
    if "smear" in f_new.attrs and "smear" in f_old.attrs:
        assert f_new.attrs["smear"] == f_old.attrs["smear"], f"smearing: new {f_new.attrs['smear']}, old {f_old.attrs['smear']}"
    for kind in ["symmetric", "asymmetric"]:
        name = f"{kind}_corr"
        if name not in f_new or name not in f_old:
            print(f"{name}: not in both files, not compared")
            continue
        assert np.array_equal(f_new[f"{kind}_qlist"][:], f_old[f"{kind}_qlist"][:]), f"{kind}_qlist differs"
        cfg_new = list(f_new[name].attrs["config_list"])
        if "config_list" in f_old[name].attrs:                               # an old flow file may not have it
            cfg_old = list(f_old[name].attrs["config_list"])
            assert cfg_new == cfg_old, f"{name}: new file is cfg {cfg_new}, old file is cfg {cfg_old}"
        n_smear = min(f_new[name].shape[3], f_old[name].shape[3])            # (cfg, munu, rhosig, smear level, wilson line, q, t)
        n_w = min(f_new[name].shape[4], f_old[name].shape[4])
        new = f_new[name][0, :, :, :n_smear, :n_w]
        old = f_old[name][0, :, :, :n_smear, :n_w]
        assert new.shape == old.shape, f"{name}: shapes {new.shape} and {old.shape}"
        assert np.isfinite(new).all() and np.isfinite(old).all(), f"{name}: a number is not finite"
        print(f"{name}: cfg {cfg_new[0]}, {new.shape[0]} x {new.shape[1]} components, smearing levels 0-{n_smear - 1}, Wilson lines 0-{n_w - 1}, "
              f"{new.shape[4]} q, {new.shape[5]} t: {new.size} complex numbers")
        print(f"    {'level':>5s} {'largest old':>12s} {'largest diff':>13s} {'(a) diff/largest':>17s} {'(b) single number':>18s}  bitwise")
        for level in range(n_smear):
            a, b = new[:, :, level], old[:, :, level]
            assert (b != 0).all(), f"{name} level {level}: an old number is exactly zero"
            ratio = float(abs(a - b).max() / abs(b).max())
            single = float((abs(a - b) / abs(b)).max())
            worst, worst_single = max(worst, ratio), max(worst_single, single)
            print(f"    {level:5d} {abs(b).max():12.4e} {abs(a - b).max():13.4e} {ratio:17.3e} {single:18.3e}  {'yes' if np.array_equal(a, b) else 'no'}")
        n_total += new.size

assert n_total > 0, "no dataset in common"
print(f"\ncompared {n_total} complex numbers; (a) largest difference / largest old number of the level: {worst:.3e}; (b) largest relative difference of one number: {worst_single:.3e}")
if worst < args.threshold and worst_single < args.threshold_single:
    print(f"VERDICT: SAME ((a) below {args.threshold:g} and (b) below {args.threshold_single:g})")
else:
    print(f"VERDICT: DIFFERENT ((a) limit {args.threshold:g}, (b) limit {args.threshold_single:g})")
    sys.exit(1)
