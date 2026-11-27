#Compares the 2pt of the two smearing test jobs of test_2pt_pyqudafft.sh: SMEARING=pyqudafft against SMEARING=production, the same 5 cfgs, the same
#24 sources per cfg, the same grid [1, 1, 1, 4] and QUDA tuning cache, so only the smearing module differs.
#Run on a Delta login node after both jobs have ended:
#    source ~/env_pyquda.sh
#    python compare_2pt_pyqudafft_with_production.py                               grid1114_pyqudafft against grid1114_production
#    python compare_2pt_pyqudafft_with_production.py --test grid2211_pyqudafft     the x-y split run, if it is done later
#It only reads. For every cfg that both jobs have finished (12 .h5 files each) it compares pt2_forward and pt2_backward number by number, and
#checks that the other datasets (q_list, momentum_list, ...) are equal. Printed per cfg: how many numbers, whether they are equal bit for bit,
#and the largest relative difference of a single number, |new - old| / |old|. Verdict: IDENTICAL if every number is equal bit for bit, SAME if
#the largest relative difference is below --threshold (1e-6), DIFFERENT otherwise.
#Expected for grid1114_pyqudafft: IDENTICAL (on this grid the PyQUDA FFT smearing does the same arithmetic; the CPU mock was bit for bit).
#Another grid changes the solver, the gauge fixing and the sums: 1e-9 to 1e-7 would be a solver-level difference, anything larger a real one.
import argparse
import glob
import os
import sys

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="/work/hdd/biqz/sliu54/2pt_production/test_smearing_pyqudafft")      # TEST_DIR of the jobs without the last folder
parser.add_argument("--ref", type=str, default="grid1114_production")
parser.add_argument("--test", type=str, default="grid1114_pyqudafft")
parser.add_argument("--threshold", type=float, default=1e-6)
args = parser.parse_args()

ref_dir, test_dir = f"{args.base}/{args.ref}", f"{args.base}/{args.test}"
ref_cfg_dirs = sorted(glob.glob(f"{ref_dir}/*_GEVP_sepq_*/cfg*"))
assert ref_cfg_dirs, f"no cfg folder in {ref_dir}"
print(f"reference: {ref_dir}")
print(f"test:      {test_dir}")

worst = 0.0
n_total = 0
all_bitwise = True
compared = []
for ref_cfg_dir in ref_cfg_dirs:
    rel = os.path.relpath(ref_cfg_dir, ref_dir)                  # <smear_tag>_GEVP_sepq_24src_phyp_fb_tsep20/cfg<N>: the same in both jobs
    ref_files = sorted(glob.glob(f"{ref_cfg_dir}/*.h5"))
    test_files = sorted(glob.glob(f"{test_dir}/{rel}/*.h5"))
    if len(ref_files) != 12 or len(test_files) != 12:
        print(f"{rel}: not finished by both jobs ({len(ref_files)} and {len(test_files)} of 12 .h5 files), not compared")
        continue
    assert [os.path.basename(f) for f in ref_files] == [os.path.basename(f) for f in test_files], f"{rel}: the file names differ"
    n, bitwise, max_rel = 0, True, 0.0
    for ref_file, test_file in zip(ref_files, test_files):
        with h5py.File(ref_file, "r") as f_ref, h5py.File(test_file, "r") as f_test:
            assert sorted(f_ref.keys()) == sorted(f_test.keys()), f"{test_file}: other datasets than in {ref_file}"
            for name in f_ref.keys():
                if name not in ["pt2_forward", "pt2_backward"]:
                    assert np.array_equal(f_ref[name][()], f_test[name][()]), f"{test_file}: dataset {name} differs"
            for pt2 in ["pt2_forward", "pt2_backward"]:
                old = f_ref[pt2][()]                             # (..., t_src, q, p, tsep)
                new = f_test[pt2][()]
                assert new.shape == old.shape, f"{test_file} {pt2}: shapes {new.shape} and {old.shape}"
                assert np.isfinite(new).all() and np.isfinite(old).all(), f"{test_file} {pt2}: a number is not finite"
                assert np.any(new != 0), f"{test_file} {pt2}: the test data is all zero"
                assert np.array_equal(new == 0, old == 0), f"{test_file} {pt2}: exact zeros at different places"
                n += new.size
                bitwise = bitwise and np.array_equal(new, old)
                nonzero = old != 0
                max_rel = max(max_rel, float((abs(new - old)[nonzero] / abs(old)[nonzero]).max()))
    print(f"{rel}: {n} numbers, bit for bit {'yes' if bitwise else 'no'}, largest |new - old| / |old| {max_rel:.3e}")
    worst, n_total, all_bitwise = max(worst, max_rel), n_total + n, all_bitwise and bitwise
    compared.append(os.path.basename(ref_cfg_dir))

assert n_total > 0, "no cfg is finished by both jobs yet"
print(f"\ncompared {n_total} complex numbers in {len(compared)} cfgs: {' '.join(compared)}")
print(f"all equal bit for bit: {'yes' if all_bitwise else 'no'};  largest relative difference of one number: {worst:.3e}")
if all_bitwise:
    print("VERDICT: IDENTICAL (every number equal bit for bit)")
elif worst < args.threshold:
    print(f"VERDICT: SAME (not bit for bit, but every number agrees to better than {args.threshold:g})")
else:
    print(f"VERDICT: DIFFERENT (a number differs by more than {args.threshold:g})")
    sys.exit(1)
