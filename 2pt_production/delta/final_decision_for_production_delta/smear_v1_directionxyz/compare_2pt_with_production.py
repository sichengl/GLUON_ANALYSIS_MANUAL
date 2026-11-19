#Compares the 2pt of the test job (test_2pt_same_as_production.sh, --boost_direction z) with the production 2pt of the same cfg.
#Run on a Delta login node after the test job has ended (TIMEOUT is its expected end):
#    source ~/env_pyquda.sh
#    python compare_2pt_with_production.py --cfg 2682
#It only reads. For each of the 12 files (pion and proton x 6 cases) and for pt2_forward and pt2_backward it takes every t_src the test has saved
#(the parts of an unfinished cfg, or the .h5 of a finished one) and compares it with the same t_src of the production .h5 file.
#Printed per file: how many numbers were compared, whether they are equal bit for bit, and the largest relative difference of a single number,
#|new - old| / |old|. Verdict: IDENTICAL if every number is equal bit for bit, SAME if the largest relative difference is below --threshold
#(1e-6), DIFFERENT otherwise. Expected: IDENTICAL (two runs of the same code on Delta were bitwise equal before). Between machines the numbers
#differed by up to 7e-8 at solver tolerance 1e-10, so 1e-9 to 1e-7 would be a solver-level difference and anything larger a real one.
import argparse
import glob
import os
import sys

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=int, default=2682)                                                                   # cfg number, 204 + 6 * ICFG of the test job
parser.add_argument("--test_dir", type=str, default="/work/hdd/biqz/sliu54/2pt_production/test_directionxyz")          # TEST_DIR of the test job
parser.add_argument("--prod_dir", type=str, default="/work/hdd/biqz/sliu54/2pt_production/final_production_3072src_200cfg")   # current_dir of the production setup
parser.add_argument("--threshold", type=float, default=1e-6)
args = parser.parse_args()

#the cfg folder of the test (z of the direction code: "_smear_direction_z_GEVP") and of the production (the old code: "_smear_GEVP")
test_folders = glob.glob(f"{args.test_dir}/*_smear_direction_z_GEVP_sepq_*/cfg{args.cfg}")
prod_folders = glob.glob(f"{args.prod_dir}/*_smear_GEVP_sepq_*/cfg{args.cfg}")
assert len(test_folders) == 1, f"expected one test folder for cfg {args.cfg} in {args.test_dir}, found {test_folders}"
assert len(prod_folders) == 1, f"expected one production folder for cfg {args.cfg} in {args.prod_dir}, found {prod_folders}"
test_folder, prod_folder = test_folders[0], prod_folders[0]
assert os.path.basename(os.path.dirname(test_folder)) == os.path.basename(os.path.dirname(prod_folder)).replace("_smear_GEVP_", "_smear_direction_z_GEVP_", 1), "the test folder is not the production folder with _direction_z"
prod_files = sorted(glob.glob(f"{prod_folder}/*.h5"))
assert len(prod_files) == 12, f"the production cfg {args.cfg} is not finished: {len(prod_files)} of 12 .h5 files in {prod_folder}"
print(f"test:       {test_folder}")
print(f"production: {prod_folder}")

worst = 0.0
n_total = 0
all_bitwise = True
t_seen = set()
print(f"{'file':28s} {'dataset':13s} {'t_src index':12s} {'numbers':>10s}  {'bitwise':8s} {'max |new-old|/|old|':>19s}")
for prod_file in prod_files:
    base = os.path.basename(prod_file)                       # {particle}_{smear_tag}_GEVP_sepq_{case}_phyp_fb_tsep20_cfg{cfg}.h5
    label = base.split("_")[0] + " " + base.split("_GEVP_sepq_")[1].split("_phyp_")[0]
    test_base = base.replace("_smear_GEVP_", "_smear_direction_z_GEVP_", 1)      # the same file in the test: the direction code adds _direction_z to the name
    test_h5 = f"{test_folder}/{test_base}"
    f_test = h5py.File(test_h5, "r") if os.path.exists(test_h5) else None      # only there if the test finished the cfg
    with h5py.File(prod_file, "r") as f_prod:
        for pt2 in ["pt2_forward", "pt2_backward"]:
            dset = f_prod[pt2]
            n_t = dset.shape[-4]                             # (..., t_src, q, p, tsep)
            used, n, bitwise, max_rel = [], 0, True, 0.0
            for t_idx in range(n_t):
                part = f"{test_folder}/parts/{test_base[:-3]}_t{t_idx}_{pt2}.npy"
                if f_test is not None:
                    new = f_test[pt2][..., t_idx, :, :, :]
                elif os.path.exists(part):
                    new = np.load(part)
                else:
                    continue
                old = dset[..., t_idx, :, :, :]
                assert new.shape == old.shape, f"{base} {pt2} t_src index {t_idx}: shapes {new.shape} and {old.shape}"
                assert np.isfinite(new).all() and np.isfinite(old).all(), f"{base} {pt2} t_src index {t_idx}: a number is not finite"
                assert np.any(new != 0), f"{base} {pt2} t_src index {t_idx}: the test data is all zero"
                used.append(t_idx)
                n += new.size
                bitwise = bitwise and np.array_equal(new, old)
                assert np.array_equal(new == 0, old == 0), f"{base} {pt2} t_src index {t_idx}: exact zeros at different places"
                nonzero = old != 0
                max_rel = max(max_rel, float((abs(new - old)[nonzero] / abs(old)[nonzero]).max()))
            if not used:
                print(f"{label:28s} {pt2:13s} nothing saved by the test")
                continue
            worst, n_total, all_bitwise = max(worst, max_rel), n_total + n, all_bitwise and bitwise
            t_seen.update(used)
            print(f"{label:28s} {pt2:13s} {str(used):12s} {n:10d}  {'yes' if bitwise else 'no':8s} {max_rel:19.3e}")
    if f_test is not None:
        f_test.close()

assert n_total > 0, "the test has saved nothing yet: wait for the first 'SAVE T_SRC' line in its log"
print(f"\ncompared {n_total} complex numbers, t_src indices {sorted(t_seen)}")
print(f"all equal bit for bit: {'yes' if all_bitwise else 'no'};  largest relative difference of one number: {worst:.3e}")
if all_bitwise:
    print("VERDICT: IDENTICAL (every number equal bit for bit)")
elif worst < args.threshold:
    print(f"VERDICT: SAME (not bit for bit, but every number agrees to better than {args.threshold:g})")
else:
    print(f"VERDICT: DIFFERENT (a number differs by more than {args.threshold:g})")
    sys.exit(1)
