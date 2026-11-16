#Compares the 2pt of the test job (test_2pt_same_as_before_cfg204.sh, --boost_direction z) with the earlier Fermilab run of the same cfg.
#The Fermilab copy of delta/final_decision_for_production_delta/smear_v1_directionxyz/compare_2pt_with_production.py, which also reads the earlier
#layout: the files of the runs before 2026-10-22 (smear_tag ..._paired) have the axes (shape, shape, frac, frac, gamma, gamma, ...) and only their
#blocks [a, b, a, b] are computed, so those blocks are compared with [a, b] of the new files (..._smear, axes (smear, smear, gamma, gamma, ...)).
#Run on a Fermilab login node after the test job has ended (TIMEOUT is its expected end), from this folder:
#    source ~/miniforge3/etc/profile.d/conda.sh; conda activate pyquda
#    python3 compare_2pt_with_old_run.py
#It only reads. It first checks that the earlier files have the smearings of this setup and the source positions this setup gives the cfg.
#Then for each of the 12 files (pion and proton x 6 cases) and for pt2_forward and pt2_backward it takes every t_src the test has saved
#(the parts of an unfinished cfg, or the .h5 of a finished one) and compares it with the same t_src of the earlier .h5 file.
#Printed per file: how many numbers were compared, whether they are equal bit for bit, and the largest relative difference of a single number,
#|new - old| / |old|. Verdict: IDENTICAL if every number is equal bit for bit, SAME if the largest relative difference is below --threshold
#(1e-6), DIFFERENT otherwise. Between machines the numbers differed by up to 7e-8 at solver tolerance 1e-10, so 1e-9 to 1e-7 would be a
#solver-level difference and anything larger a real one.
import argparse
import glob
import os
import sys

import h5py
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..", "..", "..", "tools")]        # the setup imports pt2_comm_tools from 2pt_production/tools, three levels up from this folder
from sepq_2pt_setup_fermilab_final import delta_params

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=int, default=204)                                                                    # cfg number, cfg_list[ICFG] of the test job
parser.add_argument("--test_dir", type=str, default="/lustre2/gluonp0/sliu1/2pt_production/test_directionxyz")         # TEST_DIR of the test job
parser.add_argument("--old_dir", type=str, default="/lustre2/gluonp0/sliu1/2pt_production/final_production_3072src_200cfg")   # current_dir of the earlier run
parser.add_argument("--threshold", type=float, default=1e-6)
args = parser.parse_args()

#the cfg folder of the test ("_smear_GEVP": the z names; the x and y folders, _smear_boostx_GEVP, are not matched) and of the earlier run,
#in the old layout (_paired) or the current one (_smear); only an earlier folder with all 12 .h5 files is used
test_folders = glob.glob(f"{args.test_dir}/*_smear_GEVP_sepq_*/cfg{args.cfg}")
assert len(test_folders) == 1, f"expected one test folder for cfg {args.cfg} in {args.test_dir}, found {test_folders}"
test_folder = test_folders[0]
old_folders = [folder for folder in sorted(glob.glob(f"{args.old_dir}/*_paired_GEVP_sepq_*/cfg{args.cfg}") + glob.glob(f"{args.old_dir}/*_smear_GEVP_sepq_*/cfg{args.cfg}"))
               if len(glob.glob(f"{folder}/*.h5")) == 12]
assert old_folders, f"no finished folder (12 .h5 files) of cfg {args.cfg} in {args.old_dir}"
test_tag = os.path.basename(os.path.dirname(test_folder)).split("_GEVP_sepq_")[0]          # coulomb_..._smear
print(f"test:    {test_folder}")

#the smearings [rho_T, rho_z, mom_frac] and the source positions of cfg args.cfg according to the setup of this folder
smear_list = np.array(delta_params.smear_list)
n_smear = len(smear_list)
t_src, x_src, y_src, z_src = delta_params.make_sources([args.cfg])
src_expected = {"t_src_list": t_src[0], "x_src_list": x_src[0], "y_src_list": y_src[0], "z_src_list": z_src[0]}

worst = 0.0
n_total = 0
all_bitwise = True
t_seen = set()
for old_folder in old_folders:
    old_tag = os.path.basename(os.path.dirname(old_folder)).split("_GEVP_sepq_")[0]        # coulomb_..._paired or coulomb_..._smear
    paired = old_tag.endswith("_paired")
    assert old_tag.replace("_paired", "_smear") == test_tag, f"the test tag {test_tag} and the earlier tag {old_tag} name different smearings"
    print(f"earlier: {old_folder}  ({'old layout, blocks [a, b, a, b]' if paired else 'current layout'})")
    print(f"{'file':28s} {'dataset':13s} {'t_src index':12s} {'numbers':>10s}  {'bitwise':8s} {'max |new-old|/|old|':>19s}")
    for old_file in sorted(glob.glob(f"{old_folder}/*.h5")):
        base_old = os.path.basename(old_file)                                               # {particle}_{old_tag}_GEVP_sepq_{case}_phyp_fb_tsep20_cfg{cfg}.h5
        base = base_old.replace(old_tag, test_tag, 1)                                        # the name of the same file in the test
        label = base.split("_")[0] + " " + base.split("_GEVP_sepq_")[1].split("_phyp_")[0]
        test_h5 = f"{test_folder}/{base}"
        f_test = h5py.File(test_h5, "r") if os.path.exists(test_h5) else None              # only there if the test finished the cfg
        with h5py.File(old_file, "r") as f_old:
            #the earlier file must have the smearings of this setup and the sources this setup gives the cfg
            old_smear = np.hstack([f_old["shape_list"][:], f_old["mom_frac_list"][:][:, None]]) if paired else f_old["smear_list"][:]
            assert np.array_equal(old_smear, smear_list), f"{base_old}: smearings {old_smear.tolist()}, this setup {smear_list.tolist()}"
            for name in src_expected:
                assert np.array_equal(f_old["pt2_forward"].attrs[name], src_expected[name]), f"{base_old}: {name} differs from the sources of this setup"
            for pt2 in ["pt2_forward", "pt2_backward"]:
                dset = f_old[pt2]
                n_t = dset.shape[-4]                                                        # (..., t_src, q, p, tsep)
                used, n, bitwise, max_rel = [], 0, True, 0.0
                for t_idx in range(n_t):
                    part = f"{test_folder}/parts/{base[:-3]}_t{t_idx}_{pt2}.npy"
                    if f_test is not None:
                        new = f_test[pt2][..., t_idx, :, :, :]
                    elif os.path.exists(part):
                        new = np.load(part)
                    else:
                        continue
                    if paired:
                        old = np.array([[dset[a, b, a, b, ..., t_idx, :, :, :] for b in range(n_smear)] for a in range(n_smear)])   # sink smearing a, source smearing b
                    else:
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
print(f"\nsmearings and source positions of the earlier files = this setup's for cfg {args.cfg}")
print(f"compared {n_total} complex numbers, t_src indices {sorted(t_seen)}")
print(f"all equal bit for bit: {'yes' if all_bitwise else 'no'};  largest relative difference of one number: {worst:.3e}")
if all_bitwise:
    print("VERDICT: IDENTICAL (every number equal bit for bit)")
elif worst < args.threshold:
    print(f"VERDICT: SAME (not bit for bit, but every number agrees to better than {args.threshold:g})")
else:
    print(f"VERDICT: DIFFERENT (a number differs by more than {args.threshold:g})")
    sys.exit(1)
