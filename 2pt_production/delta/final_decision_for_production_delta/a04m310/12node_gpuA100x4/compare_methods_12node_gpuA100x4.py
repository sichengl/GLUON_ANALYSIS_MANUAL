#Cross-check of the a04m310 12node_gpuA100x4 node tests: the four methods ran the same cfgs and sources on the same 48-GPU grid with the same setup
#and the same QUDA tuning cache, so only the code differs. Every output file of method_1_mrhs1 (production code) is compared with the same file of
#the other three methods: the .h5 files of finished cfgs (datasets pt2_forward, pt2_backward) and the parts/*.npy of the t_src saved so far.
#Read only. Run on Delta after the jobs:  python compare_methods_12node_gpuA100x4.py
#Printed per method: files compared, how many are bit for bit identical, and the largest max|a - b| / max|a| over the files (a = method_1).
import os
import numpy as np
import h5py

base = "/work/hdd/biqz/sliu54/2pt_production/a04m310_node_test"
ref = "method_1_mrhs1"
others = ["method_2_lean", "method_3_hostpark", "method_4_hostpark_phases"]
label = "12node_gpuA100x4"

ref_dir = f"{base}/{label}_{ref}"
ref_files = sorted(os.path.relpath(os.path.join(root, name), ref_dir)
                   for root, dirs, names in os.walk(ref_dir) for name in names if name.endswith((".h5", ".npy")))
print(f"{ref}: {len(ref_files)} files (.h5 and parts/*.npy) in {ref_dir}")

for method in others:
    n_compared, n_identical, n_missing = 0, 0, 0
    worst, worst_file = 0.0, None
    for rel in ref_files:
        other = os.path.join(f"{base}/{label}_{method}", rel.replace(f"{label}_{ref}", f"{label}_{method}"))      # the method name is in every folder and file name
        if not os.path.exists(other):
            n_missing += 1
            continue
        if rel.endswith(".npy"):
            pairs = [(np.load(os.path.join(ref_dir, rel)), np.load(other))]
        else:
            with h5py.File(os.path.join(ref_dir, rel), "r") as fa, h5py.File(other, "r") as fb:
                pairs = [(fa[name][()], fb[name][()]) for name in ["pt2_forward", "pt2_backward"]]
        for a, b in pairs:
            assert a.shape == b.shape, (rel, a.shape, b.shape)
            n_compared += 1
            n_identical += int(np.array_equal(a, b))
            rel_diff = np.abs(a - b).max() / np.abs(a).max()
            if rel_diff > worst:
                worst, worst_file = rel_diff, rel
    print(f"{method}: {n_compared} arrays compared, {n_identical} bit for bit identical, {n_missing} files not there yet, "
          f"largest max|a - b| / max|a| = {worst:.2e}" + (f" in {worst_file}" if worst_file else ""))
