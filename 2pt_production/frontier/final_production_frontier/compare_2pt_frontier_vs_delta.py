#Compares the 12 .h5 files of one cfg written on Frontier with the same files of the Delta final run (cfg 204 for the check):
#    python3 compare_2pt_frontier_vs_delta.py DELTA_CFG_DIR FRONTIER_CFG_DIR
#The small datasets (q_list, momentum_list, ...) and all attributes but gauge_file must be identical. For pt2_forward and pt2_backward it prints the
#largest |Frontier - Delta| relative to the largest |Delta|, over the whole dataset and at the worst tsep (relative to the largest |Delta| at that
#tsep, since the correlators fall with tsep). Both machines solve to a relative residual of 1e-10 with different mrhs and hardware, so expect small
#nonzero differences; 1e-3 or more means the two computations differ. Reads one t_src at a time.
import os
import sys
import h5py
import numpy as np

delta_dir, frontier_dir = sys.argv[1], sys.argv[2]
worst = 0.0
for name in sorted(f for f in os.listdir(delta_dir) if f.endswith(".h5")):
    if not os.path.exists(f"{frontier_dir}/{name}"):
        print(f"{name}: not in the Frontier folder")
        continue
    with h5py.File(f"{delta_dir}/{name}", "r") as D, h5py.File(f"{frontier_dir}/{name}", "r") as F:
        for key in D:
            if key not in F:
                print(f"PROBLEM {name}: no {key} in the Frontier file")
            elif key not in ["pt2_forward", "pt2_backward"]:
                if not np.array_equal(D[key][()], F[key][()]):
                    print(f"PROBLEM {name}: {key} differs")
            else:
                for att in D[key].attrs:
                    if att != "gauge_file" and not np.array_equal(np.asarray(D[key].attrs[att]), np.asarray(F[key].attrs[att])):
                        print(f"PROBLEM {name}: attribute {att} of {key} differs")
                diff_t, scale_t = 0, 0                     # largest |F - D| and |D| per tsep, over all t_src
                for i in range(D[key].shape[-4]):          # [..., t_src, q, p, tsep]
                    d, f = D[key][..., i, :, :, :], F[key][..., i, :, :, :]
                    axes = tuple(range(d.ndim - 1))
                    diff_t = np.maximum(diff_t, np.abs(f - d).max(axis=axes))
                    scale_t = np.maximum(scale_t, np.abs(d).max(axis=axes))
                rel_all = diff_t.max() / scale_t.max()
                rel_t = diff_t / np.where(scale_t > 0, scale_t, 1)
                worst = max(worst, rel_all, rel_t.max())
                print(f"{name[:-3]} {key}: {rel_all:.2e} overall, {rel_t.max():.2e} at tsep {int(np.argmax(rel_t))}")
print(f"WORST relative difference {worst:.2e}: " + ("AGREE (solver-tolerance size)" if worst < 1e-3 else "MISMATCH"))
