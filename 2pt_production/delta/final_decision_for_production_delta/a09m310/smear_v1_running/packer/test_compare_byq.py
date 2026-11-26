#TEST: compares the packs of test_orig_byq.py (delta_package_byq.py) and test_opt_byq.py A (delta_package_byq_opt.py) file by file: every dataset
#(values and dtype) and every attribute must be identical. test_byq.sh runs it after both; by hand: python3 test_compare_byq.py
import os
import numpy as np
import h5py

a_dir = "/work/hdd/biqz/sliu54/2pt_production/packed/test_par/orig"
b_dir = "/work/hdd/biqz/sliu54/2pt_production/packed/test_par/opt_A"
names = sorted(os.listdir(a_dir))
assert names == sorted(os.listdir(b_dir)), "different file lists"
for n in names:
    with h5py.File(f"{a_dir}/{n}", "r") as a, h5py.File(f"{b_dir}/{n}", "r") as b:
        assert sorted(a) == sorted(b) and sorted(a.attrs) == sorted(b.attrs), n
        for k in a:
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k][()], b[k][()]), (n, k)
        for k in a.attrs:
            assert np.array_equal(a.attrs[k], b.attrs[k]), (n, k)
with h5py.File(f"{a_dir}/{names[0]}", "r") as a:
    print("all", len(names), "files identical; cfgs", a["cfg_list"][:].tolist(), "layouts", a["source_layout"].asstr()[:].tolist())
