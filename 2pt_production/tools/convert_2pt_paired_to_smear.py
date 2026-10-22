#Converts the per-cfg 2pt files of the earlier final runs (smear_tag ..._paired, arrays (shape, shape, frac, frac, gamma, gamma, ...)) to the
#current layout (smear_tag ..._smear, arrays (smear, smear, gamma, gamma, ...)), then checks every converted file against its original.
#The original files are only read: nothing is changed or deleted. Smear i is shape i with boost i, the only (shape, frac) pairs the earlier runs
#computed, so new[a, b] = old[a, b, a, b]; the check also confirms that every other (shape, frac) block of the original is zero, i.e. nothing is lost.
#The converted files go where the current production writes (gevp_dir with the _smear tag, same file names), with the attributes the current
#production writes, so the production job skips these cfgs and the byq packer reads them like new ones. Each file is written as .tmp and renamed when complete.
#Only cfgs with all 12 original files are converted; a file already there in the new folder is not overwritten (the check still compares it).
#Run it from the folder of the setup, CPU only, e.g. on Delta:
#    cd 2pt_production/delta/final_decision_for_production_delta
#    python3 ../../tools/convert_2pt_paired_to_smear.py sepq_2pt_setup_delta_final
#Options: --cfgs 204 210 (only these cfgs), --check-only (check the files converted earlier, convert nothing)
import os
import sys
import argparse
import importlib
import itertools
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("setup", help="setup module in the current folder, e.g. sepq_2pt_setup_delta_final")
parser.add_argument("--cfgs", type=int, nargs="+", help="only these cfgs (default: every cfg of cfg_list with all 12 original files)")
parser.add_argument("--check-only", action="store_true", help="check the files converted earlier, convert nothing")
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.getcwd(), here]          # the setup from the current folder; it imports pt2_comm_tools from 2pt_production/tools (this folder)
setup = importlib.import_module(args.setup)
params = setup.delta_params
assert setup.smear_tag.endswith("_smear"), f"smear_tag {setup.smear_tag}: expected the current tag, ending in _smear"
old_tag = setup.smear_tag[:-len("_smear")] + "_paired"
n_src = len(params.t_base) * setup.n_spatial_src
tsep_max = params.tsep_max
cases = ["forward", "sym_0xi", "sym_non0xi", "asy_0xi", "asy_non0xi", "plain"]
particles = ["pion", "proton"]
pt2_names = ["pt2_forward", "pt2_backward"]
smear_list = np.array(params.smear_list)
n_smear = len(smear_list)

def gevp_dir(tag):
    return f"{setup.current_dir}/{tag}_GEVP_sepq_{n_src}src_phyp_fb_tsep{tsep_max}"         # gevp_dir of the job script

def h5_name(tag, case, particle, cfg):
    return f"{gevp_dir(tag)}/cfg{cfg}/{particle}_{tag}_GEVP_sepq_{case}_phyp_fb_tsep{tsep_max}_cfg{cfg}.h5"

def text(s):
    return s.decode() if isinstance(s, bytes) else str(s)

def same(x, y):
    #equal attribute or dataset values: strings compared as text, numbers exactly
    x, y = np.asarray(x), np.asarray(y)
    if x.dtype.kind in "OSU" or y.dtype.kind in "OSU":
        return x.shape == y.shape and [text(v) for v in x.ravel()] == [text(v) for v in y.ravel()]
    return x.shape == y.shape and np.array_equal(x, y)

def old_smear_list(f):
    #rows [rho_T, rho_z, mom_frac] of the original file: shape i with frac i
    return np.hstack([f["shape_list"][:], f["mom_frac_list"][:][:, None]])

#the dataset attributes the current production writes differently from the earlier one; all the others are copied as they are
def new_attrs(old, particle):
    attrs = {k: v for k, v in old.items() if k not in ("shape_names", "shape_list", "mom_frac_list")}
    head = [text(s) for s in old["dim_spec"][:6]]
    assert head == ["shape_sink", "shape_source", "frac_sink", "frac_source", "gamma_sink", "gamma_source"], f"dim_spec starts with {head}"
    attrs["dim_spec"] = np.array(["smear_sink", "smear_source", "gamma_sink", "gamma_source"] + [text(s) for s in old["dim_spec"][6:]], dtype=h5py.string_dtype())
    attrs["smear_names"] = np.array(setup.smear_names, dtype=h5py.string_dtype())
    attrs["smear_list"] = smear_list
    attrs["smear_list_note"] = "rows [rho_T, rho_z, mom_frac] of smear_sink and smear_source; quark boost k = mom_frac * mom_max in units of 2pi/L along z (k_list dataset)"
    attrs["smearing"] = setup.smearing_note
    if particle == "proton":
        propagator = text(old["propagator"])
        for a, b in [("(shape_source, frac_source)", "smear_list[smear_source]"), ("(shape_sink, frac_sink)", "smear_list[smear_sink]")]:
            assert propagator.count(a) == 1, f"propagator note: {propagator}"
            propagator = propagator.replace(a, b)
        attrs["propagator"] = propagator
    return attrs


def convert(case, particle, cfg):
    old_name, new_name = h5_name(old_tag, case, particle, cfg), h5_name(setup.smear_tag, case, particle, cfg)
    if os.path.exists(new_name):
        return "there"
    os.makedirs(os.path.dirname(new_name), exist_ok=True)
    with h5py.File(old_name, "r") as fo, h5py.File(f"{new_name}.tmp", "w") as fn:
        assert np.array_equal(old_smear_list(fo), smear_list), f"{old_name}: shape_list + mom_frac_list {old_smear_list(fo).tolist()} is not smear_list {smear_list.tolist()}"
        for name in fo:
            if name not in pt2_names + ["shape_list", "mom_frac_list"]:
                fn.create_dataset(name, data=fo[name][:])          # q_list, momentum_list, pz_list, k_list, (parity_p, parity_m)
        fn.create_dataset("smear_list", data=smear_list)
        for pt2 in pt2_names:
            old = fo[pt2]
            assert old.shape[:4] == (n_smear,) * 4, f"{old_name} {pt2}: shape {old.shape}"
            new = fn.create_dataset(pt2, shape=old.shape[:2] + old.shape[4:], dtype=old.dtype)
            for a in range(n_smear):
                for b in range(n_smear):
                    new[a, b] = old[a, b, a, b]                     # sink smearing a, source smearing b
            for key, value in new_attrs(dict(old.attrs), particle).items():
                new.attrs[key] = value
        fn.attrs["converted_from"] = old_name
    os.replace(f"{new_name}.tmp", new_name)
    return "converted"


def check(case, particle, cfg):
    #every way the new file can differ from its original; an empty list means consistent
    old_name, new_name = h5_name(old_tag, case, particle, cfg), h5_name(setup.smear_tag, case, particle, cfg)
    if not os.path.exists(new_name):
        return ["new file missing"]
    problems = []
    with h5py.File(old_name, "r") as fo, h5py.File(new_name, "r") as fn:
        converted = "converted_from" in fn.attrs                   # otherwise the current production made this file: same physics, own rounding
        expected = (set(fo) - {"shape_list", "mom_frac_list"}) | {"smear_list"}
        if set(fn) != expected:
            problems.append(f"datasets {sorted(fn)}, expected {sorted(expected)}")
        if "smear_list" in fn and not np.array_equal(fn["smear_list"][:], old_smear_list(fo)):
            problems.append(f"smear_list {fn['smear_list'][:].tolist()} is not shape_list + mom_frac_list of the original {old_smear_list(fo).tolist()}")
        for name in (set(fo) & set(fn)) - set(pt2_names):
            if not same(fo[name][:], fn[name][:]):
                problems.append(f"dataset {name} differs")
        for pt2 in pt2_names:
            old, new = fo[pt2], fn[pt2]
            if new.shape != old.shape[:2] + old.shape[4:]:
                problems.append(f"{pt2}: shape {new.shape}, expected {old.shape[:2] + old.shape[4:]}")
                continue
            for a, b in itertools.product(range(n_smear), repeat=2):
                block_new, block_old = new[a, b], old[a, b, a, b]
                if converted and not np.array_equal(block_new, block_old):
                    problems.append(f"{pt2} smear ({a}, {b}): differs from the original block")
                if not converted and not np.allclose(block_new, block_old, rtol=0, atol=1e-6 * np.abs(block_old).max()):
                    problems.append(f"{pt2} smear ({a}, {b}): recomputed by the current production, differs from the original by more than 1e-6 of its largest value")
                if not np.any(block_new):
                    problems.append(f"{pt2} smear ({a}, {b}): all zero")
                for c, d in itertools.product(range(n_smear), repeat=2):
                    if (c, d) != (a, b) and np.any(old[a, b, c, d]):
                        problems.append(f"{pt2}: original block (shape {a}, {b}, frac {c}, {d}) is not zero, and the new layout has no place for it")
            expected_attrs = new_attrs(dict(old.attrs), particle)
            if set(new.attrs) != set(expected_attrs):
                problems.append(f"{pt2}: attributes {sorted(new.attrs)}, expected {sorted(expected_attrs)}")
            for key in set(new.attrs) & set(expected_attrs):
                if not same(new.attrs[key], expected_attrs[key]):
                    problems.append(f"{pt2}: attribute {key} differs")
    return problems


cfgs = args.cfgs if args.cfgs else [int(cfg) for cfg in setup.cfg_list]
cfg_done = [cfg for cfg in cfgs if all(os.path.exists(h5_name(old_tag, case, particle, cfg)) for case in cases for particle in particles)]
print(f"original files: {gevp_dir(old_tag)}")
print(f"new files:      {gevp_dir(setup.smear_tag)}")
print(f"{len(cfg_done)} of {len(cfgs)} cfgs have all 12 original files; not complete there: {[cfg for cfg in cfgs if cfg not in cfg_done]}")
assert cfg_done, "no cfg to convert"

failed = {}
for cfg in cfg_done:
    if not args.check_only:
        status = [convert(case, particle, cfg) for case in cases for particle in particles]
        print(f"cfg {cfg}: {status.count('converted')} files converted, {status.count('there')} already in the new folder", flush=True)
    problems = [f"{particle} {case}: {p}" for case in cases for particle in particles for p in check(case, particle, cfg)]
    if problems:
        failed[cfg] = problems
        print(f"cfg {cfg}: NOT CONSISTENT", *problems, sep="\n    ", flush=True)
    else:
        print(f"cfg {cfg}: consistent", flush=True)

print(f"{len(cfg_done) - len(failed)} of {len(cfg_done)} cfgs consistent" + (f"; not consistent: {sorted(failed)}" if failed else ""))
sys.exit(1 if failed else 0)
