#Storage check of the 2pt output on /work/hdd: the finished cfgs (.h5) and the parts of unfinished cfgs (.npy). Nothing is changed or deleted.
#
#Why it works: real 2pt data has no exact zeros (Delta cfg 204, t0: 0 of 14.6M numbers in the data blocks), while storage losses on /work/hdd
#(2026-10-15/16/21/26) read back as zeros: files cut at a MiB boundary, 0-byte files, lost 1 MiB pieces (65536 complex numbers) inside a file of the
#right size, which np.load and h5py read without complaint. So every exact zero in a data block is lost data.
#_paired files (the first 199 cfgs) keep shape i only with frac i: only the blocks [a,b,a,b] of the first four axes hold data, the other 12 of 16 are
#zeros by design. Those are not read (the packers never read them either), which also cuts the reading of a _paired file to a quarter.
#
#Per file:
#   .npy   the header reads, the size is header + shape x itemsize; zeros and non-finite numbers in the data
#   .h5    the file opens (HDF5 refuses a cut file) and every attribute reads; pt2_forward and pt2_backward exist, are contiguous, end inside the
#          file, and their measurements attribute is the cfg of the file name; zeros and non-finite numbers in them, and the t_src indices of the zeros
#The data is read with plain file reads in 64 MiB pieces from many threads at once (h5py is used only for the metadata), so the speed is set by the
#file system.
#
#Usage, in the pyquda env (no GPU, no MPI):
#   python check_2pt_storage.py CFG_DIR [CFG_DIR ...]          every *.h5 and parts/*.npy of these .../cfg<cfg> folders
#   python check_2pt_storage.py --all OUT_DIR [OUT_DIR ...]    every finished cfg (12 .h5, no parts/) of these *_GEVP_sepq_* output folders
#   --ledger FILE   adds one line per checked file (path, size, mtime, OK/BAD, zeros, non-finite, problems); a later run skips the files that
#                   were OK and still have the same size and mtime, so a job stopped at its time limit continues where it stopped
#   --workers N     files read at the same time (default 16)
#Output: "BAD <path> | <problems>" lines as they are found, progress lines, then one line per cfg folder.
import os
import re
import glob
import argparse
from time import perf_counter
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import h5py

CHUNK = 64 * 2**20              # bytes read at a time, a multiple of 16 (one complex128)
N_H5 = 12                       # .h5 files of a finished cfg: 6 kinematic cases x (pion, proton)


def text(s):
    return s.decode() if isinstance(s, bytes) else str(s)


def cfg_number(path):
    return int(re.search(r"cfg(\d+)", os.path.basename(path.rstrip("/"))).group(1))


def data_slabs(shape, paired):
    #(first element, number of elements) of the blocks that hold data, in C order: the whole array, or for _paired the blocks [a,b,a,b]
    if not paired:
        return [(0, int(np.prod(shape)))]
    slab = int(np.prod(shape[4:]))
    return [(int(np.ravel_multi_index((a, b, a, b), shape[:4])) * slab, slab) for a in range(shape[0]) for b in range(shape[1])]


def check_npy(path):
    #structure problems, and the blocks to read: (byte offset, elements, dtype, first element, t_src info)
    size = os.path.getsize(path)
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran_order, dtype = read_header(f)
            data_start = f.tell()
    except Exception as e:
        return [f"header unreadable ({type(e).__name__}: {e})"], []
    problems = []
    need = data_start + int(np.prod(shape)) * dtype.itemsize
    if size != need:
        problems.append(f"size {size} B, the header {shape} {dtype} needs {need} B")
    paired = "_paired_" in os.path.basename(path)                  # a part has no dim_spec: the layout is in the file name
    blocks = [(data_start + first * dtype.itemsize, n, dtype, first, None) for first, n in data_slabs(shape, paired)]
    return problems, blocks


def check_h5(path):
    size = os.path.getsize(path)
    problems, blocks = [], []
    try:
        with h5py.File(path, "r", locking=False) as f:
            names = []
            f.visit(names.append)
            for name in names:
                for key in f[name].attrs:
                    f[name].attrs[key]                              # reads every attribute: fails if its metadata is damaged
            for name in ["pt2_forward", "pt2_backward"]:
                if name not in f:
                    problems.append(f"no {name}")
                    continue
                d = f[name]
                start = d.id.get_offset()                           # None: not stored as one contiguous piece
                if start is None:
                    problems.append(f"{name} has no contiguous storage")
                    continue
                if start + d.id.get_storage_size() > size:
                    problems.append(f"{name} ends at {start + d.id.get_storage_size()} B, past the file end {size} B")
                if "measurements" in d.attrs and [int(m) for m in d.attrs["measurements"]] != [cfg_number(path)]:
                    problems.append(f"{name} measurements {[int(m) for m in d.attrs['measurements']]}, the file name says cfg {cfg_number(path)}")
                dims = [text(s) for s in d.attrs.get("dim_spec", [])]
                paired = dims[:4] == ["shape_sink", "shape_source", "frac_sink", "frac_source"]
                t_axis = dims.index("t_src_list") if "t_src_list" in dims else d.ndim - 4
                t_info = (int(np.prod(d.shape[t_axis + 1:])), d.shape[t_axis])     # (elements per t_src step, number of t_src)
                blocks += [(start + first * d.dtype.itemsize, n, d.dtype, first, t_info) for first, n in data_slabs(d.shape, paired)]
    except Exception as e:
        problems.append(f"does not open or read ({type(e).__name__}: {e})")
    return problems, blocks


def count_block(fd, offset, n_elem, dtype, first, t_info):
    #reads n_elem numbers from the byte offset; returns zeros, non-finite numbers, t_src indices of the zeros, bytes read, and whether the file ended early
    zeros = nonfinite = nbytes = 0
    t_zero = set()
    step = CHUNK // dtype.itemsize
    for e in range(0, n_elem, step):
        want = min(step, n_elem - e) * dtype.itemsize
        data = os.pread(fd, want, offset + e * dtype.itemsize)
        nbytes += len(data)
        x = np.frombuffer(data, dtype=dtype, count=len(data) // dtype.itemsize)
        is_zero = x == 0
        n_zero = int(np.count_nonzero(is_zero))
        zeros += n_zero
        nonfinite += int(np.count_nonzero(~np.isfinite(x)))
        if n_zero and t_info:
            per_t, n_t = t_info
            t_zero.update(np.unique((first + e + np.flatnonzero(is_zero)) // per_t % n_t).tolist())
        if len(data) < want:
            return zeros, nonfinite, t_zero, nbytes, True
    return zeros, nonfinite, t_zero, nbytes, False


def check_file(path):
    #(path, result), result None if the file was deleted meanwhile (a running task merged its parts)
    try:
        st = os.stat(path)
        problems, blocks = check_h5(path) if path.endswith(".h5") else check_npy(path)
        zeros = nonfinite = nbytes = 0
        t_zero = set()
        fd = os.open(path, os.O_RDONLY)
        try:
            for offset, n, dtype, first, t_info in blocks:
                z, nf, tz, nb, ended = count_block(fd, offset, n, dtype, first, t_info)
                zeros, nonfinite, nbytes = zeros + z, nonfinite + nf, nbytes + nb
                t_zero |= tz
                if ended:
                    break
        finally:
            os.close(fd)
    except FileNotFoundError:
        return path, None
    except Exception as e:
        return path, dict(size=0, mtime=0, problems=[f"check failed ({type(e).__name__}: {e})"], zeros=0, nonfinite=0, nbytes=0)
    if not os.path.exists(path):
        return path, None
    if zeros:
        problems.append(f"{zeros} zeros" + (f" in t_src {sorted(t_zero)}" if t_zero else ""))
    if nonfinite:
        problems.append(f"{nonfinite} non-finite numbers")
    return path, dict(size=st.st_size, mtime=st.st_mtime_ns, problems=problems, zeros=zeros, nonfinite=nonfinite, nbytes=nbytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="cfg folders, or with --all output folders")
    parser.add_argument("--all", action="store_true", help="dirs are *_GEVP_sepq_* output folders: check every finished cfg in them")
    parser.add_argument("--ledger", help="file of checked files: those OK with unchanged size and mtime are skipped")
    parser.add_argument("--workers", type=int, default=16, help="files read at the same time (default 16)")
    args = parser.parse_args()

    #the files of every cfg folder
    cfg_files, unfinished = {}, []
    for d in args.dirs:
        d = d.rstrip("/")
        if not os.path.isdir(d):
            print(f"MISSING {d}: not a folder", flush=True)
            continue
        cfg_dirs = sorted((c.rstrip("/") for c in glob.glob(f"{d}/cfg*/")), key=cfg_number) if args.all else [d]
        for c in cfg_dirs:
            h5s, parts = sorted(glob.glob(f"{c}/*.h5")), sorted(glob.glob(f"{c}/parts/*.npy"))
            if args.all and (len(h5s) != N_H5 or os.path.isdir(f"{c}/parts")):
                unfinished.append(c)                                # in progress, or dropped (cfg 246): not checked in --all mode
                continue
            cfg_files[c] = h5s + parts

    #files that passed before and did not change since
    passed = {}
    if args.ledger and os.path.exists(args.ledger):
        with open(args.ledger) as f:
            for line in f:
                path, size, mtime, status = line.rstrip("\n").split("\t")[:4]
                if status == "OK":
                    passed[path] = (int(size), int(mtime))
    def unchanged(path):
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return False
        return passed.get(path) == (st.st_size, st.st_mtime_ns)
    todo = [p for files in cfg_files.values() for p in files if not unchanged(p)]
    n_files = sum(len(files) for files in cfg_files.values())
    print(f"{len(cfg_files)} cfg folders, {n_files} files, {n_files - len(todo)} OK in the ledger and unchanged, {len(todo)} to check"
          + (f"; {len(unfinished)} unfinished cfg folders not checked" if unfinished else ""), flush=True)

    #check
    s = perf_counter()
    results, gb = {}, 0.0
    with (open(args.ledger, "a") if args.ledger else nullcontext()) as ledger, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(check_file, p) for p in todo]
        for i, future in enumerate(as_completed(futures), 1):
            path, r = future.result()
            results[path] = r
            if r is not None:
                gb += r["nbytes"] / 1e9
                status = "BAD" if r["problems"] else "OK"
                if r["problems"]:
                    print(f"BAD {path} | {'; '.join(r['problems'])}", flush=True)
                if ledger:
                    ledger.write(f"{path}\t{r['size']}\t{r['mtime']}\t{status}\t{r['zeros']}\t{r['nonfinite']}\t{'; '.join(r['problems'])}\n")
                    ledger.flush()
            if i % max(1, len(todo) // 20) == 0 or i == len(todo):
                t = perf_counter() - s
                print(f"[{i}/{len(todo)} files, {gb:.1f} GB read, {gb / max(t, 1e-9):.2f} GB/s, {t:.0f} s]", flush=True)

    #one line per cfg folder
    n_bad = 0
    for c, files in cfg_files.items():
        checked = [results[p] for p in files if p in results and results[p] is not None]
        bad = sum(bool(r["problems"]) for r in checked)
        n_bad += bad
        gone = sum(p in results and results[p] is None for p in files)
        layout = "_paired" if "_paired_" in c else "_smear" if "_smear_" in c else ""
        print(f"{os.path.basename(c)} {layout}: {sum(p.endswith('.h5') for p in files)} .h5, {sum(p.endswith('.npy') for p in files)} parts, "
              f"{len(checked)} checked now, {sum(r['zeros'] for r in checked)} zeros, {bad} bad" + (f", {gone} gone" if gone else ""), flush=True)
    if unfinished:
        print(f"not checked (unfinished): {' '.join(os.path.basename(c) for c in unfinished)}", flush=True)
    print(f"done in {perf_counter() - s:.0f} s: {len(todo)} files checked, {gb:.1f} GB read, {n_bad} bad", flush=True)
