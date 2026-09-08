import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from opt_einsum import contract


SCRIPT_DIR = Path(__file__).resolve().parent
FF_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GPD_calc/FF_data")
PT2_DIR = Path("/lustre/orion/lgt132/scratch/sicheng/GPD_calc_v3/2pt_production")
TMP_DIR = SCRIPT_DIR / "tmp_percfg"

# which momentum-transfer frame to read: "symmetric" or "asymmetric".
# q = (0,0,0) exists in both and is the same data either way.
frame = "symmetric"

OUT_DIR = SCRIPT_DIR / f"gpd_3pt_{frame}"

cfg_skip = [4314]
cfg_list = [c for c in range(204, 204 + 800 * 6, 6) if c not in cfg_skip]
tsrc_list = list(range(0, 96, 12))
x_src_list = list(range(0, 32, 8))
y_src_list = list(range(0, 32, 8))
z_src_list = list(range(0, 32, 4))
T = 96
Ls = 32
tsep_max = 16
tins_max = 16
pt2_rho = "3.25"
pt2_mom_frac = "0p6"
q_phase_sign = 1

# the q rows the raw FF file stores for this frame, in file order
# (validated against the file on every read)
if frame == "symmetric":
    q_list = np.array(
        [[0, 0, 0], [-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0]],
        dtype=np.int64,
    )
else:
    q_list = np.array(
        [[qx, qy, qz]
         for qx in [0, -1, 1] for qy in [0, -1, 1] for qz in [0, -1, 1]],
        dtype=np.int64,
    )
# the momentum transfers we actually build three-point functions for
q_list_produce = np.array([[0, 0, 0]], dtype=np.int64)

tgf_list = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=np.int64)
pf_list = np.array([[0, 0, pz] for pz in range(0, 7)], dtype=np.int64)

# raw (munu, rhosig) rows of the FF file
lorentz_list = [(0, 0), (3, 3), (4, 4), (3, 0), (0, 3)]
lorentz_label = ["XYXY", "TXTX", "TYTY", "XTXY", "XYXT"]

# the gluon operators built from those rows: (name, [(row, coefficient)], formula)
operator_list = [
    (
        "TXTX",
        [(lorentz_list.index((3, 3)), 1.0)],
        "TXTX",
    ),
    (
        "TYTY",
        [(lorentz_list.index((4, 4)), 1.0)],
        "TYTY",
    ),
    (
        "XYXY",
        [(lorentz_list.index((0, 0)), 1.0)],
        "XYXY",
    ),
    (
        "TXTXpTYTYm2XYXY",
        [
            (lorentz_list.index((3, 3)), 1.0),
            (lorentz_list.index((4, 4)), 1.0),
            (lorentz_list.index((0, 0)), -2.0),
        ],
        "TXTX + TYTY - 2 * XYXY",
    ),
]

N_WORKERS = 10

ff_input = str(FF_DIR / "FF_opp_symmetric_asymmetric_flow0-40_cfg{cfg}.h5")
pt2_input = str(
    PT2_DIR
    / "N40_rho{rho}_G45_ez_momfrac{mom_frac}"
    / "pion_ix{ix}_x{shifted_x}_N40_rho{rho}_frac{mom_frac}_G45_cfg{cfg}.h5"
)
tmp_output = str(TMP_DIR / "percfg_cfg{cfg}.h5")
pt3_output = str(
    OUT_DIR / "3pt_tgf{tgf}_pf{pfx}_{pfy}_{pfz}_q{qx}_{qy}_{qz}.h5")


def q_produce_rows():
    pos = {tuple(r): i for i, r in enumerate(q_list.tolist())}
    return [pos[tuple(q)] for q in q_list_produce.tolist()]


def make_source_phase(cfg):
    ncfg = (cfg - 204) // 6
    shift = ncfg * 3
    x = np.array([(v + shift) % Ls for v in x_src_list], dtype=np.float64)
    y = np.array([(v + shift) % Ls for v in y_src_list], dtype=np.float64)
    z = np.array([(v + shift) % Ls for v in z_src_list], dtype=np.float64)
    q_dot_x = (
        x[:, None, None, None] * q_list_produce[None, None, None, :, 0]
        + y[None, :, None, None] * q_list_produce[None, None, None, :, 1]
        + z[None, None, :, None] * q_list_produce[None, None, None, :, 2]
    )
    return np.exp(-1j * q_phase_sign * (2 * np.pi / Ls) * q_dot_x).astype("<c16")


def load_splitx_pt2(cfg):
    ncfg = (cfg - 204) // 6
    pt2_raw = None
    momentum_ref = None

    for ix, xsrc in enumerate(x_src_list):
        shifted_x = (xsrc + ncfg * 3) % Ls
        path = pt2_input.format(
            rho=pt2_rho, mom_frac=pt2_mom_frac, ix=ix,
            shifted_x=shifted_x, cfg=cfg,
        )
        with h5py.File(path, "r") as f:
            block = f["pion_45"][0, :, 0, :, :, :, :]
            momentum_list = f["momentum_list"][:]

        if pt2_raw is None:
            n_tsrc, n_y, n_z, n_mom, n_t = block.shape
            pt2_raw = np.zeros(
                (n_tsrc, len(x_src_list), n_y, n_z, n_mom, n_t), "<c16")
            momentum_ref = momentum_list
        elif not np.array_equal(momentum_list, momentum_ref):
            raise ValueError(f"momentum_list mismatch in cfg {cfg}, ix {ix}")

        pt2_raw[:, ix, :, :, :, :] = block

    return pt2_raw, momentum_ref


def build_momentum_indices(momentum_list):
    mom_to_idx = {tuple(p): i for i, p in enumerate(momentum_list.tolist())}

    pf_indices = np.zeros(len(pf_list), dtype=np.int64)
    for ipf, pf in enumerate(pf_list):
        pf_tuple = tuple(pf.tolist())
        if pf_tuple not in mom_to_idx:
            raise ValueError(f"Missing final momentum pf={pf_tuple}")
        pf_indices[ipf] = mom_to_idx[pf_tuple]

    pi_indices = np.zeros((len(pf_list), len(q_list_produce)), dtype=np.int64)
    for ipf, pf in enumerate(pf_list):
        for iq, q in enumerate(q_list_produce):
            pi_tuple = tuple((pf + q_phase_sign * q).tolist())
            if pi_tuple not in mom_to_idx:
                raise ValueError(
                    f"Missing initial momentum pi={pi_tuple} for "
                    f"pf={tuple(pf.tolist())}, q={tuple(q.tolist())}"
                )
            pi_indices[ipf, iq] = mom_to_idx[pi_tuple]

    return pf_indices, pi_indices


def run_one_cfg(cfg):
    ncfg = (cfg - 204) // 6
    q_rows = q_produce_rows()

    # [x, y, z, q_produce]
    phase = make_source_phase(cfg)

    with h5py.File(ff_input.format(cfg=cfg), "r") as f:
        if not np.array_equal(q_list, f[f"{frame}_qlist"][:]):
            raise ValueError(f"{frame} q list mismatch in cfg {cfg}")
        w_list = f["wilson_line_list"][:]
        # [lorentz, tgf, z_WL, q, t] -- read only the rows the operators use
        FF_lorentz = np.stack([
            f[f"{frame}_corr"][0, i, j, tgf_list] for i, j in lorentz_list
        ])

    # keep only the produced momentum transfers.  NO vacuum subtraction here:
    # it is done at the jackknife level in assemble().
    FF_lorentz = FF_lorentz[:, :, :, q_rows, :]
    # [op, tgf, z_WL, q_produce, t]
    FF_op = np.stack([
        sum(coeff * FF_lorentz[row] for row, coeff in terms)
        for _, terms, _ in operator_list
    ])

    # [tsrc, ix, ysrc, zsrc, momentum, t]
    pt2, momentum_list = load_splitx_pt2(cfg)
    pf_indices, pi_indices = build_momentum_indices(momentum_list)
    n_spatial_src = len(x_src_list) * len(y_src_list) * len(z_src_list)

    n_op, n_tgf, n_z, n_q = FF_op.shape[:4]
    FF_roll = np.zeros(
        (len(tsrc_list), n_op, n_tgf, n_z, n_q, tins_max), "<c16")
    pt2_phase = np.zeros(
        (len(tsrc_list), n_q, len(pf_list), tsep_max), "<c16")
    pt2_f_sum = np.zeros((len(pf_list), tsep_max), "<c16")

    for it, tsrc in enumerate(tsrc_list):
        shifted_t = (tsrc + ncfg * 5) % T

        # [ix, ysrc, zsrc, momentum, tsep], source at the origin
        pt2_roll = np.roll(pt2[it], -shifted_t, axis=-1)[..., 0:tsep_max]
        # [pf, tsep], plain source average -- the <C2> of the vacuum term
        pt2_f_sum += np.mean(pt2_roll[..., pf_indices, :], axis=(0, 1, 2))
        # [q_produce, pf, tsep]
        pt2_phase[it] = contract(
            "xyzq,xyzpt->qpt", phase, pt2_roll[..., pf_indices, :]
        ) / n_spatial_src
        # [op, tgf, z_WL, q_produce, tins]
        FF_roll[it] = np.roll(FF_op, -shifted_t, axis=-1)[..., 0:tins_max]

    with h5py.File(tmp_output.format(cfg=cfg), "w") as f:
        f.create_dataset("FF_roll", data=FF_roll)
        f.create_dataset("pt2_phase", data=pt2_phase)
        f.create_dataset("pt2_f", data=pt2_f_sum / len(tsrc_list))
        f.create_dataset("phase_mean", data=np.mean(phase, axis=(0, 1, 2)))
        f.create_dataset("w_list", data=w_list)
    print(f"done cfg {cfg}", flush=True)
    return cfg


def assemble():
    paths = [tmp_output.format(cfg=cfg) for cfg in cfg_list]
    n_cfg = len(cfg_list)
    n_tsrc = len(tsrc_list)

    with h5py.File(paths[0], "r") as f:
        _, n_op, n_tgf, n_z, n_q, _ = f["FF_roll"].shape
        w_list = f["w_list"][:]

    pt2_phase = np.zeros((n_cfg, n_tsrc, n_q, len(pf_list), tsep_max), "<c16")
    pt2_f = np.zeros((n_cfg, len(pf_list), tsep_max), "<c16")
    phase_mean = np.zeros((n_cfg, n_q), "<c16")
    for icfg, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            pt2_phase[icfg] = f["pt2_phase"][:]
            pt2_f[icfg] = f["pt2_f"][:]
            phase_mean[icfg] = f["phase_mean"][:]

    op_names = np.array([name for name, _, _ in operator_list], dtype="S")

    for i_tgf, tgf in enumerate(tgf_list):
        # [cfg, tsrc, op, z_WL, q_produce, tins] for this flow time only
        FF_t = np.zeros((n_cfg, n_tsrc, n_op, n_z, n_q, tins_max), "<c16")
        for icfg, path in enumerate(paths):
            with h5py.File(path, "r") as f:
                FF_t[icfg] = f["FF_roll"][:, :, i_tgf]

        for ipf, pf in enumerate(pf_list):
            for i_q, q in enumerate(q_list_produce):
                # per configuration, [cfg, op, z_WL, tsep, tins]
                pt2FF = contract(
                    "cst,csozu->coztu",
                    pt2_phase[:, :, i_q, ipf, :],
                    FF_t[:, :, :, :, i_q, :],
                ) / n_tsrc
                # per configuration, [cfg, op, z_WL, tins]; the mean source
                # phase makes <C2> <O> the right vacuum term
                FF_cfg = (FF_t[:, :, :, :, i_q, :].mean(axis=1)
                          * phase_mean[:, i_q][:, None, None, None])
                # per configuration, [cfg, tsep]
                c2_cfg = pt2_f[:, ipf, :]

                # delete-one jackknife of each piece, then subtract the
                # vacuum term WITHIN each sample
                pt2FF_jk = (pt2FF.sum(axis=0) - pt2FF) / (n_cfg - 1)
                FF_jk = (FF_cfg.sum(axis=0) - FF_cfg) / (n_cfg - 1)
                c2_jk = (c2_cfg.sum(axis=0) - c2_cfg) / (n_cfg - 1)
                pt3_jk = pt2FF_jk - contract("nt,nozu->noztu", c2_jk, FF_jk)

                out = pt3_output.format(
                    tgf=tgf, pfx=pf[0], pfy=pf[1], pfz=pf[2],
                    qx=q[0], qy=q[1], qz=q[2])
                with h5py.File(out, "w") as f:
                    f.create_dataset("pt3_jk", data=pt3_jk)
                    f.create_dataset("pt2_f_jk", data=c2_jk)
                    f.create_dataset("cfg_list", data=np.array(cfg_list))
                    f.create_dataset("w_list", data=w_list)
                    f.create_dataset("operator_names", data=op_names)
                    f.attrs["dim_pt3_jk"] = "jk,op,z_WL,tsep,tins"
                    f.attrs["dim_pt2_f_jk"] = "jk,tsep"
                    f.attrs["n_jk"] = n_cfg
                    f.attrs["frame"] = frame
                    f.attrs["tgf"] = tgf
                    f.attrs["pf"] = pf
                    f.attrs["pz"] = int(pf[2])
                    f.attrs["q"] = q
                    f.attrs["q_phase_sign"] = q_phase_sign
                    f.attrs["vacuum_subtraction"] = (
                        "in-resample: <C2 O> - <C2><O> within each "
                        "delete-one jackknife sample")
        print(f"assembled tgf {tgf}", flush=True)


if __name__ == "__main__":
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_one_cfg, cfg) for cfg in cfg_list]
        for future in as_completed(futures):
            future.result()

    assemble()
    print("finished", flush=True)
