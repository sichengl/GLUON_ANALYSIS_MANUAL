#Plaquette of the HYP-smeared links and the tadpole clover coefficient on a04m310 (setup: setup_valence_tuning_delta_a04m310.py).
#Delta copy of frontier/final_production_frontier/8gpu/a04m310/frontier_valence_tuning_a04m310/plaquette_csw_frontier_a04m310.py: only the setup import and these header lines differ.
#c_SW = 1/u0^3 with u0 = P^(1/4), P = <(1/3) Re Tr U_plaquette> of the links the Dirac operator uses (HYP smeared): c_SW = P^(-3/4).
#This is the rule of PNDME (arXiv:1606.07049) and of Sergey. On a09m310 the production logs give P = 0.946104 over 200 cfgs, c_SW = 1.042427 (used: 1.04243).
#For each cfg of cfg_list: read the gauge file, HYP as in the production (no gauge fixing: the plaquette is gauge invariant), plaquette before and after.
#Rank 0 prints one line per cfg and the mean, and saves everything to {out_dir}/plaquette_hyp_{ens_tag}_ncfg{n}.npz.
#Run with 8 ranks (grid [1,1,1,8], one Delta gpuA100x8 node): plaquette_csw_delta_a04m310.sh
import os
import numpy as np
from pyquda_utils import core, io
from time import perf_counter
from setup_valence_tuning_delta_a04m310 import tune_params, cfg_list, gauge_file_prefix, out_dir, ens_tag, quda_resource_path

import sys, traceback
from mpi4py import MPI

#STOP ON ERROR: an uncaught error on any rank prints its traceback and kills all ranks at once (as in the production scripts)
def abort_on_error(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = abort_on_error
rank = MPI.COMM_WORLD.Get_rank()

tune_params.init_quda(quda_resource_path)

plaq_before, plaq_hyp = [], []                  # per cfg: (total, spatial, temporal) as QUDA returns them; the total is P
for cfg in cfg_list:
    s = perf_counter()
    gauge = io.readMILCGauge(f"{gauge_file_prefix}.{cfg}", checksum=True, reunitarize_sigma=1e-6)
    p0 = [float(v) for v in gauge.plaquette()]
    gauge.hypSmear(tune_params.hyp_n_steps, tune_params.hyp_alpha1, tune_params.hyp_alpha2, tune_params.hyp_alpha3, tune_params.hyp_dir_ignore,
                   tune_params.hyp_compute_plaquette, tune_params.hyp_compute_qcharge)
    p = [float(v) for v in gauge.plaquette()]
    gauge = None
    plaq_before.append(p0)
    plaq_hyp.append(p)
    core.getLogger().info(f"cfg {cfg}: plaquette {p0[0]:.12f}, after HYP {p[0]:.12f} -> c_SW = P^(-3/4) = {p[0] ** -0.75:.6f}  ({perf_counter() - s:.1f} secs)")

P = np.array(plaq_hyp)[:, 0]                    # (n_cfg,) HYP plaquette per cfg
n = len(P)
P_mean = P.mean()
P_err = P.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
csw = P_mean ** -0.75
csw_err = 0.75 * P_mean ** -1.75 * P_err        # error of P^(-3/4) from the error of P
if rank == 0:
    print(f"{ens_tag}: {n} cfgs, HYP plaquette {P_mean:.12f} +- {P_err:.2e} -> c_SW = P^(-3/4) = {csw:.6f} +- {csw_err:.1e}")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{out_dir}/plaquette_hyp_{ens_tag}_ncfg{n}.npz"
    with open(f"{name}.tmp", "wb") as f:
        np.savez(f, cfg_list=np.array(cfg_list), plaquette_before=np.array(plaq_before), plaquette_hyp=np.array(plaq_hyp),
                 csw_per_cfg=P ** -0.75, csw=csw, csw_err=csw_err,
                 hyp=np.array([tune_params.hyp_n_steps, tune_params.hyp_alpha1, tune_params.hyp_alpha2, tune_params.hyp_alpha3, tune_params.hyp_dir_ignore]),
                 note="plaquette_*: per cfg (total, spatial, temporal) from QUDA; c_SW = (total HYP plaquette)^(-3/4); hyp = n_steps, alpha1-3, dir_ignore")
    os.replace(f"{name}.tmp", name)
    print("saved", name)
