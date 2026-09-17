import os
import argparse
import h5py
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, phase
from time import perf_counter
from cupy.cuda.runtime import deviceSynchronize
from tqdm import tqdm
from mom_smearing import *


parser = argparse.ArgumentParser()
parser.add_argument("--quark",type=float, required=True)
parser.add_argument("--rho",type=float,required=True)
parser.add_argument("--icfg", type=int, default=0)      # starting position in cfg_list
args = parser.parse_args()
quark_mom_frac = args.quark
rho = args.rho
icfg0 = args.icfg
Ls = 32
Lt = 96
n = 10  #number of configs to measure, starting from 204, with step size 6
mom_min = 0
mom_max = 6
smear_steps = 40
smear_mom = [0, 0, quark_mom_frac*mom_max]
smear_mom_x_str = ("%.3f" % quark_mom_frac).rstrip("0").rstrip(".").replace(".", "p") #the string of momentum smearing info, to be used in saving
k = np.array(smear_mom)
k1 =  k
k2 = -k


cfg_list = np.arange(204, 204 + 800*6, 6)   # The complete cfg list, 800 configs
cfg_measure_spacing = 5                     # 2 steps x overall stride 2
measurement_list = cfg_list[icfg0::cfg_measure_spacing][:n]
ncfg     = (measurement_list - 204) // 6

# unshifted source grids
t_base = np.arange(0, Lt, 32)   # 3
x_base = np.arange(0, Ls,  16)   # 2
y_base = np.arange(0, Ls,  16)   # 2
z_base = np.arange(0, Ls,  16)   # 2

# broadcast (n,1) + (1,nsrc) -> (n, nsrc); index as [icfg, isrc]
t_src = (t_base[None, :] + 5*ncfg[:, None]) % Lt   # (n, 8)
x_src = (x_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 4)
y_src = (y_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 4)
z_src = (z_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 8)
run_parameters = {
    "Ls": Ls,
    "Lt": Lt,
    "cfgs_to_meas": n,
    "rho": rho,
    "smear_steps": smear_steps,
    "smear_mom": smear_mom,
    "mom_min": mom_min,
    "mom_max": mom_max,
    "x_src_list_shifted": x_src,
    "y_src_list_shifted": y_src,
    "z_src_list_shifted": z_src,
    "t_src_list_shifted": t_src,
}

core.init([1, 1, 1, 4], resource_path="/lustre2/gluonp0/sliu1/.cache/quda")
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
if latt_info.mpi_rank == 0:
    print(run_parameters)
dirac = core.getDirac(latt_info, -0.05138, 1e-10, 1000, 1.0, 1.04243, 1.04243, [[4, 4, 4, 4],[2,2,2,2]])
G4 = gamma.gamma(8)
G5 = gamma.gamma(15)
G45 = gamma.gamma(7)
G35 = gamma.gamma(11)
gamma_list = [G5,G45,G35]
charge = gamma.gamma(10)
parity_p = (gamma.gamma(0) + gamma.gamma(8)) / 2    # (1+gamma4)/2 : nucleon forward
parity_m = (gamma.gamma(0) - gamma.gamma(8)) / 2    # (1-gamma4)/2 : nucleon backward (enters with a minus sign)
eps_color = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps_color[0,1,2] = eps_color[1,2,0] = eps_color[2,0,1] = +1
eps_color[0,2,1] = eps_color[2,1,0] = eps_color[1,0,2] = -1
momentum_list = []
for px in [0,-1,1]:
    for py in [0, -1, 1]:
        for pz in range(mom_min-1,mom_max+2):
            momentum_list.append([px, py, pz])
momentum_list = np.array(momentum_list, dtype=np.int64)
current_dir = os.path.dirname(os.path.abspath(__file__))
g45_dir = f"{current_dir}/N{smear_steps}_rho{rho}_G45_ez_momfrac{smear_mom_x_str}"
g5_dir = f"{current_dir}/N{smear_steps}_rho{rho}_G5_ez_momfrac{smear_mom_x_str}"
os.makedirs(g45_dir, exist_ok=True)
os.makedirs(g5_dir, exist_ok=True)

pion_gamma     = cp.zeros((3,3,t_src.shape[1], x_src.shape[1], y_src.shape[1], z_src.shape[1], len(momentum_list), latt_info.Lt), "<c16")      #(sink_gamma, source_gamma,t,x,y,z,p,Lt)
proton_gamma_p = cp.zeros((3,3,t_src.shape[1], x_src.shape[1], y_src.shape[1], z_src.shape[1], len(momentum_list), latt_info.Lt), "<c16")
proton_gamma_m = cp.zeros((3,3,t_src.shape[1], x_src.shape[1], y_src.shape[1], z_src.shape[1], len(momentum_list), latt_info.Lt), "<c16")
for i_cfg, cfg in tqdm(enumerate(measurement_list),desc=f"Processing cfgs"):


    pion_gamma[:] = 0
    proton_gamma_p[:] = 0
    proton_gamma_m[:] = 0

    #READ GAUGE
    deviceSynchronize()
    s = perf_counter()
    gauge_ape = io.readMILCGauge(f"/lustre2/gluonp0/MILC/l3296f211b630m0074m037m440d/l3296f211b630m0074m037m440d.{cfg}",checksum=True, reunitarize_sigma=1e-6)
    deviceSynchronize()
    core.getLogger().info(f"READ GAUGE #{cfg}: {perf_counter() - s} secs")

    #HYP
    deviceSynchronize()
    s = perf_counter()
    gauge_hyp = gauge_ape.copy()
    core.getLogger().info(f"DOING HYP SMEARING")
    core.getLogger().info(f"plaq_hyp_before = {gauge_hyp.plaquette()}")
    gauge_hyp.hypSmear(1, 0.75, 0.6, 0.3, -1,True,True)
    deviceSynchronize()
    core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
    core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

    #APE
    deviceSynchronize()
    s = perf_counter()
    #with dirac.useGauge(gauge_ape):
    core.getLogger().info(f"DOING APE SMEARING")
    core.getLogger().info(f"plaq_ape_before = {gauge_ape.plaquette()}")
    gauge_ape.apeSmear(25,0.6154 , 3,True,True)
    deviceSynchronize()
    core.getLogger().info(f"plaq_ape_after = {gauge_ape.plaquette()}")
    core.getLogger().info(f"APE SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE
    #core.getLogger().info(f"LOADING GAUGE")
    dirac.loadGauge(gauge_hyp)

    for t_idx, t0 in enumerate(t_src[i_cfg]):
        for x_idx, x0 in enumerate(x_src[i_cfg]):
            for y_idx, y0 in enumerate(y_src[i_cfg]):
                for z_idx, z0 in enumerate(z_src[i_cfg]):

                    deviceSynchronize()
                    inner_loop = perf_counter()
                    core.getLogger().info(f"INNER LOOP STARTS")

                    src_pos = [x0, y0, z0, t0]
                    core.getLogger().info(f"SOURCE POSITION = {src_pos}")
                    momentum_phases = phase.MomentumPhase(latt_info).getPhases( momentum_list, src_pos )

                    #SRC GAUSSIAN SMEAR
                    deviceSynchronize()
                    s = perf_counter()
                    core.getLogger().info(f"DOING SRC GAUSSIAN SMEARING")
                    prop1 = momentum_smearing_propagator(latt_info, gauge_ape, k1, src_pos, rho, smear_steps)
                    prop2 = momentum_smearing_propagator(latt_info, gauge_ape, k2, src_pos, rho, smear_steps)
                    deviceSynchronize()
                    core.getLogger().info(f"SOURCE GAUSSIAN SMEAR: {perf_counter() - s} secs")

                    #load gauge again after smearing
                    deviceSynchronize()
                    s = perf_counter()
                    core.getLogger().info(f"RELOADING GAUGE")
                    dirac.loadGauge(gauge_hyp,thin_update_only=True)
                    deviceSynchronize()
                    core.getLogger().info(f"RELOADING GAUGE: {perf_counter() - s} secs")

                    #INVERT
                    deviceSynchronize()
                    s = perf_counter()
                    core.getLogger().info(f"SOLVING DIRAC EQ")
                    prop1 = core.invertPropagator(dirac, prop1)
                    prop2 = core.invertPropagator(dirac, prop2)
                    #deviceSynchronize()
                    core.getLogger().info(f"INVERT 2 propagagors: {perf_counter() - s} secs")

                    #SINK GAUSSIAN SMEAR
                    deviceSynchronize()
                    s = perf_counter()
                    core.getLogger().info(f"DOING SINK GAUSSIAN SMEARING")
                    prop1 = momentum_smearing_sink(latt_info, prop1, gauge_ape, k1, rho, smear_steps)
                    prop2 = momentum_smearing_sink(latt_info, prop2, gauge_ape, k2, rho, smear_steps)
                    deviceSynchronize()
                    core.getLogger().info(f"SINK GAUSSIAN SMEAR: {perf_counter() - s} secs")

                    #CONTRACT
                    deviceSynchronize()
                    s = perf_counter()

                    for i_gamma_sink, gamma_sink in enumerate(gamma_list):

                        diquark = contract("def,gh,wtzyxhjeb,wtzyxgida->wtzyxijabf",eps_color, charge @ gamma_sink, prop2.data, prop2.data)

                        for i_gamma_source, gamma_source in enumerate(gamma_list):

                                gamma_source_bar = G4 @ gamma_source.conj().T @ G4

                                pion_gamma[i_gamma_sink,i_gamma_source,t_idx,x_idx,y_idx,z_idx] += contract(
                                "pwtzyx,wtzyxjiba,jk,wtzyxklba,li->pt",
                                momentum_phases,
                                prop1.data.conj(),
                                G5 @ gamma_sink,
                                prop2.data,
                                gamma_source_bar @ G5 ,
                                )

                                proton_gamma_p[i_gamma_sink,i_gamma_source,t_idx,x_idx,y_idx,z_idx] += (
                                contract("abc,ij,kl,pwtzyx,wtzyxijabf,wtzyxlkfc->pt",
                                         eps_color, charge @ gamma_source, parity_p, momentum_phases, diquark, prop2.data)
                                - contract("abc,ij,kl,pwtzyx,wtzyxkjcbf,wtzyxlifa->pt",
                                           eps_color, charge @ gamma_source, parity_p, momentum_phases, diquark, prop2.data))

                                proton_gamma_m[i_gamma_sink,i_gamma_source,t_idx,x_idx,y_idx,z_idx] += (
                                contract("abc,ij,kl,pwtzyx,wtzyxijabf,wtzyxlkfc->pt",
                                         eps_color, charge @ gamma_source, parity_m, momentum_phases, diquark, prop2.data)
                                - contract("abc,ij,kl,pwtzyx,wtzyxkjcbf,wtzyxlifa->pt",
                                           eps_color, charge @ gamma_source, parity_m, momentum_phases, diquark, prop2.data))

                    free, total = cp.cuda.runtime.memGetInfo()
                    core.getLogger().info(f"GPU used {(total - free)/1e9:.1f} GB")
                    deviceSynchronize()
                    core.getLogger().info(f"CONTRACT ALL: {perf_counter() - s} secs")


                    core.getLogger().info(f"UNTIL CONTRACTION: {perf_counter() - inner_loop} secs")

    deviceSynchronize()
    saving_started = perf_counter()
    pion_gamma_np=core.gatherLattice(pion_gamma.get(),[7,-1,-1,-1])
    proton_gamma_p_np = core.gatherLattice(proton_gamma_p.get(), [7, -1, -1, -1])
    proton_gamma_m_np = core.gatherLattice(proton_gamma_m.get(), [7, -1, -1, -1])


    #save as h5py file
    if latt_info.mpi_rank == 0:

    #ROLL every source so that time index 0 is t_src:  rolled[..., tau] = C[..., (t_src + tau) % 96]
        for t_idx, t0 in enumerate(t_src[i_cfg]):
            pion_gamma_np[:, :, t_idx]     = np.roll(pion_gamma_np[:, :, t_idx],     -t0, axis=-1)
            proton_gamma_p_np[:, :, t_idx] = np.roll(proton_gamma_p_np[:, :, t_idx], -t0, axis=-1)
            proton_gamma_m_np[:, :, t_idx] = np.roll(proton_gamma_m_np[:, :, t_idx], -t0, axis=-1)
            proton_gamma_p_np[:, :, t_idx][..., Lt - t0:] *= -1
            proton_gamma_m_np[:, :, t_idx][..., Lt - t0:] *= -1
        proton_gamma_np = np.concatenate([proton_gamma_p_np[..., :Lt // 2], -proton_gamma_m_np[..., Lt // 2:]], axis=-1)

        pion_filename   = f"pion_N{smear_steps}_rho{rho}_frac{smear_mom_x_str}_GEVP_cfg{cfg}.h5"
        proton_filename = f"proton_N{smear_steps}_rho{rho}_frac{smear_mom_x_str}_GEVP_cfg{cfg}.h5"
        gevp_dir = f"{current_dir}/N{smear_steps}_rho{rho}_GEVP_ez_momfrac{smear_mom_x_str}"
        os.makedirs(gevp_dir, exist_ok=True)

        with h5py.File(f"{gevp_dir}/{pion_filename}", "w") as f:
            dset = f.create_dataset("pion_gamma", data=pion_gamma_np)          # (3, 3, nt, nx, ny, nz, nmom, 96)
            f.create_dataset("momentum_list", data=momentum_list)
            dset.attrs["gamma_list"] = np.array(["G5", "G45", "G35"], dtype=h5py.string_dtype())
            dset.attrs["dim_spec"] = np.array(["gamma_sink","gamma_source","t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
            dset.attrs["measurements"] = [cfg]
            dset.attrs["momentums"] = momentum_list
            dset.attrs["x_src_list"] = x_src[i_cfg]
            dset.attrs["y_src_list"] = y_src[i_cfg]
            dset.attrs["z_src_list"] = z_src[i_cfg]
            dset.attrs["t_src_list"] = t_src[i_cfg]
            dset.attrs["dim_time"] = np.arange(pion_gamma_np.shape[-1])

        with h5py.File(f"{gevp_dir}/{proton_filename}", "w") as f:
            dset = f.create_dataset("proton_gamma", data=proton_gamma_np)      # (3, 3, nt, nx, ny, nz, nmom, 96)
            f.create_dataset("momentum_list", data=momentum_list)
            dset.attrs["gamma_list"] = np.array(["G5", "G45", "G35"], dtype=h5py.string_dtype())
            dset.attrs["dim_spec"] = np.array(["gamma_sink","gamma_source","t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
            dset.attrs["measurements"] = [cfg]
            dset.attrs["momentums"] = momentum_list
            dset.attrs["x_src_list"] = x_src[i_cfg]
            dset.attrs["y_src_list"] = y_src[i_cfg]
            dset.attrs["z_src_list"] = z_src[i_cfg]
            dset.attrs["t_src_list"] = t_src[i_cfg]
            dset.attrs["dim_time"] = np.arange(proton_gamma_np.shape[-1])
            dset.attrs["diquark"] = "C Gamma_sink at the sink, C Gamma_source at the source (equal to Gamma_bar_source C for G5, G45, G35)"
            dset.attrs["propagator"] = "prop2 (k2 = -k) for all three quark lines"
    core.getLogger().info(f"SAVING SECTION:{perf_counter()-saving_started} secs")
dirac.freeGauge()






"""
To read do the following
with h5py.File(pt2_path, "r") as f:
    pion = f["pion_45"][:]

    if "momentum_list" in f:
        moms = f["momentum_list"][:]
    else:
        moms = f["pion_45"].attrs["momentums"]

mom_to_idx = {tuple(p): i for i, p in enumerate(moms.tolist())}
Then use like this
pf = (0, 0, pz)
ipf = mom_to_idx[pf]
C2_pf = pion[..., ipf, :]
"""



