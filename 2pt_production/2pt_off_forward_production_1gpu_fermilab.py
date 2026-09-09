import os
import argparse
import h5py
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, phase, source, fft, phase_v2
from pyquda_comm.array import arrayExp
from time import perf_counter
from cupy.cuda.runtime import deviceSynchronize
from tqdm import tqdm
import sys
sys.path.append('/ccs/home/sicheng/LaMETLat')
sys.path.append('/ccs/home/sicheng/LatCoding')
from pyquda_benchmark.meas.mom_smearing import *

#=======================================================================
#Production file for GPD matrix elements with q=[+-1,+-1,+-1]
#tunes the mom_fraction of quarks in the meson interpolation functions. 
#split x_src and store them in seperate files



parser = argparse.ArgumentParser()
parser.add_argument("--quark",type=float, required=True)
parser.add_argument("--rho",type=float,required=True)
args = parser.parse_args()
quark_mom_frac = args.quark
rho = args.rho
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
ncfg     = (cfg_list - 204) // 6          # (n,)  sequence index 0..n-1

# unshifted source grids
t_base = np.arange(0, Lt, 12)   # (8,)
x_base = np.arange(0, Ls,  8)   # (4,)
y_base = np.arange(0, Ls,  8)   # (4,)
z_base = np.arange(0, Ls,  4)   # (8,)

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


core.init([1, 1, 1, 1], resource_path="/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/.cache/quda")
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
if latt_info.mpi_rank == 0:
    print(run_parameters)
dirac = core.getDirac(latt_info, -0.05138, 1e-10, 1000, 1.0, 1.04243, 1.04243, [[4, 4, 4, 4],[2,2,2,2]])
G5 = gamma.gamma(15)
GT5 = gamma.gamma(7)
momentum_list = []
for px in [0,-1,1]:
    for py in [0, -1, 1]:
        for pz in range(mom_min-1,mom_max+2):
            momentum_list.append([px, py, pz])
momentum_list = np.array(momentum_list, dtype=np.int64)


measurement_list = cfg_list[0:n]
pion_45 = cp.zeros((len(measurement_list),len(t_src[1]),len(x_src[1]),len(y_src[1]),len(z_src[1]), len(momentum_list), latt_info.Lt), "<c16")
pion_5 = cp.zeros((len(measurement_list),len(t_src[1]),len(x_src[1]),len(y_src[1]),len(z_src[1]), len(momentum_list), latt_info.Lt), "<c16")
count = 0
mean_rsqr = 0.0
for i_cfg, cfg in tqdm(enumerate(measurement_list),desc=f"Processing cfgs"):

    #READ GAUGE
    #deviceSynchronize()
    #s = perf_counter()
    gauge_ape = io.readMILCGauge(f"/lustre/orion/lgt132/world-shared/DATA/MILC/a09m310/gauge/l3296f211b630m0074m037m440e.{cfg}",checksum=True, reunitarize_sigma=1e-6)
    #deviceSynchronize()
    #core.getLogger().info(f"READ GAUGE #{cfg}: {perf_counter() - s} secs")

    #HYP
    #deviceSynchronize()
    #s = perf_counter()
    gauge_hyp = gauge_ape.copy()
    core.getLogger().info(f"DOING HYP SMEARING")
    #core.getLogger().info(f"plaq_hyp_before = {gauge_hyp.plaquette()}")
    gauge_hyp.hypSmear(1, 0.75, 0.6, 0.3, -1,True,True)
    #gauge_ape.plaquette()
    #deviceSynchronize()
    #core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
    #core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

    #APE
    #deviceSynchronize()
    #s = perf_counter()
    #with dirac.useGauge(gauge_ape):
    core.getLogger().info(f"DOING APE SMEARING")
    #core.getLogger().info(f"plaq_ape_before = {gauge_ape.plaquette()}")
    gauge_ape.apeSmear(25,0.6154 , 3,True,True)
    deviceSynchronize()
    #core.getLogger().info(f"plaq_ape_after = {gauge_ape.plaquette()}")
    #core.getLogger().info(f"APE SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE
    #core.getLogger().info(f"LOADING GAUGE")
    #dirac.loadGauge(gauge_hyp)
    
    for t_idx, t_src in enumerate(t_src[i_cfg]):
        for x_idx, x_src in enumerate(x_src[i_cfg]):
            for y_idx, y_src in enumerate(y_src[i_cfg]):
                for z_idx, z_src in enumerate(z_src[i_cfg]):

                    #inner_loop = perf_counter()
                    src_pos = [x_src[x_idx],y_src[y_idx],z_src[z_idx],t_src[t_idx]]
                    core.getLogger().info(f"SOURCE POSITION = {src_pos}")
                    momentum_phases = phase.MomentumPhase(latt_info).getPhases( momentum_list, src_pos )

                    #SRC GAUSSIAN SMEAR
                    #deviceSynchronize()
                    s = perf_counter()
                    core.getLogger().info(f"DOING SRC GAUSSIAN SMEARING")
                    prop1 = momentum_smearing_propagator(latt_info, gauge_ape, k1, src_pos, rho, smear_steps)
                    prop2 = momentum_smearing_propagator(latt_info, gauge_ape, k2, src_pos, rho, smear_steps)
                    #deviceSynchronize() 
                    core.getLogger().info(f"SOURCE GAUSSIAN SMEAR: {perf_counter() - s} secs")

                    #INVERT
                    #deviceSynchronize()
                    #s = perf_counter()
                    #with dirac.useGauge(gauge_hyp):
                    dirac.loadGauge(gauge_hyp)
                    core.getLogger().info(f"SOLVING DIRAC EQ")
                    prop1 = core.invertPropagator(dirac, prop1)
                    prop2 = core.invertPropagator(dirac, prop2)
                    #deviceSynchronize()
                    core.getLogger().info(f"INVERT: {perf_counter() - s} secs")

                    #SINK GAUSSIAN SMEAR
                    #deviceSynchronize()
                    s = perf_counter()
                    core.getLogger().info(f"DOING SINK GAUSSIAN SMEARING")
                    prop1 = momentum_smearing_sink(latt_info, prop1, gauge_ape, k1, rho, smear_steps)
                    prop2 = momentum_smearing_sink(latt_info, prop2, gauge_ape, k2, rho, smear_steps)
                    #deviceSynchronize()
                    core.getLogger().info(f"SINK GAUSSIAN SMEAR: {perf_counter() - s} secs")

                    #CONTRACT
                    #deviceSynchronize()
                    #s = perf_counter()
                    pion_45[i_cfg,t_idx,x_idx,y_idx,z_idx] += contract(
                        "pwtzyx,wtzyxjiba,jk,wtzyxklba,li->pt",
                        momentum_phases,
                        prop1.data.conj(),
                        G5 @ GT5,
                        prop2.data,
                        GT5 @ G5,
                        )
                    pion_5[i_cfg,t_idx,x_idx,y_idx,z_idx] += contract(
                        "pwtzyx,wtzyxjiba,jk,wtzyxklba,li->pt",
                        momentum_phases,
                        prop1.data.conj(),
                        G5 @ G5,
                        prop2.data,
                        G5 @ G5,
                        )
                    #deviceSynchronize()
                    #core.getLogger().info(f"CONTRACT: {perf_counter() - s} secs")

                    #calculate the size of the smeared src only for [0,0,0,0]
                    """
                    if src_pos == [0,0,0,0]:
                        if latt_info.mpi_rank == 0:
                            #deviceSynchronize()
                            #s = perf_counter()
                            src_wavefunction = momentum_source1.lexico()[0,:,:,:,:,:,:,:]
                            src_density = np.sum(np.abs(src_wavefunction)**2, axis=(3,4,5,6))
                            norm = np.sum(src_density, axis=(0, 1, 2))
                            core.getLogger().info(f"norm is {norm}")
                            coordinates = phase_v2.LocationPhase(latt_info).getPhase().lexico()
                            rsqr = (np.minimum(coordinates[0],Ls-coordinates[0])**2  + np.minimum(coordinates[1],Ls-coordinates[1])**2 + np.minimum(coordinates[2],Ls-coordinates[2])**2).astype(np.float64)
                            rsqr = rsqr[0,:,:,:]
                            rsqr *= src_density
                            tmp = np.sum(np.abs(rsqr),axis=(0,1,2))/norm
                            core.getLogger().info(f"mean r^2 = {tmp}")
                            mean_rsqr += tmp
                            count += 1
                            #deviceSynchronize()
                            #core.getLogger().info(f"GET RADIUS: {perf_counter() - s} secs")
                            tmp=0
                    
                    #core.getLogger().info(f"INNER LOOP: {perf_counter() - inner_loop} secs")
                    """
dirac.freeGauge()

pion_45_np=core.gatherLattice(pion_45.get(),[6,-1,-1,-1])
pion_5_np=core.gatherLattice(pion_5.get(),[6,-1,-1,-1])
#save as h5py file
if latt_info.mpi_rank == 0:
    
    #mean_rsqr = mean_rsqr / count
    current_dir = os.path.dirname(os.path.abspath(__file__))
    g45_dir = f"{current_dir}/N{smear_steps}_rho{rho}_G45_ez_momfrac{smear_mom_x_str}"
    g5_dir = f"{current_dir}/N{smear_steps}_rho{rho}_G5_ez_momfrac{smear_mom_x_str}"
    os.makedirs(g45_dir, exist_ok=True)
    os.makedirs(g5_dir, exist_ok=True)
    g45_filename = f"pion_N{smear_steps}_rho{rho}_frac{smear_mom_x_str}_G45_cfg{measurement_list[0]}-{measurement_list[-1]}.h5"
    g5_filename = f"pion_N{smear_steps}_rho{rho}_frac{smear_mom_x_str}_G5_cfg{measurement_list[0]}-{measurement_list[-1]}.h5"

    with h5py.File(f"{g45_dir}/{g45_filename}", "w") as f:
        dset = f.create_dataset("pion_45", data=pion_45_np)
        f.create_dataset("momentum_list", data=momentum_list)
        dset.attrs["dim_spec"] = np.array(["measurement_list", "t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
        dset.attrs["measurements"] = measurement_list
        dset.attrs["momentums"] = momentum_list
        dset.attrs["x_src_list"] = x_src
        dset.attrs["y_src_list"] = y_src
        dset.attrs["z_src_list"] = z_src
        dset.attrs["t_src_list"] = t_src
        dset.attrs["dim_time"] = np.arange(latt_info.Lt)
        #dset.attrs["mean_rsqr"] = mean_rsqr

    with h5py.File(f"{g5_dir}/{g5_filename}", "w") as f:
        dset = f.create_dataset("pion_5", data=pion_5_np)
        f.create_dataset("momentum_list", data=momentum_list)
        dset.attrs["dim_spec"] = np.array(["measurement_list", "t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
        dset.attrs["measurements"] = measurement_list
        dset.attrs["momentums"] = momentum_list
        dset.attrs["x_src_list"] = x_src
        dset.attrs["y_src_list"] = y_src
        dset.attrs["z_src_list"] = z_src
        dset.attrs["t_src_list"] = t_src
        dset.attrs["dim_time"] = np.arange(latt_info.Lt)
        #dset.attrs["mean_rsqr"] = mean_rsqr
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
