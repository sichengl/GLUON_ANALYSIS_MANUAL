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
parser.add_argument("--icfg", type=int, default=0)      # starting position in cfg_list
args = parser.parse_args()
quark_mom_frac = args.quark
icfg0 = args.icfg
Ls = 32
Lt = 96
n = 10  #number of configs to measure, starting from 204, with step size 6
mom_min = 0
mom_max = 6
smear_list = [[16, 2.0], [40, 3.25], [96, 5.0]]      # [steps, rho]: rho sets the radius, steps must satisfy 4*steps/rho**2 > 6 with margin
n_smear = len(smear_list)
smear_tag = "_".join(f"N{steps}rho{rho}" for steps, rho in smear_list)
smear_mom = [0, 0, quark_mom_frac*mom_max]
smear_mom_x_str = ("%.3f" % quark_mom_frac).rstrip("0").rstrip(".").replace(".", "p") #the string of momentum smearing info, to be used in saving
k = np.array(smear_mom)
k1 =  k
k2 = -k


cfg_list = np.arange(204, 204 + 800*6, 6)   # The complete cfg list, 800 configs
cfg_measure_spacing = 5
measurement_list = cfg_list[icfg0::cfg_measure_spacing][:n]
ncfg     = (measurement_list - 204) // 6

# unshifted source grids
t_base = np.arange(0, Lt, 32)   # 3
x_base = np.arange(0, Ls,  16)   # 2
y_base = np.arange(0, Ls,  16)   # 2
z_base = np.arange(0, Ls,  16)   # 2

# broadcast (n,1) + (1,nsrc) -> (n, nsrc); index as [icfg, isrc]
t_src = (t_base[None, :] + 5*ncfg[:, None]) % Lt   # (n, 3)
x_src = (x_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 2)
y_src = (y_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 2)
z_src = (z_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 2)
run_parameters = {
    "Ls": Ls,
    "Lt": Lt,
    "cfgs_to_meas": n,
    "smear_list": smear_list,
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
current_dir = os.environ.get("SLURM_SUBMIT_DIR", os.path.dirname(os.path.abspath(__file__)))

pion_gamma         = cp.zeros((n_smear, n_smear, 3, 3, t_src.shape[1], x_src.shape[1], y_src.shape[1], z_src.shape[1], len(momentum_list), latt_info.Lt), "<c16")          #(smear_sink, smear_source, gamma_sink, gamma_source, t, x, y, z, p, Lt)
proton_gamma_dirac = cp.zeros((n_smear, n_smear, 3, 3, 4, 4, t_src.shape[1], x_src.shape[1], y_src.shape[1], z_src.shape[1], len(momentum_list), latt_info.Lt), "<c16")   #(smear_sink, smear_source, gamma_sink, gamma_source, dirac_sink, dirac_source, t, x, y, z, p, Lt)

for i_cfg, cfg in tqdm(enumerate(measurement_list),desc=f"Processing cfgs"):


    pion_gamma[:] = 0
    proton_gamma_dirac[:] = 0

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
    core.getLogger().info(f"DOING APE SMEARING")
    core.getLogger().info(f"plaq_ape_before = {gauge_ape.plaquette()}")
    gauge_ape.apeSmear(25,0.6154 , 3,True,True)
    deviceSynchronize()
    core.getLogger().info(f"plaq_ape_after = {gauge_ape.plaquette()}")
    core.getLogger().info(f"APE SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE
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

                    for i_src_smear, (steps_src, rho_src) in enumerate(smear_list):

                        #SRC GAUSSIAN SMEAR with the source width
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"DOING SRC GAUSSIAN SMEARING: steps {steps_src}, rho {rho_src}")
                        prop1_inv = momentum_smearing_propagator(latt_info, gauge_ape, k1, src_pos, rho_src, steps_src)
                        prop2_inv = momentum_smearing_propagator(latt_info, gauge_ape, k2, src_pos, rho_src, steps_src)
                        deviceSynchronize()
                        core.getLogger().info(f"SOURCE GAUSSIAN SMEAR: {perf_counter() - s} secs")

                        #load gauge again after smearing
                        deviceSynchronize()
                        s = perf_counter()
                        dirac.loadGauge(gauge_hyp,thin_update_only=True)
                        deviceSynchronize()
                        core.getLogger().info(f"RELOADING GAUGE: {perf_counter() - s} secs")

                        #INVERT
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"SOLVING DIRAC EQ")
                        prop1_inv = core.invertPropagator(dirac, prop1_inv, mrhs=12)
                        prop2_inv = core.invertPropagator(dirac, prop2_inv, mrhs=12)
                        core.getLogger().info(f"INVERT 2 propagagors: {perf_counter() - s} secs")

                        for i_sink_smear, (steps_sink, rho_sink) in enumerate(smear_list):

                            #SINK GAUSSIAN SMEAR with the sink width, from the unsmeared solution each time
                            deviceSynchronize()
                            s = perf_counter()
                            core.getLogger().info(f"DOING SINK GAUSSIAN SMEARING: steps {steps_sink}, rho {rho_sink}")
                            prop1 = momentum_smearing_sink(latt_info, prop1_inv, gauge_ape, k1, rho_sink, steps_sink)
                            prop2 = momentum_smearing_sink(latt_info, prop2_inv, gauge_ape, k2, rho_sink, steps_sink)
                            deviceSynchronize()
                            core.getLogger().info(f"SINK GAUSSIAN SMEAR: {perf_counter() - s} secs")

                            #CONTRACT
                            deviceSynchronize()
                            s = perf_counter()

                            for i_gamma_sink, gamma_sink in enumerate(gamma_list):

                                diquark = contract("def,gh,wtzyxhjeb,wtzyxgida->wtzyxijabf",eps_color, charge @ gamma_sink, prop2.data, prop2.data)

                                # pion: colors and sink gamma summed at each site, then sites summed with the phases
                                #       source spins l,i left open for the source gamma
                                pion_open = contract("pwtzyx,wtzyxjiba,jk,wtzyxklba->lipt",
                                                     momentum_phases, prop1.data.conj(), G5 @ gamma_sink, prop2.data)          # (4,4,81,Lt)

                                # proton: colors and epsilon summed at each site (step 1), then sites summed with the phases (step 2)
                                #         source spins i,j and sink spins l,k left open
                                proton_site = contract("abc,wtzyxijabf,wtzyxlkfc->wtzyxijlk", eps_color, diquark, prop2.data)    # step 1, 256 per site
                                proton_open = contract("pwtzyx,wtzyxijlk->ijlkpt", momentum_phases, proton_site)               # step 2, (4,4,4,4,81,Lt)
                                proton_site = contract("abc,wtzyxkjcbf,wtzyxlifa->wtzyxijlk", eps_color, diquark, prop2.data)    # second term, step 1
                                proton_open -= contract("pwtzyx,wtzyxijlk->ijlkpt", momentum_phases, proton_site)              # second term, step 2

                                for i_gamma_source, gamma_source in enumerate(gamma_list):

                                    gamma_source_bar = G4 @ gamma_source.conj().T @ G4

                                    pion_gamma[i_sink_smear, i_src_smear, i_gamma_sink, i_gamma_source, t_idx, x_idx, y_idx, z_idx] += contract(
                                    "li,lipt->pt", gamma_source_bar @ G5, pion_open)

                                    proton_gamma_dirac[i_sink_smear, i_src_smear, i_gamma_sink, i_gamma_source, :, :, t_idx, x_idx, y_idx, z_idx] += contract(
                                    "ij,ijlkpt->lkpt", charge @ gamma_source, proton_open)

                            deviceSynchronize()
                            core.getLogger().info(f"CONTRACT ALL: {perf_counter() - s} secs")

                    free, total = cp.cuda.runtime.memGetInfo()
                    pool = cp.get_default_memory_pool()
                    core.getLogger().info(f"GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")
                    core.getLogger().info(f"UNTIL CONTRACTION: {perf_counter() - inner_loop} secs")

    deviceSynchronize()
    saving_started = perf_counter()
    pion_gamma_np          = core.gatherLattice(pion_gamma.get(),         [9, -1, -1, -1])
    proton_gamma_dirac_cpu = core.gatherLattice(proton_gamma_dirac.get(), [11, -1, -1, -1])


    #save as h5py file
    if latt_info.mpi_rank == 0:

    #ROLL every source so that time index 0 is t_src:  rolled[..., tau] = C[..., (t_src + tau) % 96]
        for t_idx, t0 in enumerate(t_src[i_cfg]):
            pion_gamma_np[:, :, :, :, t_idx]                = np.roll(pion_gamma_np[:, :, :, :, t_idx],                -t0, axis=-1)
            proton_gamma_dirac_cpu[:, :, :, :, :, :, t_idx] = np.roll(proton_gamma_dirac_cpu[:, :, :, :, :, :, t_idx], -t0, axis=-1)
            proton_gamma_dirac_cpu[:, :, :, :, :, :, t_idx][..., Lt - t0:] *= -1

        pion_filename   = f"pion_{smear_tag}_frac{smear_mom_x_str}_GEVP_cfg{cfg}.h5"
        proton_filename = f"proton_{smear_tag}_frac{smear_mom_x_str}_GEVP_DIRAC_cfg{cfg}.h5"
        gevp_dir = f"{current_dir}/{smear_tag}_GEVP_ez_momfrac{smear_mom_x_str}"
        os.makedirs(gevp_dir, exist_ok=True)

        with h5py.File(f"{gevp_dir}/{pion_filename}", "w") as f:
            dset = f.create_dataset("pion_gamma", data=pion_gamma_np)          # (nsmear, nsmear, 3, 3, nt, nx, ny, nz, nmom, 96)
            f.create_dataset("momentum_list", data=momentum_list)
            f.create_dataset("smear_list", data=np.array(smear_list))
            dset.attrs["gamma_list"] = np.array(["G5", "G45", "G35"], dtype=h5py.string_dtype())
            dset.attrs["smear_list"] = "rows of smear_list dataset: [steps, rho]; index order (sink, source)"
            dset.attrs["dim_spec"] = np.array(["smear_sink","smear_source","gamma_sink","gamma_source","t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
            dset.attrs["measurements"] = [cfg]
            dset.attrs["momentums"] = momentum_list
            dset.attrs["x_src_list"] = x_src[i_cfg]
            dset.attrs["y_src_list"] = y_src[i_cfg]
            dset.attrs["z_src_list"] = z_src[i_cfg]
            dset.attrs["t_src_list"] = t_src[i_cfg]
            dset.attrs["dim_time"] = np.arange(pion_gamma_np.shape[-1])

        with h5py.File(f"{gevp_dir}/{proton_filename}", "w") as f:
            dset = f.create_dataset("proton_gamma", data=proton_gamma_dirac_cpu)      # (nsmear, nsmear, 3, 3, 4, 4, nt, nx, ny, nz, nmom, 96)
            f.create_dataset("momentum_list", data=momentum_list)
            f.create_dataset("smear_list", data=np.array(smear_list))
            dset.attrs["gamma_list"] = np.array(["G5", "G45", "G35"], dtype=h5py.string_dtype())
            dset.attrs["smear_list"] = "rows of smear_list dataset: [steps, rho]; index order (sink, source)"
            dset.attrs["dim_spec"] = np.array(["smear_sink","smear_source","gamma_sink","gamma_source","dirac_sink","dirac_source","t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
            dset.attrs["measurements"] = [cfg]
            dset.attrs["momentums"] = momentum_list
            dset.attrs["x_src_list"] = x_src[i_cfg]
            dset.attrs["y_src_list"] = y_src[i_cfg]
            dset.attrs["z_src_list"] = z_src[i_cfg]
            dset.attrs["t_src_list"] = t_src[i_cfg]
            dset.attrs["dim_time"] = np.arange(proton_gamma_dirac_cpu.shape[-1])
            dset.attrs["diquark"] = "C Gamma_sink at the sink, C Gamma_source at the source (equal to Gamma_bar_source C for G5, G45, G35)"
            dset.attrs["propagator"] = "prop2 (k2 = -k) for all three quark lines; source width = smear_source, sink width = smear_sink"
            dset.attrs["how_to_project"] = "NOT projected. C(t) = sum_kl parity_p[k,l] M[l,k] for t<Lt/2 and -sum_kl parity_m[k,l] M[l,k] for t>=Lt/2, M = this dataset, l = dirac_sink, k = dirac_source. Antiperiodic sign already applied."
            f.create_dataset("parity_p", data=cp.asnumpy(parity_p))      # (1+g4)/2, forward half
            f.create_dataset("parity_m", data=cp.asnumpy(parity_m))      # (1-g4)/2, backward half, enters with a minus sign
    core.getLogger().info(f"SAVING SECTION:{perf_counter()-saving_started} secs")
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    core.getLogger().info(f"END CFG #{cfg}: GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")
dirac.freeGauge()