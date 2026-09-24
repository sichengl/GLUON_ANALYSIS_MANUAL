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
from coulomb_smearing import coulomb_boosted_source, coulomb_boosted_sink, coulomb_gauge_theta


parser = argparse.ArgumentParser()
parser.add_argument("--icfg", type=int, default=0)      # starting position in cfg_list
parser.add_argument("--n", type=int, default=20)        # number of configs to measure
args = parser.parse_args()
icfg0 = args.icfg
Ls = 32
Lt = 96
n = args.n  #number of configs to measure, starting from 204, with step size 6
mom_min = 0
mom_max = 6

# smearing: boosted Gaussian in Coulomb gauge (no gauge links), the same operator at source and sink
rho_T = 3.25                                            # transverse radius; rho_z = rho_T reproduces the width of N40 rho3.25
shape_list = [[rho_T, rho_T], [rho_T, rho_T / 2]]       # [rho_transverse, rho_z]: isotropic, anisotropic with the z radius halved
shape_names = ["iso", "aniso"]
mom_frac_list = [0.6, 0.3]                              # quark boost as a fraction of pz = mom_max
k_list = [np.array([0, 0, frac * mom_max]) for frac in mom_frac_list]     # boost in units of 2pi/L
smear_types = [(i_shape, i_frac) for i_shape in range(len(shape_list)) for i_frac in range(len(mom_frac_list))]   # the 4 (shape, boost) operators
frac_str = [("%.3f" % frac).rstrip("0").rstrip(".").replace(".", "p") for frac in mom_frac_list]
smear_tag = f"coulomb_rhoT{rho_T}_{'-'.join(shape_names)}_frac{'-'.join(frac_str)}"


cfg_list = np.arange(204, 204 + 800*6, 6)   # The complete cfg list, 800 configs
cfg_measure_spacing = 5
measurement_list = cfg_list[icfg0::cfg_measure_spacing][:n]
ncfg     = (measurement_list - 204) // 6

# unshifted source grids
t_base = np.arange(0, Lt, 12)   # 8
x_base = np.arange(0, Ls,  16)   # 2
y_base = np.arange(0, Ls,  16)   # 2
z_base = np.arange(0, Ls,  16)   # 2

# broadcast (n,1) + (1,nsrc) -> (n, nsrc); index as [icfg, isrc]
t_src = (t_base[None, :] + 5*ncfg[:, None]) % Lt   # (n, 8)
x_src = (x_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 2)
y_src = (y_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 2)
z_src = (z_base[None, :] + 3*ncfg[:, None]) % Ls   # (n, 2)
run_parameters = {
    "Ls": Ls,
    "Lt": Lt,
    "cfgs_to_meas": n,
    "shape_list": shape_list,
    "mom_frac_list": mom_frac_list,
    "k_list": k_list,
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
gamma_list = [G5, G45]
gamma_names = ["G5", "G45"]
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
#current_dir = os.environ.get("SLURM_SUBMIT_DIR", os.path.dirname(os.path.abspath(__file__)))
current_dir = "/lustre2/gluonp0/sliu1/2pt_production"
n_shape, n_frac, n_gamma = len(shape_list), len(mom_frac_list), len(gamma_list)
pion_gamma         = cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, t_src.shape[1], x_src.shape[1], y_src.shape[1], z_src.shape[1], len(momentum_list), latt_info.Lt), "<c16")          #(shape_sink, shape_source, frac_sink, frac_source, gamma_sink, gamma_source, t, x, y, z, p, Lt)
proton_gamma_dirac = cp.zeros((n_shape, n_shape, n_frac, n_frac, n_gamma, n_gamma, 4, 4, t_src.shape[1], x_src.shape[1], y_src.shape[1], z_src.shape[1], len(momentum_list), latt_info.Lt), "<c16")   #(shape_sink, shape_source, frac_sink, frac_source, gamma_sink, gamma_source, dirac_sink, dirac_source, t, x, y, z, p, Lt)

for i_cfg, cfg in tqdm(enumerate(measurement_list),desc=f"Processing cfgs"):


    pion_gamma[:] = 0
    proton_gamma_dirac[:] = 0

    #READ GAUGE
    deviceSynchronize()
    s = perf_counter()
    gauge_file = f"/lustre2/gluonp0/MILC/l3296f211b630m0074m037m440d/l3296f211b630m0074m037m440d.{cfg}"
    gauge = io.readMILCGauge(gauge_file,checksum=True, reunitarize_sigma=1e-6)
    deviceSynchronize()
    core.getLogger().info(f"READ GAUGE #{cfg}: {perf_counter() - s} secs")

    #COULOMB GAUGE FIXING of the unsmeared links
    deviceSynchronize()
    s = perf_counter()
    gauge.fixingOVR(gauge_dir=3, Nsteps=20000, verbose_interval=500, relax_boost=1.7, tolerance=1e-12, reunit_interval=10, stopWtheta=1)   # 3 = Coulomb; overrelaxation, since QUDA's FFT method runs on one GPU only
    theta = coulomb_gauge_theta(latt_info, gauge)
    deviceSynchronize()
    core.getLogger().info(f"COULOMB GAUGE FIXING: {perf_counter() - s} secs, theta {theta:.3e}")
    if theta > 1e-6:                                # QUDA stops silently after Nsteps; cfg 204 ends at 7.8e-9 on both machines, 1e-12 is never reached
        raise RuntimeError(f"Coulomb gauge fixing failed: theta {theta:.3e} after 20000 steps")

    #HYP on the gauge-fixed links: HYP is gauge covariant, so the propagators live in the same Coulomb gauge as the smearing
    deviceSynchronize()
    s = perf_counter()
    gauge_hyp = gauge.copy()
    del gauge
    core.getLogger().info(f"DOING HYP SMEARING")
    core.getLogger().info(f"plaq_hyp_before = {gauge_hyp.plaquette()}")
    gauge_hyp.hypSmear(1, 0.75, 0.6, 0.3, -1,True,True)
    deviceSynchronize()
    core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
    core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

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

                    for i_src_shape, i_src_frac in smear_types:
                        rho_T_src, rho_z_src = shape_list[i_src_shape]
                        k_src = k_list[i_src_frac]

                        #SRC: boosted Gaussian in Coulomb gauge, +k for prop1 and -k for prop2 as in the Wuppertal version
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"SRC SMEARING: {shape_names[i_src_shape]} rho_T {rho_T_src} rho_z {rho_z_src}, frac {mom_frac_list[i_src_frac]}")
                        prop1_inv = coulomb_boosted_source(latt_info, +k_src, src_pos, rho_T_src, rho_z_src)
                        prop2_inv = coulomb_boosted_source(latt_info, -k_src, src_pos, rho_T_src, rho_z_src)
                        deviceSynchronize()
                        core.getLogger().info(f"SOURCE SMEAR: {perf_counter() - s} secs")

                        #INVERT
                        deviceSynchronize()
                        s = perf_counter()
                        core.getLogger().info(f"SOLVING DIRAC EQ")
                        prop1_inv = core.invertPropagator(dirac, prop1_inv, mrhs=12)
                        prop2_inv = core.invertPropagator(dirac, prop2_inv, mrhs=12)
                        core.getLogger().info(f"INVERT 2 propagagors: {perf_counter() - s} secs")

                        for i_sink_shape, i_sink_frac in smear_types:
                            rho_T_sink, rho_z_sink = shape_list[i_sink_shape]
                            k_sink = k_list[i_sink_frac]

                            #SINK: the same smearing with the sink parameters, from the unsmeared solution each time
                            deviceSynchronize()
                            s = perf_counter()
                            core.getLogger().info(f"SINK SMEARING: {shape_names[i_sink_shape]} rho_T {rho_T_sink} rho_z {rho_z_sink}, frac {mom_frac_list[i_sink_frac]}")
                            prop1 = coulomb_boosted_sink(latt_info, prop1_inv, +k_sink, rho_T_sink, rho_z_sink)
                            prop2 = coulomb_boosted_sink(latt_info, prop2_inv, -k_sink, rho_T_sink, rho_z_sink)
                            deviceSynchronize()
                            core.getLogger().info(f"SINK SMEAR: {perf_counter() - s} secs")

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

                                    pion_gamma[i_sink_shape, i_src_shape, i_sink_frac, i_src_frac, i_gamma_sink, i_gamma_source, t_idx, x_idx, y_idx, z_idx] += contract(
                                    "li,lipt->pt", gamma_source_bar @ G5, pion_open)

                                    proton_gamma_dirac[i_sink_shape, i_src_shape, i_sink_frac, i_src_frac, i_gamma_sink, i_gamma_source, :, :, t_idx, x_idx, y_idx, z_idx] += contract(
                                    "ij,ijlkpt->lkpt", charge @ gamma_source, proton_open)

                            deviceSynchronize()
                            core.getLogger().info(f"CONTRACT ALL: {perf_counter() - s} secs")

                    free, total = cp.cuda.runtime.memGetInfo()
                    pool = cp.get_default_memory_pool()
                    core.getLogger().info(f"GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")
                    core.getLogger().info(f"UNTIL CONTRACTION: {perf_counter() - inner_loop} secs")

    deviceSynchronize()
    saving_started = perf_counter()
    pion_gamma_np          = core.gatherLattice(pion_gamma.get(),         [pion_gamma.ndim - 1, -1, -1, -1])
    proton_gamma_dirac_cpu = core.gatherLattice(proton_gamma_dirac.get(), [proton_gamma_dirac.ndim - 1, -1, -1, -1])


    #save as h5py file
    if latt_info.mpi_rank == 0:

    #ROLL every source so that time index 0 is t_src:  rolled[..., tau] = C[..., (t_src + tau) % 96]
        for t_idx, t0 in enumerate(t_src[i_cfg]):
            pion_gamma_np[:, :, :, :, :, :, t_idx]                = np.roll(pion_gamma_np[:, :, :, :, :, :, t_idx],                -t0, axis=-1)
            proton_gamma_dirac_cpu[:, :, :, :, :, :, :, :, t_idx] = np.roll(proton_gamma_dirac_cpu[:, :, :, :, :, :, :, :, t_idx], -t0, axis=-1)
            proton_gamma_dirac_cpu[:, :, :, :, :, :, :, :, t_idx][..., Lt - t0:] *= -1

        pion_filename   = f"pion_{smear_tag}_GEVP_cfg{cfg}.h5"
        proton_filename = f"proton_{smear_tag}_GEVP_DIRAC_cfg{cfg}.h5"
        gevp_dir = f"{current_dir}/{smear_tag}_GEVP_ez"
        os.makedirs(gevp_dir, exist_ok=True)
        smearing_note = ("boosted Gaussian in Coulomb gauge, no gauge links: K(d) = exp(-(dx^2+dy^2)/rho_T^2 - dz^2/rho_z^2) exp(+i 2pi/L k.d), d = x - y, "
                         "Gaussian part normalized to sum 1; the same K at source and sink; rho_z = rho_T equals the width of N40 rho3.25 Wuppertal smearing")
        gauge_fix_note = "Coulomb gauge fixing of the unsmeared links by overrelaxation, fixingOVR(gauge_dir=3, Nsteps=20000, verbose_interval=500, relax_boost=1.7, tolerance=1e-12, reunit_interval=10, stopWtheta=1), before HYP"
        momentum_note = "sink phase exp(+2 pi i p.(x - x_src)/L), PyQUDA MomentumPhase: momentum_list label p is physical momentum -p; the quark boost favors label +pz, physical -pz"

        with h5py.File(f"{gevp_dir}/{pion_filename}", "w") as f:
            dset = f.create_dataset("pion_gamma", data=pion_gamma_np)          # (nshape, nshape, nfrac, nfrac, 2, 2, nt, nx, ny, nz, nmom, 96)
            f.create_dataset("momentum_list", data=momentum_list)
            f.create_dataset("shape_list", data=np.array(shape_list))
            f.create_dataset("mom_frac_list", data=np.array(mom_frac_list))
            f.create_dataset("k_list", data=np.array(k_list))
            dset.attrs["gamma_list"] = np.array(gamma_names, dtype=h5py.string_dtype())
            dset.attrs["shape_names"] = np.array(shape_names, dtype=h5py.string_dtype())
            dset.attrs["shape_list"] = "rows of shape_list dataset: [rho_T, rho_z]; index order (sink, source)"
            dset.attrs["mom_frac_list"] = "quark boost k = frac * mom_max in units of 2pi/L along z; index order (sink, source)"
            dset.attrs["dim_spec"] = np.array(["shape_sink","shape_source","frac_sink","frac_source","gamma_sink","gamma_source","t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
            dset.attrs["smearing"] = smearing_note
            dset.attrs["gauge_fixing"] = gauge_fix_note
            dset.attrs["momentum_convention"] = momentum_note
            dset.attrs["gauge_file"] = gauge_file
            dset.attrs["propagator"] = "prop1 boosted by +k and prop2 by -k, at source and sink"
            dset.attrs["time_reflection"] = "C_ab(Lt - t) = s_a s_b C_ab(t) with s = +1 for G5 and -1 for G45 (a = gamma_sink, b = gamma_source): the G5-G45 elements are odd, include the sign when averaging forward and backward"
            dset.attrs["measurements"] = [cfg]
            dset.attrs["momentums"] = momentum_list
            dset.attrs["x_src_list"] = x_src[i_cfg]
            dset.attrs["y_src_list"] = y_src[i_cfg]
            dset.attrs["z_src_list"] = z_src[i_cfg]
            dset.attrs["t_src_list"] = t_src[i_cfg]
            dset.attrs["dim_time"] = np.arange(pion_gamma_np.shape[-1])

        with h5py.File(f"{gevp_dir}/{proton_filename}", "w") as f:
            dset = f.create_dataset("proton_gamma", data=proton_gamma_dirac_cpu)      # (nshape, nshape, nfrac, nfrac, 2, 2, 4, 4, nt, nx, ny, nz, nmom, 96)
            f.create_dataset("momentum_list", data=momentum_list)
            f.create_dataset("shape_list", data=np.array(shape_list))
            f.create_dataset("mom_frac_list", data=np.array(mom_frac_list))
            f.create_dataset("k_list", data=np.array(k_list))
            dset.attrs["gamma_list"] = np.array(gamma_names, dtype=h5py.string_dtype())
            dset.attrs["shape_names"] = np.array(shape_names, dtype=h5py.string_dtype())
            dset.attrs["shape_list"] = "rows of shape_list dataset: [rho_T, rho_z]; index order (sink, source)"
            dset.attrs["mom_frac_list"] = "quark boost k = frac * mom_max in units of 2pi/L along z; index order (sink, source)"
            dset.attrs["dim_spec"] = np.array(["shape_sink","shape_source","frac_sink","frac_source","gamma_sink","gamma_source","dirac_sink","dirac_source","t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
            dset.attrs["smearing"] = smearing_note
            dset.attrs["gauge_fixing"] = gauge_fix_note
            dset.attrs["momentum_convention"] = momentum_note
            dset.attrs["gauge_file"] = gauge_file
            dset.attrs["measurements"] = [cfg]
            dset.attrs["momentums"] = momentum_list
            dset.attrs["x_src_list"] = x_src[i_cfg]
            dset.attrs["y_src_list"] = y_src[i_cfg]
            dset.attrs["z_src_list"] = z_src[i_cfg]
            dset.attrs["t_src_list"] = t_src[i_cfg]
            dset.attrs["dim_time"] = np.arange(proton_gamma_dirac_cpu.shape[-1])
            dset.attrs["diquark"] = "C Gamma_sink at the sink, C Gamma_source at the source (equal to -Gamma_bar_source C for G5 and G45, an overall sign)"
            dset.attrs["propagator"] = "prop2 (boost -k) for all three quark lines; source smearing = (shape_source, frac_source), sink smearing = (shape_sink, frac_sink)"
            dset.attrs["how_to_project"] = "NOT projected. C(t) = sum_kl parity_p[k,l] M[l,k] for t<Lt/2 and -sum_kl parity_m[k,l] M[l,k] for t>=Lt/2, M = this dataset, l = dirac_sink, k = dirac_source. Antiperiodic sign already applied. Time reflection: the projected backward correlator at Lt - t equals s_a s_b times the forward one at t, s = +1 for G5 and -1 for G45 (a = gamma_sink, b = gamma_source); include this sign when averaging forward and backward."
            f.create_dataset("parity_p", data=cp.asnumpy(parity_p))      # (1+g4)/2, forward half
            f.create_dataset("parity_m", data=cp.asnumpy(parity_m))      # (1-g4)/2, backward half, enters with a minus sign
    core.getLogger().info(f"SAVING SECTION:{perf_counter()-saving_started} secs")
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    core.getLogger().info(f"END CFG #{cfg}: GPU used {(total - free)/1e9:.1f} GB, cupy pool {pool.total_bytes()/1e9:.1f} GB, cupy live {pool.used_bytes()/1e9:.1f} GB")
dirac.freeGauge()
