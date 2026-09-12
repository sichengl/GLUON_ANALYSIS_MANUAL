#use loop function to construct the clovers and then get Fmunu
#gauge.loop takes in four sets of list of loops, and four parameters
#it calculates the product of each set of list of loops, sum them with the parameter
#then it replaces the four gauge links at each site with the four products, which are sum of one of the four list of loops, which are sumed with the parameter
#Note the fourth component of Qij and Qi4 are not used
#modified to start with sep=0 for wilson line

#==========================
#For direction, xyzt
#For data structure (index), tzyx
#==========================

#This version of FF projects Fmunu to traceless part and saves the imaginary part
from pyquda_utils import core, io, phase
from pyquda_utils.core import X, Y, Z, T
import cupy as cp
from opt_einsum import contract
from pyquda_utils.core import LatticeFermion, LatticeGauge
#import matplotlib.pyplot as plt
import h5py
import numpy as np
import json
import argparse
from cupy.cuda.runtime import deviceSynchronize
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


core.init([1, 1, 1, 1], resource_path="/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/.cache/quda")
parameters = json.loads(args.config)
start_cfg = parameters["cfg_n"]
num_cfg = parameters.get("num_cfg", 1)                          # configurations per process; default 1 = old behaviour
measurement_list = [start_cfg + 6 * i for i in range(num_cfg)]
wilson_line_list = list(range(0,16))
src_list = list(range(0,6))
sink_list = list(range(0,6))
hyp_smear_list = list(range(0,16)) 
q_phase_sign = 1
symmetric_qlist = [[0, 0, 0], [-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0]]
asymmetric_qlist = [
    [px, py, pz]
    for px in [0, -1, 1]
    for py in [0, -1, 1]
    for pz in [0, -1, 1]
]
symmetric_fourier_qlist = (
    q_phase_sign * np.asarray(symmetric_qlist, dtype=np.int64)
).tolist()
asymmetric_fourier_qlist = (
    q_phase_sign * np.asarray(asymmetric_qlist, dtype=np.int64)
).tolist()

for i_cfg,cfg in enumerate(measurement_list):
    gauge = io.readMILCGauge(f"/lustre/orion/lgt132/world-shared/DATA/MILC/a09m310/gauge/l3296f211b630m0074m037m440e.{cfg}")
    if i_cfg == 0:
        latt_info=gauge.latt_info
        #build phase and containers for symmetric and asymmetric 
        symmetric_phase = phase.MomentumPhase(latt_info).getPhases(symmetric_fourier_qlist)
        asymmetric_phase = phase.MomentumPhase(latt_info).getPhases(asymmetric_fourier_qlist)
        symmetric_corr = cp.zeros((len(measurement_list),len(src_list),len(sink_list),len(hyp_smear_list), len(wilson_line_list),len(symmetric_qlist),latt_info.Lt), "<c16")
        asymmetric_corr = cp.zeros((len(measurement_list),len(src_list),len(sink_list),len(hyp_smear_list), len(wilson_line_list),len(asymmetric_qlist),latt_info.Lt), "<c16")
    for i_smear, smear in enumerate(hyp_smear_list):
        
        #deviceSynchronize()
        s = perf_counter()  
        if i_smear != 0:
            gauge.hypSmear(1, 0.75, 0.6, 0.3, -1,True,True)
        #deviceSynchronize()
        core.getLogger().info(f"HYP smear #{cfg}: {perf_counter() - s} secs")
        
        """
        gauge_fixing_params = {
        "gauge_dir": 4,
        "Nsteps": 10000,
        "verbose_interval": 100,
        "relax_boost": 1.7,
        "tolerance": 1e-10,
        "reunit_interval": 10,
        "stopWtheta": 0
            }
        gauge.fixingOVR(**gauge_fixing_params)
        """

        #deviceSynchronize()
        s = perf_counter()
        Qij = gauge.loop(
            [
                [[X, Y, -X, -Y], [Y, -X, -Y, X], [-X, -Y, X, Y], [-Y, X, Y, -X]],
                [[Y, Z, -Y, -Z], [Z, -Y, -Z, Y], [-Y, -Z, Y, Z], [-Z, Y, Z, -Y]],
                [[X, Z, -X, -Z], [Z, -X, -Z, X], [-X, -Z, X, Z], [-Z, X, Z, -X]],
                [[T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T]],
            ],
            [ 1 , 1 , 1 , 1 ],
        )

        Qi4 = gauge.loop(
            [
                [[X, T, -X, -T], [T, -X, -T, X], [-X, -T, X, T], [-T, X, T, -X]],
                [[Y, T, -Y, -T], [T, -Y, -T, Y], [-Y, -T, Y, T], [-T, Y, T, -Y]],
                [[Z, T, -Z, -T], [T, -Z, -T, Z], [-Z, -T, Z, T], [-T, Z, T, -Z]],
                [[T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T]],
            ],
            [ 1 , 1 , 1 , 1 ],
        )

        Qij_dagger = LatticeGauge(latt_info)
        Qi4_dagger = LatticeGauge(latt_info)
        Qij_dagger.data[:] = Qij.data.conj().swapaxes(-1,-2)
        Qi4_dagger.data[:] = Qi4.data.conj().swapaxes(-1,-2)
        Fij = -1j / 8 * ( Qij - Qij_dagger )
        Fi4 = -1j / 8 * ( Qi4 - Qi4_dagger )
        
        # Project to traceless color matrix: F -> F - Tr(F)/3 * I
        eye3 = cp.eye(3, dtype=Fij.data.dtype)
        tr_Fij = cp.trace(Fij.data, axis1=-2, axis2=-1) / 3.0
        tr_Fi4 = cp.trace(Fi4.data, axis1=-2, axis2=-1) / 3.0

        Fij_norm2 = cp.vdot(Fij.data.ravel(), Fij.data.ravel()).real
        Fi4_norm2 = cp.vdot(Fi4.data.ravel(), Fi4.data.ravel()).real
        Fij_trace_norm2 = 3.0 * cp.vdot(tr_Fij.ravel(), tr_Fij.ravel()).real
        Fi4_trace_norm2 = 3.0 * cp.vdot(tr_Fi4.ravel(), tr_Fi4.ravel()).real
        trace_ratio_Fij = cp.sqrt(Fij_trace_norm2 / (Fij_norm2 + 1e-300)).get().item()
        trace_ratio_Fi4 = cp.sqrt(Fi4_trace_norm2 / (Fi4_norm2 + 1e-300)).get().item()
        trace_ratio_all = cp.sqrt(
            (Fij_trace_norm2 + Fi4_trace_norm2) / (Fij_norm2 + Fi4_norm2 + 1e-300)
        ).get().item()
        core.getLogger().info(
            f"TRACE IMPACT #{cfg} {smear} steps of HYP smear: "
            f"Fij={trace_ratio_Fij:.6e}, Fi4={trace_ratio_Fi4:.6e}, all={trace_ratio_all:.6e}"
        )

        Fij.data[:] = Fij.data - tr_Fij[..., None, None] * eye3
        Fi4.data[:] = Fi4.data - tr_Fi4[..., None, None] * eye3
        core.getLogger().info(f"Fij_trace_norm2={Fij_trace_norm2}")
        core.getLogger().info(f"Fi4_trace_norm2={Fi4_trace_norm2}")
        gauge_local = cp.asarray(gauge.data[2,:])
        gauge_local_conj = gauge_local.conj().swapaxes(-1,-2)
        
        #Fmunu = [ Fij.data[0,:] , Fij.data[1,:] , Fij.data[2,:] , Fi4.data[0,:] , Fi4.data[1,:] , Fi4.data[2,:] ]
        Fmunu = [
        cp.asarray(Fij.data[0,:]),
        cp.asarray(Fij.data[1,:]),
        cp.asarray(Fij.data[2,:]),
        cp.asarray(Fi4.data[0,:]),
        cp.asarray(Fi4.data[1,:]),
        cp.asarray(Fi4.data[2,:]),
        ]

        Fmunu_shift = [arr.copy() for arr in Fmunu]
        
        #deviceSynchronize()
        core.getLogger().info(f"BEFORE SHIFT #{cfg}: {perf_counter() - s} secs")

        #deviceSynchronize()
        s = perf_counter()
        for i_W, WL_indices in enumerate(wilson_line_list):
            
            #for wilson line with zero length skip shifting
            if i_W != 0:
                Fmunu_shift = [cp.roll((gauge_local_conj @ arr @ gauge_local)[::-1], shift=1, axis=2)for arr in Fmunu_shift]
                #Fmunu_shift = [ gauge_local_conj @ cp.roll(arr[::-1], shift=1, axis=2) @ gauge_local  for arr in Fmunu_shift]
            
            for i_src, src in enumerate(src_list):
                for i_sink, sink in enumerate(sink_list):

                    symmetric_corr[i_cfg,i_src,i_sink,i_smear,i_W] += contract("pwtzyx,wtzyxij,wtzyxji->pt", symmetric_phase,Fmunu[i_src],Fmunu_shift[i_sink] )
                    asymmetric_corr[i_cfg,i_src,i_sink,i_smear,i_W] += contract("pwtzyx,wtzyxij,wtzyxji->pt", asymmetric_phase,Fmunu[i_src],Fmunu_shift[i_sink] )
        #deviceSynchronize()
        core.getLogger().info(f"SHIFT #{cfg}: {perf_counter() - s} secs")
        


#cp.save("fmunu_corr.npy",corr)


    deviceSynchronize()
    symmetric_corr_imag_ratio = cp.sqrt(cp.vdot(symmetric_corr.imag.ravel(), symmetric_corr.imag.ravel()).real / (cp.vdot(symmetric_corr.real.ravel(), symmetric_corr.real.ravel()).real + 1e-300)).get().item()
    asymmetric_corr_imag_ratio = cp.sqrt(cp.vdot(asymmetric_corr.imag.ravel(), asymmetric_corr.imag.ravel()).real / (cp.vdot(asymmetric_corr.real.ravel(), asymmetric_corr.real.ravel()).real + 1e-300)).get().item()
    core.getLogger().info(f"LOCAL FF IMAG IMPACT #{cfg}: symmetric={symmetric_corr_imag_ratio:.6e}, asymmetric={asymmetric_corr_imag_ratio:.6e}")

    symmetric_tmp = core.gatherLattice(symmetric_corr.get(), [6, -1, -1, -1])
    asymmetric_tmp = core.gatherLattice(asymmetric_corr.get(), [6, -1, -1, -1])




    from pyquda_comm import getMPIRank
    rank = getMPIRank()

    if rank == 0:

        symmetric_tmp_cpu = symmetric_tmp
        asymmetric_tmp_cpu = asymmetric_tmp
        for name, arr in [("SYMMETRIC", symmetric_tmp_cpu), ("ASYMMETRIC", asymmetric_tmp_cpu)]:
            ratio_all = np.linalg.norm(arr.imag) / (np.linalg.norm(arr.real) + 1e-300)
            ratio_q0 = np.linalg.norm(arr[..., 0, :].imag) / (np.linalg.norm(arr[..., 0, :].real) + 1e-300)
            ratio_qnz = np.linalg.norm(arr[..., 1:, :].imag) / (np.linalg.norm(arr[..., 1:, :].real) + 1e-300)
            print(f"GLOBAL {name} FF IMAG IMPACT cfg{start_cfg}: all={ratio_all:.6e}, q0={ratio_q0:.6e}, q_nonzero={ratio_qnz:.6e}")

        filename = (f"/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/FF_production/FF_data/"
                f"FF_opp_symmetric_asymmetric_hyp0-{len(hyp_smear_list)-1}"
                f"_w{wilson_line_list[0]}-{wilson_line_list[-1]}_cfg{cfg}.h5")

        with h5py.File(filename, 'w') as f:
            symmetric_data = f.create_dataset('symmetric_corr', data=symmetric_tmp_cpu[i_cfg:i_cfg+1])
            asymmetric_data = f.create_dataset('asymmetric_corr', data=asymmetric_tmp_cpu[i_cfg:i_cfg+1])
            f.create_dataset('symmetric_qlist', data=np.asarray(symmetric_qlist, dtype=np.int64))
            f.create_dataset('asymmetric_qlist', data=np.asarray(asymmetric_qlist, dtype=np.int64))
            f.create_dataset('hyp_indices', data=np.asarray(hyp_smear_list, dtype=np.int64))
            f.create_dataset('wilson_line_list', data=np.asarray(wilson_line_list, dtype=np.int64))
            symmetric_data.attrs["dim_spec"] = "cfg, munu, rhosig, hyp, wilson_list, mom, t"
            asymmetric_data.attrs["dim_spec"] = "cfg, munu, rhosig, hyp, wilson_list, mom, t"
            symmetric_data.attrs['hyp_alpha'] = [0.75, 0.6, 0.3]
            symmetric_data.attrs['hyp_dir_ignore'] = -1
            symmetric_data.attrs['number_of_smears'] = hyp_smear_list
            symmetric_data.attrs['wilson_line_list'] = wilson_line_list
            symmetric_data.attrs['config_list'] = [cfg]
            f.attrs['q_phase_sign'] = q_phase_sign
            f.attrs['momentum_phase'] = 'exp(+i q_phase_sign q_code dot x)'
            f.attrs['momentum_relation'] = 'pi_code=pf_code+q_phase_sign*q_code'
            f.attrs['physical_transfer_convention'] = 'Delta_phys=P_f-P_i=-q_phase_sign*q_code'
            f.attrs['bilocal_direction'] = 'negative z; endpoints x and x-z before centering phase'
        #cp.save(f"fmunu_corr_smear_{smear_len*smear_parts}_insteps_from0_MILC.npy",tmp)

        """ 
        import matplotlib.pyplot as plt
        import numpy as np
        y_data = np.zeros((len(wilson_line_list)),"<c16")
        print(f"Rank {rank} is plotting...")
        tmp = cp.mean(tmp,axis=(3))
        for j in range(0,6):
            y_data = y_data + tmp[j,j,:,0]
        t_axis = range(len(y_data))


        print(f"ydata is {y_data}")


        plt.figure(figsize=(8, 6))
        plt.plot(t_axis,y_data, marker='o', linestyle='-', color='b')
        plt.yscale('log')
        plt.xlabel('l')
        plt.ylabel('Corr')
        plt.title('Field Strength Correlator')
        plt.grid(True)
        plt.savefig('fsc_correlation_plot.png')
        print("Plot saved successfully.")
    else:
        pass
        """


