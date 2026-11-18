#Delta version. F_munu F_rhosigma correlator with a Wilson line along x, y or z, with HYP smearing or Wilson flow. Both are chosen on the command line:
#    python FF_covshift_boost_direction_delta.py --config '{"cfg_n": 204, "num_cfg": 1}' --smear hyp  --boost_direction x
#    python FF_covshift_boost_direction_delta.py --config '{"cfg_n": 204, "num_cfg": 1}' --smear flow --boost_direction z
#Job script: FF_covshift_boost_direction_delta.sh
#Merged from FF_covshift_oddeven_trimag_negative_direction_hyp.py and FF_covshift_oddeven_trimag_negative_direction_flow.py.
#The calculation at each smearing level is the block of those two files, unchanged.
#
#--smear hyp : 16 levels, level i = i HYP steps (0.75, 0.6, 0.3, all four directions); Wilson-line lengths 0..15   (the hyp file)
#--smear flow: 41 levels, level i = i x 10 Wilson-flow steps of 0.01, so flow time 0.1 i; Wilson-line lengths 0..9 (the flow file)
#Level 0 is the unsmeared field in both.
#
#--boost_direction x, y or z: the lattice axis of the Wilson line. Every place that depends on it is an "if boost_direction ==" in this file:
#    1. the momentum lists used for the phases        2. the link of the Wilson line
#    3. the order of the six F components             4. the shift by one site
#The labels in the file are written as seen from the boost direction: "z" is the boost axis and x, y are the two axes across it.
#    boost_direction x: (x, y, z) of the labels = lattice (y, z, x)      a q label (a, b, c) is lattice q = (c, a, b)
#    boost_direction y: (x, y, z) of the labels = lattice (z, x, y)      a q label (a, b, c) is lattice q = (b, c, a)
#So munu = xy, yz, xz, xt, yt, zt is
#    boost_direction x: lattice  F_yz, -F_xz, -F_xy, F_yt, F_zt, F_xt     (F_zx = -F_xz, F_yx = -F_xy)
#    boost_direction y: lattice -F_xz,  F_xy, -F_yz, F_zt, F_xt, F_yt
#and an x or y file is used in the analysis exactly like a z file.

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
from pyquda_utils.core import LatticeGauge
from pyquda_comm import getMPIRank
import h5py
import numpy as np
import json
import argparse
import os
from cupy.cuda.runtime import deviceSynchronize
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)                              # '{"cfg_n": 204, "num_cfg": 1}', as before
parser.add_argument("--smear", type=str, required=True, choices=["hyp", "flow"])      # HYP smearing or Wilson flow
parser.add_argument("--boost_direction", type=str, required=True, choices=["x", "y", "z"])  # lattice axis of the Wilson line
args = parser.parse_args()
boost_direction = args.boost_direction


#======== paths (Delta) ========
resource_path = "/u/sliu54/QUDA_resources/.cache/quda_FF"                                                # QUDA tuning cache of the one-GPU FF runs (the 2pt runs have their own)
gauge_prefix = "/work/hdd/biqz/MILC/l3296f211b630m0074m037m440e/l3296f211b630m0074m037m440e"             # stream e, the files of the 2pt production
out_dir = f"/work/hdd/biqz/sliu54/FF_production/FF_data/direction_{boost_direction}"                     # /work/hdd: /projects is too small
os.makedirs(resource_path, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)


if boost_direction == "z":
    boost_direction_note = "Wilson line along negative lattice z; all labels are lattice labels"
if boost_direction == "x":
    boost_direction_note = ("Wilson line along negative lattice x. Labels are written as seen from the boost direction: (x, y, z) of the labels = lattice (y, z, x). "
                      "A q label (a, b, c) is lattice q = (c, a, b); munu = xy, yz, xz, xt, yt, zt is lattice F_yz, -F_xz, -F_xy, F_yt, F_zt, F_xt")
if boost_direction == "y":
    boost_direction_note = ("Wilson line along negative lattice y. Labels are written as seen from the boost direction: (x, y, z) of the labels = lattice (z, x, y). "
                      "A q label (a, b, c) is lattice q = (b, c, a); munu = xy, yz, xz, xt, yt, zt is lattice -F_xz, F_xy, -F_yz, F_zt, F_xt, F_yt")


#======== run parameters ========
core.init([1, 1, 1, 1], resource_path=resource_path)
parameters = json.loads(args.config)
start_cfg = parameters["cfg_n"]
num_cfg = parameters.get("num_cfg", 1)                          # configurations per process; default 1 = old behaviour
measurement_list = [start_cfg + 6 * i for i in range(num_cfg)]
src_list = list(range(0,6))
sink_list = list(range(0,6))

#smearing: the settings of the two old files
if args.smear == "hyp":
    smear_list = list(range(0,16))               # level i = i HYP steps
    wilson_line_list = list(range(0,16))
if args.smear == "flow":
    flow_steps = 10                              # Wilson-flow steps between two levels
    flow_eps = 0.01                              # size of one step
    smear_list = list(range(0,41))               # level i = flow time i * flow_steps * flow_eps = 0.1 i
    wilson_line_list = list(range(0,10))

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
#1. DIRECTION: the q lists above are labels, written for the boost direction z. The phases need the lattice q
if boost_direction == "x":
    symmetric_fourier_qlist = [[c, a, b] for a, b, c in symmetric_fourier_qlist]
    asymmetric_fourier_qlist = [[c, a, b] for a, b, c in asymmetric_fourier_qlist]
if boost_direction == "y":
    symmetric_fourier_qlist = [[b, c, a] for a, b, c in symmetric_fourier_qlist]
    asymmetric_fourier_qlist = [[b, c, a] for a, b, c in asymmetric_fourier_qlist]

for i_cfg,cfg in enumerate(measurement_list):
    gauge = io.readMILCGauge(f"{gauge_prefix}.{cfg}")
    if i_cfg == 0:
        latt_info=gauge.latt_info
        #build phase and containers for symmetric and asymmetric
        symmetric_phase = phase.MomentumPhase(latt_info).getPhases(symmetric_fourier_qlist)
        asymmetric_phase = phase.MomentumPhase(latt_info).getPhases(asymmetric_fourier_qlist)
        symmetric_corr = cp.zeros((len(measurement_list),len(src_list),len(sink_list),len(smear_list), len(wilson_line_list),len(symmetric_qlist),latt_info.Lt), "<c16")
        asymmetric_corr = cp.zeros((len(measurement_list),len(src_list),len(sink_list),len(smear_list), len(wilson_line_list),len(asymmetric_qlist),latt_info.Lt), "<c16")
        #used only for boost_direction x: the sites whose x - 1 neighbour is stored one place lower along the last (halved x) index
        Lx, Ly, Lz, Lt = latt_info.size
        parity, t, z, y = cp.indices((2, Lt, Lz, Ly))
        x_lower = ((parity + t + z + y) % 2 == 0).reshape(2, Lt, Lz, Ly, 1, 1, 1)
    for i_smear, smear in enumerate(smear_list):

        s = perf_counter()
        if i_smear != 0:
            if args.smear == "hyp":
                gauge.hypSmear(1, 0.75, 0.6, 0.3, -1,True,True)
            if args.smear == "flow":
                gauge.wilsonFlow(flow_steps, flow_eps)
        core.getLogger().info(f"{args.smear} level {smear} #{cfg}: {perf_counter() - s} secs")

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
            f"TRACE IMPACT #{cfg} {args.smear} level {smear}: "
            f"Fij={trace_ratio_Fij:.6e}, Fi4={trace_ratio_Fi4:.6e}, all={trace_ratio_all:.6e}"
        )

        Fij.data[:] = Fij.data - tr_Fij[..., None, None] * eye3
        Fi4.data[:] = Fi4.data - tr_Fi4[..., None, None] * eye3
        core.getLogger().info(f"Fij_trace_norm2={Fij_trace_norm2}")
        core.getLogger().info(f"Fi4_trace_norm2={Fi4_trace_norm2}")
        #2. DIRECTION: the link along the Wilson line (gauge.data[0], [1], [2] = x, y, z links)
        if boost_direction == "x":
            gauge_local = cp.asarray(gauge.data[0,:])
        if boost_direction == "y":
            gauge_local = cp.asarray(gauge.data[1,:])
        if boost_direction == "z":
            gauge_local = cp.asarray(gauge.data[2,:])
        gauge_local_conj = gauge_local.conj().swapaxes(-1,-2)

        Fxy, Fyz, Fxz = cp.asarray(Fij.data[0,:]), cp.asarray(Fij.data[1,:]), cp.asarray(Fij.data[2,:])      # lattice components
        Fxt, Fyt, Fzt = cp.asarray(Fi4.data[0,:]), cp.asarray(Fi4.data[1,:]), cp.asarray(Fi4.data[2,:])
        #3. DIRECTION: the six components in the order xy, yz, xz, xt, yt, zt as seen from the boost direction
        if boost_direction == "z":
            Fmunu = [ Fxy,  Fyz,  Fxz, Fxt, Fyt, Fzt]
        if boost_direction == "x":                          # (x, y, z) of the labels = lattice (y, z, x)
            Fmunu = [ Fyz, -Fxz, -Fxy, Fyt, Fzt, Fxt]
        if boost_direction == "y":                          # (x, y, z) of the labels = lattice (z, x, y)
            Fmunu = [-Fxz,  Fxy, -Fyz, Fzt, Fxt, Fyt]

        Fmunu_shift = [arr.copy() for arr in Fmunu]

        core.getLogger().info(f"BEFORE SHIFT #{cfg}: {perf_counter() - s} secs")

        s = perf_counter()
        for i_W, WL_indices in enumerate(wilson_line_list):

            #for wilson line with zero length skip shifting
            #4. DIRECTION: new(x) = U^dag(x - mu) F(x - mu) U(x - mu). Arrays are [parity, t, z, y, x//2, color, color]. A step of one site always
            #changes the parity ([::-1]). Along z or y it is then a roll along that axis. Along x the last index is x//2, so only half of the
            #sites move by one place (x_lower); the other half stay at the same place
            if i_W != 0:
                if boost_direction == "z":
                    Fmunu_shift = [cp.roll((gauge_local_conj @ arr @ gauge_local)[::-1], shift=1, axis=2)for arr in Fmunu_shift]
                if boost_direction == "y":
                    Fmunu_shift = [cp.roll((gauge_local_conj @ arr @ gauge_local)[::-1], shift=1, axis=3)for arr in Fmunu_shift]
                if boost_direction == "x":
                    Fmunu_shift = [(gauge_local_conj @ arr @ gauge_local)[::-1] for arr in Fmunu_shift]
                    Fmunu_shift = [cp.where(x_lower, cp.roll(arr, shift=1, axis=4), arr) for arr in Fmunu_shift]

            for i_src, src in enumerate(src_list):
                for i_sink, sink in enumerate(sink_list):

                    symmetric_corr[i_cfg,i_src,i_sink,i_smear,i_W] += contract("pwtzyx,wtzyxij,wtzyxji->pt", symmetric_phase,Fmunu[i_src],Fmunu_shift[i_sink] )
                    asymmetric_corr[i_cfg,i_src,i_sink,i_smear,i_W] += contract("pwtzyx,wtzyxij,wtzyxji->pt", asymmetric_phase,Fmunu[i_src],Fmunu_shift[i_sink] )
        core.getLogger().info(f"SHIFT #{cfg}: {perf_counter() - s} secs")

        #free this level's GPU arrays before the next HYP or flow step: QUDA allocates its own GPU memory and cannot use what cupy holds
        del Qij, Qi4, Qij_dagger, Qi4_dagger, Fij, Fi4, tr_Fij, tr_Fi4, Fxy, Fyz, Fxz, Fxt, Fyt, Fzt, Fmunu, Fmunu_shift, gauge_local, gauge_local_conj
        cp.get_default_memory_pool().free_all_blocks()


    deviceSynchronize()
    symmetric_corr_imag_ratio = cp.sqrt(cp.vdot(symmetric_corr.imag.ravel(), symmetric_corr.imag.ravel()).real / (cp.vdot(symmetric_corr.real.ravel(), symmetric_corr.real.ravel()).real + 1e-300)).get().item()
    asymmetric_corr_imag_ratio = cp.sqrt(cp.vdot(asymmetric_corr.imag.ravel(), asymmetric_corr.imag.ravel()).real / (cp.vdot(asymmetric_corr.real.ravel(), asymmetric_corr.real.ravel()).real + 1e-300)).get().item()
    core.getLogger().info(f"LOCAL FF IMAG IMPACT #{cfg}: symmetric={symmetric_corr_imag_ratio:.6e}, asymmetric={asymmetric_corr_imag_ratio:.6e}")

    symmetric_tmp = core.gatherLattice(symmetric_corr.get(), [6, -1, -1, -1])
    asymmetric_tmp = core.gatherLattice(asymmetric_corr.get(), [6, -1, -1, -1])

    rank = getMPIRank()

    if rank == 0:

        symmetric_tmp_cpu = symmetric_tmp
        asymmetric_tmp_cpu = asymmetric_tmp
        for name, arr in [("SYMMETRIC", symmetric_tmp_cpu), ("ASYMMETRIC", asymmetric_tmp_cpu)]:
            ratio_all = np.linalg.norm(arr.imag) / (np.linalg.norm(arr.real) + 1e-300)
            ratio_q0 = np.linalg.norm(arr[..., 0, :].imag) / (np.linalg.norm(arr[..., 0, :].real) + 1e-300)
            ratio_qnz = np.linalg.norm(arr[..., 1:, :].imag) / (np.linalg.norm(arr[..., 1:, :].real) + 1e-300)
            print(f"GLOBAL {name} FF IMAG IMPACT cfg{cfg}: all={ratio_all:.6e}, q0={ratio_q0:.6e}, q_nonzero={ratio_qnz:.6e}")

        #the smearing and the direction are in the file name
        filename = (f"{out_dir}/"
                f"FF_opp_symmetric_asymmetric_{args.smear}{smear_list[0]}-{smear_list[-1]}"
                f"_w{wilson_line_list[0]}-{wilson_line_list[-1]}_boost{boost_direction}_cfg{cfg}.h5")

        with h5py.File(filename, 'w') as f:
            symmetric_data = f.create_dataset('symmetric_corr', data=symmetric_tmp_cpu[i_cfg:i_cfg+1])
            asymmetric_data = f.create_dataset('asymmetric_corr', data=asymmetric_tmp_cpu[i_cfg:i_cfg+1])
            f.create_dataset('symmetric_qlist', data=np.asarray(symmetric_qlist, dtype=np.int64))
            f.create_dataset('asymmetric_qlist', data=np.asarray(asymmetric_qlist, dtype=np.int64))
            f.create_dataset('smear_indices', data=np.asarray(smear_list, dtype=np.int64))
            f.create_dataset('wilson_line_list', data=np.asarray(wilson_line_list, dtype=np.int64))
            for data in [symmetric_data, asymmetric_data]:
                data.attrs["dim_spec"] = f"cfg, munu, rhosig, {args.smear}, wilson_list, mom, t"
                data.attrs['smear'] = args.smear
                data.attrs['boost_direction'] = boost_direction
                data.attrs['boost_direction_note'] = boost_direction_note
                data.attrs['number_of_smears'] = smear_list
                data.attrs['wilson_line_list'] = wilson_line_list
                data.attrs['config_list'] = [cfg]
                if args.smear == "hyp":
                    data.attrs['hyp_alpha'] = [0.75, 0.6, 0.3]
                    data.attrs['hyp_dir_ignore'] = -1
                if args.smear == "flow":
                    data.attrs['number_of_steps_each_cycle'] = flow_steps
                    data.attrs['size_of_each_step'] = flow_eps
            f.attrs['smear'] = args.smear
            f.attrs['boost_direction'] = boost_direction
            f.attrs['boost_direction_note'] = boost_direction_note
            f.attrs['q_phase_sign'] = q_phase_sign
            f.attrs['momentum_phase'] = 'exp(+i q_phase_sign q_code dot x)'
            f.attrs['momentum_relation'] = 'pi_code=pf_code+q_phase_sign*q_code'
            f.attrs['physical_transfer_convention'] = 'Delta_phys=P_f-P_i=-q_phase_sign*q_code'
            f.attrs['bilocal_direction'] = f'negative {boost_direction} of the lattice (z of the labels); endpoints x and x-z before centering phase'
