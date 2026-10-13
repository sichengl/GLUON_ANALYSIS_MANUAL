#Prints the size and momentum of every operator of the setup, built with numpy on the CPU (no GPU, no PyQUDA): run it after editing
#operator_list, before submitting.   python3 kernel_radii.py
#rms_T, rms_z: rms radii of |K|^2, transverse and along z (the operators that change the envelope are matched to G_ref here);
#r_source: their combination, the radius mean_radius gives for the smeared source; mean_k, width_k: mean and width of the momentum
#intensity along z, in units of 2pi/L. The job logs the same numbers (KERNEL CHECK lines) and writes them to the .h5 files.
import os
import sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..", "..", "tools")]        # the setup imports pt2_comm_tools from 2pt_production/tools
from new_kernels_2pt_setup_fermilab import fermilab_params, operator_list
from new_kernels_tools import build_kernel, kernel_moments

spatial_size = [fermilab_params.GLs, fermilab_params.GLs, fermilab_params.GLs]
print(f"{'operator':14s} {'rms_T':>7s} {'rms_z':>7s} {'r_source':>8s} {'mean_k':>7s} {'width_k':>7s}   function")
for row in operator_list:
    kernel = build_kernel(np, spatial_size, row)
    rms_T, rms_z, mean_k, width_k = kernel_moments(np, spatial_size, kernel)
    print(f"{row[0]:14s} {rms_T:7.3f} {rms_z:7.3f} {(rms_T**2 + rms_z**2) ** 0.5:8.3f} {mean_k:7.2f} {width_k:7.2f}   {row[1]}")
