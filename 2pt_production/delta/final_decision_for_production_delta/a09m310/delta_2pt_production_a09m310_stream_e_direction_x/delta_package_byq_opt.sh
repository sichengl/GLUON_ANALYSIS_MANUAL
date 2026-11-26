#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH -J pack_byq_opt
#SBATCH -o pack_byq_opt_%j.out
#SBATCH -e pack_byq_opt_%j.out
#Packs every finished a09m310 stream-e direction-x cfg with delta_package_byq_opt_a09m310_streame.py (n_workers = 16 reader processes, about 2.5 GB each with the proton) on one GPU share of a
#gpuA40x4 node; the GPU is not used, it is requested only because the account is a GPU allocation. Submit from this folder:
#    sbatch delta_package_byq_opt.sh          (x only in this folder)
source ~/env_pyquda.sh
[ "${BOOST_DIRECTION:-x}" = x ] || { echo "BOOST_DIRECTION=$BOOST_DIRECTION refused: this folder is the x production (current_dir ends in _direction_x)"; exit 1; }
BOOST_DIRECTION=x                                 # the run to pack: x only in this folder
stamp() { while read line; do echo "$(date +%T) $line"; done; }     # clock time in front of every printed line
time (python3 -u delta_package_byq_opt_a09m310_streame.py --boost_direction $BOOST_DIRECTION | stamp)
