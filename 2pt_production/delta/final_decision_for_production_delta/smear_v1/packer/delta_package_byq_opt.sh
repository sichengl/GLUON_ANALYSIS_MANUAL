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
#Packs every finished cfg with delta_package_byq_opt.py (n_workers = 16 reader processes, about 2.5 GB each with the proton) on one GPU share of a
#gpuA40x4 node; the GPU is not used, it is requested only because the account is a GPU allocation. Submit from this folder: sbatch delta_package_byq_opt.sh
source ~/env_pyquda.sh
stamp() { while read line; do echo "$(date +%T) $line"; done; }     # clock time in front of every printed line
time (python3 -u delta_package_byq_opt.py | stamp)
