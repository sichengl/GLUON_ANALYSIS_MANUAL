#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH -J test_byq
#SBATCH -o test_byq_%j.out
#SBATCH -e test_byq_%j.out
#Timing and correctness test of delta_package_byq_opt.py (parallel readers, n_workers = 16) against delta_package_byq.py on a fresh node, whose file cache starts empty:
#1. the original packs cfg set A and 2. the opt packs cfg set B, both read from disk (16 cfgs each): compare their times (real, user, sys after each step);
#3. the opt packs set A as well (from the cache the original filled) and 4. test_compare_byq.py checks that its files match the original's.
#Submit from the folder with the setup and the test files: sbatch test_byq.sh
source ~/env_pyquda.sh
stamp() { while read line; do echo "$(date +%T) $line"; done; }     # clock time in front of every printed line
echo "=== 1. original, set A"; time (python3 -u test_orig_byq.py | stamp)
echo "=== 2. opt, set B"; time (python3 -u test_opt_byq.py B | stamp)
echo "=== 3. opt, set A"; time (python3 -u test_opt_byq.py A | stamp)
echo "=== 4. compare"; python3 test_compare_byq.py
