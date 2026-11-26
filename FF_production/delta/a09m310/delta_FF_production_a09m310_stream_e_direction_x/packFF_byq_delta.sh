#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH -J packFF_byq
#SBATCH -o packFF_byq_%j.out
#SBATCH -e packFF_byq_%j.out
#Packs every cfg that has an FF file with packFF_byq_delta_streame.py (one reader, little memory) on one GPU share of a gpuA40x4 node; the GPU is not used,
#it is requested only because the account is a GPU allocation. The smearing and the direction come from two variables, both required.
#Submit from this folder:
#    SMEAR=hyp sbatch packFF_byq_delta.sh          (SMEAR hyp or flow; x only in this folder)
source ~/env_pyquda.sh
[ "${BOOST_DIRECTION:-x}" = x ] || { echo "BOOST_DIRECTION=$BOOST_DIRECTION refused: this folder is the x production (out_dir ends in _direction_x)"; exit 1; }
BOOST_DIRECTION=x                                 # lattice axis of the Wilson line: x only in this folder
if [ -z "$SMEAR" ]; then
    echo "set SMEAR (hyp or flow):  SMEAR=hyp sbatch packFF_byq_delta.sh"
    exit 1
fi
set -o pipefail                                   # if python fails, the job ends as FAILED (without it the exit code would be the one of stamp)
stamp() { while read line; do echo "$(date +%T) $line"; done; }     # clock time in front of every printed line
time (python3 -u packFF_byq_delta_a09m310_streame.py --smear $SMEAR --boost_direction $BOOST_DIRECTION | stamp)
