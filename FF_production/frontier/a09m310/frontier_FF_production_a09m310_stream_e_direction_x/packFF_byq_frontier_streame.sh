#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J packFF_byq
#SBATCH -o packFF_byq_%j.out
#SBATCH -e packFF_byq_%j.out
#Packs every cfg that has an FF file with packFF_byq_frontier_streame.py (one reader, little memory) on one Frontier node; the GPUs are not used.
#The smearing and the direction come from two variables, both required.
#The Frontier copy of FF_production/delta/packFF_byq_delta.sh: the Frontier header and environment. Submit from this folder:
#    SMEAR=hyp sbatch packFF_byq_frontier_streame.sh          (SMEAR hyp or flow; x only in this folder)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh             # environment of the FF job script (Sergey's PyQUDA build); the packer uses only numpy and h5py
[ "${BOOST_DIRECTION:-x}" = x ] || { echo "BOOST_DIRECTION=$BOOST_DIRECTION refused: this folder is the x production (out_dir ends in _direction_x)"; exit 1; }
BOOST_DIRECTION=x                                 # lattice axis of the Wilson line: x only in this folder
if [ -z "$SMEAR" ]; then
    echo "set SMEAR (hyp or flow):  SMEAR=hyp sbatch packFF_byq_frontier_streame.sh"
    exit 1
fi
set -o pipefail                                   # if python fails, the job ends as FAILED (without it the exit code would be the one of stamp)
stamp() { while read line; do echo "$(date +%T) $line"; done; }     # clock time in front of every printed line
time (python3 -u packFF_byq_frontier_a09m310_streame.py --smear $SMEAR --boost_direction $BOOST_DIRECTION | stamp)
