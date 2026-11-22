#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J packFF_byq_d
#SBATCH -o packFF_byq_d_%j.out
#SBATCH -e packFF_byq_d_%j.out
#STREAM D copy of packFF_byq_frontier.sh: only this line, the job name, the log names and the python file differ.
#Packs every cfg that has an FF file with packFF_byq_frontier_streamd.py (one reader, little memory) on one Frontier node; the GPUs are not used.
#The smearing and the direction come from two variables, both required.
#The Frontier copy of FF_production/delta/packFF_byq_delta.sh: the Frontier header and environment. Submit from this folder:
#    SMEAR=hyp BOOST_DIRECTION=x sbatch packFF_byq_frontier_streamd.sh          (SMEAR hyp or flow; BOOST_DIRECTION x, y or z)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh             # environment of the FF job script (Sergey's PyQUDA build); the packer uses only numpy and h5py
if [ -z "$SMEAR" ] || [ -z "$BOOST_DIRECTION" ]; then
    echo "set SMEAR (hyp or flow) and BOOST_DIRECTION (x, y or z):  SMEAR=hyp BOOST_DIRECTION=x sbatch packFF_byq_frontier_streamd.sh"
    exit 1
fi
set -o pipefail                                   # if python fails, the job ends as FAILED (without it the exit code would be the one of stamp)
stamp() { while read line; do echo "$(date +%T) $line"; done; }     # clock time in front of every printed line
time (python3 -u packFF_byq_frontier_streamd.py --smear $SMEAR --boost_direction $BOOST_DIRECTION | stamp)
