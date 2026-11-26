#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J pack_streamd
#SBATCH -o pack_streamd_%j.out
#SBATCH -e pack_streamd_%j.out
#Packs every finished a09m310 stream-d cfg with frontier_package_byq_opt_a09m310_streamd.py (n_workers = 16 reader processes, about 2.5 GB each with the proton) on one Frontier
#node; the GPUs are not used. Submit from this folder:
#    sbatch frontier_package_byq_opt.sh          (x only in this folder)
set -o pipefail                    # a python failure fails the job (sacct FAILED), as in the production scripts
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh
cd $SLURM_SUBMIT_DIR
[ "${BOOST_DIRECTION:-x}" = x ] || { echo "BOOST_DIRECTION=$BOOST_DIRECTION refused: this folder is the x production (current_dir ends in _direction_x)"; exit 1; }
BOOST_DIRECTION=x                                 # the run to pack: x only in this folder
time python3 -u frontier_package_byq_opt_a09m310_streamd.py --boost_direction $BOOST_DIRECTION
