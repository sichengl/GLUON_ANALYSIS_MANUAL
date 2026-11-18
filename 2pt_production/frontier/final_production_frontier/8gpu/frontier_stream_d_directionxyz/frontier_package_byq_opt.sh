#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J pack_streamd
#SBATCH -o pack_streamd_%j.out
#SBATCH -e pack_streamd_%j.out
#Packs every finished stream-d cfg with frontier_package_byq_opt.py (n_workers = 16 reader processes, about 2.5 GB each with the proton) on one Frontier
#node; the GPUs are not used. Submit from this folder:
#    BOOST_DIRECTION=x sbatch frontier_package_byq_opt.sh          (x, y or z; z if it is not set)
set -o pipefail                    # a python failure fails the job (sacct FAILED), as in the production scripts
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh
cd $SLURM_SUBMIT_DIR
BOOST_DIRECTION=${BOOST_DIRECTION:-z}             # the run to pack: lattice axis of the boost
time python3 -u frontier_package_byq_opt.py --boost_direction $BOOST_DIRECTION
