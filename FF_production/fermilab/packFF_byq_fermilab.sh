#!/bin/bash
#SBATCH --account=gluonp0.lq2_gpu
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH -J packFF_byq
#SBATCH -o packFF_byq_%j.out
#SBATCH -e packFF_byq_%j.out
#Packs every cfg that has an FF file with packFF_byq_fermilab.py (one reader, little memory) on one GPU share of an lq2_gpu node; the GPU is not used,
#it is requested only because lq2_gpu is a GPU partition. The smearing and the direction come from two variables, both required.
#The Fermilab copy of FF_production/delta/packFF_byq_delta.sh: the Fermilab header and environment. Submit from this folder:
#    sbatch --export=ALL,SMEAR=hyp,BOOST_DIRECTION=x packFF_byq_fermilab.sh          (SMEAR hyp or flow; BOOST_DIRECTION x, y or z)
module purge                                      # environment of the FF job script (installed once by ~/setup_pyquda_env.sh); the packer uses only numpy and h5py
module load gompi/2023a cuda/12.2.1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyquda
if [ -z "$SMEAR" ] || [ -z "$BOOST_DIRECTION" ]; then
    echo "set SMEAR (hyp or flow) and BOOST_DIRECTION (x, y or z):  sbatch --export=ALL,SMEAR=hyp,BOOST_DIRECTION=x packFF_byq_fermilab.sh"
    exit 1
fi
set -o pipefail                                   # if python fails, the job ends as FAILED (without it the exit code would be the one of stamp)
stamp() { while read line; do echo "$(date +%T) $line"; done; }     # clock time in front of every printed line
time (python3 -u packFF_byq_fermilab.py --smear $SMEAR --boost_direction $BOOST_DIRECTION | stamp)
