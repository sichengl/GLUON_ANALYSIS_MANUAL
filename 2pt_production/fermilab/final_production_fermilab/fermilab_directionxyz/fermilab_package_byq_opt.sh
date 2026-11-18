#!/bin/bash
#SBATCH -A gluonp0.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --qos=normal
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH -J pack_byq_opt
#SBATCH -o pack_byq_opt_%j.out
#SBATCH -e pack_byq_opt_%j.out
#Packs every finished cfg with fermilab_package_byq_opt.py (n_workers = 16 reader processes, about 2.5 GB each with the proton) on one lq2_gpu node;
#the GPU is not used, it is requested only because lq2_gpu is a GPU partition. Submit from this folder:
#    sbatch --export=ALL,BOOST_DIRECTION=x fermilab_package_byq_opt.sh          (x, y or z; z if it is not set)
#The Fermilab copy of delta/final_decision_for_production_delta/smear_v1_directionxyz/delta_package_byq_opt.sh: the Fermilab header and environment
#of the 2pt job script of this folder, and pipefail.
set -o pipefail                    # a python failure fails the job (sacct FAILED) instead of being hidden by the pipe into stamp
cd "$SLURM_SUBMIT_DIR"

# environment of the 2pt job script (installed once by ~/setup_pyquda_env.sh); the packer uses only numpy and h5py from it
module purge
module load gompi/2023a cuda/12.2.1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyquda

BOOST_DIRECTION=${BOOST_DIRECTION:-z}             # the run to pack: lattice axis of the boost
stamp() { while read line; do echo "$(date +%T) $line"; done; }     # clock time in front of every printed line
time (python3 -u fermilab_package_byq_opt.py --boost_direction $BOOST_DIRECTION | stamp)
