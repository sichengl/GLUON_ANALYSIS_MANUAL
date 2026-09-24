#!/bin/bash
#SBATCH -A gluonp0.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --qos=normal
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -J pt2_coulomb_test
#SBATCH -o ./logs/pt2_coulomb_boosted_fermilab_test_%j.out
#SBATCH -e ./logs/pt2_coulomb_boosted_fermilab_test_%j.err

# 30-minute test of the Coulomb-gauge boosted smearing run on a normal lq2 node, 4 GPUs, grid [1,1,1,4] in the python script
# ends before the first configuration is saved: read the log for the convention check, gauge fixing, memory and timing
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_test_64srcs_fermilab.py

# fail fast if a module the script imports is missing from the submit directory
for f in "$SCRIPT" coulomb_smearing.py; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# keep a copy of exactly what ran; the smearing module copy is a record only, python imports the one in $SLURM_SUBMIT_DIR
RUN_SCRIPT=logs/${SCRIPT%.py}_${SLURM_JOB_ID}.py
cp "$SCRIPT" "$RUN_SCRIPT"
cp coulomb_smearing.py logs/coulomb_smearing_${SLURM_JOB_ID}.py
date; hostname

# environment (installed once by ~/setup_pyquda_env.sh)
module purge
module load gompi/2023a cuda/12.2.1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyquda
export QUDA_PATH=/project/gluonp0/sicheng/quda-install
export LD_LIBRARY_PATH=$QUDA_PATH/lib:$LD_LIBRARY_PATH

# QUDA
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=/lustre2/gluonp0/sliu1/.cache/quda     # same string as resource_path in core.init
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p $QUDA_RESOURCE_PATH

# run parameters: first configuration of the block
ICFG=${ICFG:-0}
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
nvidia-smi -L
# this OpenMPI has no Slurm PMI support: launch with mpirun, not srun
mpirun -np 4 python3 -m mpi4py "$RUN_SCRIPT" --icfg $ICFG     # -m mpi4py: an exception on any rank aborts all ranks
date
