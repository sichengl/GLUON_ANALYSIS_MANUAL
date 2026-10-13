#!/bin/bash
#SBATCH -A gluonp0.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --qos=normal
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -J pt2_new_kernels
#SBATCH --array=0-3
#SBATCH -o ./logs/pt2_new_kernels_%A_%a.out
#SBATCH -e ./logs/pt2_new_kernels_%A_%a.err

# Test of the new smearing kernels: the 14 operators of operator_list in the setup, each one at source and sink (diagonal correlators only),
# 32 sources per cfg (4t x 2x x 2y x 2z), one stream-d cfg per array task, pion only, q = 0 at the 99 sink momenta, 4 GPUs, grid [1,1,1,4].
# Submit from this folder (2pt_production/fermilab/new_kernels).
# Test: --array=0-3 measures the first 4 of the 40 cfgs 204, 234, ..., 1374. More statistics: sbatch --array=4-39 (the same folder of
# results keeps growing; the job refuses to write into it if operator_list was changed, so a new list needs a new operator_tag).
# Before the first cfg the job checks every kernel on the GPUs (log lines KERNEL CHECK): a failed check stops it within minutes.
# Time: the smearing tuning budgets at most 2.5 h for 30 smearings x 32 sources per cfg (2pt_coulomb_boosted_smear_tune_32srcs_fermilab.sh),
# so 14 operators take at most about 1.2 h per cfg, plus the gauge fixing; 4 h leaves room.
# Every task runs in its own folder testruns/new_kernels_<jobid>_<task>, on copies of the python files; the folder also keeps this sbatch script.
# A task killed at the time limit can be resubmitted (sbatch --array=<task> ...): the per-t_src parts already on disk are skipped.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_new_kernels_32srcs_fermilab.py
SETUP=new_kernels_2pt_setup_fermilab.py
SUBMIT=2pt_coulomb_new_kernels_32srcs_fermilab.sh                  # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing_kernels.py                                   # the kernel package lives in this folder only
TOOLS=new_kernels_tools.py                                           # build_kernel and the GPU checks of the kernels
COMM=pt2_comm_tools.py
[ -f "$COMM" ] || COMM=../../tools/pt2_comm_tools.py                     # the shared TwoPtParams class, imported by the setup

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$TOOLS" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/project/gluonp0/sicheng/gluon_production/2pt_production/testruns/new_kernels_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$RUNDIR"
cp "$SCRIPT" "$SETUP" "$SMEAR" "$TOOLS" "$COMM" "$RUNDIR"/
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_new_kernels_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_new_kernels_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$RUNDIR"/
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
export QUDA_RESOURCE_PATH=/lustre2/gluonp0/sliu1/.cache/quda     # same string as quda_resource_path in new_kernels_2pt_setup_fermilab.py
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p $QUDA_RESOURCE_PATH

# run parameters: array task i measures cfg_list[i] of the setup (one task alone: sbatch --array=3 ...)
ICFG=${SLURM_ARRAY_TASK_ID:-0}
cd "$RUNDIR"
export PYTHONPATH=$RUNDIR:$PYTHONPATH
# fail fast if this task's gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python3 -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "task $ICFG: gauge file $GAUGE"
# beta_kernel needs scipy: fail fast if the environment does not have it
python3 -c "import scipy" || { echo "scipy is missing in the pyquda environment (beta_kernel needs it)"; exit 1; }
nvidia-smi -L
# this OpenMPI has no Slurm PMI support: launch with mpirun, not srun
mpirun -np 4 python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n 1     # -m mpi4py: an exception on any rank aborts all ranks
date
