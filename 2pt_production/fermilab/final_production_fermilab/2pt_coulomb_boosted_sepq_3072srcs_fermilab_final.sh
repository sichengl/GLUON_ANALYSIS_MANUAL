#!/bin/bash
#SBATCH -A gluonp0.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --qos=normal
#SBATCH -t 14:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH -J pt2_final
#SBATCH --array=0-199
#SBATCH -o ./logs/pt2_final_%A_%a.out
#SBATCH -e ./logs/pt2_final_%A_%a.err

# Final run on Fermilab: the Delta final job (delta/final_decision_for_production_delta/2pt_coulomb_boosted_sepq_3072srcs_delta_final.sh) with the
# Fermilab header, environment and launcher of 2pt_coulomb_boosted_sepq_1024srcs_fermilab_streame_test_200cfgs.sh. 200 stream-e cfgs (204, 210, ..., 1398)
# with the 24t x 4x x 4y x 8z = 3072 sources per cfg, the two tuned smearings [4.5, 2.25, 0.8] and [1.5, 1.5, 0] at source and sink (the full 2 x 2
# smearing matrix) and three gammas: the 6 x 6 GEVP matrix. 2pt (pion + proton), sepq kinematics, Coulomb-gauge boosted smearing, 4 GPUs on one
# lq2_gpu node, grid [1,1,1,4].
# Array task i measures cfg_list[i] (--n 1): one cfg per task. Only the cfgs whose gauge file is on Fermilab can run: a task whose file is missing
# stops at the check below, so submit the tasks of the cfgs that are there, e.g. sbatch --array=0-9 ... for the first 10.
# Time: Delta measured 10.66 s per source, 9.2 h per cfg, and the Fermilab steps are as fast (inversion pair 3.7 s, pion + proton contraction 0.91 s),
# so about 9 h per cfg; 14 h leaves margin (lq2_gpu accepts 18 h and refuses 27 h). Output about 22 GB per cfg (0.93 GB per t_src), 4.5 TB for the 200 cfgs.
# Every task runs in its own folder testruns/final_<jobid>_<task>, on copies of the python files; the folder also keeps this sbatch script.
# A task killed at the time limit can be resubmitted (sbatch --array=<task> ...): the per-t_src parts already on disk are skipped.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_fermilab_final.py
SETUP=sepq_2pt_setup_fermilab_final.py
SUBMIT=2pt_coulomb_boosted_sepq_3072srcs_fermilab_final.sh            # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing.py
[ -f "$SMEAR" ] || SMEAR=../../tools/coulomb_smearing.py                 # repo layout: 2pt_production/tools, two levels up from 2pt_production/fermilab/<this folder>
[ -f "$SMEAR" ] || SMEAR=../tools/coulomb_smearing.py                    # or one level up, from a folder in 2pt_production
COMM=pt2_comm_tools.py
[ -f "$COMM" ] || COMM=../../tools/pt2_comm_tools.py                     # the shared TwoPtParams class, imported by the setup
[ -f "$COMM" ] || COMM=../tools/pt2_comm_tools.py

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/project/gluonp0/sicheng/gluon_production/2pt_production/testruns/final_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_final_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_final_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$RUNDIR"/
date; hostname

# environment (installed once by ~/setup_pyquda_env.sh)
module purge
module load gompi/2023a cuda/12.2.1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyquda
export QUDA_PATH=/project/gluonp0/sicheng/quda-install
export LD_LIBRARY_PATH=$QUDA_PATH/lib:$LD_LIBRARY_PATH
cd "$RUNDIR" || exit 1
export PYTHONPATH=$RUNDIR:$PYTHONPATH

# QUDA
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python3 -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # same string as resource_path in core.init
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"

# run parameters: array task i measures cfg_list[i] of the setup (one task alone: sbatch --array=3 ...)
ICFG=${SLURM_ARRAY_TASK_ID:-0}
# fail fast if this task's gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python3 -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "task $ICFG: gauge file $GAUGE"
# versions of the modules the script imports, without starting MPI outside mpirun (importing pyquda calls MPI_Init).
# cupy is imported for its version: the name of its package depends on the CUDA build (cupy-cuda12x here, cupy-cuda13x on Delta), the module name does not
python3 -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'mpi4py', 'numpy', 'h5py', 'opt_einsum', 'tqdm']), sep=', '); import cupy; print('cupy', cupy.__version__, 'CUDA runtime', cupy.cuda.runtime.runtimeGetVersion())" || exit 1
nvidia-smi -L
nvidia-smi topo -m

# GPU memory of all 4 GPUs every 2 s, to see the peak (QUDA and cupy together)
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -lms 2000 > "$RUNDIR/gpu_mem_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
MONITOR=$!

# this OpenMPI has no Slurm PMI support: launch with mpirun, not srun. PyQUDA gives node-local rank i GPU i.
mpirun -np 4 python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n 1       # -m mpi4py: an exception on any rank aborts all ranks
STATUS=$?                                             # exit code of the run, so that sacct shows FAILED instead of COMPLETED when python fails
kill $MONITOR
date
exit $STATUS
