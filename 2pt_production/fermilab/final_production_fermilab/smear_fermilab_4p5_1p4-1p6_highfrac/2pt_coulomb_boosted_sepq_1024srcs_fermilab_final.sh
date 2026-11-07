#!/bin/bash
#SBATCH -A gluonp0.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --qos=normal
#SBATCH -t 18:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH -J pt2_diag
#SBATCH --array=0-199
#SBATCH -o ./logs/pt2_final_%A_%a.out
#SBATCH -e ./logs/pt2_final_%A_%a.err

# Fermilab boost test at narrow rho_z, high-boost run rho_z 1.4, 1.5, 1.6 (large-pz pion, DIAGONAL run): the first 200 stream-e cfgs (204, 210, ..., 1398), the same stream
# and cfgs as Delta, with the 8t x 4x x 4y x 8z = 1024 sources per cfg (every third t_src of Delta's 3072), the 11 smearings
# [4.5, 1.4, 1.2], [4.5, 1.4, 1.3], [4.5, 1.4, 1.4], [4.5, 1.4, 1.5], [4.5, 1.5, 1.4], [4.5, 1.5, 1.5], [4.5, 1.6, 1.1], [4.5, 1.6, 1.2], [4.5, 1.6, 1.3], [4.5, 1.6, 1.4], [4.5, 1.6, 1.5],
# each at source and sink only (no smearing GEVP), three gammas, plain q = 0 case only. 2pt (pion + proton), Coulomb-gauge boosted smearing, 4 GPUs on one lq2_gpu node, grid [1,1,1,4].
# The other 4 points of the 15-point grid rho_z 1.4, 1.5, 1.6 x mom_frac 1.1 ... 1.5 are already on disk for the same cfgs and sources: [4.5, 1.5, 1.1], [4.5, 1.5, 1.2], [4.5, 1.5, 1.3] in ../smear_fermilab_4p5_1p5_momfrac, [4.5, 1.4, 1.1] in ../smear_fermilab_4p5_1p2-1p4_momfrac.
# Array task i measures cfg_list[i] (--n 1): one cfg per task. Only the cfgs whose gauge file is on Fermilab can run: a task whose file is missing
# stops at the check below, so submit the tasks of the cfgs that are there, e.g. sbatch --array=0-9 ... for the first 10.
# Time (measured in the two momfrac runs, 1354253 and 1354254: 4.9 s per smearing per source, 12.3-12.8 h for 9 smearings, 16.4-17.1 h for 12):
# about 53.7 s per source for 11 smearings, so about 15.3 h per cfg (up to about 15.7 h); 18 h limit (lq2_gpu accepts 18 h).
# Output about 0.85 GB per cfg (107 MB per t_src).
# Every task runs in its own folder testruns/final_<jobid>_<task>, on copies of the python files; the folder also keeps this sbatch script.
# A task killed at the time limit can be resubmitted (sbatch --array=<task> ...): the per-t_src parts already on disk are skipped.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_1024srcs_fermilab_final.py
SETUP=sepq_2pt_setup_fermilab_final.py
SUBMIT=2pt_coulomb_boosted_sepq_1024srcs_fermilab_final.sh            # name of this file, used for its copy in the run folder
SMEAR=../../../tools/coulomb_smearing.py       # the only copy: 2pt_production/tools, three levels up from 2pt_production/fermilab/final_production_fermilab/fermilab_smear_test_v2
COMM=../../../tools/pt2_comm_tools.py          # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)

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
