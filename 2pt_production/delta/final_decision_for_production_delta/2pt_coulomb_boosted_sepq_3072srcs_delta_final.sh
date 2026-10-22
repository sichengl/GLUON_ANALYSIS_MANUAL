#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=208G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -J pt2_final
#SBATCH --array=0-198:3
#SBATCH -o ./logs/pt2_final_%A_%a.out
#SBATCH -e ./logs/pt2_final_%A_%a.err

# Final run: 200 stream-e cfgs (204, 210, ..., 1398) with the 24t x 4x x 4y x 8z = 3072 sources per cfg, the two tuned smearings [4.5, 2.25, 0.8]
# and [1.5, 1.5, 0] at source and sink (the full 2 x 2 smearing matrix) and three gammas: the 6 x 6 GEVP matrix. 2pt (pion + proton), sepq kinematics,
# Coulomb-gauge boosted smearing, 4 A100-40GB GPUs on one Delta gpuA100x4 node, grid [1,1,1,4].
# Array task i measures cfg_list[i], cfg_list[i+1], cfg_list[i+2] (--n 3): 67 tasks, the last one has 2 cfgs.
# Time (from the tuning and stream-e test timings): about 13 s per source (2 x 3.8 s inversions, 4 x (0.17 s sink smearing + 1.06 s contraction),
# 0.6 s phases), so about 11.4 h per cfg and 34 h per task. Output about 5.6 GB per cfg (0.23 GB per t_src), 1.1 TB for the 200 cfgs.
# Every task runs in its own folder testruns/final_<jobid>_<task>, on copies of the python files; the folder also keeps this sbatch script.
# A task killed at the time limit can be resubmitted (sbatch --array=<task> ...): the per-t_src parts already on disk are skipped.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_delta_final.py
SETUP=sepq_2pt_setup_delta_final.py
SUBMIT=2pt_coulomb_boosted_sepq_3072srcs_delta_final.sh               # name of this file, used for its copy in the run folder
SMEAR=../../tools/coulomb_smearing.py          # the only copy: 2pt_production/tools, two levels up from 2pt_production/delta/<this folder>
COMM=../../tools/pt2_comm_tools.py             # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/projects/biqz/$USER/2pt_production/testruns/final_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_final_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_final_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$RUNDIR"/
date; hostname

# environment: conda env pyquda, QUDA_PATH, LD_LIBRARY_PATH (Cray PE defaults: PrgEnv-gnu, cray-mpich, cudatoolkit)
source ~/env_pyquda.sh
cd "$RUNDIR" || exit 1
export PYTHONPATH=$RUNDIR:$PYTHONPATH

# QUDA
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # same string as resource_path in core.init
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"

# run parameters: array task i starts at cfg_list[i] of the setup (one task alone: sbatch --array=3 ...)
ICFG=${SLURM_ARRAY_TASK_ID:-0}
# fail fast if this task's first gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "task $ICFG: gauge file $GAUGE"
# versions, and the modules the script imports, without starting MPI outside srun (importing pyquda calls MPI_Init)
python -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'cupy-cuda13x', 'mpi4py', 'numpy', 'h5py', 'opt_einsum', 'tqdm']), sep=', ')" || exit 1
nvidia-smi -L
nvidia-smi topo -m

# GPU memory of all 4 GPUs every 2 s, to see the peak (QUDA and cupy together) on the 40 GB A100s
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -lms 2000 > "$RUNDIR/gpu_mem_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
MONITOR=$!

# cray-mpich has Slurm PMI: launch with srun. --gpu-bind=none: PyQUDA gives node-local rank i GPU i, so every rank must see all 4 GPUs.
# mask_cpu puts rank i on the 16 cores next to GPU i (Delta gpuA100x4: GPU0 cores 48-63, GPU1 32-47, GPU2 16-31, GPU3 0-15; check with nvidia-smi topo -m above)
srun -n 4 --gpu-bind=none --cpu-bind=mask_cpu:0xffff000000000000,0xffff00000000,0xffff0000,0xffff \
    python -m mpi4py "$SCRIPT" --icfg $ICFG --n 3     # -m mpi4py: an exception on any rank aborts all ranks
STATUS=$?                                             # exit code of the run, so that sacct shows FAILED instead of COMPLETED when python fails
kill $MONITOR
date
exit $STATUS
