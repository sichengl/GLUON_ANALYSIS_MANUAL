#!/bin/bash
#SBATCH -A gluonp0.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --qos=normal
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH -J pt2_p2pcheck
#SBATCH -o ./logs/pt2_p2pcheck_%j.out
#SBATCH -e ./logs/pt2_p2pcheck_%j.err

# P2P check: the Fermilab final job (fermilab/final_production_fermilab/fermilab_smear_3072cfgs/2pt_coulomb_boosted_sepq_3072srcs_fermilab_final.sh)
# run twice on the same node, on cfg 204 with 8 sources (setup: 2t x 2x x 2y x 1z), first with QUDA_ENABLE_P2P=3 (as in production), then with
# QUDA_ENABLE_P2P=0. The python script is an unchanged copy of the production one; the setup puts each run in its own folder, .../p2pcheck/p2p3 and
# .../p2pcheck/p2p0, so the two sets of 12 .h5 files can be compared entry by entry. With 8 sources the files cannot be compared with production
# (each t_src there is the mean over 128 spatial sources), hence the P2P=3 run as the reference.
# Time: per run about 1 min start, 3.6 min gauge fixing, 8 x 10 s sources, saving: about 6 min, so about 13 min for both; 30 min limit.
# The run folder is testruns/p2pcheck_<jobid>; a resubmitted job skips the cfg of a run whose .h5 files are already saved.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_fermilab_final.py
SETUP=sepq_2pt_setup_fermilab_final.py
SUBMIT=2pt_p2pcheck_fermilab.sh                 # name of this file, used for its copy in the run folder
SMEAR=../../tools/coulomb_smearing.py          # the only copy: 2pt_production/tools, two levels up from 2pt_production/p2pcheck/fermilab
COMM=../../tools/pt2_comm_tools.py             # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/project/gluonp0/sicheng/gluon_production/2pt_production/testruns/p2pcheck_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_p2pcheck_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_p2pcheck_${SLURM_JOB_ID}.err" "$RUNDIR"/
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

# QUDA (QUDA_ENABLE_P2P is set for each run below)
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python3 -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # same string as resource_path in core.init
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"
# start the test cache from the production tunings, so that the P2P=3 run does not retune and the production cache is never written
[ -f "$QUDA_RESOURCE_PATH/tunecache.tsv" ] || cp /lustre2/gluonp0/sliu1/.cache/quda/tunecache.tsv "$QUDA_RESOURCE_PATH"/

# run parameters: cfg_list[0] = 204
ICFG=${SLURM_ARRAY_TASK_ID:-0}
# fail fast if the gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python3 -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "task $ICFG: gauge file $GAUGE"
# versions of the modules the script imports, without starting MPI outside mpirun (importing pyquda calls MPI_Init).
# cupy is imported for its version: the name of its package depends on the CUDA build (cupy-cuda12x here, cupy-cuda13x on Delta), the module name does not
python3 -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'mpi4py', 'numpy', 'h5py', 'opt_einsum', 'tqdm']), sep=', '); import cupy; print('cupy', cupy.__version__, 'CUDA runtime', cupy.cuda.runtime.runtimeGetVersion())" || exit 1
nvidia-smi -L
nvidia-smi topo -m

# GPU memory of all 4 GPUs every 2 s, to see the peak (QUDA and cupy together)
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -lms 2000 > "$RUNDIR/gpu_mem_${SLURM_JOB_ID}.csv" &
MONITOR=$!

# this OpenMPI has no Slurm PMI support: launch with mpirun, not srun. PyQUDA gives node-local rank i GPU i.
STATUS=0
for P2P in 3 0; do                                    # the same run twice: with P2P as in production, then without
    export QUDA_ENABLE_P2P=$P2P
    echo "RUN WITH QUDA_ENABLE_P2P=$QUDA_ENABLE_P2P"; date
    mpirun -np 4 python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n 1 || { STATUS=$?; break; }     # -m mpi4py: an exception on any rank aborts all ranks
done
kill $MONITOR
date
exit $STATUS                                          # exit code of the failed run, so that sacct shows FAILED instead of COMPLETED
