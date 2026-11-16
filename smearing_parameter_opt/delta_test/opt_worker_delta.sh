#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=12:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=208G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -J smear_opt
#SBATCH -o ./logs/smear_opt_%j.out
#SBATCH -e ./logs/smear_opt_%j.err

# Smearing parameter optimization: ONE job with 10 nodes, so that all nodes run at the same time (they wait for each other at every step; the tasks
# of a job array are started one by one by Slurm and did not run together). Worker w is node w of the job and holds one stream-e cfg
# (cfg_list[opt_cfg_index[w]] of the setup), 16 sources per cfg, 4 A100-40GB GPUs per node, grid [1,1,1,4]. Every worker keeps its cfg (gauge fixing,
# HYP and multigrid setup once) and repeats: wait for the kernel table of the step, invert the 16 sources, contract, write a small file. Worker 0 also
# checks every step and writes the next table; ITS .out FILE (logs/smear_opt_<job>_0.out) SHOWS THE STATUS OF EVERY STEP (the same text is in
# opt_dir/status.log). The run ends by itself (STOP file) or at the time limit; if one worker ends with an error, the whole job is stopped.
# The job script of the Delta production (smear_v2) with: the optimization files, 10 nodes with one srun per node (--worker w) instead of one
# array task per cfg (--icfg/--n).
# --nodes must be the number of workers in the setup, len(opt_cfg_index). A test on fewer nodes: shorten opt_cfg_index and sbatch --nodes=<number>.
# Restart: sbatch again: every worker continues at the first step it has not written. A new run needs a new opt_run_tag in the setup.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=opt_worker_delta.py
SETUP=opt_setup_delta.py
SUBMIT=opt_worker_delta.sh                              # name of this file, used for its copy in the run folder
SMEAR=../../tools/coulomb_smearing.py                   # the only copy: 2pt_production/tools, two levels up from 2pt_production/smear_gradient_descent/delta_test (the folder on Delta)
COMM=../../tools/pt2_comm_tools.py                      # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)
KERNEL=../tools_smearing_opt/opt_kernel.py              # the kernel: sum of boosted Gaussians, its derivatives, the mom_frac move
CONTRACT=../tools_smearing_opt/opt_contract.py          # the momentum-space contraction
CHECK=../tools_smearing_opt/opt_check.py                # the check of node 0: score, gradient, noise, decision, status

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$KERNEL" "$CONTRACT" "$CHECK"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder, ONE for all workers: copies of exactly what runs (python imports these copies), this sbatch script, links to the log files
RUNDIR=/projects/biqz/$USER/smearing_parameter_opt/testruns/opt_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$KERNEL" "$CONTRACT" "$CHECK" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/smear_opt_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/smear_opt_${SLURM_JOB_ID}.err" "$RUNDIR"/
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

# run parameters: worker w runs on node w of this job (NODES[w]) and holds cfg_list[opt_cfg_index[w]] of the setup
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))      # the names of the nodes of this job, as a list
NWORKERS=$(python -c "from ${SETUP%.py} import opt_cfg_index; print(len(opt_cfg_index))")
[ "${#NODES[@]}" = "$NWORKERS" ] || { echo "this job has ${#NODES[@]} nodes, the setup has $NWORKERS workers (opt_cfg_index): they must be equal"; exit 1; }
# fail fast if a gauge file is not there (the setup builds the name from gauge_file_prefix, cfg_list and opt_cfg_index)
for ((WORKER = 0; WORKER < NWORKERS; WORKER++)); do
    GAUGE=$(python -c "from ${SETUP%.py} import cfg_list, opt_cfg_index, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[opt_cfg_index[$WORKER]]}')")
    [ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
    echo "worker $WORKER: node ${NODES[$WORKER]}, gauge file $GAUGE"
done
# versions, and the modules the script imports, without starting MPI outside srun (importing pyquda calls MPI_Init)
python -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'cupy-cuda13x', 'mpi4py', 'numpy', 'opt_einsum']), sep=', ')" || exit 1
nvidia-smi -L
nvidia-smi topo -m

# GPU memory of all 4 GPUs every 2 s, to see the peak (QUDA and cupy together) on the 40 GB A100s. Only on the node this script runs on, the first
# node of the job = worker 0 (hostname above); the other nodes do the same work.
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -lms 2000 > "$RUNDIR/gpu_mem_${SLURM_JOB_ID}_0.csv" &
MONITOR=$!

# cray-mpich has Slurm PMI: launch with srun, ONE srun per worker, all started at the same time (the & at the end runs each in the background).
# -N 1 -n 4 -w node: the 4 ranks of worker w on node w of the job. --gpu-bind=none: PyQUDA gives node-local rank i GPU i, so every rank must see all 4 GPUs.
# mask_cpu puts rank i on the 16 cores next to GPU i (Delta gpuA100x4: GPU0 cores 48-63, GPU1 32-47, GPU2 16-31, GPU3 0-15; check with nvidia-smi topo -m above)
# Every worker has its own log, logs/smear_opt_<job>_<worker>.out and .err (linked in the run folder). -m mpi4py: an exception on any rank aborts all ranks of that worker.
# A worker that ends with an error can never deliver its steps and the others would only wait for it: "kill $$" then ends this script ($$ is the
# process number of this script), and when the job script ends Slurm stops the job with all its workers.
PIDS=""
for ((WORKER = 0; WORKER < NWORKERS; WORKER++)); do
    LOG=$SLURM_SUBMIT_DIR/logs/smear_opt_${SLURM_JOB_ID}_${WORKER}
    ln -s "$LOG.out" "$LOG.err" "$RUNDIR"/
    ( srun -N 1 -n 4 -w "${NODES[$WORKER]}" --gpu-bind=none --cpu-bind=mask_cpu:0xffff000000000000,0xffff00000000,0xffff0000,0xffff \
          python -m mpi4py "$SCRIPT" --worker $WORKER > "$LOG.out" 2> "$LOG.err" \
      || { echo "worker $WORKER on ${NODES[$WORKER]} ended with an error at $(date): stopping the job"; kill $$; } ) &
    PIDS="$PIDS $!"                                   # the process numbers of the 10 background commands
done
wait $PIDS                                            # until every worker has ended (STOP file): reached only if none of them failed
kill $MONITOR
date
