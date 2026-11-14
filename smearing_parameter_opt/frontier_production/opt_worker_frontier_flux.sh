#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 12:00:00
#SBATCH -N 800
#SBATCH -J smear_opt_flux
#SBATCH -o ./logs/smear_opt_flux_%j.out
#SBATCH -e ./logs/smear_opt_flux_%j.err

# Smearing parameter optimization on Frontier, stream e: one cfg per node (cfg_list[opt_cfg_index[w]] of the setup on node w), the 8 GCDs of the node per
# cfg, 16 sources per cfg. Every node keeps its cfg (gauge fixing, HYP and multigrid setup once) and repeats: wait for the kernel table of the step,
# invert the 16 sources, contract, write a small file. Node 0 also checks every step and writes the next table; ITS LOG flux_logs/worker_0_cfg204.log
# SHOWS THE STATUS OF EVERY STEP (the same text is in opt_dir/status.log). The run ends by itself (STOP file) or at the time limit.
# The nodes are started through Flux inside the Slurm job, as in the stream-d production
# (2pt_production/frontier/final_production_frontier/8gpu/frontier_stream_d/2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd_flux.sh):
# the environment, the Flux lines and the per-node options are those of that file; it differs in the files copied, in --worker instead of --icfg/--n,
# and in taking the list of cfgs from opt_cfg_index of the setup (no ICFG_START, no STEP_TIME).
# The number of nodes (-N) must equal the number of entries of opt_cfg_index.
# Continue a run that hit the time limit: sbatch this file again (every node continues at the first step it has not written; lost nodes stay out).
# A new run needs a new opt_run_tag in the setup.
# Every job runs in its own folder runs/smear_opt_flux_<jobid>, on copies of the python files; the folder also keeps this sbatch script, the Flux
# submit script and one log per node (flux_logs/).
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=opt_worker_frontier.py
SETUP=opt_setup_frontier.py
SUBMIT=opt_worker_frontier_flux.sh                      # name of this file, used for its copy in the run folder
SMEAR=../../tools/coulomb_smearing.py                   # the only copy: 2pt_production/tools, two levels up from 2pt_production/smear_gradient_descent/<this folder> (the folder on Frontier)
COMM=../../tools/pt2_comm_tools.py                      # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)
KERNEL=../tools_smearing_opt/opt_kernel.py              # the kernel: sum of boosted Gaussians, its derivatives, the mom_frac move
CONTRACT=../tools_smearing_opt/opt_contract.py          # the momentum-space contraction
CHECK=../tools_smearing_opt/opt_check.py                # the check of node 0: score, gradient, noise, decision, status

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$KERNEL" "$CONTRACT" "$CHECK"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=$SLURM_SUBMIT_DIR/runs/smear_opt_flux_${SLURM_JOB_ID}
mkdir -p "$RUNDIR/flux_logs" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$KERNEL" "$CONTRACT" "$CHECK" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/smear_opt_flux_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/smear_opt_flux_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment (Sergey's PyQUDA build), then Flux and the GPU-aware hwloc it needs to see the GPUs (modules of the Frontier guide's Flux example)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh
module load hwloc/2.9.1-gpu flux
command -v flux > /dev/null || { echo "flux not found after module load"; exit 1; }
flux --version

# QUDA: the tuning cache is set by core.init(resource_path=quda_resource_path) from the setup file, so QUDA_RESOURCE_PATH is not exported here;
# its path is read from the setup file only to create the folder and to place the profile files next to the tuning files
TUNE_DIR=$(cd "$RUNDIR" && python3 -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")
[ -n "$TUNE_DIR" ] || { echo "could not read quda_resource_path from $SETUP"; exit 1; }
export QUDA_ENABLE_TUNING=1
export QUDA_PROFILE_OUTPUT_BASE=${TUNE_DIR}/profile_
export QUDA_ENABLE_P2P=0                  # must stay 0 on Frontier: ROCm IPC gives silent halo errors
export QUDA_ENABLE_MPS=1                  # one GCD visible per rank: PyQUDA then uses device 0 on every rank
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export PYTHONPATH=$RUNDIR:$PYTHONPATH:/ccs/home/sicheng/packages      # the run folder first, so python imports the copies
export CUPY_CACHE_DIR=/lustre/orion/lgt132/scratch/sicheng/cupy_cache/${SLURM_JOB_ID}
export RUN_TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}     # TMPDIR of the python runs (set in each run below); Flux itself keeps the node-local /tmp
mkdir -p $TUNE_DIR $CUPY_CACHE_DIR $RUN_TMPDIR
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer for Cray MPICH GPU support; harmless with GDR off (GDR hangs intermittently on this build, keep it off)
export SCRIPT

# nodes of this run: node w holds cfg_list[opt_cfg_index[w]]. A node whose gauge file is missing is marked as lost here (node 0 then does not wait
# for it) and gets no run. The job must have exactly as many nodes as opt_cfg_index has entries.
cd "$RUNDIR"
python3 -c "
import os
from ${SETUP%.py} import cfg_list, opt_cfg_index, gauge_file_prefix, opt_dir
from opt_check import mark_lost
for w, i in enumerate(opt_cfg_index):
    gauge = f'{gauge_file_prefix}.{cfg_list[i]}'
    if os.path.exists(gauge):
        print(w, cfg_list[i])
    else:
        mark_lost(opt_dir, w, f'missing gauge file {gauge}')" > worker_list.txt || exit 1
N_WORKERS=$(python3 -c "from ${SETUP%.py} import opt_cfg_index; print(len(opt_cfg_index))")
[ "$N_WORKERS" -eq "$SLURM_NNODES" ] || { echo "the job has $SLURM_NNODES nodes but opt_cfg_index has $N_WORKERS entries: submit with -N $N_WORKERS"; exit 1; }
RUN_LIST=""
while read W CFG; do RUN_LIST="$RUN_LIST $W:$CFG"; done < worker_list.txt
export RUN_LIST
echo "nodes $SLURM_NNODES, runs: $(echo $RUN_LIST | wc -w) (the others have no gauge file and are marked as lost)"

# the script Flux runs on the first node: one Flux job per node of the run (1 node, 8 ranks, 7 cores and the closest GCD per rank), then wait for all
# of them. Flux copies this environment into every job. Each rank takes its GCD from CUDA_VISIBLE_DEVICES, which gpu-affinity=per-task sets (Frontier
# guide: --gpus-per-task alone leaves all 8 GCDs visible).
cat > flux_submit.sh << 'EOF2'
flux resource list
for PAIR in $RUN_LIST; do
    W=${PAIR%%:*}; CFG=${PAIR##*:}
    flux submit -N 1 -n 8 -c 7 --gpus-per-task=1 -o gpu-affinity=per-task -o cpu-affinity=per-task \
        --job-name=w$W --output=flux_logs/worker_${W}_cfg${CFG}.log \
        bash -c 'export ROCR_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; export TMPDIR=$RUN_TMPDIR
                 exec python3 -m mpi4py $SCRIPT --worker '$W''
done
flux queue drain                          # returns when every run has finished or failed
flux jobs -a                              # one line per node: ST = CD finished, F failed, CA canceled
EOF2

# one Flux broker per node (all 56 free cores and 8 GCDs of the node), started by a single srun; only this srun talks to the Slurm controller
srun -N $SLURM_NNODES -n $SLURM_NNODES -c 56 --gpus-per-node=8 env TMPDIR=/tmp flux start bash "$RUNDIR/flux_submit.sh"
STATUS=$?                                 # exit code of the Flux instance: 0 even when single runs failed, so check the flux jobs table above
date
exit $STATUS
