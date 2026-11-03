#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 12:00:00
#SBATCH -N 200
#SBATCH -J pt2_final_8gcd_d_flux
#SBATCH -o ./logs/pt2_final_8gcd_d_flux_%j.out
#SBATCH -e ./logs/pt2_final_8gcd_d_flux_%j.err

# Final run on Frontier, stream d, many nodes at once: one cfg per node, each run the same 8-rank run as 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.sh
# (same python files, --n 1), launched through Flux inside the Slurm job instead of one srun per node: the Frontier guide does not recommend more than
# ~100 simultaneous sruns (each one talks to the shared Slurm controller) and recommends Flux for this ("Ensemble Jobs" -> "Flux in a Slurm allocation").
# The job measures cfg_list[ICFG_START], ..., cfg_list[ICFG_START + nodes - 1]; a cfg whose gauge file is missing is skipped (its node stays idle).
# One cfg takes about 18.2 h (21.3 s per source, tol 1e-10 test), longer than the 12-h limit of 184+ node jobs, so each batch of cfgs takes two jobs:
#   job A:  ICFG_START=0 sbatch <this file>                                                          (12 h: about 15 of the 24 t_src per cfg)
#   job B:  ICFG_START=0 STEP_TIME=8.5h sbatch -t 09:00:00 --dependency=afterany:<job A id> <this file>   (skips the saved t_src, finishes, merges the .h5)
# STEP_TIME is a time limit for each cfg's run: a stuck run is killed at that time instead of keeping the 199 finished nodes waiting (charged) until the
# job limit; its saved t_src stay on disk for a later job. Next batches: ICFG_START=200, 400, 600.
# Test first:  ICFG_START=0 sbatch -N 2 -t 02:00:00 <this file>      (cfgs 204, 210: Flux starts both runs, about 1-2 t_src each are saved)
# Every job runs in its own folder runs/final_8gcd_flux_<jobid>, on copies of the python files; the folder also keeps this sbatch script, the Flux
# submit script and one log per cfg (flux_logs/).
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.py
SETUP=sepq_2pt_setup_frontier_final_8gcd.py
SUBMIT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd_flux.sh          # name of this file, used for its copy in the run folder
SMEAR=../../../../tools/coulomb_smearing.py    # the only copy: 2pt_production/tools, four levels up from 2pt_production/frontier/final_production_frontier/8gpu/<this folder>
COMM=../../../../tools/pt2_comm_tools.py       # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)

# no default batch: a forgotten ICFG_START would silently rerun batch 0
[ -n "$ICFG_START" ] || { echo "set ICFG_START, e.g. ICFG_START=0 sbatch $SUBMIT"; exit 1; }

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=$SLURM_SUBMIT_DIR/runs/final_8gcd_flux_${SLURM_JOB_ID}
mkdir -p "$RUNDIR/flux_logs" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_final_8gcd_d_flux_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_final_8gcd_d_flux_${SLURM_JOB_ID}.err" "$RUNDIR"/
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
export SCRIPT STEP_TIME

# cfgs of this job: one per node from cfg_list[ICFG_START]; skip the ones whose gauge file is missing (the setup builds the name from gauge_file_prefix and cfg_list)
cd "$RUNDIR"
python3 -c "
from ${SETUP%.py} import cfg_list, gauge_file_prefix
for i in range($ICFG_START, min($ICFG_START + $SLURM_NNODES, len(cfg_list))): print(i, cfg_list[i], f'{gauge_file_prefix}.{cfg_list[i]}')" > gauge_files.txt || exit 1
RUN_LIST=""
while read ICFG CFG GAUGE; do
    if [ -f "$GAUGE" ]; then RUN_LIST="$RUN_LIST $ICFG:$CFG"; else echo "missing gauge file $GAUGE: icfg $ICFG skipped"; fi
done < gauge_files.txt
export RUN_LIST
echo "nodes $SLURM_NNODES, icfg from $ICFG_START, runs: $(echo $RUN_LIST | wc -w), step time limit: ${STEP_TIME:-none}"

# the script Flux runs on the first node: one Flux job per cfg (1 node, 8 ranks, 7 cores and the closest GCD per rank), then wait for all of them.
# Flux copies this environment into every job. Each rank takes its GCD from CUDA_VISIBLE_DEVICES, which gpu-affinity=per-task sets (Frontier guide:
# --gpus-per-task alone leaves all 8 GCDs visible). -t: Flux kills a run that passes STEP_TIME; the other runs go on.
cat > flux_submit.sh << 'EOF'
flux resource list
for PAIR in $RUN_LIST; do
    ICFG=${PAIR%%:*}; CFG=${PAIR##*:}
    flux submit -N 1 -n 8 -c 7 --gpus-per-task=1 -o gpu-affinity=per-task -o cpu-affinity=per-task ${STEP_TIME:+-t $STEP_TIME} \
        --job-name=cfg$CFG --output=flux_logs/icfg${ICFG}_cfg${CFG}.log \
        bash -c 'export ROCR_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; export TMPDIR=$RUN_TMPDIR
                 exec python3 -m mpi4py $SCRIPT --icfg '$ICFG' --n 1'
done
flux queue drain                          # returns when every run has finished, failed or timed out
flux jobs -a                              # one line per cfg: ST = CD finished, F failed, TO timed out, CA canceled
EOF

# one Flux broker per node (all 56 free cores and 8 GCDs of the node), started by a single srun; only this srun talks to the Slurm controller
srun -N $SLURM_NNODES -n $SLURM_NNODES -c 56 --gpus-per-node=8 env TMPDIR=/tmp flux start bash "$RUNDIR/flux_submit.sh"
STATUS=$?                                 # exit code of the Flux instance: 0 even when single runs failed, so check the flux jobs table above
date
exit $STATUS
