#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p extended
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -J pt2_final_1gcd
#SBATCH -o ./logs/pt2_final_1gcd_%j.out
#SBATCH -e ./logs/pt2_final_1gcd_%j.err

# Final run on Frontier, 1-GCD version: 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_1gcd.py with sepq_2pt_setup_frontier_final_1gcd.py, the
# physics of the Delta final run. NRUNS runs side by side, each on one GCD with its own srun (on Delta, one srun per task); run r measures NCFG cfgs
# from cfg_list[ICFG + r * NCFG] (--icfg, --n). At most 8 runs per node.
# A run killed at the time limit can be resubmitted with the same ICFG, NRUNS and NCFG: the per-t_src parts already on disk are skipped.
# Frontier limits: extended partition up to 64 nodes and 24 h (1 running job); batch 1-91 nodes 2 h, 92-183 nodes 6 h, 184+ nodes 12 h.
# OLCF advises against more than about 100 sruns in one job.
# Usage, from this folder (one cfg, 204):   ICFG=0 NRUNS=1 MRHS=2 sbatch 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_1gcd.sh
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
ICFG=${ICFG:-0}                 # index in cfg_list of the first cfg: 0 -> cfg 204
NRUNS=${NRUNS:-1}               # runs side by side, one GCD each
NCFG=${NCFG:-1}                 # cfgs per run
export PT2_MRHS=${MRHS:-2}      # read by the setup; take it from the timing test
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_1gcd.py
SETUP=sepq_2pt_setup_frontier_final_1gcd.py
SMEAR=coulomb_smearing.py
COMM=pt2_comm_tools.py
SUBMIT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_1gcd.sh        # name of this file, used for its copy in the run folder

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, one log per run in logs/
RUNDIR=$SLURM_SUBMIT_DIR/runs/final_1gcd_${SLURM_JOB_ID}
mkdir -p "$RUNDIR/logs" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
date; hostname; echo "ICFG $ICFG, NRUNS $NRUNS, NCFG $NCFG, MRHS $PT2_MRHS"

# environment
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_P2P=0                  # must stay 0 on Frontier: ROCm IPC gives silent halo errors
export QUDA_ENABLE_MPS=1                  # one GCD visible per rank: PyQUDA then uses device 0 on every rank
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0   # as on Delta
export PYTHONPATH=$RUNDIR:$PYTHONPATH:/ccs/home/sicheng/packages      # the run folder first, so python imports the copies
export CUPY_CACHE_DIR=/lustre/orion/lgt132/scratch/sicheng/cupy_cache/2pt_final
export TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}
mkdir -p $CUPY_CACHE_DIR $TMPDIR
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer for Cray MPICH GPU support; harmless with GDR off
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES                          # the GCD of each rank comes from Slurm only (ROCR_VISIBLE_DEVICES)
cd "$RUNDIR"
export QUDA_RESOURCE_PATH=$(python3 -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # QUDA's tuning cache: same string as resource_path in core.init
mkdir -p "$QUDA_RESOURCE_PATH"                                                                                    # as on Delta: QUDA saves no tuning into a missing folder

# fail fast if the first gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python3 -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "first gauge file $GAUGE"

# one srun per run, each on 7 cores and the GCD closest to them; -m mpi4py: an exception aborts that run only
PIDS=()
for ((r = 0; r < NRUNS; r++)); do
    srun -N 1 -n 1 -c 7 --exact --gpus-per-task=1 --gpu-bind=closest -o "logs/run$r.out" -e "logs/run$r.err" \
        python3 -u -m mpi4py "$SCRIPT" --icfg $((ICFG + r * NCFG)) --n $NCFG &
    PIDS+=($!)
    sleep 2                               # gives Slurm time to place this run before the next (OLCF guide)
done
STATUS=0
for pid in "${PIDS[@]}"; do
    wait $pid || STATUS=1                 # 1 if any run failed, so that sacct shows FAILED
done
date
exit $STATUS
