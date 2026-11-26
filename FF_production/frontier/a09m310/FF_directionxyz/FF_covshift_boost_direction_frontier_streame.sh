#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 20
#SBATCH -J FF_boost_flux
#SBATCH -o ./logs/FF_boost_flux_%j.out
#SBATCH -e ./logs/FF_boost_flux_%j.err

# FF (FF_covshift_boost_direction_frontier_streame.py) for the 800 stream-e cfgs (204, 210, ..., 4998) on Frontier: one GCD (64 GB) per run, 8 runs per node.
# The Frontier copy of FF_production/delta/FF_covshift_boost_direction_delta.sh. On Delta every array task is one run on one A100; Frontier gives whole
# nodes only, so here one Slurm job holds all the runs and Flux puts them on the GCDs (one Flux job per run), as in the stream-d 2pt job
# 2pt_production/frontier/final_production_frontier/8gpu/frontier_stream_d/2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd_flux.sh
# (the Frontier guide does not recommend more than ~100 simultaneous sruns and recommends Flux for many small runs).
# The smearing and the direction come from two variables, both required:
#     SMEAR            hyp or flow      (hyp: 16 levels, lines 0..15;  flow: 41 levels of 10 steps of 0.01, lines 0..9)
#     BOOST_DIRECTION  x, y or z        (lattice axis of the Wilson line)
# Run k does NCFG consecutive cfgs, starting at cfg 204 + 6 * NCFG * k (run k = array task k of the Delta script with the same NCFG).
# The job does the runs FIRST_RUN .. LAST_RUN, by default all of them (0 .. 799 / NCFG). Flux starts one run per GCD and the next one when a GCD is
# free, so the runs need not fit on the nodes at once. With NCFG = 5 (the default): 160 runs, all at once on -N 20 (the default).
#
# First test, one cfg (204):  NCFG=1 FIRST_RUN=0 LAST_RUN=0 SMEAR=hyp BOOST_DIRECTION=z sbatch -N 1 FF_covshift_boost_direction_frontier_streame.sh
# then compare with the old FF (compare_FF_with_old_frontier_streame.py), read the time per smearing level in the run's log
# (runs/FF_flux_<jobid>/flux_logs/run0_cfg204.log, "SHIFT #204: ... secs"), choose NCFG, -N and -t, and submit the rest:
#                             SMEAR=hyp BOOST_DIRECTION=x sbatch FF_covshift_boost_direction_frontier_streame.sh
# Time limits: batch partition 2 h for 1-91 nodes, 6 h for 92-183, 12 h for 184+; extended partition (-p extended) 1-64 nodes, 24 h.
# NCFG = 5 in 2 h is a guess (Delta A100: about 25 s per HYP level, 16 levels, so about 7 min per cfg; a GCD is not measured).
# A cfg is saved when it is done, so a run killed at the time limit loses only the cfg it is on.
# A cfg is NOT skipped if its file exists: a run that is submitted again computes all its cfgs again.
# Output: one file per cfg, FF_opp_symmetric_asymmetric_<hyp0-15_w0-15 | flow0-40_w0-9>_boost<x|y|z>_cfg<cfg>.h5, in out_dir of the python script.
# About 0.45 GB per cfg for hyp and 0.73 GB for flow: 360 GB and 580 GB for 800 cfgs of one direction.
# Every job runs in its own folder runs/FF_flux_<jobid>, on a copy of the python script; the folder also keeps this sbatch script, the Flux submit
# script and one log per run (flux_logs/). Submit from the folder that holds the python script.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=FF_covshift_boost_direction_frontier_streame.py
SUBMIT=FF_covshift_boost_direction_frontier_streame.sh                # name of this file, used for its copy in the run folder
NCFG=${NCFG:-5}
FIRST_RUN=${FIRST_RUN:-0}
LAST_RUN=${LAST_RUN:-$((799 / NCFG))}
[ -f "$SCRIPT" ] || { echo "missing $SCRIPT in $SLURM_SUBMIT_DIR"; exit 1; }
if [ -z "$SMEAR" ] || [ -z "$BOOST_DIRECTION" ]; then
    echo "set SMEAR (hyp or flow) and BOOST_DIRECTION (x, y or z):  SMEAR=hyp BOOST_DIRECTION=x sbatch $SUBMIT"
    exit 1
fi

# run folder: a copy of exactly what runs, this sbatch script, links to the two log files
RUNDIR=$SLURM_SUBMIT_DIR/runs/FF_flux_${SLURM_JOB_ID}
mkdir -p "$RUNDIR/flux_logs" || exit 1
cp "$SCRIPT" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/FF_boost_flux_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/FF_boost_flux_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment (Sergey's PyQUDA build), then Flux and the GPU-aware hwloc it needs to see the GPUs (as in the stream-d 2pt Flux job)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh
module load hwloc/2.9.1-gpu flux
command -v flux > /dev/null || { echo "flux not found after module load"; exit 1; }
flux --version

# QUDA: the tuning cache is resource_path in the python script (core.init)
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_P2P=0                  # must stay 0 on Frontier: ROCm IPC gives silent halo errors
export QUDA_ENABLE_MPS=1                  # one GCD visible per run: PyQUDA then uses device 0
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0   # QUDA allocates with hipMalloc and cannot use the blocks cupy keeps cached
export CUPY_CACHE_DIR=/lustre/orion/lgt132/scratch/sicheng/cupy_cache/${SLURM_JOB_ID}
export RUN_TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}     # TMPDIR of the python runs (set in each run below); Flux itself keeps the node-local /tmp
mkdir -p $CUPY_CACHE_DIR $RUN_TMPDIR
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer for Cray MPICH GPU support; harmless with GDR off (GDR hangs intermittently on this build, keep it off)
export SCRIPT SMEAR BOOST_DIRECTION

# runs of this job: run k does N = NCFG cfgs from CFG = 204 + 6 * NCFG * k (fewer at the end of the stream; a run without cfgs is left out)
RUN_LIST=""
for K in $(seq $FIRST_RUN $LAST_RUN); do
    CFG=$((204 + 6 * NCFG * K))
    [ $CFG -le 4998 ] || continue
    N=$(( (4998 - CFG) / 6 + 1 ))
    [ $N -le $NCFG ] || N=$NCFG
    RUN_LIST="$RUN_LIST $K:$CFG:$N"
done
export RUN_LIST
echo "smear $SMEAR, boost direction $BOOST_DIRECTION, NCFG $NCFG, nodes $SLURM_NNODES ($((8 * SLURM_NNODES)) GCDs), runs $FIRST_RUN .. $LAST_RUN: $(echo $RUN_LIST | wc -w) runs"
[ -n "$RUN_LIST" ] || { echo "no run has cfgs"; exit 1; }

# the script Flux runs on the first node: one Flux job per run (1 rank, 7 cores and one GCD), then wait for all of them.
# Flux copies this environment into every job. The run takes its GCD from CUDA_VISIBLE_DEVICES, which gpu-affinity=per-task sets.
cd "$RUNDIR"
cat > flux_submit.sh << 'EOF'
flux resource list
for RUN in $RUN_LIST; do
    K=$(echo $RUN | cut -d: -f1); CFG=$(echo $RUN | cut -d: -f2); N=$(echo $RUN | cut -d: -f3)
    flux submit -n 1 -c 7 --gpus-per-task=1 -o gpu-affinity=per-task -o cpu-affinity=per-task \
        --job-name=run${K} --output=flux_logs/run${K}_cfg${CFG}.log \
        bash -c 'export ROCR_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; export TMPDIR=$RUN_TMPDIR
                 echo "run '$K': cfgs '$CFG' .. $(('$CFG' + 6 * ('$N' - 1))) on $(hostname), GCD $ROCR_VISIBLE_DEVICES"
                 exec python3 $SCRIPT --config "{\"cfg_n\": '$CFG', \"num_cfg\": '$N'}" --smear $SMEAR --boost_direction $BOOST_DIRECTION'
done
flux queue drain                          # returns when every run has finished or failed
flux jobs -a                              # one line per run: ST = CD finished, F failed, CA canceled
EOF

# one Flux broker per node (all 56 free cores and 8 GCDs of the node), started by a single srun; only this srun talks to the Slurm controller
srun -N $SLURM_NNODES -n $SLURM_NNODES -c 56 --gpus-per-node=8 env TMPDIR=/tmp flux start bash "$RUNDIR/flux_submit.sh"
STATUS=$?                                 # exit code of the Flux instance: 0 even when single runs failed, so check the flux jobs table above
date
exit $STATUS
