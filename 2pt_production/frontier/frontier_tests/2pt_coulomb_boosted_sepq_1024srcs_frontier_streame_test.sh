#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J pt2_streame_test
#SBATCH -o ./logs/pt2_streame_test_%j.out
#SBATCH -e ./logs/pt2_streame_test_%j.err

# Stream-e test on Frontier: one cfg of the 8t x 4x x 4y x 8z = 1024 sources of the old stream-e data, split between two independent runs at the same time
# on one node. Each run uses 4 of the 8 GCDs with grid [1,1,1,4], as one Delta job on 4 A100s (a GCD has 64 GB, a Delta A100 40 GB), and measures half of
# the t_src: run 0 the t_src 0-3, run 1 the t_src 4-7 (--part 0/1 --nparts 2). Both read and gauge-fix the same cfg; the run that saves the last t_src
# merges the parts of both into the .h5 files. Default icfg 0 (cfg 204, also measured on Delta); another cfg: ICFG=3 sbatch <this file> (0 -> cfg 204, 9 -> cfg 258).
# A 2-hour batch job is shorter than one t_src (128 sources), so it saves nothing: it checks that both runs start, run side by side and fit in memory,
# and how long one source takes. For the full cfg: sbatch -p extended -t 24:00:00 <this file>, and submit it again after a time-limit kill: the t_src
# already saved are skipped. The extended partition (1-64 nodes, 24 hours) allows 1 running job per user.
# Every job runs in its own folder testruns/streame_test_<jobid>, on copies of the python files; the folder also keeps this sbatch script.
# Each run writes its own log, logs/pt2_streame_test_<jobid>_icfg<i>_part<p>.out/.err; the job's own .out/.err only has the setup and the exit codes.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_1024srcs_frontier_streame_test.py
SETUP=sepq_2pt_setup_frontier_classused_streame_test.py
SUBMIT=2pt_coulomb_boosted_sepq_1024srcs_frontier_streame_test.sh               # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing.py
[ -f "$SMEAR" ] || SMEAR=../tools/coulomb_smearing.py                    # repo layout: 2pt_production/tools next to 2pt_production/<cluster>
COMM=pt2_comm_tools.py
[ -f "$COMM" ] || COMM=../tools/pt2_comm_tools.py                        # the shared TwoPtParams class, imported by the setup

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the log files
RUNDIR=$SLURM_SUBMIT_DIR/testruns/streame_test_${SLURM_JOB_ID}
mkdir -p "$RUNDIR"
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_streame_test_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_streame_test_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment (Sergey's PyQUDA build)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh

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
export TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}
mkdir -p $TUNE_DIR $CUPY_CACHE_DIR $TMPDIR
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer for Cray MPICH GPU support; harmless with GDR off (GDR hangs intermittently on this build, keep it off)

# run parameters: one cfg, ICFG is its index in cfg_list of the setup; NPARTS runs share its t_src
ICFG=${ICFG:-0}
NPARTS=2
cd "$RUNDIR"
# fail fast if the gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python3 -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "icfg $ICFG: gauge file $GAUGE"

# Two 4-rank runs side by side, as in the OLCF guide for simultaneous job steps on one node: each rank gets 7 cores (one L3 region) and the GCD
# closest to them, so each run has 4 GCDs and 28 cores of its own and the two runs never share a GCD (Slurm sets ROCR_VISIBLE_DEVICES per rank).
# verbose: Slurm writes each rank's GCD and cores into that run's .err log, to check that the two runs do not overlap.
# -m mpi4py: an exception on any rank aborts the 4 ranks of that run; the other run keeps going (and leaves the merge to a resubmission)
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES                          # the GCD of each rank comes from Slurm only (ROCR_VISIBLE_DEVICES)
PIDS=()
for ((PART = 0; PART < NPARTS; PART++)); do
    LOG=$SLURM_SUBMIT_DIR/logs/pt2_streame_test_${SLURM_JOB_ID}_icfg${ICFG}_part${PART}
    srun -u -N 1 -n 4 -c 7 --exact --gpus-per-task=1 --gpu-bind=verbose,closest --cpu-bind=verbose --kill-on-bad-exit=1 -o "$LOG.out" -e "$LOG.err" \
        python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n 1 --part $PART --nparts $NPARTS &
    PIDS+=($!)
    ln -s "$LOG.out" "$LOG.err" "$RUNDIR"/
    sleep 1                               # gives Slurm time to place the first run before the second (OLCF guide)
done
rc=0
for PART in "${!PIDS[@]}"; do
    wait ${PIDS[$PART]} || { echo "icfg $ICFG part $PART failed"; rc=1; }
done
date
exit $rc                                  # so that sacct shows FAILED when a python run fails
