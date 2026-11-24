#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J pt2_test
#SBATCH -o ./logs/pt2_test_%j.out
#SBATCH -e ./logs/pt2_test_%j.err

# TEST: does the direction version give the same 2pt as the stream-d production on Frontier?
# Runs 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd_streamd.py of this folder with --boost_direction z on ONE cfg that the stream-d production has
# already finished with the current setup (solver tol 1e-8, _smear files): cfg_list[2] = cfg 216 of batch 1 (Flux jobs 5845021 and 5845023).
# The job is cut by the 2 h limit on purpose (TIMEOUT is the expected end): by then the first 2-3 t_src (128 sources each, about 38 min each at
# 18 s per source) are saved as parts. compare_2pt_with_production_streamd.py then compares these parts with the same t_src of the production .h5 files.
#
# The test writes to its own folder TEST_DIR, never to the production folder: only the COPY of the setup in the run folder gets another
# current_dir (the sed line below). Everything else is the one-node job script 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.sh of this folder.
#     sbatch test_2pt_same_as_production.sh                          z, cfg index 2
#     ICFG=10 sbatch test_2pt_same_as_production.sh                  another finished cfg (cfg 204 + 6 * ICFG)
#     BOOST_DIRECTION=x sbatch test_2pt_same_as_production.sh        only to see x start and save; there is no production to compare it with
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd_streamd.py
SETUP=sepq_2pt_setup_frontier_final_8gcd_streamd.py
SUBMIT=test_2pt_same_as_production.sh             # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing_boost_direction.py      # in this folder: the copy of 2pt_production/tools/coulomb_smearing.py with the boost_direction argument
COMM=../../../../tools/pt2_comm_tools.py       # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)
ICFG=${ICFG:-2}                                # position in cfg_list of the cfg to test
BOOST_DIRECTION=${BOOST_DIRECTION:-z}          # lattice axis of the boost: z for the comparison with the production
TEST_DIR=/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/2pt_production/test_directionxyz      # output of the test

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=$SLURM_SUBMIT_DIR/runs/test_directionxyz_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_test_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_test_${SLURM_JOB_ID}.err" "$RUNDIR"/

# THE ONLY DIFFERENCE FROM THE PRODUCTION SETUP: the copy in the run folder writes to TEST_DIR
sed -i "s|^current_dir = .*|current_dir = \"$TEST_DIR\"|" "$RUNDIR/$SETUP"
grep -q "^current_dir = \"$TEST_DIR\"" "$RUNDIR/$SETUP" || { echo "current_dir was not replaced in $RUNDIR/$SETUP"; exit 1; }
echo "setup of this test compared with the setup of this folder:"; diff "$SETUP" "$RUNDIR/$SETUP"
date; hostname

# environment (Sergey's PyQUDA build)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh

# QUDA: the tuning cache of the production (quda_resource_path of the setup), so that the test runs the same tuned kernels
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

cd "$RUNDIR" || exit 1
# fail fast if the gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python3 -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "test: cfg index $ICFG, gauge file $GAUGE, boost direction $BOOST_DIRECTION, output in $TEST_DIR"
python3 -c "from ${SETUP%.py} import delta_params; print('solver tol', delta_params.tol, ' grid', delta_params.grid_size)"

# GPU memory of the 8 GCDs every 2 s, to see the peak (QUDA and cupy together), as on Delta with nvidia-smi
( while true; do rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/^card/ {print $1 "," $NF}'; sleep 2; done ) > "$RUNDIR/gpu_mem.csv" &
MONITOR=$!

# one run of 8 ranks for grid [1,1,1,8]: each rank gets 7 cores (one L3 region) and the GCD closest to them (Slurm sets ROCR_VISIBLE_DEVICES per rank).
# verbose: Slurm writes each rank's GCD and cores into the job's .err log. -m mpi4py: an exception on any rank aborts all ranks
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES                          # the GCD of each rank comes from Slurm only (ROCR_VISIBLE_DEVICES)
srun -u -N 1 -n 8 -c 7 --gpus-per-task=1 --gpu-bind=verbose,closest --cpu-bind=verbose --kill-on-bad-exit=1 \
    python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n 1 --boost_direction $BOOST_DIRECTION
STATUS=$?                                 # exit code of the run, so that sacct shows FAILED instead of COMPLETED when python fails
kill $MONITOR
date
exit $STATUS
