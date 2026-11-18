#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J pt2_xyz_e
#SBATCH -o ./logs/pt2_xyz_e_%j.out
#SBATCH -e ./logs/pt2_xyz_e_%j.err

# TEST: the direction version on Frontier in x, y and z, on the STREAM-E cfgs that Delta and Fermilab also run (cfg_list index 413-422 = cfg 2682-2736),
# so that every number can be compared cfg by cfg with Delta (z: its production; x, y: its test runs) and with Fermilab.
# Runs 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.py of this folder on ONE cfg with the boost direction BOOST_DIRECTION.
# The job is cut by the 2 h limit on purpose (TIMEOUT is the expected end; batch jobs of 1-91 nodes get at most 2 h): by then the first 2-3 t_src
# (128 sources each, about 38 min each at 18 s per source) are saved as parts.
#
# This is test_2pt_same_as_production.sh of this folder with two changes, both only in the COPY of the setup in the run folder (the two sed lines below):
#   current_dir        -> TEST_DIR, a folder of its own: the file names do not say the stream, so stream-e and stream-d tests must not share a folder
#   gauge_file_prefix  -> the stream-e gauge files on Orion (the setup of this folder reads stream d)
# Everything else (cfg_list, sources, smearings, tol 1e-8) is the setup of this folder, which equals Delta's setup apart from paths and grid.
#     ICFG=413 BOOST_DIRECTION=x sbatch test_2pt_xyz_streame.sh        cfg 2682, x
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.py
SETUP=sepq_2pt_setup_frontier_final_8gcd.py
SUBMIT=test_2pt_xyz_streame.sh                 # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing_boost_direction.py      # in this folder: the copy of 2pt_production/tools/coulomb_smearing.py with the boost_direction argument
COMM=../../../../tools/pt2_comm_tools.py       # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)
ICFG=${ICFG:-413}                              # position in cfg_list of the cfg to test
BOOST_DIRECTION=${BOOST_DIRECTION:-z}          # lattice axis of the boost: x, y or z
TEST_DIR=/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/2pt_production/test_directionxyz_streame      # output of the test
GAUGE_E=/lustre/orion/lgt132/world-shared/DATA/MILC/a09m310/gauge/l3296f211b630m0074m037m440e                     # stream e (gauge_prefix of the Frontier FF script)

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=$SLURM_SUBMIT_DIR/runs/test_xyz_streame_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_xyz_e_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_xyz_e_${SLURM_JOB_ID}.err" "$RUNDIR"/

# THE ONLY DIFFERENCES FROM THE PRODUCTION SETUP: the copy in the run folder writes to TEST_DIR and reads the stream-e gauge files
sed -i "s|^current_dir = .*|current_dir = \"$TEST_DIR\"|" "$RUNDIR/$SETUP"
sed -i "s|^gauge_file_prefix = .*|gauge_file_prefix = \"$GAUGE_E\"|" "$RUNDIR/$SETUP"
grep -q "^current_dir = \"$TEST_DIR\"$" "$RUNDIR/$SETUP" || { echo "current_dir was not replaced in $RUNDIR/$SETUP"; exit 1; }
grep -q "^gauge_file_prefix = \"$GAUGE_E\"$" "$RUNDIR/$SETUP" || { echo "gauge_file_prefix was not replaced in $RUNDIR/$SETUP"; exit 1; }
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
# fail fast if the gauge file is not there (the run-folder setup builds the name from gauge_file_prefix and cfg_list)
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
