#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=208G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -J pt2_test
#SBATCH -o ./logs/pt2_test_%j.out
#SBATCH -e ./logs/pt2_test_%j.err

# TEST: does the direction version give the same 2pt as the production?
# Runs 2pt_coulomb_boosted_sepq_3072srcs_delta_final_streame.py of this folder with --boost_direction z on ONE cfg that the production has already
# finished with the current setup (solver tol 1e-8, _smear files): cfg_list[413] = cfg 2682, production task 23411671_413.
# The job is cut by the 1 h limit on purpose (TIMEOUT is the expected end): by then the first 2 t_src (128 sources each, about 21 min each)
# are saved as parts. compare_2pt_with_production_streame.py then compares these parts with the same t_src of the production .h5 files.
#
# The test writes to its own folder TEST_DIR, never to the production folder: only the COPY of the setup in the run folder gets another
# current_dir (the sed line below). Everything else is the production job script of this folder.
#     sbatch test_2pt_same_as_production.sh                          z, cfg index 413
#     ICFG=440 sbatch test_2pt_same_as_production.sh                 another finished cfg (cfg 204 + 6 * ICFG)
#     BOOST_DIRECTION=x sbatch test_2pt_same_as_production.sh        only to see x start and save; there is no production to compare it with
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_delta_final_streame.py
SETUP=sepq_2pt_setup_delta_final_streame.py
SUBMIT=test_2pt_same_as_production.sh             # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing_boost_direction.py      # in this folder: the copy of 2pt_production/tools/coulomb_smearing.py with the boost_direction argument
COMM=../../../../tools/pt2_comm_tools.py          # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools, four levels up from this folder)
ICFG=${ICFG:-413}                              # position in cfg_list of the cfg to test
BOOST_DIRECTION=${BOOST_DIRECTION:-z}          # lattice axis of the boost: z for the comparison with the production
TEST_DIR=/work/hdd/biqz/$USER/2pt_production/test_directionxyz      # output of the test (/work/hdd: /projects is too small)

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/projects/biqz/$USER/2pt_production/testruns/test_directionxyz_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_test_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_test_${SLURM_JOB_ID}.err" "$RUNDIR"/

# THE ONLY DIFFERENCE FROM THE PRODUCTION SETUP: the copy in the run folder writes to TEST_DIR
sed -i "s|^current_dir = .*|current_dir = \"$TEST_DIR\"|" "$RUNDIR/$SETUP"
grep -q "^current_dir = \"$TEST_DIR\"" "$RUNDIR/$SETUP" || { echo "current_dir was not replaced in $RUNDIR/$SETUP"; exit 1; }
echo "setup of this test compared with the setup of this folder:"; diff "$SETUP" "$RUNDIR/$SETUP"
date; hostname

# environment: conda env pyquda, QUDA_PATH, LD_LIBRARY_PATH (Cray PE defaults: PrgEnv-gnu, cray-mpich, cudatoolkit)
source ~/env_pyquda.sh
cd "$RUNDIR" || exit 1
export PYTHONPATH=$RUNDIR:$PYTHONPATH

# QUDA: the tuning cache of the production, so that the test runs the same tuned kernels
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # same string as resource_path in core.init
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"

# fail fast if the gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "test: cfg index $ICFG, gauge file $GAUGE, boost direction $BOOST_DIRECTION, output in $TEST_DIR"
python -c "from ${SETUP%.py} import delta_params; print('solver tol', delta_params.tol, ' grid', delta_params.grid_size)"
# versions, and the modules the script imports, without starting MPI outside srun (importing pyquda calls MPI_Init)
python -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'cupy-cuda13x', 'mpi4py', 'numpy', 'h5py', 'opt_einsum', 'tqdm']), sep=', ')" || exit 1
nvidia-smi -L

# cray-mpich has Slurm PMI: launch with srun. --gpu-bind=none: PyQUDA gives node-local rank i GPU i, so every rank must see all 4 GPUs.
# mask_cpu puts rank i on the 16 cores next to GPU i (Delta gpuA100x4: GPU0 cores 48-63, GPU1 32-47, GPU2 16-31, GPU3 0-15)
srun -n 4 --gpu-bind=none --cpu-bind=mask_cpu:0xffff000000000000,0xffff00000000,0xffff0000,0xffff \
    python -m mpi4py "$SCRIPT" --icfg $ICFG --n 1 --boost_direction $BOOST_DIRECTION     # -m mpi4py: an exception on any rank aborts all ranks
STATUS=$?
date
exit $STATUS
