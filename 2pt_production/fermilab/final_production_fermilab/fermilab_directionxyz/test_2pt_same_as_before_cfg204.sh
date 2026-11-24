#!/bin/bash
#SBATCH -A gluonp0.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --qos=normal
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH -J pt2_test
#SBATCH -o ./logs/pt2_test_%j.out
#SBATCH -e ./logs/pt2_test_%j.err

# TEST: does the code of this folder give the same 2pt as the earlier Fermilab run of cfg 204?
# The earlier run: job 1349355_0 (2026-10-21), the code of 2026-10-20 (git b25384c), solver tol 1e-10, files in
# /lustre2/gluonp0/sliu1/2pt_production/final_production_3072src_200cfg/coulomb_rho4p5z2p25-rho1p5_frac0p8-0_paired_GEVP_sepq_3072src_phyp_fb_tsep20/cfg204
# (the old layout: shape, shape, frac, frac, ...). This job runs 2pt_coulomb_boosted_sepq_3072srcs_fermilab_final_streame.py of this folder with
# --boost_direction z on cfg_list[0] = cfg 204, at the tol of the earlier run.
# The job is cut by the 1 h limit on purpose (TIMEOUT is the expected end): by then the first 2 t_src (128 sources each, about 21 min each)
# are saved as parts. compare_2pt_with_old_run_streame.py then compares these parts with the same t_src of the earlier .h5 files.
#
# The test writes to its own folder TEST_DIR, never to a production folder: only the COPY of the setup in the run folder gets another
# current_dir and the tol of the earlier run (the two sed lines below). Everything else is the job script of this folder.
#     sbatch test_2pt_same_as_before_cfg204.sh                       z, cfg 204
#     sbatch --export=ALL,ICFG=413,BOOST_DIRECTION=x,TEST_TOL=1e-8 --time=14:00:00 test_2pt_same_as_before_cfg204.sh
#                                                                    10-cfg x/y/z test: one whole cfg (here 2682, x) at the production tol
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_fermilab_final_streame.py
SETUP=sepq_2pt_setup_fermilab_final_streame.py
SUBMIT=test_2pt_same_as_before_cfg204.sh       # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing_boost_direction.py      # in this folder: the copy of 2pt_production/tools/coulomb_smearing.py with the boost_direction argument
COMM=../../../tools/pt2_comm_tools.py          # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools, three levels up from this folder)
ICFG=${ICFG:-0}                                # position in cfg_list of the cfg to test: 0 = cfg 204
BOOST_DIRECTION=${BOOST_DIRECTION:-z}          # lattice axis of the boost: z for the comparison with the earlier run
TEST_DIR=/lustre2/gluonp0/sliu1/2pt_production/test_directionxyz      # output of the test
TEST_TOL=${TEST_TOL:-1e-10}                    # solver tol: 1e-10 = the earlier run of cfg 204 (its setup copy: testruns/final_1349355_0/sepq_2pt_setup_fermilab_final.py); 1e-8 = production

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/project/gluonp0/sicheng/gluon_production/2pt_production/testruns/test_directionxyz_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_test_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_test_${SLURM_JOB_ID}.err" "$RUNDIR"/

# THE ONLY DIFFERENCES FROM THE SETUP OF THIS FOLDER: the copy in the run folder writes to TEST_DIR and solves at the tol of the earlier run
sed -i "s|^current_dir = .*|current_dir = \"$TEST_DIR\"|" "$RUNDIR/$SETUP"
sed -i "s|^delta_params.tol = .*|delta_params.tol = $TEST_TOL|" "$RUNDIR/$SETUP"
grep -q "^current_dir = \"$TEST_DIR\"" "$RUNDIR/$SETUP" || { echo "current_dir was not replaced in $RUNDIR/$SETUP"; exit 1; }
grep -q "^delta_params.tol = $TEST_TOL$" "$RUNDIR/$SETUP" || { echo "tol was not replaced in $RUNDIR/$SETUP"; exit 1; }
echo "setup of this test compared with the setup of this folder:"; diff "$SETUP" "$RUNDIR/$SETUP"
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

# QUDA: the tuning cache of the earlier Fermilab runs, so that the test runs the same tuned kernels
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python3 -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # same string as resource_path in core.init
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"

# fail fast if the gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python3 -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "test: cfg index $ICFG, gauge file $GAUGE, boost direction $BOOST_DIRECTION, output in $TEST_DIR"
python3 -c "from ${SETUP%.py} import delta_params; print('solver tol', delta_params.tol, ' grid', delta_params.grid_size)"
# versions of the modules the script imports, without starting MPI outside mpirun (importing pyquda calls MPI_Init)
python3 -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'mpi4py', 'numpy', 'h5py', 'opt_einsum', 'tqdm']), sep=', '); import cupy; print('cupy', cupy.__version__, 'CUDA runtime', cupy.cuda.runtime.runtimeGetVersion())" || exit 1
nvidia-smi -L

# this OpenMPI has no Slurm PMI support: launch with mpirun, not srun. PyQUDA gives node-local rank i GPU i.
mpirun -np 4 python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n 1 --boost_direction $BOOST_DIRECTION       # -m mpi4py: an exception on any rank aborts all ranks
STATUS=$?
date
exit $STATUS
