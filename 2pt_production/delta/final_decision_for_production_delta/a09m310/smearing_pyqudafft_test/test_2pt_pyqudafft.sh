#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=208G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -J pt2_smeartest
#SBATCH -o ./logs/pt2_smeartest_%j.out
#SBATCH -e ./logs/pt2_smeartest_%j.err

# TEST: does the PyQUDA FFT smearing (2pt_production/tools/coulomb_smearing_pyqudafft.py) give the same 2pt as the production smearing?
# Runs the Delta direction-x production script (../delta_2pt_production_a09m310_stream_e_direction_x) with --boost_direction x on 5 cfgs,
# cfg_list[413] .. cfg_list[417] = cfgs 2682 .. 2706, with 24 sources per cfg (3 t_src x 2 x 2 x 2) instead of 3072, once with each smearing:
#     SMEARING=production sbatch test_2pt_pyqudafft.sh       the production smearing (the reference)
#     SMEARING=pyqudafft sbatch test_2pt_pyqudafft.sh        the PyQUDA FFT smearing
# compare_2pt_pyqudafft_with_production.py then compares the two outputs. Both run on the production grid [1, 1, 1, 4] with the production QUDA
# tuning cache, so the same kernels run in both: expected IDENTICAL. GRID=2,2,1,1 (x and y split, own tuning cache) only works with pyqudafft.
# Copy of ../smear_v1_directionxyz/test_2pt_same_as_production.sh. Only the COPIES in the run folder are changed, never the production files:
#   - pyqudafft: the main script copy imports coulomb_smearing_pyqudafft instead of coulomb_smearing_boost_direction (and gets the name ..._pyqudafft.py);
#   - the setup copy writes to TEST_DIR and has the 24-source grid (t_base, x_base, y_base, z_base); with GRID also another grid_size and tuning cache.
# Time: production about 13 s per source; PyQUDA's FFT goes through host memory and MPI, so a source takes longer (estimate ~1-2 min). 5 cfgs x 24
# sources should fit in the 6 h limit; a job stopped at the limit continues from the saved t_src when submitted again. Look at "UNTIL CONTRACTION".
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
PROD=../delta_2pt_production_a09m310_stream_e_direction_x      # the production folder: its files are only read and copied
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_delta_final_a09m310_streame.py
SETUP=sepq_2pt_setup_delta_final_a09m310_streame.py
SUBMIT=test_2pt_pyqudafft.sh                      # name of this file, used for its copy in the run folder
COMM=../../../../tools/pt2_comm_tools.py          # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools, four levels up from this folder)
SMEARING=${SMEARING:-pyqudafft}                # production or pyqudafft
if [ "$SMEARING" = production ]; then
    SMEAR=$PROD/coulomb_smearing_boost_direction.py                 # the production smearing
    TEST_SCRIPT=$SCRIPT                                             # the main script copy is not changed
elif [ "$SMEARING" = pyqudafft ]; then
    SMEAR=../../../../tools/coulomb_smearing_pyqudafft.py           # the PyQUDA FFT smearing (2pt_production/tools, four levels up from this folder)
    TEST_SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_delta_final_a09m310_streame_pyqudafft.py      # the changed copy in the run folder gets its own name
else
    echo "SMEARING=$SMEARING refused: production or pyqudafft"; exit 1
fi
ICFG=${ICFG:-413}                              # position in cfg_list of the first cfg (413 = cfg 2682); the job measures 5 cfgs from there
BOOST_DIRECTION=x                              # the x production code
GRID=${GRID:-1,1,1,4}                          # process grid x,y,z,t over the 4 GPUs
IFS=, read GX GY GZ GT <<< "$GRID"
[ $((GX * GY * GZ * GT)) -eq 4 ] || { echo "GRID=$GRID refused: the product must be 4 (one node, 4 GPUs)"; exit 1; }
[ "$SMEARING" = pyqudafft ] || [ "$GRID" = "1,1,1,4" ] || { echo "GRID=$GRID refused: the production smearing needs the grid [1, 1, 1, 4] here"; exit 1; }
TEST_DIR=/work/hdd/biqz/$USER/2pt_production/test_smearing_pyqudafft/grid${GX}${GY}${GZ}${GT}_${SMEARING}      # output of the test (/work/hdd: /projects is too small)

# fail fast if a file the script imports is missing
for f in "$PROD/$SCRIPT" "$PROD/$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/projects/biqz/$USER/2pt_production/testruns/test_smearing_grid${GX}${GY}${GZ}${GT}_${SMEARING}_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$PROD/$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$PROD/$SCRIPT" "$RUNDIR/$TEST_SCRIPT" || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_smeartest_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_smeartest_${SLURM_JOB_ID}.err" "$RUNDIR"/

# THE CHANGES FROM THE PRODUCTION, in the run-folder copies only
if [ "$SMEARING" = pyqudafft ]; then
    sed -i "s|^from coulomb_smearing_boost_direction import |from coulomb_smearing_pyqudafft import |" "$RUNDIR/$TEST_SCRIPT"
    grep -q "^from coulomb_smearing_pyqudafft import " "$RUNDIR/$TEST_SCRIPT" || { echo "the smearing import was not replaced in $RUNDIR/$TEST_SCRIPT"; exit 1; }
fi
sed -i "s|^current_dir = .*|current_dir = \"$TEST_DIR\"|" "$RUNDIR/$SETUP"
grep -q "^current_dir = \"$TEST_DIR\"" "$RUNDIR/$SETUP" || { echo "current_dir was not replaced in $RUNDIR/$SETUP"; exit 1; }
sed -i "s|^delta_params.t_base = np.arange(0, delta_params.GLt, 4) .*|delta_params.t_base = np.arange(0, delta_params.GLt, 32)     # TEST: 3 t_src|" "$RUNDIR/$SETUP"
sed -i "s|^delta_params.x_base = np.arange(0, delta_params.GLs, 8) .*|delta_params.x_base = np.arange(0, delta_params.GLs, 16)     # TEST: 2 x|" "$RUNDIR/$SETUP"
sed -i "s|^delta_params.y_base = np.arange(0, delta_params.GLs, 8) .*|delta_params.y_base = np.arange(0, delta_params.GLs, 16)     # TEST: 2 y|" "$RUNDIR/$SETUP"
sed -i "s|^delta_params.z_base = np.arange(0, delta_params.GLs, 4) .*|delta_params.z_base = np.arange(0, delta_params.GLs, 16)     # TEST: 2 z: 3 x 2 x 2 x 2 = 24 sources|" "$RUNDIR/$SETUP"
[ $(grep -c "^delta_params.[txyz]_base = .*# TEST: " "$RUNDIR/$SETUP") -eq 4 ] || { echo "the source grid was not replaced in $RUNDIR/$SETUP"; exit 1; }
if [ "$GRID" != "1,1,1,4" ]; then
    sed -i "s|^delta_params.grid_size = \[1, 1, 1, 4\]|delta_params.grid_size = [$GX, $GY, $GZ, $GT]|" "$RUNDIR/$SETUP"
    grep -q "^delta_params.grid_size = \[$GX, $GY, $GZ, $GT\]" "$RUNDIR/$SETUP" || { echo "grid_size was not replaced in $RUNDIR/$SETUP"; exit 1; }
    sed -i "s|^quda_resource_path = .*|quda_resource_path = \"/u/$USER/QUDA_resources/.cache/quda_test_pyqudafft_grid${GX}${GY}${GZ}${GT}\"|" "$RUNDIR/$SETUP"      # another grid tunes other kernels: keep the production cache unchanged
    grep -q "^quda_resource_path = \"/u/$USER/QUDA_resources/.cache/quda_test_pyqudafft_grid" "$RUNDIR/$SETUP" || { echo "quda_resource_path was not replaced in $RUNDIR/$SETUP"; exit 1; }
fi
echo "main script of this test compared with the production one:"; diff "$PROD/$SCRIPT" "$RUNDIR/$TEST_SCRIPT"
echo "setup of this test compared with the production one:"; diff "$PROD/$SETUP" "$RUNDIR/$SETUP"
date; hostname

# environment: conda env pyquda, QUDA_PATH, LD_LIBRARY_PATH (Cray PE defaults: PrgEnv-gnu, cray-mpich, cudatoolkit)
source ~/env_pyquda.sh
cd "$RUNDIR" || exit 1
export PYTHONPATH=$RUNDIR:$PYTHONPATH

# QUDA: on grid [1, 1, 1, 4] the tuning cache of the production, so that both smearings run with the same tuned kernels
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # same string as resource_path in core.init
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"

# fail fast if the gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "test: smearing $SMEARING, first cfg index $ICFG, gauge file $GAUGE, boost direction $BOOST_DIRECTION, grid $GRID, output in $TEST_DIR"
python -c "from ${SETUP%.py} import delta_params, quda_resource_path; print('solver tol', delta_params.tol, ' grid', delta_params.grid_size, ' multigrid', delta_params.multigrid, ' QUDA cache', quda_resource_path, ' sources per cfg', len(delta_params.t_base) * len(delta_params.x_base) * len(delta_params.y_base) * len(delta_params.z_base))"
# versions, and the modules the script imports, without starting MPI outside srun (importing pyquda calls MPI_Init)
python -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'cupy-cuda13x', 'mpi4py', 'numpy', 'h5py', 'opt_einsum', 'tqdm']), sep=', ')" || exit 1
# pyqudafft: fail fast if this PyQUDA has no distributed FFT or no field shift: read the files without importing them (no MPI outside srun)
if [ "$SMEARING" = pyqudafft ]; then
python -c "
import importlib.util, pathlib
fft_file = pathlib.Path(importlib.util.find_spec('pyquda_utils').origin).parent / 'fft.py'
field_file = pathlib.Path(importlib.util.find_spec('pyquda_comm').origin).parent / 'field.py'
fft_lines = [l for l in fft_file.read_text().splitlines() if l.startswith(('def fft(', 'def ifft('))] if fft_file.exists() else []
shift_lines = [l.strip() for l in field_file.read_text().splitlines() if l.strip().startswith(('def shift(', 'def location('))] if field_file.exists() else []
print(fft_file, fft_lines); print(field_file, shift_lines)
raise SystemExit(0 if len(fft_lines) == 2 and any('shift' in l for l in shift_lines) and any('location' in l for l in shift_lines) else 'this PyQUDA has no pyquda_utils.fft fft/ifft, no field shift or no field location')
" || exit 1
fi
nvidia-smi -L

# cray-mpich has Slurm PMI: launch with srun. --gpu-bind=none: PyQUDA gives node-local rank i GPU i, so every rank must see all 4 GPUs.
# mask_cpu puts rank i on the 16 cores next to GPU i (Delta gpuA100x4: GPU0 cores 48-63, GPU1 32-47, GPU2 16-31, GPU3 0-15)
srun -n 4 --gpu-bind=none --cpu-bind=mask_cpu:0xffff000000000000,0xffff00000000,0xffff0000,0xffff \
    python -m mpi4py "$TEST_SCRIPT" --icfg $ICFG --n 5 --boost_direction $BOOST_DIRECTION     # -m mpi4py: an exception on any rank aborts all ranks
STATUS=$?
date
exit $STATUS
