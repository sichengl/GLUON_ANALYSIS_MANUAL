#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=208G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -J pt2_sepq_test
#SBATCH -o ./logs/pt2_sepq_test_%j.out
#SBATCH -e ./logs/pt2_sepq_test_%j.err

# 2pt test (pion + proton), sepq kinematics, Coulomb-gauge boosted smearing, 4 A100-40GB GPUs on one Delta gpuA100x4 node,
# grid [1,1,1,4] in the python script. One configuration, the sources of sepq_2pt_setup_delta.py.
# Every job runs in its own folder testruns/sepq_test_<jobid>, on copies of the python files; the folder also keeps this sbatch script.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_1024srcs_delta_largetest.py
SETUP=sepq_2pt_setup_delta.py
SUBMIT=2pt_coulomb_boosted_sepq_1024srcs_delta_largetest.sh              # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing.py
[ -f "$SMEAR" ] || SMEAR=../tools/coulomb_smearing.py                    # repo layout: 2pt_production/tools next to 2pt_production/delta

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/projects/biqz/$USER/2pt_production/testruns/sepq_test_${SLURM_JOB_ID}
mkdir -p "$RUNDIR"
cp "$SCRIPT" "$SETUP" "$SMEAR" "$RUNDIR"/
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_sepq_test_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_sepq_test_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment: conda env pyquda, QUDA_PATH, LD_LIBRARY_PATH (Cray PE defaults: PrgEnv-gnu, cray-mpich, cudatoolkit)
source ~/env_pyquda.sh
cd "$RUNDIR"
export PYTHONPATH=$RUNDIR:$PYTHONPATH

# QUDA
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python -c "from sepq_2pt_setup_delta import quda_resource_path; print(quda_resource_path)")   # same string as resource_path in core.init
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"

# run parameters: one configuration (override with: ICFG=5 sbatch 2pt_coulomb_boosted_sepq_1024srcs_delta_largetest.sh)
ICFG=${ICFG:-0}                     # ICFG=5 is cfg 234: its q phases are not all 1, unlike cfg 204
python -c "import pyquda, cupy, mpi4py; print('pyquda', pyquda.__version__, 'cupy', cupy.__version__, 'mpi4py', mpi4py.__version__)"
nvidia-smi -L
nvidia-smi topo -m

# GPU memory of all 4 GPUs every 30 s, to see the peak (QUDA and cupy together) on the 40 GB A100s
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -l 30 > "$RUNDIR/gpu_mem_${SLURM_JOB_ID}.csv" &
MONITOR=$!

# cray-mpich has Slurm PMI: launch with srun. --gpu-bind=none: PyQUDA gives node-local rank i GPU i, so every rank must see all 4 GPUs.
# mask_cpu puts rank i on the 16 cores next to GPU i (Delta gpuA100x4: GPU0 cores 48-63, GPU1 32-47, GPU2 16-31, GPU3 0-15; check with nvidia-smi topo -m above)
srun -n 4 --gpu-bind=none --cpu-bind=mask_cpu:0xffff000000000000,0xffff00000000,0xffff0000,0xffff \
    python -m mpi4py "$SCRIPT" --icfg $ICFG --n 1     # -m mpi4py: an exception on any rank aborts all ranks
kill $MONITOR
date
