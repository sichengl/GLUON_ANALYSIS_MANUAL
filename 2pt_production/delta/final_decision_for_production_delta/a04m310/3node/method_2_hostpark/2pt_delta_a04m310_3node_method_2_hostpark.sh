#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --time=02:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -J pt2_a04_3n_m2
#SBATCH -o ./logs/pt2_a04m310_3node_method_2_hostpark_%j.out
#SBATCH -e ./logs/pt2_a04m310_3node_method_2_hostpark_%j.err

# NODE TEST: the Delta direction-x production 2pt for a04m310 (l64192f211b700m00316m0158m188, 64^3 x 192) on 3 gpuA100x8 node(s),
# 8 ranks per node, grid [1, 1, 1, 24] (8 local time slices). Method method_2_hostpark: hostpark code, mrhs 1, nt_chunk 1.
# Copy of a09m310/delta_2pt_production_a09m310_stream_e_direction_x/2pt_coulomb_boosted_sepq_3072srcs_delta_final.sh (md5 678ebda9): one job
# measures five cfgs (--n 5), cfg_list[ICFG] .. cfg_list[ICFG + 4] of the setup (default ICFG=0: cfgs 1002 .. 1026), 24 sources each,
# with no array; a job stopped at the time limit continues from the saved t_src when submitted again. The output folder (current_dir of the setup) is
# separate for every node count and method, so the results of the methods can be compared with each other.
# Test:  BOOST_DIRECTION=x sbatch 2pt_delta_a04m310_3node_method_2_hostpark.sh        another cfg: ICFG=3 BOOST_DIRECTION=x sbatch 2pt_delta_a04m310_3node_method_2_hostpark.sh
# Memory: every inversion logs "AFTER INVERT: GPU used ..." (rank 0), and gpu_mem_<jobid>.csv in the run folder has the 8 GPUs of the
# first node every 2 s; QUDA prints its peak ("Device memory used") at the end of the .out log.
# Every job runs in its own folder testruns/<name>_<jobid>, on copies of the python files; the folder also keeps this sbatch script.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_delta_a04m310_hostpark.py
SETUP=sepq_2pt_setup_delta_a04m310.py
SUBMIT=2pt_delta_a04m310_3node_method_2_hostpark.sh               # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing_boost_direction_tchunk.py      # in this folder: the production smearing + sink smearing on nt_chunk time slices at a time
COMM=../../../../../tools/pt2_comm_tools.py          # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools, five levels up from this folder)

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# x only, checked before the run folder is made
[ "${BOOST_DIRECTION:-x}" = x ] || { echo "BOOST_DIRECTION=$BOOST_DIRECTION refused: this folder is the x production (current_dir ends in _direction_x)"; exit 1; }
BOOST_DIRECTION=x                                 # lattice axis of the boost: x only in this folder

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/projects/biqz/$USER/2pt_production/testruns/pt2_a04m310_3node_method_2_hostpark_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_a04m310_3node_method_2_hostpark_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_a04m310_3node_method_2_hostpark_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment: conda env pyquda, QUDA_PATH, LD_LIBRARY_PATH (Cray PE defaults: PrgEnv-gnu, cray-mpich, cudatoolkit)
source ~/env_pyquda.sh
cd "$RUNDIR" || exit 1
export PYTHONPATH=$RUNDIR:$PYTHONPATH

# QUDA
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # same string as resource_path in core.init
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"

# run parameters: the job measures cfg_list[ICFG] .. cfg_list[ICFG + 4] of the setup (NODE TEST: ICFG from the environment, default 0)
ICFG=${ICFG:-0}
# fail fast if this task's first gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "task $ICFG: gauge file $GAUGE, boost direction $BOOST_DIRECTION"
# versions, and the modules the script imports, without starting MPI outside srun (importing pyquda calls MPI_Init)
python -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'cupy-cuda13x', 'mpi4py', 'numpy', 'h5py', 'opt_einsum', 'tqdm']), sep=', ')" || exit 1
nvidia-smi -L
nvidia-smi topo -m

# GPU memory of the 8 GPUs of the first node every 2 s, to see the peak (QUDA and cupy together) on the 40 GB A100s
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -lms 2000 > "$RUNDIR/gpu_mem_${SLURM_JOB_ID}.csv" &
MONITOR=$!

# cray-mpich has Slurm PMI: launch with srun. --gpu-bind=none: PyQUDA gives node-local rank i GPU i, so every rank must see all GPUs of its node.
# NODE TEST: 8 ranks per node on 3 node(s); --cpu-bind=cores gives each rank its own 16 cores (the GPU-core map of the gpuA100x8 node is
# printed by nvidia-smi topo -m above; the 4-GPU masks of the production script are for gpuA100x4)
srun -N $SLURM_NNODES -n $((SLURM_NNODES * 8)) --ntasks-per-node=8 --gpu-bind=none --cpu-bind=cores \
    python -m mpi4py "$SCRIPT" --icfg $ICFG --n 5 --boost_direction $BOOST_DIRECTION     # -m mpi4py: an exception on any rank aborts all ranks
STATUS=$?                                             # exit code of the run, so that sacct shows FAILED instead of COMPLETED when python fails
kill $MONITOR
date
exit $STATUS
