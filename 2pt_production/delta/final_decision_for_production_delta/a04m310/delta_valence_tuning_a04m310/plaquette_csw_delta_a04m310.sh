#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=208G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -J plaq_a04m310
#SBATCH -o ./logs/plaq_a04m310_%j.out
#SBATCH -e ./logs/plaq_a04m310_%j.err

# HYP plaquette and the tadpole clover coefficient c_SW = P^(-3/4) on the cfgs of cfg_list of setup_valence_tuning_delta_a04m310.py
# (plaquette_csw_delta_a04m310.py), 8 ranks on one Delta gpuA100x8 node (8 A100-40GB, NVLink), grid [1,1,1,8]. Copy of the Delta production job script
# (a09m310/delta_2pt_production_a09m310_stream_e_direction_x/2pt_coulomb_boosted_sepq_3072srcs_delta_final.sh) without the array, the smearing file,
# the direction lines and the run arguments.
# The result is in the .out log (one line per cfg, then the mean) and in {out_dir}/plaquette_hyp_<ens_tag>_ncfg<n>.npz of the setup;
# the GPU memory of the 8 GPUs every 2 s is in gpu_mem_<jobid>.csv in the run folder.
# Usage, from this folder:   sbatch plaquette_csw_delta_a04m310.sh
# The job runs in its own folder testruns/plaquette_a04m310_<jobid>, on copies of the python files; the folder also keeps this sbatch script.
# Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" in the submit directory before the first sbatch.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=plaquette_csw_delta_a04m310.py
SETUP=setup_valence_tuning_delta_a04m310.py
SUBMIT=plaquette_csw_delta_a04m310.sh                  # name of this file, used for its copy in the run folder
COMM=../../../../tools/pt2_comm_tools.py          # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools, four levels up from this folder)

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/projects/biqz/$USER/2pt_production/testruns/plaquette_a04m310_${SLURM_JOB_ID}
mkdir -p "$RUNDIR" || exit 1
cp "$SCRIPT" "$SETUP" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/plaq_a04m310_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/plaq_a04m310_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment: conda env pyquda, QUDA_PATH, LD_LIBRARY_PATH (Cray PE defaults: PrgEnv-gnu, cray-mpich, cudatoolkit)
source ~/env_pyquda.sh
cd "$RUNDIR" || exit 1
export PYTHONPATH=$RUNDIR:$PYTHONPATH

# QUDA. Importing the setup also checks the gauge files (all complete, enough of them): it stops here if they are not
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=$(python -c "from ${SETUP%.py} import quda_resource_path, cfg_list; print(quda_resource_path)")   # same string as resource_path in core.init
[ -n "$QUDA_RESOURCE_PATH" ] || { echo "could not read quda_resource_path and cfg_list from $SETUP"; exit 1; }
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p "$QUDA_RESOURCE_PATH"
python -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print('cfgs', cfg_list.tolist(), 'from', gauge_file_prefix)"
# versions, and the modules the script imports, without starting MPI outside srun (importing pyquda calls MPI_Init)
python -c "import importlib.metadata as m; print(*(f'{p} {m.version(p)}' for p in ['pyquda', 'pyquda-utils', 'cupy-cuda13x', 'mpi4py', 'numpy']), sep=', ')" || exit 1
nvidia-smi -L
nvidia-smi topo -m

# GPU memory of all 8 GPUs every 2 s, to see the peak (QUDA and cupy together) on the 40 GB A100s
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -lms 2000 > "$RUNDIR/gpu_mem_${SLURM_JOB_ID}.csv" &
MONITOR=$!

# cray-mpich has Slurm PMI: launch with srun. --gpu-bind=none: PyQUDA gives node-local rank i GPU i, so every rank must see all 8 GPUs.
# --cpu-bind=cores: each rank gets its own 16 of the 128 cores. Which cores sit next to which GPU on the gpuA100x8 node is not known yet (the 4-GPU
# masks of the production script are for gpuA100x4): nvidia-smi topo -m above prints it, for mask_cpu in the 2pt runs.
srun -n 8 --gpu-bind=none --cpu-bind=cores \
    python -m mpi4py "$SCRIPT"                        # -m mpi4py: an exception on any rank aborts all ranks
STATUS=$?                                             # exit code of the run, so that sacct shows FAILED instead of COMPLETED when python fails
kill $MONITOR
date
exit $STATUS
