#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -J plaq_a04m310
#SBATCH -o ./logs/plaq_a04m310_%j.out
#SBATCH -e ./logs/plaq_a04m310_%j.err

# HYP plaquette and the tadpole clover coefficient c_SW = P^(-3/4) on the cfgs of cfg_list of setup_valence_tuning_frontier_a04m310.py
# (plaquette_csw_frontier_a04m310.py): one run of 8 ranks on one node, as the one-node 2pt job of the a09m310 production
# (a09m310/frontier_stream_d_directionxyz/2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.sh), without the smearing and direction lines.
# The result is in the .out log (one line per cfg, then the mean) and in {out_dir}/plaquette_hyp_<ens_tag>_ncfg<n>.npz of the setup.
# Usage, from this folder:   sbatch plaquette_csw_frontier_a04m310.sh
# Every job runs in its own folder runs/plaquette_<jobid>, on copies of the python files; the folder also keeps this sbatch script.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=plaquette_csw_frontier_a04m310.py
SETUP=setup_valence_tuning_frontier_a04m310.py
SUBMIT=plaquette_csw_frontier_a04m310.sh               # name of this file, used for its copy in the run folder
COMM=../../../../../tools/pt2_comm_tools.py       # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=$SLURM_SUBMIT_DIR/runs/plaquette_${SLURM_JOB_ID}
mkdir -p "$RUNDIR"
cp "$SCRIPT" "$SETUP" "$COMM" "$RUNDIR"/
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/plaq_a04m310_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/plaq_a04m310_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment (Sergey's PyQUDA build)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh

# QUDA: the tuning cache is set by core.init(resource_path=quda_resource_path) from the setup file; its path is read here only to create the folder.
# Importing the setup also checks the gauge files (all complete, enough of them): it stops here if they are not
TUNE_DIR=$(cd "$RUNDIR" && python3 -c "from ${SETUP%.py} import quda_resource_path, cfg_list; print(quda_resource_path)")
[ -n "$TUNE_DIR" ] || { echo "could not read quda_resource_path and cfg_list from $SETUP"; exit 1; }
export QUDA_ENABLE_TUNING=1
export QUDA_PROFILE_OUTPUT_BASE=${TUNE_DIR}/profile_
export QUDA_ENABLE_P2P=0                  # must stay 0 on Frontier: ROCm IPC gives silent halo errors
export QUDA_ENABLE_MPS=1                  # one GCD visible per rank: PyQUDA then uses device 0 on every rank
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export PYTHONPATH=$RUNDIR:$PYTHONPATH:/ccs/home/sicheng/packages      # the run folder first, so python imports the copies
export CUPY_CACHE_DIR=/lustre/orion/lgt132/scratch/sicheng/cupy_cache/${SLURM_JOB_ID}
export TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}
mkdir -p $TUNE_DIR $CUPY_CACHE_DIR $TMPDIR
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer for Cray MPICH GPU support; harmless with GDR off
cd "$RUNDIR"

# one run of 8 ranks for grid [1,1,1,8]: each rank gets 7 cores and the GCD closest to them. -m mpi4py: an exception on any rank aborts all ranks
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES                          # the GCD of each rank comes from Slurm only (ROCR_VISIBLE_DEVICES)
srun -u -N 1 -n 8 -c 7 --gpus-per-task=1 --gpu-bind=verbose,closest --cpu-bind=verbose --kill-on-bad-exit=1 \
    python3 -m mpi4py "$SCRIPT"
STATUS=$?                                 # exit code of the run, so that sacct shows FAILED instead of COMPLETED when python fails
date
exit $STATUS
