#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J pt2_final_8gcd
#SBATCH -o ./logs/pt2_final_8gcd_%j.out
#SBATCH -e ./logs/pt2_final_8gcd_%j.err

# Final run on Frontier, 8-GCD version: the Delta final script as it is (2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.py, only its setup
# import differs) with sepq_2pt_setup_frontier_final_8gcd.py (Delta's setup with grid [1,1,1,8] and Orion directories): one run on the 8 GCDs of one
# node, as one Delta task on the 4 A100s of one node. The job measures N cfgs from cfg_list[ICFG] (Delta's 200 cfgs: 0 -> cfg 204).
# A 2-hour batch job (1 node) is a timing test: it gets through part of the first t_src. For a full cfg: sbatch -p extended -t 24:00:00 <this file>,
# and submit it again after a time-limit kill: the t_src already saved are skipped. The extended partition (1-64 nodes, 24 hours) allows 1 running job.
# Every job runs in its own folder runs/final_8gcd_<jobid>, on copies of the python files; the folder also keeps this sbatch script.
# Usage, from this folder:   ICFG=0 N=1 sbatch 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.sh
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.py
SETUP=sepq_2pt_setup_frontier_final_8gcd.py
SUBMIT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.sh               # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing.py
COMM=pt2_comm_tools.py

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=$SLURM_SUBMIT_DIR/runs/final_8gcd_${SLURM_JOB_ID}
mkdir -p "$RUNDIR"
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_final_8gcd_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_final_8gcd_${SLURM_JOB_ID}.err" "$RUNDIR"/
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

# run parameters: N cfgs from cfg_list[ICFG] of the setup
ICFG=${ICFG:-0}
N=${N:-1}
cd "$RUNDIR"
# fail fast if the first gauge file is not there (the setup builds the name from gauge_file_prefix and cfg_list)
GAUGE=$(python3 -c "from ${SETUP%.py} import cfg_list, gauge_file_prefix; print(f'{gauge_file_prefix}.{cfg_list[$ICFG]}')")
[ -f "$GAUGE" ] || { echo "missing gauge file $GAUGE"; exit 1; }
echo "icfg $ICFG, n $N: first gauge file $GAUGE"

# GPU memory of the 8 GCDs every 2 s, to see the peak (QUDA and cupy together), as on Delta with nvidia-smi
( while true; do rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/^card/ {print $1 "," $NF}'; sleep 2; done ) > "$RUNDIR/gpu_mem.csv" &
MONITOR=$!

# one run of 8 ranks for grid [1,1,1,8]: each rank gets 7 cores (one L3 region) and the GCD closest to them (Slurm sets ROCR_VISIBLE_DEVICES per rank).
# verbose: Slurm writes each rank's GCD and cores into the job's .err log. -m mpi4py: an exception on any rank aborts all ranks
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES                          # the GCD of each rank comes from Slurm only (ROCR_VISIBLE_DEVICES)
srun -u -N 1 -n 8 -c 7 --gpus-per-task=1 --gpu-bind=verbose,closest --cpu-bind=verbose --kill-on-bad-exit=1 \
    python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n $N
STATUS=$?                                 # exit code of the run, so that sacct shows FAILED instead of COMPLETED when python fails
kill $MONITOR
date
exit $STATUS
