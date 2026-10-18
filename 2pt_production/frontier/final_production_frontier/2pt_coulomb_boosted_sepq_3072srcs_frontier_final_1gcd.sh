#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 12:00:00
#SBATCH -N 200
#SBATCH -J pt2_final_1gcd
#SBATCH -o ./logs/pt2_final_1gcd_%j.out
#SBATCH -e ./logs/pt2_final_1gcd_%j.err

# Final run on Frontier, 1 GCD per run (grid [1,1,1,1]): the physics of the Delta final run (3072 sources per cfg, two smearings, three gammas,
# pion + proton, sepq kinematics). One srun starts the 8 runs of every node (OLCF advises against hundreds of sruns). Every cfg is cut into NPARTS
# blocks of its 24 t_src, one block per run, so with N nodes the job measures 8N / NPARTS cfgs from cfg_list[ICFG] on. Every t_src is saved to
# parts/ when done; the run that saves the last t_src of a cfg merges it into the 12 .h5 files.
# After a time-limit kill, submit the same command again: the saved t_src are skipped. The next range starts at ICFG + 8N / NPARTS.
# Usage, from this folder:   ICFG=200 NPARTS=4 sbatch -N 200 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_1gcd.sh
# 200-1881 nodes get 12 hours; 92-183 nodes 6 hours; fewer than 92 nodes 2 hours.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
ICFG=${ICFG:-200}       # first cfg, index in cfg_list: 200 -> cfg 1404, the first after the 200 cfgs of the Delta run
NPARTS=${NPARTS:-4}     # t_src blocks per cfg: 24 / NPARTS t_src per run must fit in 12 h, set it from the timing test
MRHS=${MRHS:-4}         # from the timing test
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final.py
SETUP=sepq_2pt_setup_frontier_final.py
TOOLS=frontier_2pt_tools.py
SMEAR=coulomb_smearing.py
COMM=pt2_comm_tools.py
SUBMIT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_1gcd.sh        # name of this file, used for its copy in the run folder

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$TOOLS" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, one log per rank in logs/
RUNDIR=$SLURM_SUBMIT_DIR/runs/final_1gcd_${SLURM_JOB_ID}
mkdir -p "$RUNDIR/logs" || exit 1
cp "$SCRIPT" "$SETUP" "$TOOLS" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
date; hostname; echo "nodes $SLURM_NNODES, ICFG $ICFG, NPARTS $NPARTS, MRHS $MRHS"

# environment (Sergey's PyQUDA build)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_P2P=0                  # must stay 0 on Frontier: ROCm IPC gives silent halo errors
export QUDA_ENABLE_MPS=1                  # one GCD visible per rank: PyQUDA then uses device 0 on every rank
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export PYTHONPATH=$RUNDIR:$PYTHONPATH:/ccs/home/sicheng/packages      # the run folder first, so python imports the copies
export CUPY_CACHE_DIR=/lustre/orion/lgt132/scratch/sicheng/cupy_cache/2pt_final
export TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}
mkdir -p $CUPY_CACHE_DIR $TMPDIR
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer for Cray MPICH GPU support; harmless with GDR off
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES                          # the GCD of each rank comes from Slurm only (ROCR_VISIBLE_DEVICES)

# all runs, one srun: rank r is run r, 7 cores and the closest GCD each. -m mpi4py: an exception on any rank stops the srun (submit again)
cd "$RUNDIR"
srun -N $SLURM_NNODES -n $((SLURM_NNODES * 8)) --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest \
    -o "$RUNDIR/logs/rank%t.out" -e "$RUNDIR/logs/rank%t.err" \
    python3 -u -m mpi4py "$SCRIPT" --gcds 1 --mrhs $MRHS --icfg $ICFG --n 1 --nparts $NPARTS
STATUS=$?
date
exit $STATUS
