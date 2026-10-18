#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p extended
#SBATCH -t 24:00:00
#SBATCH -N 3
#SBATCH -J pt2_check_cfg204
#SBATCH -o ./logs/pt2_check_cfg204_%j.out
#SBATCH -e ./logs/pt2_check_cfg204_%j.err

# Check against the Delta final run on cfg 204: all 24 t_src of cfg 204 with the production script, launched as in production (one srun, one run per
# GCD), one t_src per run on 3 nodes, into its own folder (never the production one). The run that saves the last t_src merges the 12 .h5 files,
# which are then compared with Delta's cfg 204 files if DELTA_DIR holds them (copied from Delta:
# /work/hdd/biqz/sliu54/2pt_production/final_production_3072src_200cfg/coulomb_rho4p5z2p25-rho1p5_frac0p8-0_paired_GEVP_sepq_3072src_phyp_fb_tsep20/cfg204).
# Extended partition: at most 64 nodes and 24 hours, 1 running job per user. Run the timing test first: MRHS must fit in the GCD.
# Usage, from this folder:   DELTA_DIR=/lustre/orion/lgt132/scratch/sicheng/delta_cfg204 MRHS=4 sbatch 2pt_coulomb_boosted_sepq_frontier_check_cfg204.sh
# To measure cfg 204 again, delete the check folder first: the t_src already saved there are skipped.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MRHS=${MRHS:-4}
DELTA_DIR=${DELTA_DIR:-}
export PT2_OUTPUT_DIR=/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/2pt_production/check_cfg204          # read by the setup
NPARTS=$((SLURM_NNODES * 8))           # one block of t_src per run: 24 runs, one t_src each
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final.py
SETUP=sepq_2pt_setup_frontier_final.py
TOOLS=frontier_2pt_tools.py
SMEAR=coulomb_smearing.py
COMM=pt2_comm_tools.py
COMPARE=compare_2pt_frontier_vs_delta.py
SUBMIT=2pt_coulomb_boosted_sepq_frontier_check_cfg204.sh        # name of this file, used for its copy in the run folder

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$TOOLS" "$SMEAR" "$COMM" "$COMPARE"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, one log per rank in logs/
RUNDIR=$SLURM_SUBMIT_DIR/runs/check_cfg204_${SLURM_JOB_ID}
mkdir -p "$RUNDIR/logs" || exit 1
cp "$SCRIPT" "$SETUP" "$TOOLS" "$SMEAR" "$COMM" "$COMPARE" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
date; hostname; echo "nodes $SLURM_NNODES, NPARTS $NPARTS, MRHS $MRHS, output $PT2_OUTPUT_DIR, Delta files ${DELTA_DIR:-not given}"

# environment: as in the production job files
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

# all runs, one srun, as in production: rank r is run r, cfg_list[0] = 204, t_src block r
cd "$RUNDIR"
srun -N $SLURM_NNODES -n $((SLURM_NNODES * 8)) --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest \
    -o "$RUNDIR/logs/rank%t.out" -e "$RUNDIR/logs/rank%t.err" \
    python3 -u -m mpi4py "$SCRIPT" --gcds 1 --mrhs $MRHS --icfg 0 --n 1 --nparts $NPARTS
STATUS=$?
date

# gauge fixing of cfg 204 in every run (all must agree), then the comparison with Delta
grep -ho "theta [0-9.e+-]*" logs/rank*.out | sort | uniq -c
FRONTIER_CFG=$(python3 -c "from sepq_2pt_setup_frontier_final import *; print(f'{current_dir}/{smear_tag}_GEVP_sepq_{len(frontier_params.t_base) * n_spatial_src}src_phyp_fb_tsep{frontier_params.tsep_max}/cfg204')")
if [ -n "$DELTA_DIR" ] && [ -d "$DELTA_DIR" ]; then
    python3 "$COMPARE" "$DELTA_DIR" "$FRONTIER_CFG"
else
    echo "Delta's cfg 204 files not given (DELTA_DIR): compare later with"
    echo "  python3 $COMPARE <folder with Delta's cfg 204 .h5 files> $FRONTIER_CFG"
fi
exit $STATUS
