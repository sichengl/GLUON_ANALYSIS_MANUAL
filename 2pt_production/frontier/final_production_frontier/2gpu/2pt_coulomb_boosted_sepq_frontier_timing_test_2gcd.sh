#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -J pt2_timing_2gcd
#SBATCH -o ./logs/pt2_timing_2gcd_%j.out
#SBATCH -e ./logs/pt2_timing_2gcd_%j.err

# Timing test of the 2-GCD version on one node, 1 hour in the debug QOS (1 debug job at a time, at most 2 hours): its script on cfg 204 from
# t_src 0, four runs side by side, each on the 2 GCDs of one MI250X with a different mrhs. Each run is its own srun, so a run that runs out of GPU memory stops alone
# (that mrhs is too large). Every run is stopped after 55 minutes (about 15-20 of them go to gauge fixing, HYP and the multigrid setup with QUDA's tuning) and writes into its own test folder, never the production one. At the end the job
# prints, per run, the gauge fixing time, the time per source position (UNTIL CONTRACTION, without the first source, which includes QUDA's tuning)
# with the time for one t_src and one cfg it gives, and the GPU memory.
# Usage, from this folder:   sbatch 2pt_coulomb_boosted_sepq_frontier_timing_test_2gcd.sh
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# name, mrhs (read by the setup from PT2_MRHS); every run has 2 GCDs, grid [1, 1, 1, 2] from the setup
RUNS=("mrhs12 12" "mrhs6 6" "mrhs4 4" "mrhs3 3")
SCRIPT=2pt_coulomb_boosted_sepq_3072srcs_frontier_final_2gcd.py
SETUP=sepq_2pt_setup_frontier_final_2gcd.py
SMEAR=../../tools/coulomb_smearing.py          # the only copy: 2pt_production/tools, two levels up from 2pt_production/frontier/<this folder>
COMM=../../tools/pt2_comm_tools.py             # the shared TwoPtParams class, imported by the setup (only copy, in 2pt_production/tools)
SUBMIT=2pt_coulomb_boosted_sepq_frontier_timing_test_2gcd.sh        # name of this file, used for its copy in the run folder

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, one log per run in logs/
RUNDIR=$SLURM_SUBMIT_DIR/runs/timing_2gcd_${SLURM_JOB_ID}
mkdir -p "$RUNDIR/logs" || exit 1
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/ || exit 1
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
date; hostname

# environment
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_P2P=0                  # must stay 0 on Frontier: ROCm IPC gives silent halo errors
export QUDA_ENABLE_MPS=1                  # one GCD visible per rank: PyQUDA then uses device 0 on every rank
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0   # as on Delta
export PYTHONPATH=$RUNDIR:$PYTHONPATH:/ccs/home/sicheng/packages      # the run folder first, so python imports the copies
export CUPY_CACHE_DIR=/lustre/orion/lgt132/scratch/sicheng/cupy_cache/2pt_final
export TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}
mkdir -p $CUPY_CACHE_DIR $TMPDIR
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer for Cray MPICH GPU support; harmless with GDR off
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES                          # the GCD of each rank comes from Slurm only (ROCR_VISIBLE_DEVICES)
cd "$RUNDIR"
export QUDA_RESOURCE_PATH=$(python3 -c "from ${SETUP%.py} import quda_resource_path; print(quda_resource_path)")   # QUDA's tuning cache: same string as resource_path in core.init
mkdir -p "$QUDA_RESOURCE_PATH"                                                                                    # as on Delta: QUDA saves no tuning into a missing folder

# GPU memory of the 8 GCDs every 2 s: QUDA's solver memory peaks inside the solves, between the lines the runs log
( while true; do rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/^card/ {print $1 "," $NF}'; sleep 2; done ) > gpu_mem.csv &
MONITOR=$!

# the runs start one after another, so that each gets the cores and the two GCDs of one MI250X (the gpu-bind lines in logs/<name>.err show which)
PIDS=()
for entry in "${RUNS[@]}"; do
    read -r NAME MRHS <<< "$entry"
    PT2_OUTPUT_DIR=$RUNDIR/out_$NAME PT2_MRHS=$MRHS timeout 55m \
        srun -u -N 1 -n 2 -c 7 --exact --gpus-per-task=1 --gpu-bind=verbose,closest -o "logs/$NAME.out" -e "logs/$NAME.err" \
        python3 -u -m mpi4py "$SCRIPT" --icfg 0 --n 1 &
    PIDS+=($!)
    sleep 2                               # gives Slurm time to place this run before the next (OLCF guide)
done
wait "${PIDS[@]}"                      # the four runs only: the memory monitor above never ends by itself
kill $MONITOR
date

# per run: gauge fixing, seconds per source position without the first one, what that gives for one t_src (128 sources) and one cfg (3072), GPU memory
for entry in "${RUNS[@]}"; do
    read -r NAME MRHS <<< "$entry"
    echo "==== $NAME (2 GCDs, mrhs $MRHS)"
    grep -h "COULOMB GAUGE FIXING\|out of memory\|Error" "logs/$NAME.out" "logs/$NAME.err" | head -5
    awk '/UNTIL CONTRACTION/ {n++; if (n > 1) {sum += $(NF-1); m++}}
         /GPU used/ {for (i = 1; i < NF; i++) if ($i == "used") {gb = $(i+1) + 0; if (gb > peak) peak = gb}}
         END {if (m > 0) printf "%d sources, %.1f s per source (first left out): one t_src %.2f h, one cfg %.1f h; peak GPU used in the log %.1f GB\n",
                                n, sum / m, 128 * sum / m / 3600, 3072 * sum / m / 3600, peak;
              else printf "%d sources done: no timing\n", n}' "logs/$NAME.out"
done
echo "==== peak GPU memory per GCD (rocm-smi every 2 s)"
awk -F, '{if ($2 > max[$1]) max[$1] = $2} END {for (c in max) printf "%s %.1f GB\n", c, max[c] / 1e9}' gpu_mem.csv | sort
