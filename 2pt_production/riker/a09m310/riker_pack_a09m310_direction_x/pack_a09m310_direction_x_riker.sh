#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -J pack_a09m310_x
#SBATCH -o ./logs/pack_a09m310_direction_x_riker_%j.out
#SBATCH -e ./logs/pack_a09m310_direction_x_riker_%j.out
#The six packs of the a09m310 direction-x production, all at the same time on ONE Riker CPU node (128 cores, ~2.2 TB memory):
#   2pt stream d, e:            frontier_package_byq_opt_a09m310_stream{d,e}.py --boost_direction x        (16 reader processes, ~40 GB each)
#   FF  stream d, e x hyp, flow: packFF_byq_frontier_a09m310_stream{d,e}.py --smear {hyp,flow} --boost_direction x   (one cfg at a time)
#The packers are the Frontier ones, unchanged, run in their Frontier folders on Orion (Riker mounts /lustre/orion), so they read and write the
#same Orion folders as on Frontier. Each pack has its own log, logs/<pack>_<jobid>.log in this folder, with the clock time in front of every line.
#A pack whose output folder already has its files (finished, or .tmp files being written by another run) is not started; FORCE=1 starts it anyway.
#The job ends FAILED if any pack fails.
#Submit on Riker from this folder (after setup_riker_pack_venv.sh; Slurm needs logs/ to exist before it starts):    mkdir -p logs; sbatch pack_a09m310_direction_x_riker.sh
#Cost: 0.0078125 x 128 cores = 1 Riker node-hour per hour of this job (lgt132 has 1000).
cd $SLURM_SUBMIT_DIR
PY=/ccs/home/$USER/riker_pack_venv/bin/python                     # made by setup_riker_pack_venv.sh
$PY -c "import numpy, h5py" || { echo "numpy/h5py do not import with $PY: run setup_riker_pack_venv.sh on a Riker login node"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1                  # one thread per process (38 processes on 128 cores)
export PYTHONDONTWRITEBYTECODE=1                                 # no __pycache__ in the code folders
ROOT=/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL
PT2=$ROOT/2pt_production/frontier/final_production_frontier/8gpu/a09m310
FFC=$ROOT/FF_production/frontier/a09m310
OUT2=$ROOT/2pt_production/packed
OUTF=$ROOT/FF_production
stamp() { while read line; do echo "$(date +%T) $line"; done; }   # clock time in front of every line

#run_pack <name> <folder> <packer> <its output files> <packer arguments ...>: starts one pack in the background
NAMES=(); PIDS=()
run_pack() {
    name=$1; dir=$2; script=$3; out=$4; shift 4
    n_out=$(ls $out 2>/dev/null | wc -l)
    if [ $n_out -gt 0 ] && [ "${FORCE:-0}" != 1 ]; then
        echo "$name: not started, $n_out files already there: $out"
        return
    fi
    (set -o pipefail; cd $dir && $PY -u $script "$@" 2>&1 | stamp > $SLURM_SUBMIT_DIR/logs/${name}_$SLURM_JOB_ID.log) &
    NAMES+=($name); PIDS+=($!)
    echo "$name: started, log logs/${name}_$SLURM_JOB_ID.log"
}

date; hostname; $PY --version
run_pack 2pt_streamd     $PT2/frontier_2pt_production_a09m310_stream_d_direction_x frontier_package_byq_opt_a09m310_streamd.py "$OUT2/a09m310_final_production_byq_streamd_direction_x/*" --boost_direction x
run_pack 2pt_streame     $PT2/frontier_2pt_production_a09m310_stream_e_direction_x frontier_package_byq_opt_a09m310_streame.py "$OUT2/a09m310_final_production_byq_streame_direction_x/*" --boost_direction x
run_pack FF_streamd_hyp  $FFC/frontier_FF_production_a09m310_stream_d_direction_x  packFF_byq_frontier_a09m310_streamd.py        "$OUTF/a09m310_FF_byq_streamd_direction_x/FF_a09m310_streamd_hyp*"  --smear hyp  --boost_direction x
run_pack FF_streamd_flow $FFC/frontier_FF_production_a09m310_stream_d_direction_x  packFF_byq_frontier_a09m310_streamd.py        "$OUTF/a09m310_FF_byq_streamd_direction_x/FF_a09m310_streamd_flow*" --smear flow --boost_direction x
run_pack FF_streame_hyp  $FFC/frontier_FF_production_a09m310_stream_e_direction_x  packFF_byq_frontier_a09m310_streame.py        "$OUTF/a09m310_FF_byq_streame_direction_x/FF_a09m310_streame_hyp*"  --smear hyp  --boost_direction x
run_pack FF_streame_flow $FFC/frontier_FF_production_a09m310_stream_e_direction_x  packFF_byq_frontier_a09m310_streame.py        "$OUTF/a09m310_FF_byq_streame_direction_x/FF_a09m310_streame_flow*" --smear flow --boost_direction x

STATUS=0
for i in ${!PIDS[@]}; do
    wait ${PIDS[$i]}; s=$?
    echo "$(date +%T) ${NAMES[$i]}: exit $s"
    [ $s = 0 ] || STATUS=1
done
for d in $OUT2/a09m310_final_production_byq_stream{d,e}_direction_x $OUTF/a09m310_FF_byq_stream{d,e}_direction_x; do
    echo "$d: $(ls $d/*.h5 2>/dev/null | wc -l) .h5, $(ls $d/*.tmp 2>/dev/null | wc -l) .tmp"
done
date
exit $STATUS
