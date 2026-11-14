#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --no-requeue
#SBATCH -J FF_boost
#SBATCH --array=0-79
#SBATCH -o ./logs/FF_boost_%A_%a.out
#SBATCH -e ./logs/FF_boost_%A_%a.err

# FF (FF_covshift_boost_direction_delta.py) for the 800 stream-e cfgs (204, 210, ..., 4998) on Delta: one A100 GPU per array task.
# The smearing and the direction come from two variables, both required:
#     SMEAR            hyp or flow      (hyp: 16 levels, lines 0..15;  flow: 41 levels of 10 steps of 0.01, lines 0..9)
#     BOOST_DIRECTION  x, y or z        (lattice axis of the Wilson line)
# Array task k does NCFG consecutive cfgs, starting at cfg 204 + 6 * NCFG * k. With NCFG = 10 (the default) the tasks 0-79 cover 204 .. 4998.
# --array must match NCFG: the last task with cfgs is 799 / NCFG (a task without cfgs exits at once).
#
# First test, one cfg (204):    NCFG=1 SMEAR=hyp BOOST_DIRECTION=x sbatch --array=0 FF_covshift_boost_direction_delta.sh
# then read the time per smearing level in the log ("SHIFT #204: ... secs"), set --time, and submit the rest:
#                               SMEAR=hyp BOOST_DIRECTION=x sbatch FF_covshift_boost_direction_delta.sh
# The 6 h limit is a guess for 10 cfgs (the old script took about 6 min per cfg for hyp on a Fermilab A100; not measured on Delta).
# A cfg is NOT skipped if its file exists: a task that is submitted again computes all its cfgs again.
# Output: one file per cfg, FF_opp_symmetric_asymmetric_<hyp0-15_w0-15 | flow0-40_w0-9>_boost<x|y|z>_cfg<cfg>.h5, in out_dir of the python script.
# About 0.45 GB per cfg for hyp and 0.73 GB for flow: 360 GB and 580 GB for 800 cfgs of one direction. Run "quota" first.
# Submit from the folder that holds the python script. Slurm opens the -o/-e files before this script starts: run "mkdir -p logs" first.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SCRIPT=FF_covshift_boost_direction_delta.py
NCFG=${NCFG:-10}
[ -f "$SCRIPT" ] || { echo "missing $SCRIPT in $SLURM_SUBMIT_DIR"; exit 1; }
if [ -z "$SMEAR" ] || [ -z "$BOOST_DIRECTION" ]; then
    echo "set SMEAR (hyp or flow) and BOOST_DIRECTION (x, y or z):  SMEAR=hyp BOOST_DIRECTION=x sbatch FF_covshift_boost_direction_delta.sh"
    exit 1
fi
CFG=$((204 + 6 * NCFG * ${SLURM_ARRAY_TASK_ID:-0}))
if [ $CFG -gt 4998 ]; then
    echo "array task $SLURM_ARRAY_TASK_ID: no cfgs for NCFG=$NCFG (first cfg $CFG > 4998)"
    exit 0
fi
N=$(( (4998 - CFG) / 6 + 1 ))
[ $N -le $NCFG ] || N=$NCFG
date; hostname
echo "smear $SMEAR, boost direction $BOOST_DIRECTION, cfgs $CFG .. $((CFG + 6 * (N - 1)))"

# environment: conda env pyquda, QUDA_PATH, LD_LIBRARY_PATH (as in the 2pt scripts)
source ~/env_pyquda.sh
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0          # QUDA allocates with cudaMalloc and cannot use the blocks cupy keeps cached
nvidia-smi -L

# GPU memory every 10 s, to see the peak (QUDA and cupy together) on the 40 GB A100
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -l 10 > "logs/FF_boost_gpu_mem_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
MONITOR=$!

# cray-mpich has Slurm PMI: launch with srun, one rank on the one GPU (the python script uses the grid [1,1,1,1])
srun -n 1 python "$SCRIPT" --config "{\"cfg_n\": $CFG, \"num_cfg\": $N}" --smear $SMEAR --boost_direction $BOOST_DIRECTION
STATUS=$?                                        # exit code of the run, so that sacct shows FAILED instead of COMPLETED when python fails
kill $MONITOR
date
exit $STATUS
