#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH --qos debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -J pt2_coulomb_test
#SBATCH -o ./logs/pt2_coulomb_boosted_frontier_test_%j.out
#SBATCH -e ./logs/pt2_coulomb_boosted_frontier_test_%j.err

# 30-minute debug test on one Frontier node, 8 GCDs, grid [1,1,1,8] in the python script
# ends before the configuration is saved: read the log for the convention check, gauge fixing, memory and timing
cd $SLURM_SUBMIT_DIR
mkdir -p logs wrappers
SCRIPT=2pt_coulomb_boosted_test_64srcs_frontier.py

# fail fast if a module the script imports is missing from the submit directory
for f in "$SCRIPT" coulomb_smearing.py; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# keep a copy of exactly what ran; the smearing module copy is a record only, python imports the one in $SLURM_SUBMIT_DIR
RUN_SCRIPT=logs/${SCRIPT%.py}_${SLURM_JOB_ID}.py
cp "$SCRIPT" "$RUN_SCRIPT"
cp coulomb_smearing.py logs/coulomb_smearing_${SLURM_JOB_ID}.py
date; hostname

# environment (Sergey's PyQUDA build)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh

# QUDA
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=/lustre/orion/lgt132/scratch/sicheng/GLUON_ANALYSIS_MANUAL/2pt_production/.cache/quda   # same string as resource_path in core.init
export QUDA_PROFILE_OUTPUT_BASE=${QUDA_RESOURCE_PATH}/profile_
export QUDA_ENABLE_P2P=0                  # must stay 0 on Frontier: ROCm IPC gives silent halo errors
export QUDA_ENABLE_MPS=1                  # one GCD visible per rank
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH:/ccs/home/sicheng/packages
export CUPY_CACHE_DIR=/lustre/orion/lgt132/scratch/sicheng/cupy_cache/${SLURM_JOB_ID}
export TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}
mkdir -p $QUDA_RESOURCE_PATH $CUPY_CACHE_DIR $TMPDIR

# one GCD per rank, bound to its NUMA domain
WRAPPER="wrappers/select_gpu_${SLURM_JOB_ID}"
cat << EOW > "${WRAPPER}"
#!/bin/bash
export GPU_MAP=(0 1 2 3 7 6 5 4)
export NUMA_MAP=(3 3 1 1 2 2 0 0)
export GPU=\${GPU_MAP[\$SLURM_LOCALID]}
export NUMA=\${NUMA_MAP[\$SLURM_LOCALID]}
export HIP_VISIBLE_DEVICES=\$GPU
unset ROCR_VISIBLE_DEVICES
echo RANK \$SLURM_LOCALID using GPU \$GPU
exec numactl -m \$NUMA -N \$NUMA \$*
EOW
chmod +x "${WRAPPER}"
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer: 74.5 -> 45.1 s per source

# first configuration of the stream-d set (override with: ICFG=5 sbatch submit_2pt_coulomb_frontier_test.sh)
ICFG=${ICFG:-0}
srun -u --exclusive -N 1 -n 8 --kill-on-bad-exit=1 "${WRAPPER}" python3 -m mpi4py "$RUN_SCRIPT" --icfg $ICFG --n 1
date
