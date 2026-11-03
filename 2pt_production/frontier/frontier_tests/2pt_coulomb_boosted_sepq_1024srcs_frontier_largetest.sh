#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J pt2_sepq_frontier
#SBATCH -o ./logs/pt2_sepq_frontier_%j.out
#SBATCH -e ./logs/pt2_sepq_frontier_%j.err

# 2pt test (pion + proton), sepq kinematics, Coulomb-gauge boosted smearing, 4 GCDs on one Frontier node, grid [1,1,1,4] in the python script.
# One configuration, all 1024 sources. Every job runs in its own folder testruns/sepq_test_<jobid>, on copies of the python files;
# the folder also keeps this sbatch script. Jobs below 92 nodes are limited to 2 hours on Frontier.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_1024srcs_frontier_largetest.py
SETUP=sepq_2pt_setup_frontier.py
SUBMIT=2pt_coulomb_boosted_sepq_1024srcs_frontier_largetest.sh            # name of this file, used for its copy in the run folder

# fail fast if a file the script imports is missing from the submit directory
for f in "$SCRIPT" "$SETUP" coulomb_smearing.py; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, the GPU wrapper, links to the two log files
RUNDIR=$SLURM_SUBMIT_DIR/testruns/sepq_test_${SLURM_JOB_ID}
mkdir -p "$RUNDIR"
cp "$SCRIPT" "$SETUP" coulomb_smearing.py "$RUNDIR"/
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_sepq_frontier_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_sepq_frontier_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment (Sergey's PyQUDA build)
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh

# QUDA: the tuning cache is set by core.init(resource_path=quda_resource_path) from the setup file, so QUDA_RESOURCE_PATH is not exported here;
# its path is read from the setup file only to create the folder and to place the profile files next to the tuning files
TUNE_DIR=$(cd "$RUNDIR" && python3 -c "from sepq_2pt_setup_frontier import quda_resource_path; print(quda_resource_path)")
[ -n "$TUNE_DIR" ] || { echo "could not read quda_resource_path from $SETUP"; exit 1; }
export QUDA_ENABLE_TUNING=1
export QUDA_PROFILE_OUTPUT_BASE=${TUNE_DIR}/profile_
export QUDA_ENABLE_P2P=0                  # must stay 0 on Frontier: ROCm IPC gives silent halo errors
export QUDA_ENABLE_MPS=1                  # one GCD visible per rank
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export PYTHONPATH=$RUNDIR:$PYTHONPATH:/ccs/home/sicheng/packages      # the run folder first, so python imports the copies
export CUPY_CACHE_DIR=/lustre/orion/lgt132/scratch/sicheng/cupy_cache/${SLURM_JOB_ID}
export TMPDIR=/lustre/orion/lgt132/scratch/sicheng/tmp/${SLURM_JOB_ID}
mkdir -p $TUNE_DIR $CUPY_CACHE_DIR $TMPDIR

# one GCD per rank, bound to its NUMA domain
WRAPPER="$RUNDIR/select_gpu"
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
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so   # GPU transport layer for Cray MPICH GPU support; harmless with GDR off (GDR hangs intermittently on this build, keep it off)

# run parameters: one configuration, cfg = 204 + 6*ICFG: ICFG=0 -> cfg 204, ICFG=5 -> cfg 234 (override with: ICFG=5 sbatch 2pt_coulomb_boosted_sepq_1024srcs_frontier_largetest.sh)
ICFG=${ICFG:-0}
cd "$RUNDIR"
# 4 ranks for grid [1,1,1,4]; -m mpi4py: an exception on any rank aborts all ranks
srun -u --exclusive -N 1 -n 4 --kill-on-bad-exit=1 "${WRAPPER}" python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n 1
rc=$?
date
exit $rc                                  # so that sacct shows FAILED when the python run fails
