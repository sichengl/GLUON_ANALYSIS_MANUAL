#!/bin/bash
#SBATCH -A gluonp0.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --qos=normal
#SBATCH -t 18:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -J pt2_sepq_test
#SBATCH -o ./logs/pt2_sepq_test_%j.out
#SBATCH -e ./logs/pt2_sepq_test_%j.err

# 2pt test (pion + proton), sepq kinematics, Coulomb-gauge boosted smearing, 4 GPUs on one lq2 node, grid [1,1,1,4] in the python script.
# One configuration, all 1024 sources. Every job runs in its own folder testruns/sepq_test_<jobid>, on copies of the python files;
# the folder also keeps this sbatch script.
cd $SLURM_SUBMIT_DIR
mkdir -p logs
SCRIPT=2pt_coulomb_boosted_sepq_1024srcs_fermilab_tsrcsave_fb_classused.py
SETUP=sepq_2pt_setup_fermilab_classused.py
SUBMIT=2pt_coulomb_boosted_sepq_1024srcs_fermilab_tsrcsave_fb_classused.sh            # name of this file, used for its copy in the run folder
SMEAR=coulomb_smearing.py
[ -f "$SMEAR" ] || SMEAR=../tools/coulomb_smearing.py                    # repo layout: 2pt_production/tools next to 2pt_production/<cluster>
COMM=pt2_comm_tools.py
[ -f "$COMM" ] || COMM=../tools/pt2_comm_tools.py                        # the shared TwoPtParams class, imported by the setup

# fail fast if a file the script imports is missing
for f in "$SCRIPT" "$SETUP" "$SMEAR" "$COMM"; do
    [ -f "$f" ] || { echo "missing $f in $SLURM_SUBMIT_DIR"; exit 1; }
done

# run folder: copies of exactly what runs (python imports these copies), this sbatch script, links to the two log files
RUNDIR=/project/gluonp0/sicheng/gluon_production/2pt_production/testruns/sepq_test_${SLURM_JOB_ID}
mkdir -p "$RUNDIR"
cp "$SCRIPT" "$SETUP" "$SMEAR" "$COMM" "$RUNDIR"/
cp "$0" "$RUNDIR/$SUBMIT" || cp "$SUBMIT" "$RUNDIR/$SUBMIT"      # $0 is the copy of this script that Slurm is running
ln -s "$SLURM_SUBMIT_DIR/logs/pt2_sepq_test_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/pt2_sepq_test_${SLURM_JOB_ID}.err" "$RUNDIR"/
date; hostname

# environment (installed once by ~/setup_pyquda_env.sh)
module purge
module load gompi/2023a cuda/12.2.1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyquda
export QUDA_PATH=/project/gluonp0/sicheng/quda-install
export LD_LIBRARY_PATH=$QUDA_PATH/lib:$LD_LIBRARY_PATH

# QUDA
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=/lustre2/gluonp0/sliu1/.cache/quda     # same string as quda_resource_path in sepq_2pt_setup_fermilab_classused.py
export QUDA_ENABLE_P2P=3
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
mkdir -p $QUDA_RESOURCE_PATH

# run parameters: one configuration (override with: ICFG=100 sbatch 2pt_coulomb_boosted_sepq_1024srcs_fermilab_tsrcsave_fb_classused.sh)
ICFG=${ICFG:-0}                     # cfg 234: its q phases are not all 1, unlike cfg 204
cd "$RUNDIR"
export PYTHONPATH=$RUNDIR:$PYTHONPATH
nvidia-smi -L
# this OpenMPI has no Slurm PMI support: launch with mpirun, not srun
mpirun -np 4 python3 -m mpi4py "$SCRIPT" --icfg $ICFG --n 1     # -m mpi4py: an exception on any rank aborts all ranks
date