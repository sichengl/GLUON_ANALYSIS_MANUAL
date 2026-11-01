#!/bin/bash
#SBATCH --account=biqz-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH -J check_2pt
#SBATCH -o ./check_2pt_%j.out
#SBATCH -e ./check_2pt_%j.out

# Storage check of every finished 2pt cfg in the _paired (first 199 cfgs) and _smear output folders on /work/hdd (check_2pt_storage.py --all):
# reads the data blocks of every .h5 and prints a BAD line for zeros, non-finite numbers, cut or unreadable files, then one line per cfg.
# The GPU is not used: the allocation has only GPU partitions, and 1 GPU with 16 cores and 60 GB (one quarter of an A40 node) is the smallest
# charge, 1 SU per hour with the A40 discount on top. About 2 TB to read (the _paired files only in their [a,b,a,b] blocks).
# Files that are OK in the ledger with the same size and mtime are skipped: after a time limit, or when more cfgs have finished, submit again.
# Submit from 2pt_production/tools on Delta: sbatch check_2pt_storage.sh
cd "$SLURM_SUBMIT_DIR"
source ~/env_pyquda.sh
OUT=/work/hdd/biqz/sliu54/2pt_production/final_production_3072src_200cfg
LEDGER=~/check_2pt_ledger.tsv
date; hostname
python check_2pt_storage.py --all --workers 32 --ledger "$LEDGER" "$OUT"/*_GEVP_sepq_3072src_phyp_fb_tsep20
STATUS=$?
date
exit $STATUS
