#!/bin/bash

#SBATCH --job-name=subset-extraction
#SBATCH --output=/lustre/fsn1/projects/rech/lba/ulq92pd/logs_and_ckpts/slurm_logs/subset-extraction.out
#SBATCH --error=/lustre/fsn1/projects/rech/lba/ulq92pd/logs_and_ckpts/slurm_logs/subset-extraction.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=letellier@lpccaen.in2p3.fr

#SBATCH --account=lba@cpu
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=0-08:00:00

#SBATCH --no-requeue
#SBATCH --signal=SIGUSR1@60


echo "Initial working directory: $PWD"
echo "Job: $SLURM_JOB_ID"
echo "Job array task ID: $SLURM_ARRAY_TASK_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Visible devices: $CUDA_VISIBLE_DEVICES"

module purge
module load pytorch-gpu/py3/2.8.0

echo "Python: $(which python)"
export PYTHONUNBUFFERED=1  # needed because slurm error when printing in python

pip list  # check your env

cd /lustre/fsn1/projects/rech/lba/ulq92pd/

srun python scripts/extract_subset_db_fixed.py \
        --src  ../commun/ORCA/merged.db \
        --dst  ../commun/ORCA/merged_100k_bis.db \
        --n    100000 \
        --pulse-table merged_photons \
        --truth-table mc_truth \
        --seed 42
mv ../commun/ORCA/merged_100k_bis_selection.parquet ../commun/ORCA/selections/merged_100k_bis_selection.parquet

module purge
echo 'Done.'
exit 0
