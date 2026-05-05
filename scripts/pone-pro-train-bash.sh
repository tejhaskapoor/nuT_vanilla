#!/bin/bash

#SBATCH --job-name=pone_nuT_training
#SBATCH --output=/lustre/fsn1/projects/rech/dtr/commun/logs_and_ckpts/slurm_logs/pone_nuT_output.out
#SBATCH --error=/lustre/fsn1/projects/rech/dtr/commun/logs_and_ckpts/slurm_logs/pone_nuT_error.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kapoor@lpccaen.in2p3.fr

#SBATCH --account=dtr@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=0-19:00:00

#SBATCH --no-requeue
#SBATCH --signal=SIGUSR1@60


echo "Initial working directory: $PWD"
echo "Job: $SLURM_JOB_ID"
echo "Job array task ID: $SLURM_ARRAY_TASK_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0

echo "Python: $(which python)"
export PYTHONUNBUFFERED=1  # needed because slurm error when printing in python

pip list  # check your env

cd /lustre/fsn1/projects/rech/dtr/commun/nuT_Neutrino_Transformer

CONFIG="/lustre/fsn1/projects/rech/dtr/commun/nuT_Neutrino_Transformer/configs/jz_config_files/pone-pro-energy-config.yaml"

srun python scripts/pone-pro-train.py \
    --config "$CONFIG"
    
module purge
echo 'Done.'
exit 0
