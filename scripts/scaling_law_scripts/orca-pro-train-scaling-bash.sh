#!/bin/bash

#SBATCH --job-name=orca_nuT_scaling
#SBATCH --output=/lustre/fsn1/projects/rech/dtr/commun/logs_and_ckpts/slurm_logs/orca_nuT_scaling_output.out
#SBATCH --error=/lustre/fsn1/projects/rech/dtr/commun/logs_and_ckpts/slurm_logs/orca_nuT_scaling_error.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kapoor@lpccaen.in2p3.fr

#SBATCH --account=dtr@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=0-19:50:00

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

cd /lustre/fsn1/projects/rech/dtr/commun/nuT_vanilla

CONFIG_DIR="/lustre/fsn1/projects/rech/dtr/commun/nuT_vanilla/configs/scaling_law_config_files/configs_orca_scaling_studies"

CONFIGS=(
    "$CONFIG_DIR/orca-pro-energy-config-10k.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-20k.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-50k.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-100k.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-200k.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-500k.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-1M.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-2M.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-5M.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-10M.yaml"
    "$CONFIG_DIR/orca-pro-energy-config-20M.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running config: $CONFIG"
    echo "=========================================="
    srun python scripts/orca-pro-train.py \
        --config "$CONFIG"
    echo "Finished: $CONFIG"
done

module purge
echo 'Done.'
exit 0
