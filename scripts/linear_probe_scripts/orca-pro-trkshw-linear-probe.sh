#!/bin/bash

#SBATCH --job-name=orca_nuT_linear_probe
#SBATCH --output=/lustre/fsn1/projects/rech/lba/commun/logs_and_ckpts/slurm_logs/orca_linear_probe_output.out
#SBATCH --error=/lustre/fsn1/projects/rech/lba/commun/logs_and_ckpts/slurm_logs/orca_linear_probe_error.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kapoor@lpccaen.in2p3.fr

#SBATCH --account=lba@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=0-01:00:00

#SBATCH --no-requeue

echo "Initial working directory: $PWD"
echo "Job: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0

echo "Python: $(which python)"
export PYTHONUNBUFFERED=1

cd /lustre/fsn1/projects/rech/lba/commun/nuT_Neutrino_Transformer

CKPT="/lustre/fsn1/projects/rech/lba/commun/logs_and_ckpts/ckpt/Prometheus FlowerS - nuT_no_graphnet/Track shower classification 1M/Track shower classification 1M_epoch=30_val_loss=0.2888.ckpt"
CONFIG="configs/jz_config_files/orca-pro-trkshw-config.yaml"
OUTPUT_DIR="/lustre/fsn1/projects/rech/lba/commun/logs_and_ckpts/probe_output"

srun python -m linear_probe_analysis \
    --config  "$CONFIG" \
    --ckpt    "$CKPT" \
    --output_dir  "$OUTPUT_DIR" \
    --events  100000 \
    --no_random_baseline

module purge
echo 'Done.'
exit 0
