#!/bin/bash

#SBATCH --job-name=prometheus_nuT_training
#SBATCH --output=slurm_logs/prometheus_nuT_output.out
#SBATCH --error=slurm_logs/prometheus_nuT_error.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kapoor@lpccaen.in2p3.fr

#SBATCH --partition=htc_hgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 #1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16    # 252 is the number of cpus
#SBATCH --mem-per-cpu="4000M"
#SBATCH --time=0-00:20:00
#SBATCH --hint=multithread

#SBATCH --no-requeue
#SBATCH --signal=SIGUSR1@60


echo "Initial working directory: $PWD"
echo "Job: $SLURM_JOB_ID"
echo "Job array task ID: $SLURM_ARRAY_TASK_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

#module load almalinux-9-x86-64/Programming_Languages/anaconda/3.12
source /usr/etc/profile.d/conda.sh
#source ~/.bashrc  # module not working, used for accessing conda and already installed conda envs
# For some reason, venv didn't load (or conda didn't load the second time i ran the script, thus, using another way to activate conda)
#source /usr/etc/profile.d/conda.sh

#For activating virtural environment
export PYTHONNOUSERSITE=1
conda activate /data_hgx/KM3NeT/ML_graphnet/venv
export PYTHONPATH=/data_hgx/KM3NeT/ML_graphnet/

echo "Confa env:"
echo $CONDA_PREFIX
export PYTHONUNBUFFERED=1  # needed because slurm error when printing in python

pip list  # check your env

cd /data_hgx/KM3NeT/ML_graphnet/nuT_no_graphnet

CONFIG="/data_hgx/KM3NeT/ML_graphnet/nuT_no_graphnet/configs/jz_config_files/orca-pro-dir-config.yaml"

srun /data_hgx/KM3NeT/ML_graphnet/venv/bin/python orca-pro-train-optimized.py \
    --config "$CONFIG"

#python prometheus_train.py --config /data_hgx/KM3NeT/kapoor/prometheus_practice_space/configs/config_prometheus_energy.yaml

module purge
echo 'Done.'
exit 0
