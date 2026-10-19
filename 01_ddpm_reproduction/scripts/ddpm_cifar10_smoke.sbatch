#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=ddpm-c10-smoke
#SBATCH --output=/projects/keyi005ssd/WBCDiffusion/results/ddpm-cifar10-reference-smoke/slurm-%j.out

set -euo pipefail

cd /home/keyi005/projects/WBCDiffusion/01_ddpm_reproduction
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-not-set}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi

exec /projects/keyi005ssd/envs/wbcdiffusion/bin/python -m ddpm.train \
  --config configs/cifar10.toml \
  --max-steps 2
