#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=wbc-ddpm-128
#SBATCH --output=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-128-%j.out
#SBATCH --error=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-128-%j.err

set -euo pipefail

cd /home/keyi005/projects/WBCDiffusion/01_ddpm_reproduction
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-not-set}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-not-set}"
srun --ntasks=1 --gres=gpu:a6000:1 nvidia-smi

srun --ntasks=1 --gres=gpu:a6000:1 /projects/keyi005ssd/envs/wbcdiffusion/bin/python -m ddpm.train \
  --config configs/wbc_raabin_128_pilot.toml \
  --max-steps 200
