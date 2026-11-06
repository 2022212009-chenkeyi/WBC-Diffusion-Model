#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=wbc-ddpm-smoke
#SBATCH --output=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-smoke-%j.out
#SBATCH --error=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-smoke-%j.err

set -euo pipefail

cd /home/keyi005/projects/WBCDiffusion/01_ddpm_reproduction
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-not-set}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-not-set}"
srun --ntasks=1 --gres=gpu:a6000:1 nvidia-smi

srun --ntasks=1 --gres=gpu:a6000:1 /projects/keyi005ssd/envs/wbcdiffusion/bin/python -m ddpm.train \
  --config configs/wbc_raabin_128_smoke.toml \
  --max-steps 2 \
  --subset-size 64
