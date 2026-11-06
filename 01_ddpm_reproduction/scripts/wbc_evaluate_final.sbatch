#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@300
#SBATCH --job-name=wbc-ddpm-evaluate
#SBATCH --output=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-evaluate-%j.out
#SBATCH --error=/projects/keyi005ssd/WBCDiffusion/results/wbc-ddpm/slurm-evaluate-%j.err

set -euo pipefail

cd /home/keyi005/projects/WBCDiffusion/01_ddpm_reproduction
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

sample_file=/projects/keyi005ssd/WBCDiffusion/generated_samples/wbc-raabin-256-baseline-v1/samples_ema_step0200000_n2000_seed2026.npy
test -s "$sample_file"
srun --ntasks=1 --gres=gpu:pro6000:1 nvidia-smi
srun --ntasks=1 --gres=gpu:pro6000:1 /projects/keyi005ssd/envs/wbcdiffusion/bin/python -m ddpm.evaluate_wbc \
  --config configs/wbc_raabin_256_baseline_v1.toml \
  --samples "$sample_file" \
  --batch-size 64 \
  --num-workers 4 \
  --feature-dim 768 \
  --kid-subsets 100 \
  --kid-subset-size 1000
